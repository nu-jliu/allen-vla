#!/usr/bin/env python
"""
Training script for Diffusion policy on SO101 robot dataset.

This script uses LeRobot's full training infrastructure including Accelerate for
distributed training, Weights & Biases for experiment tracking, and comprehensive
checkpoint management.

FEATURE SPECIFICATION:
---------------------------------------------------
Features are automatically extracted from the dataset by the LeRobot training
pipeline. The dataset should be collected using collect.py, which uses the
centralized build_dataset_features_for_so101() function from utils.py.

EXPECTED FEATURES (from data collection):
  INPUT FEATURES:
    - observation.state: [6] float32
        Joint positions: shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
                         wrist_flex.pos, wrist_roll.pos, gripper.pos
    - observation.images.front: [3, 480, 640] video
        Front camera RGB image (default resolution, configurable in collect.py)

  OUTPUT FEATURES:
    - action: [6] float32
        Target joint positions: shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
                                wrist_flex.pos, wrist_roll.pos, gripper.pos
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import logging
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter

from tqdm import tqdm
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.scripts.lerobot_train import train

from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def parse_args() -> Namespace:
    """Parse command line arguments with sensible defaults.

    :return: Parsed command line arguments
    :rtype: argparse.Namespace
    """
    parser = ArgumentParser(
        description="Train Diffusion policy on SO101 robot dataset",
        formatter_class=RawDescriptionHelpFormatter,
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save checkpoints and logs",
    )

    # Dataset source (mutually exclusive)
    dataset_group = parser.add_argument_group(
        "dataset source (choose one)",
        "Specify either --repo-id to load from HuggingFace Hub, "
        "or --local-dir to load from a local directory",
    )
    dataset_source = dataset_group.add_mutually_exclusive_group(required=True)
    dataset_source.add_argument(
        "--repo-id",
        type=str,
        help="HuggingFace Hub dataset repo ID (e.g., username/policy-robot-task)",
    )
    dataset_source.add_argument(
        "--local-dir",
        type=Path,
        help="Path to local dataset directory",
    )
    dataset_group.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Dataset revision/branch to use (default: main)",
    )

    # Model push configuration (required when using --local-dir with --push)
    push_group = parser.add_argument_group(
        "model push configuration",
        "Required when using --local-dir with --push to specify the model repo ID",
    )
    push_group.add_argument(
        "--username",
        type=str,
        help="Hugging Face username (required with --local-dir and --push)",
    )
    push_group.add_argument(
        "--policy-type",
        type=str,
        help="Policy type e.g. act, diffusion (required with --local-dir and --push)",
    )
    push_group.add_argument(
        "--robot-type",
        type=str,
        help="Robot type e.g. so101 (required with --local-dir and --push)",
    )
    push_group.add_argument(
        "--task",
        type=str,
        help="Task name for the model repo (required with --local-dir and --push)",
    )

    # Training hyperparameters
    training = parser.add_argument_group("training hyperparameters")
    training.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    training.add_argument(
        "--steps",
        type=int,
        default=100_000,
        help="Total training steps (default: 100,000)",
    )
    training.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )
    training.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Logging and checkpointing
    logging_group = parser.add_argument_group("logging and checkpointing")
    logging_group.add_argument(
        "--log-freq",
        type=int,
        default=250,
        help="Log training metrics every N steps (default: 250)",
    )
    logging_group.add_argument(
        "--save-freq",
        type=int,
        default=5000,
        help="Save checkpoint every N steps (default: 5000)",
    )
    logging_group.add_argument(
        "--progress-bar",
        action="store_true",
        default=True,
        help="Show tqdm progress bar during training (default: enabled)",
    )
    logging_group.add_argument(
        "--no-progress-bar",
        dest="progress_bar",
        action="store_false",
        help="Disable tqdm progress bar",
    )

    # Diffusion-specific hyperparameters
    diffusion_group = parser.add_argument_group("Diffusion hyperparameters")
    diffusion_group.add_argument(
        "--horizon",
        type=int,
        default=16,
        help="Diffusion model action prediction horizon (default: 16)",
    )
    diffusion_group.add_argument(
        "--n-action-steps",
        type=int,
        default=8,
        help="Number of action steps to execute per query (default: 8)",
    )
    diffusion_group.add_argument(
        "--n-obs-steps",
        type=int,
        default=2,
        help="Number of observation steps for context (default: 2)",
    )
    diffusion_group.add_argument(
        "--num-train-timesteps",
        type=int,
        default=100,
        help="Number of diffusion training timesteps (default: 100)",
    )
    diffusion_group.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Number of diffusion inference steps (default: same as train timesteps)",
    )
    diffusion_group.add_argument(
        "--noise-scheduler",
        type=str,
        default="DDPM",
        choices=["DDPM", "DDIM"],
        help="Noise scheduler type (default: DDPM)",
    )
    diffusion_group.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    diffusion_group.add_argument(
        "--vision-backbone",
        type=str,
        default="resnet18",
        help="Vision backbone for image encoding (default: resnet18)",
    )
    diffusion_group.add_argument(
        "--crop-shape",
        type=int,
        nargs=2,
        default=[84, 84],
        help="Image crop shape for preprocessing (default: 84 84)",
    )

    # Advanced options
    advanced = parser.add_argument_group("advanced options")
    advanced.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint in output-dir",
    )
    advanced.add_argument(
        "--push",
        action="store_true",
        help="Push checkpoints to HuggingFace Hub",
    )

    return parser.parse_args()


def build_training_config(args: Namespace) -> TrainPipelineConfig:
    """Build TrainPipelineConfig from parsed arguments.

    This function creates the complete training configuration by assembling:

    - DiffusionConfig: Policy architecture and hyperparameters
    - DatasetConfig: Dataset loading configuration
    - WandBConfig: Experiment tracking configuration
    - TrainPipelineConfig: Main training pipeline configuration

    Features are automatically extracted from the dataset by the LeRobot training
    pipeline using dataset_to_policy_features(). We only need to specify
    Diffusion-specific hyperparameters here.

    Expected features (from dataset collected with collect.py):
      INPUT: observation.state [6], observation.images.front [3, H, W]
      OUTPUT: action [6]

    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    :return: Complete training configuration
    :rtype: TrainPipelineConfig
    """
    logger.info("Building training configuration...")

    # Extract args to local variables
    repo_id = args.repo_id
    local_dir = args.local_dir
    username = args.username
    policy_type = args.policy_type
    robot_type = args.robot_type
    task = args.task
    output_dir = args.output_dir
    batch_size = args.batch_size
    steps = args.steps
    num_workers = args.num_workers
    seed = args.seed
    log_freq = args.log_freq
    save_freq = args.save_freq
    progress_bar = args.progress_bar
    horizon = args.horizon
    n_action_steps = args.n_action_steps
    n_obs_steps = args.n_obs_steps
    num_train_timesteps = args.num_train_timesteps
    num_inference_steps = args.num_inference_steps
    noise_scheduler = args.noise_scheduler
    lr = args.lr
    vision_backbone = args.vision_backbone
    crop_shape = tuple(args.crop_shape)
    resume = args.resume
    push = args.push
    revision = args.revision

    # Determine policy_repo_id for pushing
    if push:
        if repo_id is not None:
            # Using HuggingFace dataset, use same repo_id for model
            policy_repo_id = repo_id
        elif all([username, policy_type, robot_type, task]):
            # Using local dataset with push, construct repo_id from components
            policy_repo_id = f"{username}/{policy_type}-{robot_type}-{task}"
        else:
            raise ValueError(
                "When using --push with --local-dir, you must also specify --username, --policy-type, --robot-type, and --task"
            )
    else:
        policy_repo_id = None

    # Determine dataset source
    if local_dir is not None:
        # Local dataset mode
        dataset_source = str(local_dir)
        dataset_root = str(local_dir.parent)
        logger.info(f"Using local dataset: {local_dir}")
    else:
        # HuggingFace Hub mode
        dataset_source = repo_id
        dataset_root = None
        logger.info(f"Using HuggingFace Hub dataset: {repo_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Diffusion Policy Configuration
    # Features will be automatically populated from the dataset by the training pipeline
    # We only specify Diffusion-specific hyperparameters here
    diffusion_config = DiffusionConfig(
        device=device,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_train_timesteps=num_train_timesteps,
        num_inference_steps=num_inference_steps,
        noise_scheduler_type=noise_scheduler,
        optimizer_lr=lr,
        vision_backbone=vision_backbone,
        crop_shape=crop_shape,
        push_to_hub=push,
        repo_id=policy_repo_id if push else None,
        # All other parameters use DiffusionConfig defaults:
        # - down_dims=(512, 1024, 2048)
        # - kernel_size=5
        # - n_groups=8
        # - diffusion_step_embed_dim=128
        # - use_film_scale_modulation=True
        # - beta_schedule="squaredcos_cap_v2"
        # - prediction_type="epsilon"
        # - clip_sample=True
        # - optimizer_betas=(0.95, 0.999)
        # - optimizer_weight_decay=1e-6
        # - scheduler_name="cosine"
        # - scheduler_warmup_steps=500
    )

    # Dataset Configuration
    dataset_config = DatasetConfig(
        repo_id=dataset_source,
        root=dataset_root,
        revision=revision,
    )

    # Weights & Biases Configuration - Disabled
    wandb_config = WandBConfig(
        enable=False,
    )

    # Main Training Pipeline Configuration
    train_config = TrainPipelineConfig(
        dataset=dataset_config,
        policy=diffusion_config,
        output_dir=output_dir,
        resume=resume,
        seed=seed,
        num_workers=num_workers,
        batch_size=batch_size,
        steps=steps,
        eval_freq=0,  # No evaluation (no simulation environment)
        log_freq=log_freq,
        save_checkpoint=True,
        save_freq=save_freq,
        wandb=wandb_config,
        env=None,  # No simulation environment
    )

    # Log configuration details for reproducibility
    logger.info("Configuration built successfully:")
    if local_dir is not None:
        logger.info(f"  Dataset source: local ({local_dir})")
    else:
        logger.info(f"  Dataset source: HuggingFace Hub ({repo_id})")
        logger.info(f"  Dataset revision: {revision}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Training steps: {steps:,}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Num workers: {num_workers}")
    logger.info(f"  Seed: {seed}")
    logger.info("")
    logger.info("Diffusion Configuration:")
    logger.info(f"  Horizon: {horizon}")
    logger.info(f"  Action steps: {n_action_steps}")
    logger.info(f"  Observation steps: {n_obs_steps}")
    logger.info(f"  Train timesteps: {num_train_timesteps}")
    logger.info(f"  Inference timesteps: {num_inference_steps or num_train_timesteps}")
    logger.info(f"  Noise scheduler: {noise_scheduler}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Vision backbone: {vision_backbone}")
    logger.info(f"  Crop shape: {crop_shape}")
    logger.info("")
    logger.info("Features:")
    logger.info(
        "  Input/output features will be automatically extracted from the dataset"
    )
    logger.info("  Expected features (from collect.py with default 640x480 camera):")
    logger.info(
        "    - INPUT: observation.state [6], observation.images.front [3, H, W]"
    )
    logger.info("    - OUTPUT: action [6]")
    logger.info("")
    logger.info("Logging:")
    logger.info(f"  Log frequency: {log_freq} steps")
    logger.info(f"  Save frequency: {save_freq} steps")
    logger.info(f"  Progress bar: {'enabled' if progress_bar else 'disabled'}")
    logger.info(f"  Push to Hub: {'enabled' if push else 'disabled'}")
    if push:
        logger.info(f"  Policy repo ID: {policy_repo_id}")
    logger.info(f"  Wandb: disabled")

    return train_config


def train_with_progress(
    cfg: TrainPipelineConfig,
    total_steps: int,
    show_progress: bool = True,
) -> None:
    """Wrapper around LeRobot's train() function with tqdm progress tracking.

    :param cfg: Training configuration
    :type cfg: TrainPipelineConfig
    :param total_steps: Total number of training steps
    :type total_steps: int
    :param show_progress: Whether to show tqdm progress bar
    :type show_progress: bool
    """
    import os
    import sys

    if show_progress:
        # Enable tqdm for better progress visibility
        os.environ["TQDM_DISABLE"] = "0"

        logger.info("Training with progress bar enabled")
        logger.info(f"Progress will be tracked over {total_steps:,} steps")
        logger.info("")

        # Create a progress bar for overall training
        with tqdm(
            total=total_steps,
            desc="Overall Training Progress",
            unit="step",
            file=sys.stdout,
            dynamic_ncols=True,
            colour="green",
        ) as pbar:
            pbar.set_postfix_str("Initializing training...")

            # Call the actual training function
            train(cfg)

            # Complete the progress bar
            pbar.n = total_steps
            pbar.refresh()
    else:
        # Disable tqdm if requested
        os.environ["TQDM_DISABLE"] = "1"
        logger.info("Training with progress bar disabled")
        train(cfg)


def main() -> None:
    """Main training entry point.

    Flow:

    1. Parse command line arguments
    2. Log startup information and configuration
    3. Build training configuration
    4. Validate configuration
    5. Delegate to LeRobot's train() function with progress tracking
    6. Handle errors gracefully with proper logging

    :raises Exception: If training fails
    """
    logger.info("=" * 60)
    logger.info("Diffusion Policy Training for SO101 Robot")
    logger.info("=" * 60)

    # Parse arguments
    args = parse_args()

    # Extract args to local variables
    repo_id = args.repo_id
    local_dir = args.local_dir
    output_dir = args.output_dir
    steps = args.steps
    batch_size = args.batch_size
    progress_bar = args.progress_bar

    logger.info("")
    logger.info("Starting training with configuration:")
    if local_dir is not None:
        logger.info(f"  Dataset: {local_dir} (local)")
    else:
        logger.info(f"  Dataset: {repo_id} (HuggingFace Hub)")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Steps: {steps:,}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info("")

    try:
        # Build configuration
        cfg = build_training_config(args)

        # Validate configuration
        logger.info("Validating configuration...")
        cfg.validate()
        logger.info("Configuration validated successfully")
        logger.info("")

        # Start training with progress tracking
        logger.info("Starting training loop...")
        logger.info("(Training will be handled by LeRobot's train() function)")
        logger.info("")
        train_with_progress(cfg, steps, show_progress=progress_bar)

        logger.info("")
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Checkpoints saved to: {output_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("Training failed with error:")
        logger.error(str(e), exc_info=True)
        logger.error("=" * 60)
        raise


if __name__ == "__main__":
    main()
