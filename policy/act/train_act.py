#!/usr/bin/env python
"""
Training script for ACT (Action Chunking Transformer) policy on SO101 robot dataset.

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
from lerobot.policies.act.configuration_act import ACTConfig
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
        description="Train ACT policy on SO101 robot dataset",
        formatter_class=RawDescriptionHelpFormatter,
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace dataset repo ID (e.g., jliu6718/lerobot-so101-abc123)",
    )
    required.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save checkpoints and logs",
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
        default=1000,
        help="Random seed for reproducibility (default: 1000)",
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

    # ACT-specific hyperparameters
    act_group = parser.add_argument_group("ACT hyperparameters")
    act_group.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Action prediction chunk size (default: 100)",
    )
    act_group.add_argument(
        "--n-action-steps",
        type=int,
        default=100,
        help="Number of action steps to execute per query (default: 100)",
    )
    act_group.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    act_group.add_argument(
        "--kl-weight",
        type=float,
        default=10.0,
        help="KL divergence loss weight for VAE (default: 10.0)",
    )
    act_group.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate in transformer (default: 0.1)",
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
    advanced.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Local dataset cache directory (optional)",
    )

    return parser.parse_args()


def build_training_config(args: Namespace) -> TrainPipelineConfig:
    """Build TrainPipelineConfig from parsed arguments.

    This function creates the complete training configuration by assembling:

    - ACTConfig: Policy architecture and hyperparameters
    - DatasetConfig: Dataset loading configuration
    - WandBConfig: Experiment tracking configuration
    - TrainPipelineConfig: Main training pipeline configuration

    Features are automatically extracted from the dataset by the LeRobot training
    pipeline using dataset_to_policy_features(). We only need to specify
    ACT-specific hyperparameters here.

    Expected features (from dataset collected with collect.py):
      INPUT: observation.state [6], observation.images.front [3, H, W]
      OUTPUT: action [6]

    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    :return: Complete training configuration
    :rtype: TrainPipelineConfig
    """
    logger.info("Building training configuration...")

    # ACT Policy Configuration
    # Features will be automatically populated from the dataset by the training pipeline
    # We only specify ACT-specific hyperparameters here
    act_config = ACTConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        n_obs_steps=1,  # ACT default
        optimizer_lr=args.lr,
        kl_weight=args.kl_weight,
        dropout=args.dropout,
        push_to_hub=args.push,
        # All other parameters use ACTConfig defaults:
        # - vision_backbone="resnet18"
        # - pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1"
        # - dim_model=512, n_heads=8, dim_feedforward=3200
        # - n_encoder_layers=4, n_decoder_layers=1
        # - use_vae=True, latent_dim=32
        # - optimizer_weight_decay=1e-4
        # - optimizer_lr_backbone=1e-5
        # - normalization_mapping uses MEAN_STD for VISUAL, STATE, and ACTION
    )

    # Dataset Configuration
    dataset_config = DatasetConfig(
        repo_id=args.repo_id,
        root=args.dataset_root,
    )

    # Weights & Biases Configuration - Disabled
    wandb_config = WandBConfig(
        enable=False,
    )

    # Main Training Pipeline Configuration
    # Note: Set progress_bar=True to enable tqdm progress bars during training
    train_config = TrainPipelineConfig(
        dataset=dataset_config,
        policy=act_config,
        output_dir=args.output_dir,
        resume=args.resume,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=0,  # No evaluation (no simulation environment)
        log_freq=args.log_freq,
        save_checkpoint=True,
        save_freq=args.save_freq,
        wandb=wandb_config,
        env=None,  # No simulation environment
    )

    # Log configuration details for reproducibility
    logger.info("Configuration built successfully:")
    logger.info(f"  Dataset repo: {args.repo_id}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Training steps: {args.steps:,}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Num workers: {args.num_workers}")
    logger.info(f"  Seed: {args.seed}")
    logger.info("")
    logger.info("ACT Configuration:")
    logger.info(f"  Chunk size: {args.chunk_size}")
    logger.info(f"  Action steps: {args.n_action_steps}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  KL weight: {args.kl_weight}")
    logger.info(f"  Dropout: {args.dropout}")
    logger.info("")
    logger.info("Features:")
    logger.info("  Input/output features will be automatically extracted from the dataset")
    logger.info("  Expected features (from collect.py with default 640x480 camera):")
    logger.info("    - INPUT: observation.state [6], observation.images.front [3, H, W]")
    logger.info("    - OUTPUT: action [6]")
    logger.info("")
    logger.info("Logging:")
    logger.info(f"  Log frequency: {args.log_freq} steps")
    logger.info(f"  Save frequency: {args.save_freq} steps")
    logger.info(f"  Progress bar: {'enabled' if args.progress_bar else 'disabled'}")
    logger.info(f"  Push to Hub: {'enabled' if args.push else 'disabled'}")
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
        # LeRobot's training loop likely already has tqdm, but we ensure it's visible
        os.environ["TQDM_DISABLE"] = "0"

        logger.info("Training with progress bar enabled")
        logger.info(f"Progress will be tracked over {total_steps:,} steps")
        logger.info("")

        # Create a progress bar for overall training
        # Note: LeRobot's internal training loop will also show its own progress
        with tqdm(
            total=total_steps,
            desc="Overall Training Progress",
            unit="step",
            file=sys.stdout,
            dynamic_ncols=True,
            colour="green",
        ) as pbar:
            # Store original stdout to restore later
            pbar.set_postfix_str("Initializing training...")

            # Call the actual training function
            # LeRobot's train() will handle the actual progress updates
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
    logger.info("ACT Policy Training for SO101 Robot")
    logger.info("=" * 60)

    # Parse arguments
    args = parse_args()

    logger.info("")
    logger.info("Starting training with configuration:")
    logger.info(f"  Dataset: {args.repo_id}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Steps: {args.steps:,}")
    logger.info(f"  Batch size: {args.batch_size}")
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
        train_with_progress(cfg, args.steps, show_progress=args.progress_bar)

        logger.info("")
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Checkpoints saved to: {args.output_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("Training failed with error:")
        logger.error(str(e), exc_info=True)
        logger.error("=" * 60)
        raise


if __name__ == "__main__":
    main()
