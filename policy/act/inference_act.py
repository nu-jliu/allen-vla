#!/usr/bin/env python
"""
Inference/Evaluation script for ACT policy on SO101 robot using LeRobot's record infrastructure.

This script uses lerobot-record with a trained policy to evaluate the model on real hardware.
The policy controls the robot autonomously and the results are saved as a dataset for analysis.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter

from lerobot.scripts.lerobot_record import record, RecordConfig, DatasetRecordConfig
from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from uuid import uuid4
from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def parse_args() -> Namespace:
    """Parse command line arguments for inference.

    :return: Parsed command line arguments
    :rtype: argparse.Namespace
    """
    parser = ArgumentParser(
        description="Run inference with trained ACT policy on SO101 robot",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with local checkpoint
  python policy/act/inference_act.py \\
    --checkpoint ./outputs/act_training/pretrained_model \\
    --robot-port /dev/ttyACM0 \\
    --camera-index 0 \\
    --num-episodes 10 \\
    --repo-id my_username/eval_results

  # Evaluation with HuggingFace Hub model
  python policy/act/inference_act.py \\
    --checkpoint username/act_policy \\
    --robot-port /dev/ttyACM0 \\
    --camera-index 0 \\
    --num-episodes 5 \\
    --repo-id username/eval_results \\
    --push-to-hub
        """,
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained policy checkpoint or HuggingFace repo ID",
    )
    required.add_argument(
        "--robot-port",
        type=str,
        required=True,
        help="Robot port (e.g., /dev/ttyACM0)",
    )
    required.add_argument(
        "--camera-index",
        type=str,
        required=True,
        help="Camera index or path (e.g., '0' for /dev/video0)",
    )
    required.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Dataset repo ID for saving evaluation results (e.g., username/eval_results)",
    )

    # Robot configuration
    robot = parser.add_argument_group("robot configuration")
    robot.add_argument(
        "--robot-type",
        type=str,
        default="so101_follower",
        help="Robot type (default: so101_follower)",
    )
    robot.add_argument(
        "--robot-id",
        type=str,
        default="eval_robot",
        help="Robot ID (default: eval_robot)",
    )

    # Camera configuration
    camera = parser.add_argument_group("camera configuration")
    camera.add_argument(
        "--camera-name",
        type=str,
        default="front",
        help="Camera name in config (default: front)",
    )
    camera.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Camera width (default: 640)",
    )
    camera.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Camera height (default: 480)",
    )
    camera.add_argument(
        "--camera-fps",
        type=int,
        default=30,
        help="Camera FPS (default: 30)",
    )

    # Evaluation parameters
    eval_group = parser.add_argument_group("evaluation parameters")
    eval_group.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate (default: 10)",
    )
    eval_group.add_argument(
        "--task-description",
        type=str,
        default="Policy evaluation",
        help="Task description for this evaluation run",
    )
    eval_group.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Control frequency in Hz (default: 30)",
    )
    eval_group.add_argument(
        "--episode-time",
        type=int,
        default=60,
        help="Maximum time per episode in seconds (default: 60)",
    )
    eval_group.add_argument(
        "--reset-time",
        type=int,
        default=60,
        help="Time for resetting between episodes in seconds (default: 60)",
    )

    # Data saving options
    save_group = parser.add_argument_group("data saving")
    save_group.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory to save dataset locally (default: ~/.cache/lerobot)",
    )
    save_group.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push evaluation dataset to HuggingFace Hub",
    )
    save_group.add_argument(
        "--video",
        action="store_true",
        default=True,
        help="Encode videos in the dataset (default: enabled)",
    )

    # Display and debugging
    display = parser.add_argument_group("display and debugging")
    display.add_argument(
        "--display-data",
        action="store_true",
        default=True,
        help="Display camera feed and robot state during evaluation (default: enabled)",
    )
    display.add_argument(
        "--no-display",
        dest="display_data",
        action="store_false",
        help="Disable display",
    )
    display.add_argument(
        "--play-sounds",
        action="store_true",
        default=False,
        help="Enable vocal synthesis for events",
    )

    return parser.parse_args()


def create_record_config(args: Namespace) -> RecordConfig:
    """Create RecordConfig from parsed arguments.

    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    :return: Recording configuration
    :rtype: RecordConfig
    """
    logger.info("Building evaluation configuration...")

    # Extract args to local variables
    checkpoint = args.checkpoint
    robot_port = args.robot_port
    robot_id = args.robot_id
    robot_type = args.robot_type
    camera_index_str = args.camera_index
    camera_name = args.camera_name
    camera_width = args.camera_width
    camera_height = args.camera_height
    camera_fps = args.camera_fps
    repo_id = args.repo_id
    task_description = args.task_description
    root = args.root
    fps = args.fps
    episode_time = args.episode_time
    reset_time = args.reset_time
    num_episodes = args.num_episodes
    video = args.video
    push_to_hub = args.push_to_hub
    display_data = args.display_data
    play_sounds = args.play_sounds

    # Create camera configuration
    # Convert camera_index to int if numeric, otherwise keep as path string
    camera_index = (
        int(camera_index_str) if camera_index_str.isdigit() else camera_index_str
    )
    cameras = {
        camera_name: OpenCVCameraConfig(
            index_or_path=camera_index,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
        )
    }

    # Create robot configuration
    robot_config = SO101FollowerConfig(
        port=robot_port,
        id=robot_id,
        cameras=cameras,
    )

    # Create dataset configuration
    dataset_config = DatasetRecordConfig(
        repo_id=f"{repo_id}-{uuid4()}",
        single_task=task_description,
        root=root,
        fps=fps,
        episode_time_s=episode_time,
        reset_time_s=reset_time,
        num_episodes=num_episodes,
        video=video,
        push_to_hub=push_to_hub,
    )

    # Create policy configuration by loading from pretrained checkpoint
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    policy_config.pretrained_path = checkpoint

    # Create main record configuration
    config = RecordConfig(
        robot=robot_config,
        dataset=dataset_config,
        policy=policy_config,
        teleop=None,  # No teleoperation, only policy control
        display_data=display_data,
        play_sounds=play_sounds,
        resume=False,
    )

    logger.info("Configuration created successfully")
    logger.info(f"  Policy: {checkpoint}")
    logger.info(f"  Robot: {robot_type} @ {robot_port}")
    logger.info(f"  Camera: {camera_name} (index: {camera_index_str})")
    logger.info(f"  Episodes: {num_episodes}")
    logger.info(f"  Dataset: {repo_id}")
    logger.info(f"  Push to Hub: {push_to_hub}")

    return config


def main():
    """Main inference entry point."""
    args = parse_args()

    # Extract args to local variables
    checkpoint = args.checkpoint
    robot_type = args.robot_type
    robot_port = args.robot_port
    camera_name = args.camera_name
    camera_index = args.camera_index
    num_episodes = args.num_episodes
    fps = args.fps
    task_description = args.task_description
    push_to_hub = args.push_to_hub
    repo_id = args.repo_id

    logger.info("=" * 60)
    logger.info("ACT Policy Inference for SO101 Robot")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Checkpoint: {checkpoint}")
    logger.info(f"  Robot: {robot_type}:{robot_port}")
    logger.info(f"  Camera: {camera_name} @ index {camera_index}")
    logger.info(f"  Episodes: {num_episodes}")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Task: {task_description}")
    logger.info("")

    try:
        # Create configuration
        config = create_record_config(args)

        # Run evaluation using lerobot-record
        logger.info("Starting policy evaluation...")
        logger.info("=" * 60)
        logger.info("")

        dataset = record(config)

        logger.info("")
        logger.info("=" * 60)
        logger.info("Evaluation completed successfully!")
        logger.info(f"Dataset saved: {dataset.repo_id}")
        logger.info(f"Total episodes: {dataset.num_episodes}")
        logger.info(f"Total frames: {dataset.num_frames}")
        if push_to_hub:
            logger.info(
                f"Dataset pushed to: https://huggingface.co/datasets/{repo_id}"
            )
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("=" * 60)
        logger.error("Evaluation failed with error:")
        logger.error(str(e), exc_info=True)
        logger.error("=" * 60)
        raise


if __name__ == "__main__":
    main()
