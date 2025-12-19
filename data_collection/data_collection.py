import os
import json
import time
import logging
from pynput import keyboard
from argparse import ArgumentParser
from pathlib import Path

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

HOME = os.getenv("HOME")
CALIBRATION_DIR = os.path.join(
    HOME,
    ".cache",
    "huggingface",
    "lerobot",
    "calibration",
)


def main():
    parser = ArgumentParser(description="Teleoperation for SO101 robot")
    parser.add_argument(
        "--leader-port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for the leader arm (default: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--leader-id",
        type=str,
        default="my_leader",
        help="ID for the leader arm (default: my_leader)",
    )
    parser.add_argument(
        "--follower-port",
        type=str,
        default="/dev/ttyACM1",
        help="Serial port for the follower arm (default: /dev/ttyACM1)",
    )
    parser.add_argument(
        "--follower-id",
        type=str,
        default="my_follower",
        help="ID for the follower arm (default: my_follower)",
    )
    parser.add_argument(
        "--hf-username",
        type=str,
        default="jliu6718",
        help="Hugging Face username",
    )
    parser.add_argument(
        "--repo-id", type=str, default="lerobot-so101", help="Name of the dataset repo"
    )
    args = parser.parse_args()

    logger.info("Starting data collection with configuration:")
    logger.info(f"  Leader port: {args.leader_port}")
    logger.info(f"  Leader ID: {args.leader_id}")
    logger.info(f"  Follower port: {args.follower_port}")
    logger.info(f"  Follower ID: {args.follower_id}")
    logger.info(f"  HF username: {args.hf_username}")
    logger.info(f"  Repo ID: {args.repo_id}")

    leader_config = SO101LeaderConfig(
        port=args.leader_port,
        use_degrees=False,
        id=args.leader_id,
        calibration_dir=Path(
            os.path.join(
                CALIBRATION_DIR,
                "teleoperators",
                "so101_leader",
            )
        ),
    )
    follower_config = SO101FollowerConfig(
        port=args.follower_port,
        disable_torque_on_disconnect=True,
        use_degrees=False,
        id=args.follower_id,
        calibration_dir=Path(
            os.path.join(
                CALIBRATION_DIR,
                "robots",
                "so101_follower",
            )
        ),
    )

    logger.info("Initializing leader and follower robots...")
    leader = SO101Leader(config=leader_config)
    follower = SO101Follower(config=follower_config)

    obs_features = hw_to_dataset_features(
        follower.observation_features,
        prefix="observation",
    )
    action_features = hw_to_dataset_features(
        follower.action_features,
        prefix="action",
    )

    logger.info(f"Observation features: {follower.observation_features}")
    logger.info(f"Action features: {follower.action_features}")

    logger.info(f"Creating dataset: {args.hf_username}/{args.repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=f"{args.hf_username}/{args.repo_id}",
        fps=30,
        features={**action_features, **obs_features},
        robot_type=follower.name,
    )

    logger.info("Connecting to leader robot and calibrating...")
    leader.connect(calibrate=True)
    logger.info("Leader robot connected successfully")

    logger.info("Connecting to follower robot and calibrating...")
    follower.connect(calibrate=True)
    logger.info("Follower robot connected successfully")

    # Disable torque on leader so it can be moved freely by hand
    logger.info("Disabling torque on leader arm for manual control")
    leader.bus.disable_torque()

    recording = False
    frame_count = 0

    def on_enter():
        nonlocal recording, frame_count
        if not recording:
            recording = True
            frame_count = 0
            logger.info("=== RECORDING STARTED ===")
        else:
            recording = False
            dataset.save_episode()
            logger.info(f"=== RECORDING STOPPED === (Saved {frame_count} frames)")
            frame_count = 0

    keyboard.on_press_key("enter", lambda _: on_enter())
    logger.info("Press ENTER to start/stop recording. Press Ctrl+C to exit and upload dataset.")

    logger.info("Starting teleoperation loop at 30 Hz...")

    while True:
        try:
            obs = leader.get_observation()
            follower.send_action(obs)

            if recording:
                dataset.add_frame()
                frame_count += 1

                # Log progress every 30 frames (every 1 second at 30 Hz)
                if frame_count % 30 == 0:
                    logger.debug(f"Recording: {frame_count} frames captured")

            time.sleep(1.0 / 30.0)

        except KeyboardInterrupt:
            logger.info("Shutdown signal received (Ctrl+C)")
            logger.info("Disconnecting from leader robot...")
            leader.disconnect()
            logger.info("Disconnecting from follower robot...")
            follower.disconnect()
            logger.info("Pushing dataset to Hugging Face Hub...")
            dataset.push_to_hub()
            logger.info("Dataset uploaded successfully. Exiting.")
            return
        except Exception as e:
            logger.error(f"Error in teleoperation loop: {e}", exc_info=True)
            logger.info("Attempting cleanup...")
            try:
                leader.disconnect()
                follower.disconnect()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")
            raise


if __name__ == "__main__":
    main()
