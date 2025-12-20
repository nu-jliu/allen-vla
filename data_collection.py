import os
import pprint
import uuid
import time
import logging
import threading
import cv2
import numpy as np

from argparse import ArgumentParser
from pathlib import Path

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features

from utils import setup_logging

setup_logging()
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
    parser.add_argument(
        "--hz",
        type=int,
        default=30,
        help="Control loop frequency in Hz (default: 30)",
    )
    args = parser.parse_args()

    logger.info("Starting data collection with configuration:")
    logger.info(f"  Leader port: {args.leader_port}")
    logger.info(f"  Leader ID: {args.leader_id}")
    logger.info(f"  Follower port: {args.follower_port}")
    logger.info(f"  Follower ID: {args.follower_id}")
    logger.info(f"  HF username: {args.hf_username}")
    logger.info(f"  Repo ID: {args.repo_id}")
    logger.info(f"  Control frequency: {args.hz} Hz")

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

    cap = cv2.VideoCapture(0)
    # Get camera properties
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    obs_features = hw_to_dataset_features(
        follower.observation_features,
        prefix="observation",
    )
    action_features = hw_to_dataset_features(
        follower.action_features,
        prefix="action",
    )

    # Manually add camera feature since we're using cv2.VideoCapture directly
    obs_features["observation.images.cam_leader"] = {
        "dtype": "video",
        "shape": (cam_height, cam_width, 3),
        "names": ["height", "width", "channels"],
    }

    logger.info(f"Observation features: {follower.observation_features}")
    logger.info(f"Action features: {follower.action_features}")

    logger.info(f"Creating dataset: {args.hf_username}/{args.repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=f"{args.hf_username}/{args.repo_id}-{uuid.uuid4()}",
        fps=args.hz,
        features={**action_features, **obs_features},
        robot_type=follower.name,
    )
    print(dataset.features)

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
    running = True

    def keyboard_thread():
        nonlocal recording, frame_count, running
        while running:
            try:
                logger.info("Press ENTER to start/stop recording...")
                input()
                if not recording:
                    recording = True
                    frame_count = 0
                    logger.info("=== RECORDING STARTED ===")
                else:
                    recording = False
                    dataset.save_episode()
                    logger.info(
                        f"=== RECORDING STOPPED === (Saved {frame_count} frames)"
                    )
                    frame_count = 0
            except EOFError:
                break
            except Exception as e:
                logger.error(f"Error in keyboard thread: {e}")
                break

    input_thread = threading.Thread(target=keyboard_thread, daemon=True)
    input_thread.start()

    logger.info(
        "Press ENTER to start/stop recording. Press Ctrl+C to exit and upload dataset."
    )

    logger.info(f"Starting teleoperation loop at {args.hz} Hz...")
    period = 1.0 / args.hz

    while True:
        try:
            action = leader.get_action()
            follower.send_action(action)
            succ, image = cap.read()

            if not succ:
                logger.warning("Failed to get image, skipping this loop")

            if recording:
                obs = follower.get_observation()

                pprint.pprint(obs, indent=2)
                pprint.pprint(action, indent=2)

                # Construct frame according to dataset features
                # Action: convert dict of motor positions to numpy array
                action_array = np.array(
                    [action[name] for name in dataset.features["action"]["names"]],
                    dtype=np.float32,
                )

                # Observation state: convert dict of motor positions to numpy array
                obs_state_array = np.array(
                    [
                        obs[name]
                        for name in dataset.features["observation.state"]["names"]
                    ],
                    dtype=np.float32,
                )

                frame = {
                    "action": action_array,
                    "observation.state": obs_state_array,
                    "observation.images.cam_leader": image,
                    "task": "soarm_grasp",
                }

                dataset.add_frame(frame=frame)
                frame_count += 1

                # Log progress every Hz frames (every 1 second)
                if frame_count % args.hz == 0:
                    logger.debug(f"Recording: {frame_count} frames captured")

            time.sleep(period)

        except KeyboardInterrupt:
            logger.info("Shutdown signal received (Ctrl+C)")
            running = False
            logger.info("Disconnecting from leader robot...")
            leader.disconnect()
            logger.info("Disconnecting from follower robot...")
            follower.disconnect()
            logger.info("Pushing dataset to Hugging Face Hub...")
            dataset.finalize()
            # dataset.push_to_hub()
            logger.info("Dataset uploaded successfully. Exiting.")
            return
        except Exception as e:
            logger.error(f"Error in teleoperation loop: {e}", exc_info=True)
            logger.info("Attempting cleanup...")
            running = False
            try:
                leader.disconnect()
                follower.disconnect()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")
            raise


if __name__ == "__main__":
    main()
