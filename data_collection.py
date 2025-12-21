import pprint
import uuid
import time
import logging
import threading
import cv2
import numpy as np

from argparse import ArgumentParser

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features

from utils import setup_logging
from robot_utils import add_common_robot_args, initialize_robots

setup_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser(description="Data collection for SO101 robot")
    parser = add_common_robot_args(parser)
    parser.add_argument(
        "--repo-id",
        type=str,
        default="jliu6718/lerobot-so101",
        help="Hugging Face repository ID (format: username/repo-name)",
    )
    parser.add_argument(
        "--hz",
        type=int,
        default=30,
        help="Control loop frequency in Hz (default: 30)",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push dataset to Hugging Face Hub after collection",
    )
    args = parser.parse_args()

    logger.info("Starting data collection with configuration:")
    logger.info(f"  Leader port: {args.leader_port}")
    logger.info(f"  Leader ID: {args.leader_id}")
    logger.info(f"  Follower port: {args.follower_port}")
    logger.info(f"  Follower ID: {args.follower_id}")
    logger.info(f"  Repo ID: {args.repo_id}")
    logger.info(f"  Control frequency: {args.hz} Hz")

    leader, follower = initialize_robots(args, calibrate=True)

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

    logger.info(f"Creating dataset: {args.repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=f"{args.repo_id}-{uuid.uuid4()}",
        fps=args.hz,
        features={**action_features, **obs_features},
        robot_type=follower.name,
        use_videos=True,
    )
    print(dataset.features)

    recording = False
    frame_count = 0
    running = True
    recording_lock = threading.Lock()

    def keyboard_thread() -> None:
        nonlocal recording, frame_count, running
        while running:
            try:
                logger.info("Press ENTER to start/stop recording...")
                input()
                with recording_lock:
                    logger.info("Handling keyboard ...")

                    if not recording:
                        recording = True
                        frame_count = 0
                        # Ensure episode buffer is properly initialized
                        dataset.episode_buffer = dataset.create_episode_buffer()
                        logger.info("=== RECORDING STARTED ===")
                    else:
                        recording = False
                        if frame_count > 0:
                            dataset.save_episode()
                            logger.info(
                                f"=== RECORDING STOPPED === (Saved {frame_count} frames)"
                            )
                        else:
                            logger.info(
                                "=== RECORDING STOPPED === (No frames captured)"
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

            with recording_lock:
                is_recording = recording

            if is_recording:
                obs = follower.get_observation()

                # logger.info(f"Observation: {pprint.pformat(obs, indent=2)}")
                # logger.info(f"Action: {pprint.pformat(action, indent=2)}")

                # Construct frame according to dataset features
                # Action: convert dict of motor positions to numpy array
                with recording_lock:
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
                    current_frame_count = frame_count

                # Log progress every Hz frames (every 1 second)
                if current_frame_count % args.hz == 0:
                    logger.debug(f"Recording: {current_frame_count} frames captured")

            time.sleep(period)

        except KeyboardInterrupt:
            logger.info("Shutdown signal received (Ctrl+C)")
            running = False
            logger.info("Disconnecting from leader robot...")
            leader.disconnect()
            logger.info("Disconnecting from follower robot...")
            follower.disconnect()
            dataset.finalize()
            if args.push:
                logger.info("Pushing dataset to Hugging Face Hub...")
                dataset.push_to_hub()
                logger.info("Dataset uploaded successfully. Exiting.")
            else:
                logger.info("Dataset saved locally. Exiting.")
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
