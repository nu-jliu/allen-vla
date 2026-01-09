import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import pprint
import time
import logging

from argparse import ArgumentParser

from utils import setup_logging
from robot_utils import add_common_robot_args, initialize_robots

setup_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser(description="Teleoperation for SO101 robot")
    parser = add_common_robot_args(parser)
    parser.add_argument(
        "--frequency",
        type=float,
        default=20.0,
        help="Control frequency in Hz (default: 20.0)",
    )
    args = parser.parse_args()

    frequency = args.frequency

    leader, follower = initialize_robots(args, calibrate=True)

    # Record follower's initial joint positions right after connection
    initial_obs = follower.get_observation()
    initial_position = {
        "shoulder_pan.pos": initial_obs["shoulder_pan.pos"],
        "shoulder_lift.pos": initial_obs["shoulder_lift.pos"],
        "elbow_flex.pos": initial_obs["elbow_flex.pos"],
        "wrist_flex.pos": initial_obs["wrist_flex.pos"],
        "wrist_roll.pos": initial_obs["wrist_roll.pos"],
        "gripper.pos": initial_obs["gripper.pos"],
    }
    logger.info(f"Recorded follower initial position: {initial_position}")

    period = 1.0 / frequency
    while True:
        try:
            act = leader.get_action()
            follower.send_action(act)

            logger.info(f"Action: {pprint.pformat(act, indent=2)}")

            time.sleep(period)

        except KeyboardInterrupt:
            logger.info("Moving follower to initial position before disconnect...")
            follower.send_action(initial_position)
            # Wait for robot to reach initial position
            while True:
                obs = follower.get_observation()
                max_diff = max(
                    abs(obs[name] - initial_position[name])
                    for name in initial_position.keys()
                )
                if max_diff < 1.0:  # Within 1 degree tolerance
                    break
                time.sleep(0.05)
            leader.disconnect()
            follower.disconnect()
            return


if __name__ == "__main__":
    main()
