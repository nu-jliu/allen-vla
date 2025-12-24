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

    rest_action = {
        "shoulder_pan.pos": 1.7148014440433172,
        "shoulder_lift.pos": -99.82721382289417,
        "elbow_flex.pos": 100.0,
        "wrist_flex.pos": 76.12988152698551,
        "wrist_roll.pos": -1.7338217338217419,
        "gripper.pos": 1.766304347826087,
    }

    period = 1.0 / frequency
    while True:
        try:
            act = leader.get_action()
            follower.send_action(act)

            logger.info(f"Action: {pprint.pformat(act, indent=2)}")

            time.sleep(period)

        except KeyboardInterrupt:
            follower.send_action(rest_action)
            leader.disconnect()
            follower.disconnect()
            return


if __name__ == "__main__":
    main()
