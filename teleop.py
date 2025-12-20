import os
import cv2
import pprint
import time
import logging

from argparse import ArgumentParser
from pathlib import Path

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

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
        "--frequency",
        type=float,
        default=20.0,
        help="Control frequency in Hz (default: 20.0)",
    )
    args = parser.parse_args()

    cap_leader = cv2.VideoCapture(0)
    cap_follower = cv2.VideoCapture(2)

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

    leader = SO101Leader(config=leader_config)
    follower = SO101Follower(config=follower_config)

    leader.connect(calibrate=True)
    follower.connect(calibrate=True)

    # Disable torque on leader so it can be moved freely by hand
    leader.bus.disable_torque()

    rest_action = {
        "shoulder_pan.pos": 1.7148014440433172,
        "shoulder_lift.pos": -99.82721382289417,
        "elbow_flex.pos": 100.0,
        "wrist_flex.pos": 76.12988152698551,
        "wrist_roll.pos": -1.7338217338217419,
        "gripper.pos": 1.766304347826087,
    }

    sleep_time = 1.0 / args.frequency
    while True:
        try:
            obs = leader.get_action()
            follower.send_action(obs)

            succ_leader, img_leader = cap_leader.read()
            succ_follower, img_follower = cap_follower.read()

            if not succ_follower or not succ_leader:
                continue

            print(img_leader.shape)
            print(img_follower.shape)

            pprint.pprint(obs, indent=2)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            follower.send_action(rest_action)
            leader.disconnect()
            follower.disconnect()
            return


if __name__ == "__main__":
    main()
