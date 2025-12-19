import os
import json
import time
from argparse import ArgumentParser
from pathlib import Path

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

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
        "--leader.port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for the leader arm (default: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--leader.id",
        type=str,
        default="my_leader",
        help="ID for the leader arm (default: my_leader)",
    )
    parser.add_argument(
        "--follower.port",
        type=str,
        default="/dev/ttyACM1",
        help="Serial port for the follower arm (default: /dev/ttyACM1)",
    )
    parser.add_argument(
        "--follower.id",
        type=str,
        default="my_follower",
        help="ID for the follower arm (default: my_follower)",
    )
    args = parser.parse_args()

    leader_config = SO101FollowerConfig(
        port=args.leader_port,
        disable_torque_on_disconnect=True,
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

    leader = SO101Follower(config=leader_config)
    follower = SO101Follower(config=follower_config)

    leader.connect(calibrate=True)
    follower.connect(calibrate=True)

    # Disable torque on leader so it can be moved freely by hand
    leader.bus.disable_torque()

    while True:
        try:
            obs = leader.get_observation()
            print(json.loads(json.dumps(obs)))
            follower.send_action(obs)
            time.sleep(0.05)

        except KeyboardInterrupt:
            leader.disconnect()
            follower.disconnect()
            return


if __name__ == "__main__":
    main()
