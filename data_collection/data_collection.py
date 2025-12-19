import os
import json
import time
from pathlib import Path

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

HOME = os.getenv("HOME")
CALIBRATION_DIR = os.path.join(
    HOME,
    ".cache",
    "huggingface",
    "lerobot",
    "calibration",
)


def main():
    leader_config = SO101FollowerConfig(
        port="/dev/ttyACM0",
        disable_torque_on_disconnect=True,
        use_degrees=False,
        id="my_leader",
        calibration_dir=Path(
            os.path.join(
                CALIBRATION_DIR,
                "teleoperators",
                "so101_leader",
            )
        ),
    )
    follower_config = SO101FollowerConfig(
        port="/dev/ttyACM1",
        disable_torque_on_disconnect=True,
        use_degrees=False,
        id="my_follower",
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
