from argparse import ArgumentParser
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--port",
        type=str,
        required=True,
        help="Serial port for the robot",
    )
    parser.add_argument(
        "--id",
        type=str,
        required=True,
        help="Robot ID",
    )

    args = parser.parse_args()

    config = SO101FollowerConfig(
        port=args.port,
        id=args.id,
    )
    robot = SO101Follower(config)

    robot.connect(calibrate=False)
    robot.calibrate()
    robot.disconnect()


if __name__ == "__main__":
    main()
