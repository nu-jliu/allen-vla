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

    port = args.port
    robot_id = args.id

    config = SO101FollowerConfig(
        port=port,
        id=robot_id,
    )
    robot = SO101Follower(config=config)

    robot.connect(calibrate=False)
    robot.calibrate()
    robot.disconnect()


if __name__ == "__main__":
    main()
