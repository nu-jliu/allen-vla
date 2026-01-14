import os
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

logger = logging.getLogger(__name__)

HOME = os.getenv("HOME")
CALIBRATION_DIR = os.path.join(
    HOME,
    ".cache",
    "huggingface",
    "lerobot",
    "calibration",
)


def add_common_robot_args(parser: ArgumentParser) -> ArgumentParser:
    """Add common robot configuration arguments to an argument parser.

    :param parser: Argument parser to add robot arguments to
    :type parser: ArgumentParser
    :return: Argument parser with added robot arguments
    :rtype: ArgumentParser
    """
    parser.add_argument(
        "--leader-port",
        type=str,
        default="/dev/ttyACM1",
        help="Serial port for the leader arm (default: /dev/ttyACM1)",
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
        default="/dev/ttyACM0",
        help="Serial port for the follower arm (default: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--follower-id",
        type=str,
        default="my_follower",
        help="ID for the follower arm (default: my_follower)",
    )
    return parser


def create_robot_configs(args: Namespace) -> tuple[SO101LeaderConfig, SO101FollowerConfig]:
    """Create leader and follower robot configurations from parsed arguments.

    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    :return: Tuple of (leader_config, follower_config)
    :rtype: tuple[SO101LeaderConfig, SO101FollowerConfig]
    """
    leader_port = args.leader_port
    leader_id = args.leader_id
    follower_port = args.follower_port
    follower_id = args.follower_id

    leader_calibration_dir = Path(
        os.path.join(
            CALIBRATION_DIR,
            "teleoperators",
            "so101_leader",
        )
    )
    follower_calibration_dir = Path(
        os.path.join(
            CALIBRATION_DIR,
            "robots",
            "so101_follower",
        )
    )

    leader_config = SO101LeaderConfig(
        port=leader_port,
        use_degrees=False,
        id=leader_id,
        calibration_dir=leader_calibration_dir,
    )
    follower_config = SO101FollowerConfig(
        port=follower_port,
        disable_torque_on_disconnect=True,
        use_degrees=False,
        id=follower_id,
        calibration_dir=follower_calibration_dir,
    )
    return leader_config, follower_config


def initialize_robots(args: Namespace, calibrate: bool = True) -> tuple[SO101Leader, SO101Follower]:
    """Initialize and connect leader and follower robots.

    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    :param calibrate: Whether to calibrate robots during connection
    :type calibrate: bool
    :return: Tuple of (leader, follower) robot instances
    :rtype: tuple[SO101Leader, SO101Follower]
    """
    logger.info("Creating robot configurations...")
    leader_config, follower_config = create_robot_configs(args)

    logger.info("Initializing leader and follower robots...")
    leader = SO101Leader(config=leader_config)
    follower = SO101Follower(config=follower_config)

    logger.info("Connecting to leader robot and calibrating...")
    leader.connect(calibrate=calibrate)
    logger.info("Leader robot connected successfully")

    logger.info("Connecting to follower robot and calibrating...")
    follower.connect(calibrate=calibrate)
    logger.info("Follower robot connected successfully")

    # Disable torque on leader so it can be moved freely by hand
    logger.info("Disabling torque on leader arm for manual control")
    leader.bus.disable_torque()

    return leader, follower
