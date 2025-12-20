import os
import logging
from argparse import ArgumentParser
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
    """Add common robot configuration arguments to an argument parser."""
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
    return parser


def create_robot_configs(args):
    """Create leader and follower robot configurations from parsed arguments."""
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
    return leader_config, follower_config


def initialize_robots(args, calibrate=True):
    """Initialize and connect leader and follower robots."""
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
