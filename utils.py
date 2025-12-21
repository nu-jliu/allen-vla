import logging
from typing import Any

from lerobot.datasets.utils import hw_to_dataset_features


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    COLORS = {
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_color = self.COLORS.get(record.levelno, self.RESET)
        formatter = logging.Formatter(
            fmt=f"[%(asctime)s] [{log_color}%(levelname)s{self.RESET}] %(message)s{self.RESET}",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return formatter.format(record)


def setup_logging() -> None:
    """Setup colored logging for the application."""
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def get_camera_feature(
    camera_name: str = "front", height: int = 480, width: int = 640
) -> dict[str, Any]:
    """Get camera feature definition for LeRobot dataset.

    This uses the standard camera name 'front' to match ACT policy expectations.

    :param camera_name: Camera identifier (default: 'front' for ACT compatibility)
    :type camera_name: str
    :param height: Camera image height in pixels
    :type height: int
    :param width: Camera image width in pixels
    :type width: int
    :return: Camera feature dictionary for LeRobot dataset
    :rtype: dict
    """
    return {
        f"observation.images.{camera_name}": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    }


def build_observation_features(
    robot_obs_features: dict[str, Any],
    cam_height: int,
    cam_width: int,
    camera_name: str = "front",
) -> dict[str, Any]:
    """Build complete observation features for LeRobot dataset.

    Combines robot state observations with camera features.

    :param robot_obs_features: Robot's observation feature definitions
    :type robot_obs_features: dict
    :param cam_height: Camera image height in pixels
    :type cam_height: int
    :param cam_width: Camera image width in pixels
    :type cam_width: int
    :param camera_name: Camera identifier (default: 'front' for ACT compatibility)
    :type camera_name: str
    :return: Complete observation features dictionary
    :rtype: dict
    """
    obs_features = hw_to_dataset_features(robot_obs_features, prefix="observation")
    camera_features = get_camera_feature(
        camera_name=camera_name, height=cam_height, width=cam_width
    )
    obs_features.update(camera_features)
    return obs_features


def build_action_features(robot_action_features: dict[str, Any]) -> dict[str, Any]:
    """Build action features for LeRobot dataset.

    :param robot_action_features: Robot's action feature definitions
    :type robot_action_features: dict
    :return: Action features dictionary
    :rtype: dict
    """
    return hw_to_dataset_features(robot_action_features, prefix="action")
