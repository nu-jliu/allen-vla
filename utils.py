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


def get_so101_robot_config() -> dict[str, Any]:
    """Get SO101 robot configuration (joint names for observation and action).

    Returns the robot's native feature configuration used for both observation
    and action, matching what the robot's observation_features and action_features
    properties would return.

    :return: Robot configuration dict with 'observation' and 'action' keys
    :rtype: dict
    """
    joint_features = {
        "shoulder_pan.pos": float,
        "shoulder_lift.pos": float,
        "elbow_flex.pos": float,
        "wrist_flex.pos": float,
        "wrist_roll.pos": float,
        "gripper.pos": float,
    }
    return {
        "observation": joint_features,
        "action": joint_features,
    }


def dataset_features_to_act_shapes(
    obs_features: dict[str, Any], action_features: dict[str, Any]
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Convert dataset features to ACTConfig input/output shapes.

    Dataset features use (H, W, C) format for images, but ACTConfig expects (C, H, W).
    This function handles the conversion.

    :param obs_features: Observation features from build_observation_features()
    :type obs_features: dict
    :param action_features: Action features from build_action_features()
    :type action_features: dict
    :return: Tuple of (input_shapes, output_shapes) for ACTConfig
    :rtype: tuple[dict, dict]
    """
    input_shapes = {}
    output_shapes = {}

    # Process observation features
    for key, feature in obs_features.items():
        shape = feature["shape"]
        # Convert image shapes from (H, W, C) to (C, H, W)
        if "images" in key and len(shape) == 3:
            height, width, channels = shape
            input_shapes[key] = [channels, height, width]
        else:
            input_shapes[key] = list(shape)

    # Process action features
    for key, feature in action_features.items():
        shape = feature["shape"]
        output_shapes[key] = list(shape)

    return input_shapes, output_shapes


def build_dataset_features_for_so101(
    cam_height: int = 480,
    cam_width: int = 640,
    camera_name: str = "front",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build complete dataset features for SO101 robot.

    This is the centralized function used by both data collection and training
    to ensure features are always consistent.

    :param cam_height: Camera image height in pixels (default: 480)
    :type cam_height: int
    :param cam_width: Camera image width in pixels (default: 640)
    :type cam_width: int
    :param camera_name: Camera identifier (default: 'front' for ACT compatibility)
    :type camera_name: str
    :return: Tuple of (observation_features, action_features)
    :rtype: tuple[dict, dict]
    """
    # Get robot configuration
    robot_config = get_so101_robot_config()

    # Build observation features (robot state + camera)
    obs_features = build_observation_features(
        robot_obs_features=robot_config["observation"],
        cam_height=cam_height,
        cam_width=cam_width,
        camera_name=camera_name,
    )

    # Build action features
    action_features = build_action_features(robot_config["action"])

    return obs_features, action_features


def build_act_config_features(
    cam_height: int = 480,
    cam_width: int = 640,
    camera_name: str = "front",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[int]], dict[str, list[int]]]:
    """Build all features needed for ACTConfig.

    This returns both the dataset features AND the ACT-compatible shapes,
    ensuring consistency between data collection and training.

    :param cam_height: Camera image height in pixels (default: 480)
    :type cam_height: int
    :param cam_width: Camera image width in pixels (default: 640)
    :type cam_width: int
    :param camera_name: Camera identifier (default: 'front' for ACT compatibility)
    :type camera_name: str
    :return: Tuple of (obs_features, action_features, input_shapes, output_shapes)
    :rtype: tuple[dict, dict, dict, dict]
    """
    # Build dataset features
    obs_features, action_features = build_dataset_features_for_so101(
        cam_height=cam_height,
        cam_width=cam_width,
        camera_name=camera_name,
    )

    # Convert to ACT shapes (handles H,W,C -> C,H,W conversion)
    input_shapes, output_shapes = dataset_features_to_act_shapes(
        obs_features, action_features
    )

    return obs_features, action_features, input_shapes, output_shapes
