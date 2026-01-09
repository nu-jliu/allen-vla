import logging
import time
from argparse import ArgumentParser
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    # Record robot's initial joint positions right after connection
    initial_obs = robot.get_observation()
    initial_position = {
        "shoulder_pan.pos": initial_obs["shoulder_pan.pos"],
        "shoulder_lift.pos": initial_obs["shoulder_lift.pos"],
        "elbow_flex.pos": initial_obs["elbow_flex.pos"],
        "wrist_flex.pos": initial_obs["wrist_flex.pos"],
        "wrist_roll.pos": initial_obs["wrist_roll.pos"],
        "gripper.pos": initial_obs["gripper.pos"],
    }
    logger.info(f"Recorded robot initial position: {initial_position}")

    robot.calibrate()

    logger.info("Moving robot to initial position before disconnect...")
    robot.send_action(initial_position)
    # Wait for robot to reach initial position
    while True:
        obs = robot.get_observation()
        max_diff = max(
            abs(obs[name] - initial_position[name])
            for name in initial_position.keys()
        )
        if max_diff < 1.0:  # Within 1 degree tolerance
            break
        time.sleep(0.05)
    robot.disconnect()


if __name__ == "__main__":
    main()
