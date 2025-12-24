#!/usr/bin/env python
"""
Robot Client for ACT policy inference.

This client connects to the robot and an inference server to execute policy actions.
It gathers observations from the robot, sends them to the server, and actuates the
robot with the returned actions.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import pickle
import socket
import struct
import time
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
import numpy as np

from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.robots import make_robot_from_config
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.robot_utils import precise_sleep

from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def recv_message(conn: socket.socket) -> bytes:
    """Receive a length-prefixed message from a socket."""
    raw_msglen = recv_all(conn, 4)
    if not raw_msglen:
        return b""
    msglen = struct.unpack(">I", raw_msglen)[0]
    return recv_all(conn, msglen)


def recv_all(conn: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from a socket."""
    data = bytearray()
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return b""
        data.extend(packet)
    return bytes(data)


def send_message(conn: socket.socket, data: bytes) -> None:
    """Send a length-prefixed message over a socket."""
    msg = struct.pack(">I", len(data)) + data
    conn.sendall(msg)


class InferenceClient:
    """Client that connects to robot and inference server."""

    def __init__(
        self,
        server_host: str,
        server_port: int = 8000,
    ):
        """Initialize the inference client.

        :param server_host: Inference server host address
        :param server_port: Inference server port
        """
        self.server_host = server_host
        self.server_port = server_port
        self.robot = None
        self.server_conn: socket.socket | None = None

        # Observation feature names (for SO101)
        self.state_names = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

    def connect_robot(
        self,
        robot_port: str,
        camera_index: int | str,
        camera_name: str = "front",
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        robot_id: str = "inference_robot",
    ) -> None:
        """Connect to the robot.

        :param robot_port: Robot serial port
        :param camera_index: Camera index or path
        :param camera_name: Camera name in config
        :param camera_width: Camera width
        :param camera_height: Camera height
        :param camera_fps: Camera FPS
        :param robot_id: Robot ID
        """
        logger.info("Connecting to robot...")

        # Create camera config
        cameras = {
            camera_name: OpenCVCameraConfig(
                index_or_path=camera_index,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
            )
        }

        # Create robot config
        robot_config = SO101FollowerConfig(
            port=robot_port,
            id=robot_id,
            cameras=cameras,
        )

        # Create and connect robot
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()

        logger.info(f"Robot connected on {robot_port}")
        logger.info(f"Robot ID: {robot_id}")
        logger.info(f"Camera: {camera_name} @ index {camera_index}")

    def connect_server(self) -> None:
        """Connect to the inference server."""
        logger.info(f"Connecting to inference server at {self.server_host}:{self.server_port}...")

        self.server_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_conn.connect((self.server_host, self.server_port))

        # Send ping to verify connection
        request = {"type": "ping"}
        send_message(self.server_conn, pickle.dumps(request))
        response_data = recv_message(self.server_conn)
        response = pickle.loads(response_data)

        if response.get("type") != "ack":
            raise ConnectionError(f"Server ping failed: {response}")

        logger.info("Connected to inference server")

    def disconnect(self) -> None:
        """Disconnect from robot and server."""
        if self.robot is not None:
            try:
                self.robot.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting robot: {e}")
            self.robot = None

        if self.server_conn is not None:
            try:
                self.server_conn.close()
            except Exception as e:
                logger.warning(f"Error closing server connection: {e}")
            self.server_conn = None

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get observation from the robot.

        :return: Dictionary with observation.state and observation.images.front
        """
        if self.robot is None:
            raise RuntimeError("Robot not connected")

        # Get raw observation from robot
        raw_obs = self.robot.get_observation()

        # Extract state (joint positions)
        state = np.array(
            [raw_obs[name] for name in self.state_names],
            dtype=np.float32,
        )

        # Extract camera image
        # Find the camera key (e.g., "front")
        image_key = None
        for key in raw_obs:
            if key not in self.state_names:
                image_key = key
                break

        if image_key is None:
            raise RuntimeError("No camera image in observation")

        image = raw_obs[image_key]
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Build observation dict with proper keys
        observation = {
            "observation.state": state,
            f"observation.images.{image_key}": image,
        }

        return observation

    def send_observation(self, observation: dict[str, np.ndarray]) -> dict[str, float]:
        """Send observation to server and get action.

        :param observation: Observation dictionary
        :return: Action dictionary
        """
        if self.server_conn is None:
            raise RuntimeError("Not connected to server")

        request = {"type": "inference", "observation": observation}
        send_message(self.server_conn, pickle.dumps(request))

        response_data = recv_message(self.server_conn)
        if not response_data:
            raise ConnectionError("Server disconnected")

        response = pickle.loads(response_data)

        if response.get("type") == "error":
            raise RuntimeError(f"Server error: {response.get('message')}")

        if response.get("type") != "action":
            raise RuntimeError(f"Unexpected response type: {response.get('type')}")

        return response.get("action", {})

    def send_action(self, action: dict[str, float]) -> None:
        """Send action to the robot.

        :param action: Action dictionary with motor positions
        """
        if self.robot is None:
            raise RuntimeError("Robot not connected")

        self.robot.send_action(action)

    def reset_episode(self) -> None:
        """Tell the server to reset policy state for a new episode."""
        if self.server_conn is None:
            raise RuntimeError("Not connected to server")

        request = {"type": "reset"}
        send_message(self.server_conn, pickle.dumps(request))

        response_data = recv_message(self.server_conn)
        response = pickle.loads(response_data)

        if response.get("type") == "error":
            raise RuntimeError(f"Reset error: {response.get('message')}")

        logger.info("Policy state reset for new episode")

    def run_episode(self, duration_s: float, fps: int) -> int:
        """Run a single episode.

        :param duration_s: Episode duration in seconds
        :param fps: Control frequency in Hz
        :return: Number of steps executed
        """
        logger.info(f"Starting episode (duration: {duration_s}s, fps: {fps})")

        steps = 0
        start_time = time.perf_counter()

        while (time.perf_counter() - start_time) < duration_s:
            loop_start = time.perf_counter()

            try:
                # Get observation from robot
                obs = self.get_observation()

                # Send to server and get action
                action = self.send_observation(obs)

                # Send action to robot
                self.send_action(action)

                steps += 1

            except KeyboardInterrupt:
                logger.info("Episode interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                break

            # Sleep to maintain FPS
            elapsed = time.perf_counter() - loop_start
            precise_sleep(1.0 / fps - elapsed)

        logger.info(f"Episode completed: {steps} steps")
        return steps

    def run(
        self,
        num_episodes: int,
        episode_time_s: float,
        reset_time_s: float,
        fps: int,
    ) -> None:
        """Run multiple episodes.

        :param num_episodes: Number of episodes to run
        :param episode_time_s: Duration of each episode in seconds
        :param reset_time_s: Time for reset between episodes in seconds
        :param fps: Control frequency in Hz
        """
        logger.info("=" * 60)
        logger.info("Starting inference run")
        logger.info("=" * 60)

        total_steps = 0

        try:
            for episode in range(num_episodes):
                logger.info(f"\n--- Episode {episode + 1}/{num_episodes} ---")

                # Reset policy state
                self.reset_episode()

                # Run episode
                steps = self.run_episode(episode_time_s, fps)
                total_steps += steps

                # Reset period (except after last episode)
                if episode < num_episodes - 1:
                    logger.info(f"Reset period: {reset_time_s}s")
                    logger.info("Please reset the environment...")
                    time.sleep(reset_time_s)

        except KeyboardInterrupt:
            logger.info("\nRun interrupted by user")

        logger.info("=" * 60)
        logger.info("Inference run completed")
        logger.info(f"Total episodes: {episode + 1}")
        logger.info(f"Total steps: {total_steps}")
        logger.info("=" * 60)


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(
        description="Robot Client for ACT Policy Inference",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python policy/act/inference_client.py \\
    --server-host localhost \\
    --robot-port /dev/ttyACM0 \\
    --camera-index 0 \\
    --num-episodes 10

  # Inference with custom settings
  python policy/act/inference_client.py \\
    --server-host 192.168.1.100 \\
    --server-port 8000 \\
    --robot-port /dev/ttyACM0 \\
    --camera-index 0 \\
    --fps 30 \\
    --episode-time 60 \\
    --num-episodes 5
        """,
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--server-host",
        type=str,
        required=True,
        help="Inference server host address",
    )
    required.add_argument(
        "--robot-port",
        type=str,
        required=True,
        help="Robot serial port (e.g., /dev/ttyACM0)",
    )
    required.add_argument(
        "--camera-index",
        type=str,
        required=True,
        help="Camera index or path (e.g., '0' for /dev/video0)",
    )

    # Server configuration
    server = parser.add_argument_group("server configuration")
    server.add_argument(
        "--server-port",
        type=int,
        default=8000,
        help="Inference server port (default: 8000)",
    )

    # Robot configuration
    robot = parser.add_argument_group("robot configuration")
    robot.add_argument(
        "--robot-id",
        type=str,
        default="inference_robot",
        help="Robot ID (default: inference_robot)",
    )

    # Camera configuration
    camera = parser.add_argument_group("camera configuration")
    camera.add_argument(
        "--camera-name",
        type=str,
        default="front",
        help="Camera name in config (default: front)",
    )
    camera.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Camera width (default: 640)",
    )
    camera.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Camera height (default: 480)",
    )
    camera.add_argument(
        "--camera-fps",
        type=int,
        default=30,
        help="Camera FPS (default: 30)",
    )

    # Episode parameters
    episode = parser.add_argument_group("episode parameters")
    episode.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)",
    )
    episode.add_argument(
        "--episode-time",
        type=float,
        default=60.0,
        help="Duration of each episode in seconds (default: 60)",
    )
    episode.add_argument(
        "--reset-time",
        type=float,
        default=60.0,
        help="Time for reset between episodes in seconds (default: 60)",
    )
    episode.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Control frequency in Hz (default: 30)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Extract args to local variables
    server_host = args.server_host
    server_port = args.server_port
    robot_port = args.robot_port
    robot_id = args.robot_id
    camera_index_str = args.camera_index
    camera_name = args.camera_name
    camera_width = args.camera_width
    camera_height = args.camera_height
    camera_fps = args.camera_fps
    num_episodes = args.num_episodes
    episode_time = args.episode_time
    reset_time = args.reset_time
    fps = args.fps

    # Convert camera index to int if numeric
    camera_index: int | str
    if camera_index_str.isdigit():
        camera_index = int(camera_index_str)
    else:
        camera_index = camera_index_str

    logger.info("=" * 60)
    logger.info("ACT Policy Inference Client")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Server: {server_host}:{server_port}")
    logger.info(f"  Robot: so101_follower @ {robot_port}")
    logger.info(f"  Robot ID: {robot_id}")
    logger.info(f"  Camera: {camera_name} @ index {camera_index_str}")
    logger.info(f"  Episodes: {num_episodes}")
    logger.info(f"  Episode time: {episode_time}s")
    logger.info(f"  Reset time: {reset_time}s")
    logger.info(f"  FPS: {fps}")
    logger.info("")

    # Create client
    client = InferenceClient(
        server_host=server_host,
        server_port=server_port,
    )

    try:
        # Connect to robot
        client.connect_robot(
            robot_port=robot_port,
            camera_index=camera_index,
            camera_name=camera_name,
            camera_width=camera_width,
            camera_height=camera_height,
            camera_fps=camera_fps,
            robot_id=robot_id,
        )

        # Connect to server
        client.connect_server()

        # Run episodes
        client.run(
            num_episodes=num_episodes,
            episode_time_s=episode_time,
            reset_time_s=reset_time,
            fps=fps,
        )

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        client.disconnect()
        logger.info("Client disconnected")


if __name__ == "__main__":
    main()
