#!/usr/bin/env python
"""
Robot Client for PI0 policy inference.

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

from utils import setup_logging, load_camera_config, query_camera_info

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
        self.initial_position: dict[str, float] | None = None

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
        camera_config_path: str,
        robot_id: str = "inference_robot",
    ) -> None:
        """Connect to the robot.

        :param robot_port: Robot serial port
        :param camera_config_path: Path to camera config TOML file
        :param robot_id: Robot ID
        """
        logger.info("Connecting to robot...")

        # Load camera configuration from TOML and build stream URLs
        camera_configs = load_camera_config(camera_config_path)
        cameras = {}
        for cam in camera_configs:
            name = cam["name"]
            host = cam["host"]
            cam_port = cam["port"]
            info = query_camera_info(host, cam_port)
            stream_url = f"http://{host}:{cam_port}/stream"
            cameras[name] = OpenCVCameraConfig(
                index_or_path=stream_url,
                width=info["width"],
                height=info["height"],
                fps=info["fps"],
            )
            logger.info(f"  Camera '{name}': {stream_url} ({info['width']}x{info['height']} @ {info['fps']}fps)")

        # Create robot config
        robot_config = SO101FollowerConfig(
            port=robot_port,
            id=robot_id,
            cameras=cameras,
        )

        # Create and connect robot
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()

        # Record robot's initial joint positions right after connection
        raw_obs = self.robot.get_observation()
        self.initial_position = {name: raw_obs[name] for name in self.state_names}
        logger.info(f"Recorded robot initial position: {self.initial_position}")

        logger.info(f"Robot connected on {robot_port}")
        logger.info(f"Robot ID: {robot_id}")
        logger.info(f"Camera config: {camera_config_path}")

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
                # Move robot to initial position before disconnecting
                if self.initial_position is not None:
                    logger.info("Moving robot to initial position before disconnect...")
                    self.robot.send_action(self.initial_position)
                    # Wait for robot to reach initial position
                    while True:
                        raw_obs = self.robot.get_observation()
                        max_diff = max(
                            abs(raw_obs[name] - self.initial_position[name])
                            for name in self.initial_position.keys()
                        )
                        if max_diff < 1.0:  # Within 1 degree tolerance
                            break
                        time.sleep(0.05)
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

        :return: Dictionary with observation.state and observation.images.{camera_name} per camera
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

        # Build observation dict with proper keys
        observation = {
            "observation.state": state,
        }

        # Extract all camera images (non-state keys)
        for key in raw_obs:
            if key not in self.state_names:
                image = raw_obs[key]
                if not isinstance(image, np.ndarray):
                    image = np.array(image)
                observation[f"observation.images.{key}"] = image

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
        description="Robot Client for PI0 Policy Inference",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python policy/pi0/inference_client.py \\
    --server-host localhost \\
    --robot-port /dev/ttyACM0 \\
    --episode 10

  # Inference with custom settings
  python policy/pi0/inference_client.py \\
    --server-host 192.168.1.100 \\
    --server-port 8000 \\
    --robot-port /dev/ttyACM0 \\
    --camera-config config/camera.toml \\
    --fps 30 \\
    --episode-time 60 \\
    --episode 5
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
        "--camera-config",
        type=str,
        default="config/camera.toml",
        help="Path to camera config TOML file (default: config/camera.toml)",
    )

    # Episode parameters
    episode = parser.add_argument_group("episode parameters")
    episode.add_argument(
        "--episode",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)",
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
    camera_config = args.camera_config
    num_episodes = args.episode
    episode_time = args.episode_time
    reset_time = args.reset_time
    fps = args.fps

    logger.info("=" * 60)
    logger.info("PI0 Policy Inference Client")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Server: {server_host}:{server_port}")
    logger.info(f"  Robot: so101_follower @ {robot_port}")
    logger.info(f"  Robot ID: {robot_id}")
    logger.info(f"  Camera config: {camera_config}")
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
            camera_config_path=camera_config,
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
