import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import logging
import threading
import socket
import urllib.request
import cv2
import numpy as np
import shutil

from argparse import ArgumentParser

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from utils import setup_logging, get_camera_feature, build_action_features, get_so101_robot_config, load_camera_config, query_camera_info
from lerobot.datasets.utils import hw_to_dataset_features
from robot_utils import add_common_robot_args, initialize_robots

setup_logging()
logger = logging.getLogger(__name__)


def camera_subscribe_thread(
    host: str,
    port: int,
    camera_name: str,
    camera_frames: dict,
    image_lock: threading.Lock,
    running_flag: callable,
) -> None:
    """Subscribe to an MJPEG stream and update the shared frame dict.

    :param host: Camera server hostname
    :param port: Camera server port
    :param camera_name: Name key for this camera in camera_frames
    :param camera_frames: Shared dict mapping camera name -> latest frame
    :param image_lock: Lock protecting camera_frames
    :param running_flag: Callable returning bool, False to stop
    """
    url = f"http://{host}:{port}/stream"
    logger.info(f"Subscribing to camera '{camera_name}' at {url}")

    while running_flag():
        try:
            req = urllib.request.Request(url)
            stream = urllib.request.urlopen(req, timeout=5)
            buf = b""

            while running_flag():
                chunk = stream.read(4096)
                if not chunk:
                    break
                buf += chunk

                # Find complete JPEG frames (SOI: 0xFFD8, EOI: 0xFFD9)
                while True:
                    start = buf.find(b"\xff\xd8")
                    end = buf.find(b"\xff\xd9")
                    if start == -1 or end == -1 or end < start:
                        break
                    jpeg = buf[start : end + 2]
                    buf = buf[end + 2 :]
                    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with image_lock:
                            camera_frames[camera_name] = frame

        except Exception as e:
            if running_flag():
                logger.warning(f"Camera '{camera_name}' stream error: {e}, reconnecting...")
                time.sleep(1.0)

    logger.info(f"Camera '{camera_name}' subscribe thread stopped")


def main() -> None:
    parser = ArgumentParser(description="Data collection for SO101 robot")
    parser = add_common_robot_args(parser)
    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Hugging Face username",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        required=True,
        help="Robot type (e.g., so101)",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name for the dataset (e.g., pick_place, stack_blocks)",
    )
    parser.add_argument(
        "--hz",
        type=int,
        default=30,
        help="Control loop frequency in Hz (default: 30)",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push dataset to Hugging Face Hub after collection",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data",
        help="Root directory for dataset storage (default: data)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1234,
        help="Port to listen for commands (default: 1234)",
    )
    args = parser.parse_args()

    leader_port = args.leader_port
    leader_id = args.leader_id
    follower_port = args.follower_port
    follower_id = args.follower_id
    username = args.username
    robot_type = args.robot_type
    task = args.task
    # Construct repo_id as {username}/{robot}-{task}
    repo_id = f"{username}/{robot_type}-{task}"
    hz = args.hz
    root = args.root
    push = args.push
    port = args.port
    camera_config_path = str(Path(__file__).resolve().parent.parent / "config" / "camera.toml")

    # Load camera configuration
    cameras = load_camera_config(camera_config_path)
    logger.info(f"Loaded {len(cameras)} camera(s) from {camera_config_path}")

    # Query each camera server for its info
    camera_infos = {}
    for cam in cameras:
        name = cam["name"]
        host = cam["host"]
        cam_port = cam["port"]
        logger.info(f"Querying camera '{name}' at {host}:{cam_port}...")
        info = query_camera_info(host, cam_port)
        camera_infos[name] = {"host": host, "port": cam_port, **info}
        logger.info(f"  Camera '{name}': {info['width']}x{info['height']} @ {info['fps']}fps")

    logger.info("Starting data collection with configuration:")
    logger.info(f"  Leader port: {leader_port}")
    logger.info(f"  Leader ID: {leader_id}")
    logger.info(f"  Follower port: {follower_port}")
    logger.info(f"  Follower ID: {follower_id}")
    logger.info(f"  Repo ID: {repo_id}")
    logger.info(f"  Control frequency: {hz} Hz")
    logger.info(f"  Camera config: {camera_config_path}")
    for name, info in camera_infos.items():
        logger.info(f"  Camera '{name}': {info['host']}:{info['port']} ({info['width']}x{info['height']})")
    logger.info(f"  Root directory: {root}")
    logger.info(f"  Command port: {port}")

    # Remove root directory if it exists
    root_path = Path(root)
    if root_path.exists():
        logger.warning(f"Root directory {root} already exists. Removing it...")
        shutil.rmtree(root_path)
        logger.info(f"Removed existing directory: {root}")

    leader, follower = initialize_robots(args, calibrate=True)

    # Record follower's initial joint positions right after connection
    initial_obs = follower.get_observation()
    initial_position = {
        "shoulder_pan.pos": initial_obs["shoulder_pan.pos"],
        "shoulder_lift.pos": initial_obs["shoulder_lift.pos"],
        "elbow_flex.pos": initial_obs["elbow_flex.pos"],
        "wrist_flex.pos": initial_obs["wrist_flex.pos"],
        "wrist_roll.pos": initial_obs["wrist_roll.pos"],
        "gripper.pos": initial_obs["gripper.pos"],
    }
    logger.info(f"Recorded follower initial position: {initial_position}")

    # Build dataset features using robot config + camera info from servers
    robot_config = get_so101_robot_config()
    obs_features = hw_to_dataset_features(robot_config["observation"], prefix="observation")
    for name, info in camera_infos.items():
        cam_feature = get_camera_feature(
            camera_name=name, height=info["height"], width=info["width"]
        )
        obs_features.update(cam_feature)
    action_features = build_action_features(robot_config["action"])

    logger.info(f"Observation features: {obs_features}")
    logger.info(f"Action features: {action_features}")

    logger.info(f"Creating dataset: {repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=hz,
        features={**action_features, **obs_features},
        robot_type=follower.name,
        use_videos=True,
        root=root,
    )
    logger.info(f"Dataset features: {dataset.features}")

    recording = False
    frame_count = 0
    episode_count = 0
    running = True
    recording_lock = threading.Lock()

    # Camera frame state: shared dict of camera_name -> latest BGR frame
    camera_frames = {}
    image_lock = threading.Lock()

    # Start a subscribe thread per camera
    camera_threads = []
    for name, info in camera_infos.items():
        t = threading.Thread(
            target=camera_subscribe_thread,
            args=(info["host"], info["port"], name, camera_frames, image_lock, lambda: running),
            daemon=True,
        )
        t.start()
        camera_threads.append(t)

    def command_server_thread() -> None:
        """Listen for commands via socket connection (telnet compatible)."""
        nonlocal recording, frame_count, episode_count, running

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("0.0.0.0", port))
        server_socket.listen(1)
        server_socket.settimeout(1.0)  # Allow periodic checking of running flag

        logger.info(f"Command server listening on port {port}")
        logger.info(f"Connect via: telnet localhost {port}")
        logger.info("Commands: s=start/stop, a=abort episode, q=quit")

        prompt = b"--> "
        client_connected = False
        client_lock = threading.Lock()

        def send_response(sock: socket.socket, message: str) -> None:
            """Send a response message followed by the prompt."""
            sock.sendall(f"<-- {message}\r\n".encode() + prompt)

        while running:
            try:
                try:
                    client_socket, addr = server_socket.accept()

                    # Check if another client is already connected
                    with client_lock:
                        if client_connected:
                            logger.warning(
                                f"Rejecting connection from {addr} - another client already connected"
                            )
                            client_socket.sendall(
                                b"<-- Connection rejected: another client is already connected.\r\n"
                            )
                            client_socket.close()
                            continue
                        client_connected = True

                    logger.info(f"Client connected from {addr[0]}:{addr[1]}")
                    client_socket.settimeout(0.5)

                    # Send welcome message
                    welcome_msg = (
                        "\r\n"
                        "========================================\r\n"
                        "   Data Collection Controller\r\n"
                        "========================================\r\n"
                        "\r\n"
                        "Commands:\r\n"
                        "  s     - Start/Stop recording\r\n"
                        "  a     - Abort current episode\r\n"
                        "  q     - Quit and save dataset\r\n"
                        "\r\n"
                        f"Status: Ready (Episodes recorded: {episode_count})\r\n"
                        "\r\n"
                    )
                    client_socket.sendall(welcome_msg.encode() + prompt)

                    while running:
                        try:
                            data = client_socket.recv(1024)
                            if not data:
                                break

                            # Process each byte/character received
                            for byte in data:
                                char = chr(byte)

                                # Handle 's' - start/stop recording
                                if char == "s":
                                    with recording_lock:
                                        if not recording:
                                            recording = True
                                            frame_count = 0
                                            dataset.episode_buffer = (
                                                dataset.create_episode_buffer()
                                            )
                                            logger.info("=== RECORDING STARTED ===")
                                            send_response(
                                                client_socket,
                                                "\033[32m[START]\033[0m Recording started - Press s to stop",
                                            )
                                        else:
                                            recording = False
                                            if frame_count > 0:
                                                dataset.save_episode()
                                                episode_count += 1
                                                msg = f"\033[32m[SAVED]\033[0m Episode {episode_count} saved with {frame_count} frames"
                                                logger.info(
                                                    f"=== RECORDING STOPPED === (Saved {frame_count} frames, Episode {episode_count})"
                                                )
                                            else:
                                                msg = "\033[32m[STOP]\033[0m Recording stopped (No frames captured)"
                                                logger.info(
                                                    "=== RECORDING STOPPED === (No frames captured)"
                                                )
                                            send_response(client_socket, msg)
                                            frame_count = 0

                                # Handle 'a' - abort episode
                                elif char == "a":
                                    with recording_lock:
                                        if recording:
                                            recording = False
                                            discarded = frame_count
                                            # Clear episode buffer without saving
                                            dataset.episode_buffer = (
                                                dataset.create_episode_buffer()
                                            )
                                            frame_count = 0
                                            logger.warning(
                                                f"=== EPISODE ABORTED === (Discarded {discarded} frames)"
                                            )
                                            send_response(
                                                client_socket,
                                                f"\033[33m[ABORT]\033[0m Episode aborted - Discarded {discarded} frames",
                                            )
                                        else:
                                            send_response(
                                                client_socket,
                                                "\033[36m[INFO]\033[0m Not recording - nothing to abort",
                                            )

                                # Handle 'q' - quit
                                elif char == "q":
                                    logger.info("Quit command received from client")
                                    client_socket.sendall(
                                        b"\r\n<-- [QUIT] Shutting down and saving dataset...\r\n"
                                    )
                                    client_socket.close()
                                    with client_lock:
                                        client_connected = False
                                    running = False
                                    server_socket.close()
                                    return

                                # Handle unknown commands (ignore telnet control chars)
                                elif ord(char) >= 32 and char not in ("\r", "\n"):
                                    logger.warning(
                                        f"Unknown command received: '{char}'"
                                    )
                                    send_response(
                                        client_socket,
                                        f"\033[31m[ERROR]\033[0m Unknown command: '{char}' - Use s, a, or q",
                                    )

                        except socket.timeout:
                            continue
                        except ConnectionResetError:
                            logger.info(f"Client {addr[0]}:{addr[1]} connection reset")
                            break

                    client_socket.close()
                    with client_lock:
                        client_connected = False
                    logger.info(f"Client {addr[0]}:{addr[1]} disconnected")

                except socket.timeout:
                    continue

            except Exception as e:
                if running:
                    logger.error(f"Error in command server: {e}")
                with client_lock:
                    client_connected = False

        server_socket.close()
        logger.info("Command server stopped")

    # Start command server thread
    command_thread = threading.Thread(target=command_server_thread, daemon=True)
    command_thread.start()

    logger.info(
        f"Connect via 'telnet localhost {port}' to control recording. Press Ctrl+C to exit."
    )

    # Wait for at least one frame from each camera
    camera_names = list(camera_infos.keys())
    logger.info("Waiting for frames from all cameras...")
    while True:
        with image_lock:
            have_all = all(name in camera_frames for name in camera_names)
        if have_all:
            break
        time.sleep(0.1)
    logger.info("All cameras streaming")

    logger.info(f"Starting teleoperation loop at {hz} Hz...")
    period = 1.0 / hz

    while True:
        try:
            action = leader.get_action()
            follower.send_action(action)

            # Get the current recording state
            with recording_lock:
                is_recording = recording

            # Grab latest frames from all cameras
            with image_lock:
                current_frames = {name: camera_frames[name].copy() for name in camera_names if name in camera_frames}
                has_all_frames = len(current_frames) == len(camera_names)

            # Only record if we're in recording mode AND have all camera frames
            if is_recording and has_all_frames:
                obs = follower.get_observation()

                # Construct frame according to dataset features
                with recording_lock:
                    # Action: convert dict of motor positions to numpy array
                    action_array = np.array(
                        [action[name] for name in dataset.features["action"]["names"]],
                        dtype=np.float32,
                    )

                    # Observation state: convert dict of motor positions to numpy array
                    obs_state_array = np.array(
                        [
                            obs[name]
                            for name in dataset.features["observation.state"]["names"]
                        ],
                        dtype=np.float32,
                    )

                    frame = {
                        "action": action_array,
                        "observation.state": obs_state_array,
                        "task": task,
                    }

                    # Add all camera images
                    for cam_name, cam_image in current_frames.items():
                        frame[f"observation.images.{cam_name}"] = cam_image

                    dataset.add_frame(frame=frame)

                    frame_count += 1
                    current_frame_count = frame_count

                # Log progress every Hz frames (every 1 second)
                if current_frame_count % hz == 0:
                    logger.debug(f"Recording: {current_frame_count} frames captured")
            elif is_recording and not has_all_frames:
                logger.warning(
                    "Recording active but not all camera frames available, skipping frame"
                )

            time.sleep(period)

        except KeyboardInterrupt:
            logger.info("Shutdown signal received (Ctrl+C)")
            running = False
            logger.info("Moving follower to initial position before disconnect...")
            follower.send_action(initial_position)
            # Wait for robot to reach initial position
            while True:
                obs = follower.get_observation()
                max_diff = max(
                    abs(obs[name] - initial_position[name])
                    for name in initial_position.keys()
                )
                if max_diff < 1.0:  # Within 1 degree tolerance
                    break
                time.sleep(0.05)
            logger.info("Disconnecting from leader robot...")
            leader.disconnect()
            logger.info("Disconnecting from follower robot...")
            follower.disconnect()
            dataset.finalize()
            logger.info(f"Total episodes recorded: {episode_count}")
            if push:
                logger.info("Pushing dataset to Hugging Face Hub...")
                dataset.push_to_hub()
                logger.info("Dataset uploaded successfully. Exiting.")
            else:
                logger.info("Dataset saved locally. Exiting.")
            return
        except Exception as e:
            logger.error(f"Error in teleoperation loop: {e}", exc_info=True)
            logger.info("Attempting cleanup...")
            running = False
            try:
                logger.info("Moving follower to initial position before disconnect...")
                follower.send_action(initial_position)
                # Wait for robot to reach initial position
                while True:
                    obs = follower.get_observation()
                    max_diff = max(
                        abs(obs[name] - initial_position[name])
                        for name in initial_position.keys()
                    )
                    if max_diff < 1.0:  # Within 1 degree tolerance
                        break
                    time.sleep(0.05)
                leader.disconnect()
                follower.disconnect()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")
            raise


if __name__ == "__main__":
    main()
