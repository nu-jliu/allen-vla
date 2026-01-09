import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import logging
import threading
import socket
import cv2
import numpy as np
import shutil

from argparse import ArgumentParser

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from utils import setup_logging, build_dataset_features_for_so101
from robot_utils import add_common_robot_args, initialize_robots

setup_logging()
logger = logging.getLogger(__name__)


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
        "--policy-type",
        type=str,
        required=True,
        help="Policy type (e.g., act, diffusion)",
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
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Camera width in pixels (default: 640)",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Camera height in pixels (default: 480)",
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
    policy_type = args.policy_type
    robot_type = args.robot_type
    task = args.task
    # Construct repo_id as {username}/{policy}-{robot}-{task}
    repo_id = f"{username}/{policy_type}-{robot_type}-{task}"
    hz = args.hz
    camera_index = args.camera_index
    camera_width = args.camera_width
    camera_height = args.camera_height
    root = args.root
    push = args.push
    port = args.port

    logger.info("Starting data collection with configuration:")
    logger.info(f"  Leader port: {leader_port}")
    logger.info(f"  Leader ID: {leader_id}")
    logger.info(f"  Follower port: {follower_port}")
    logger.info(f"  Follower ID: {follower_id}")
    logger.info(f"  Repo ID: {repo_id}")
    logger.info(f"  Control frequency: {hz} Hz")
    logger.info(f"  Camera index: {camera_index}")
    logger.info(f"  Camera resolution: {camera_width}x{camera_height}")
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

    cap = cv2.VideoCapture(camera_index)
    # Configure camera resolution and FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    cap.set(cv2.CAP_PROP_FPS, hz)
    # Verify camera properties match requested values
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if cam_width != camera_width or cam_height != camera_height:
        logger.warning(
            f"Camera resolution mismatch: requested {camera_width}x{camera_height}, "
            f"got {cam_width}x{cam_height}. This may cause issues during training/inference."
        )

    # Build dataset features using centralized helper function
    # This ensures features are always consistent between collection and training
    obs_features, action_features = build_dataset_features_for_so101(
        cam_height=cam_height,
        cam_width=cam_width,
        camera_name="front",
    )

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

    # Image capture state
    latest_image = None
    image_valid = False
    image_lock = threading.Lock()

    def image_capture_thread() -> None:
        """Continuously capture images from camera in separate thread."""
        nonlocal latest_image, image_valid
        logger.info("Starting image capture thread...")

        while running:
            try:
                succ, image = cap.read()
                with image_lock:
                    if succ:
                        latest_image = image.copy()
                        image_valid = True
                    else:
                        image_valid = False
                        logger.warning("Failed to capture image")
                # Small sleep to avoid excessive CPU usage
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"Error in image capture thread: {e}")
                with image_lock:
                    image_valid = False

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
                                                "[START] Recording started - Press s to stop",
                                            )
                                        else:
                                            recording = False
                                            if frame_count > 0:
                                                dataset.save_episode()
                                                episode_count += 1
                                                msg = f"[SAVED] Episode {episode_count} saved with {frame_count} frames"
                                                logger.info(
                                                    f"=== RECORDING STOPPED === (Saved {frame_count} frames, Episode {episode_count})"
                                                )
                                            else:
                                                msg = "[STOP] Recording stopped (No frames captured)"
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
                                            logger.info(
                                                f"=== EPISODE ABORTED === (Discarded {discarded} frames)"
                                            )
                                            send_response(
                                                client_socket,
                                                f"[ABORT] Episode aborted - Discarded {discarded} frames",
                                            )
                                        else:
                                            send_response(
                                                client_socket,
                                                "[INFO] Not recording - nothing to abort",
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
                                        f"[ERROR] Unknown command: '{char}' - Use s, a, or q",
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

    # Start image capture thread
    image_thread = threading.Thread(target=image_capture_thread, daemon=True)
    image_thread.start()

    # Start command server thread
    command_thread = threading.Thread(target=command_server_thread, daemon=True)
    command_thread.start()

    logger.info(
        f"Connect via 'telnet localhost {port}' to control recording. Press Ctrl+C to exit."
    )

    logger.info(f"Starting teleoperation loop at {hz} Hz...")
    period = 1.0 / hz

    while True:
        try:
            action = leader.get_action()
            follower.send_action(action)

            # Get the current recording state and image
            with recording_lock:
                is_recording = recording

            with image_lock:
                has_valid_image = image_valid
                current_image = latest_image.copy() if image_valid else None

            # Only record if we're in recording mode AND have a valid image
            if is_recording and has_valid_image:
                obs = follower.get_observation()

                # logger.info(f"Observation: {pprint.pformat(obs, indent=2)}")
                # logger.info(f"Action: {pprint.pformat(action, indent=2)}")

                # Construct frame according to dataset features
                # Action: convert dict of motor positions to numpy array
                with recording_lock:
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
                        "observation.images.front": current_image,
                        "task": task,
                    }

                    dataset.add_frame(frame=frame)

                    frame_count += 1
                    current_frame_count = frame_count

                # Log progress every Hz frames (every 1 second)
                if current_frame_count % hz == 0:
                    logger.debug(f"Recording: {current_frame_count} frames captured")
            elif is_recording and not has_valid_image:
                logger.warning(
                    "Recording active but no valid image available, skipping frame"
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
