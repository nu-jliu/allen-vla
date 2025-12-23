#!/usr/bin/env python
"""
Inference Server for ACT policy.

This server loads a trained ACT policy and handles inference requests over TCP.
It receives observations from clients and returns predicted actions.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import pickle
import socket
import struct
import threading
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference, make_robot_action
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.utils import get_safe_torch_device

from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def recv_message(conn: socket.socket) -> bytes:
    """Receive a length-prefixed message from a socket."""
    # First, receive the 4-byte length prefix
    raw_msglen = recv_all(conn, 4)
    if not raw_msglen:
        return b""
    msglen = struct.unpack(">I", raw_msglen)[0]
    # Now receive the actual message
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


class InferenceServer:
    """TCP server that handles policy inference requests."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        task: str = "Policy evaluation",
    ):
        """Initialize the inference server.

        :param checkpoint: Path to trained policy checkpoint or HuggingFace repo ID
        :param device: Device to run inference on (cuda/cpu)
        :param task: Task description for inference
        """
        self.checkpoint = checkpoint
        self.device = get_safe_torch_device(device)
        self.task = task

        self.policy: PreTrainedPolicy | None = None
        self.preprocessor: PolicyProcessorPipeline | None = None
        self.postprocessor: PolicyProcessorPipeline | None = None
        self.ds_features: dict[str, Any] | None = None

        self._lock = threading.Lock()
        self._running = False

    def load_policy(self) -> None:
        """Load the policy and create preprocessor/postprocessor pipelines."""
        logger.info(f"Loading policy from {self.checkpoint}...")

        # Load policy config
        policy_cfg = PreTrainedConfig.from_pretrained(self.checkpoint)
        policy_cfg.pretrained_path = self.checkpoint
        policy_cfg.device = str(self.device)

        # Load dataset metadata for features and stats
        # The checkpoint should have a dataset reference or we need to load stats
        try:
            # Try to load the dataset info from the checkpoint
            dataset_repo_id = getattr(policy_cfg, "dataset_repo_id", None)
            if dataset_repo_id:
                ds = LeRobotDataset(dataset_repo_id)
                ds_meta = ds.meta
            else:
                # Load from the checkpoint directory
                checkpoint_path = Path(self.checkpoint)
                if checkpoint_path.exists():
                    # Local checkpoint - look for dataset info
                    ds = LeRobotDataset.from_pretrained(self.checkpoint)
                    ds_meta = ds.meta
                else:
                    # HuggingFace checkpoint - load metadata from hub
                    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
                    ds_meta = LeRobotDatasetMetadata.from_pretrained(self.checkpoint)
        except Exception as e:
            logger.warning(f"Could not load dataset metadata: {e}")
            logger.info("Attempting to load policy without dataset metadata...")
            ds_meta = None

        # Create policy
        if ds_meta is not None:
            self.policy = make_policy(policy_cfg, ds_meta=ds_meta)
            self.ds_features = ds_meta.features

            # Create preprocessor and postprocessor
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=self.checkpoint,
                dataset_stats=ds_meta.stats,
                preprocessor_overrides={
                    "device_processor": {"device": str(self.device)},
                },
            )
        else:
            # Fallback: try to load policy directly
            from lerobot.policies.act.modeling_act import ACTPolicy
            self.policy = ACTPolicy.from_pretrained(self.checkpoint)
            self.policy.to(self.device)

            # Load preprocessor/postprocessor from checkpoint
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=self.checkpoint,
                preprocessor_overrides={
                    "device_processor": {"device": str(self.device)},
                },
            )

            # Get features from policy config
            self.ds_features = {
                "action": {
                    "names": [f.name for f in policy_cfg.output_features.values()
                              if hasattr(f, 'name')] or
                             ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                              "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
                }
            }

        self.policy.eval()
        logger.info("Policy loaded successfully")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Policy type: {policy_cfg.type}")

    def process_observation(self, obs_dict: dict[str, np.ndarray]) -> dict[str, float]:
        """Process an observation and return the predicted action.

        :param obs_dict: Dictionary of observation arrays
        :return: Dictionary of action values
        """
        with self._lock:
            if self.policy is None:
                raise RuntimeError("Policy not loaded")

            # Reset if this is a new episode (handled by client sending reset message)
            with torch.inference_mode():
                # Prepare observation for inference
                observation = prepare_observation_for_inference(
                    obs_dict.copy(),
                    self.device,
                    task=self.task,
                    robot_type="so101_follower",
                )

                # Run through preprocessor
                if self.preprocessor is not None:
                    observation = self.preprocessor(observation)

                # Get action from policy
                action = self.policy.select_action(observation)

                # Run through postprocessor
                if self.postprocessor is not None:
                    action = self.postprocessor(action)

                # Convert to robot action dict
                if self.ds_features is not None:
                    action_dict = make_robot_action(action, self.ds_features)
                else:
                    # Fallback: convert tensor to dict with default names
                    action = action.squeeze(0).cpu().numpy()
                    action_names = [
                        "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
                    ]
                    action_dict = {name: float(action[i]) for i, name in enumerate(action_names)}

                return action_dict

    def reset_policy(self) -> None:
        """Reset the policy state for a new episode."""
        with self._lock:
            if self.policy is not None:
                self.policy.reset()
            if self.preprocessor is not None:
                self.preprocessor.reset()
            if self.postprocessor is not None:
                self.postprocessor.reset()
        logger.info("Policy state reset")

    def handle_client(self, conn: socket.socket, addr: tuple) -> None:
        """Handle a client connection.

        :param conn: Client socket connection
        :param addr: Client address tuple
        """
        logger.info(f"Client connected: {addr}")
        try:
            while self._running:
                # Receive message
                data = recv_message(conn)
                if not data:
                    break

                try:
                    request = pickle.loads(data)
                except Exception as e:
                    logger.error(f"Failed to unpickle request: {e}")
                    response = {"type": "error", "message": str(e)}
                    send_message(conn, pickle.dumps(response))
                    continue

                msg_type = request.get("type", "")

                if msg_type == "inference":
                    # Process observation and return action
                    try:
                        obs = request.get("observation", {})
                        action = self.process_observation(obs)
                        response = {"type": "action", "action": action}
                    except Exception as e:
                        logger.error(f"Inference error: {e}", exc_info=True)
                        response = {"type": "error", "message": str(e)}

                elif msg_type == "reset":
                    # Reset policy state
                    try:
                        self.reset_policy()
                        response = {"type": "ack"}
                    except Exception as e:
                        logger.error(f"Reset error: {e}")
                        response = {"type": "error", "message": str(e)}

                elif msg_type == "ping":
                    response = {"type": "ack"}

                else:
                    response = {"type": "error", "message": f"Unknown message type: {msg_type}"}

                # Send response
                send_message(conn, pickle.dumps(response))

        except Exception as e:
            logger.error(f"Error handling client {addr}: {e}", exc_info=True)
        finally:
            conn.close()
            logger.info(f"Client disconnected: {addr}")

    def start(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the inference server.

        :param host: Host address to bind to
        :param port: Port to listen on
        """
        # Load policy first
        self.load_policy()

        # Create server socket
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(5)

        self._running = True
        logger.info("=" * 60)
        logger.info("ACT Policy Inference Server")
        logger.info("=" * 60)
        logger.info(f"Listening on {host}:{port}")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)

        try:
            while self._running:
                try:
                    server.settimeout(1.0)
                    conn, addr = server.accept()
                    # Handle each client in a new thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(conn, addr),
                        daemon=True,
                    )
                    client_thread.start()
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            logger.info("\nShutting down server...")
        finally:
            self._running = False
            server.close()
            logger.info("Server stopped")


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(
        description="ACT Policy Inference Server",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with local checkpoint
  python policy/act/inference_server.py \\
    --checkpoint ./outputs/act_training/pretrained_model \\
    --port 8000

  # Start server with HuggingFace model
  python policy/act/inference_server.py \\
    --checkpoint username/act_policy \\
    --device cuda \\
    --port 8000
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained policy checkpoint or HuggingFace repo ID",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (default: cuda)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Policy evaluation",
        help="Task description for inference",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    server = InferenceServer(
        checkpoint=args.checkpoint,
        device=args.device,
        task=args.task,
    )

    server.start(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
