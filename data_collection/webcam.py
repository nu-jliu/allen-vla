import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import socket
import threading
import time
from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Shared state for the latest frame
latest_frame = None
frame_lock = threading.Lock()
shutdown_event = threading.Event()

# Track active client connections
active_clients: set[socket.socket] = set()
clients_lock = threading.Lock()


def capture_thread(camera_index: int, width: int, height: int, fps: int, jpeg_quality: int) -> None:
    """Continuously capture frames from webcam and encode as JPEG."""
    global latest_frame

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_w != width or actual_h != height:
        logger.warning(
            f"Camera resolution mismatch: requested {width}x{height}, got {actual_w}x{actual_h}"
        )

    logger.info(f"Capture thread started: camera={camera_index}, resolution={actual_w}x{actual_h}")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    while not shutdown_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to capture frame")
            time.sleep(0.01)
            continue

        success, jpeg = cv2.imencode(".jpg", frame, encode_params)
        if success:
            with frame_lock:
                latest_frame = jpeg.tobytes()

        time.sleep(0.001)

    cap.release()
    logger.info("Capture thread stopped")


# Store camera info globally so the handler can access it
camera_info = {}


def shutdown_clients() -> None:
    """Force-close all active client sockets to unblock handler threads."""
    with clients_lock:
        for sock in active_clients:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        active_clients.clear()


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/stream":
            self._handle_stream()
        elif self.path == "/info":
            self._handle_info()
        else:
            self.send_error(404, "Not Found")

    def _handle_stream(self) -> None:
        client_addr = f"{self.client_address[0]}:{self.client_address[1]}"
        client_sock = self.request

        with clients_lock:
            active_clients.add(client_sock)
            count = len(active_clients)
        logger.info(f"Client connected: {client_addr} (active: {count})")

        try:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            while not shutdown_event.is_set():
                with frame_lock:
                    frame = latest_frame

                if frame is None:
                    time.sleep(0.01)
                    continue

                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n".encode())
                    self.wfile.write(b"\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break

                time.sleep(1.0 / camera_info.get("fps", 30))
        finally:
            with clients_lock:
                active_clients.discard(client_sock)
                count = len(active_clients)
            logger.info(f"Client disconnected: {client_addr} (active: {count})")

    def _handle_info(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(camera_info).encode())

    def log_message(self, format: str, *args: object) -> None:
        # Suppress default request logging
        pass


def list_v4l2_devices() -> str:
    """List available V4L2 devices with their names."""
    lines = []
    v4l2_dir = Path("/sys/class/video4linux")
    if not v4l2_dir.exists():
        return "  No V4L2 devices found"
    devices = sorted(v4l2_dir.iterdir(), key=lambda p: int(p.name.removeprefix("video")))
    if not devices:
        return "  No V4L2 devices found"
    for dev in devices:
        index = dev.name.removeprefix("video")
        name_file = dev / "name"
        name = name_file.read_text().strip() if name_file.exists() else "unknown"
        lines.append(f"  /dev/video{index}  {name}")
    return "\n".join(lines)


class CustomHelpFormatter(ArgumentParser):
    """ArgumentParser that appends V4L2 device list to help output."""

    def format_help(self) -> str:
        help_text = super().format_help()
        help_text += "\navailable v4l2 devices:\n"
        help_text += list_v4l2_devices() + "\n"
        return help_text


def main() -> None:
    global camera_info

    parser = CustomHelpFormatter(description="Webcam MJPEG streaming server")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port (default: 8080)")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Camera width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality 1-100 (default: 80)")
    args = parser.parse_args()

    camera_info = {"width": args.width, "height": args.height, "fps": args.fps}

    # Start capture thread
    thread = threading.Thread(
        target=capture_thread,
        args=(args.camera_index, args.width, args.height, args.fps, args.jpeg_quality),
        daemon=True,
    )
    thread.start()

    # Wait for first frame
    logger.info("Waiting for first frame...")
    while latest_frame is None:
        time.sleep(0.1)
    logger.info("First frame captured")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), MJPEGHandler)
    server.daemon_threads = True
    logger.info(f"MJPEG server running on http://0.0.0.0:{args.port}")
    logger.info(f"  Stream: http://localhost:{args.port}/stream")
    logger.info(f"  Info:   http://localhost:{args.port}/info")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        shutdown_event.set()
        shutdown_clients()
        server.shutdown()


if __name__ == "__main__":
    main()
