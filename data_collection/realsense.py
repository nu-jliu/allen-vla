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
import numpy as np
import pyrealsense2 as rs

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


def capture_thread(serial: str, width: int, height: int, fps: int, jpeg_quality: int) -> None:
    """Continuously capture frames from RealSense camera and encode as JPEG."""
    global latest_frame

    pipeline = rs.pipeline()
    config = rs.config()

    if serial:
        config.enable_device(serial)

    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)
    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    actual_w = stream.width()
    actual_h = stream.height()

    if actual_w != width or actual_h != height:
        logger.warning(
            f"RealSense resolution mismatch: requested {width}x{height}, got {actual_w}x{actual_h}"
        )

    logger.info(f"Capture thread started: serial={serial or 'default'}, resolution={actual_w}x{actual_h}")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    try:
        while not shutdown_event.is_set():
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            success, jpeg = cv2.imencode(".jpg", frame, encode_params)
            if success:
                with frame_lock:
                    latest_frame = jpeg.tobytes()
    finally:
        pipeline.stop()
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


def get_realsense_devices() -> list[dict[str, str]]:
    """Query connected RealSense devices and return their info."""
    try:
        ctx = rs.context()
        devices = []
        for dev in ctx.query_devices():
            devices.append({
                "serial": dev.get_info(rs.camera_info.serial_number),
                "name": dev.get_info(rs.camera_info.name),
            })
        return devices
    except Exception:
        return []


def build_epilog() -> str:
    """Build help epilog with available RealSense devices."""
    devices = get_realsense_devices()
    if not devices:
        return "available RealSense devices: none detected"
    lines = ["available RealSense devices:"]
    for dev in devices:
        lines.append(f"  {dev['serial']}  ({dev['name']})")
    return "\n".join(lines)


def main() -> None:
    global camera_info

    parser = ArgumentParser(
        description="RealSense MJPEG streaming server",
        epilog=build_epilog(),
        formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(prog),
    )
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port (default: 8080)")
    parser.add_argument("--serial", type=str, default="", help="RealSense serial number (default: first device)")
    parser.add_argument("--width", type=int, default=640, help="Camera width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality 1-100 (default: 80)")
    args = parser.parse_args()

    camera_info = {"width": args.width, "height": args.height, "fps": args.fps}

    # Start capture thread
    thread = threading.Thread(
        target=capture_thread,
        args=(args.serial, args.width, args.height, args.fps, args.jpeg_quality),
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
