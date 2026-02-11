import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
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
capture_running = True


def capture_thread(camera_index: int, width: int, height: int, fps: int, jpeg_quality: int) -> None:
    """Continuously capture frames from webcam and encode as JPEG."""
    global latest_frame, capture_running

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

    while capture_running:
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


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/stream":
            self._handle_stream()
        elif self.path == "/info":
            self._handle_info()
        else:
            self.send_error(404, "Not Found")

    def _handle_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        while capture_running:
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
            except (BrokenPipeError, ConnectionResetError):
                break

            time.sleep(1.0 / camera_info.get("fps", 30))

    def _handle_info(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(camera_info).encode())

    def log_message(self, format: str, *args: object) -> None:
        # Suppress default request logging
        pass


def main() -> None:
    global camera_info, capture_running

    parser = ArgumentParser(description="Webcam MJPEG streaming server")
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
    logger.info(f"MJPEG server running on http://0.0.0.0:{args.port}")
    logger.info(f"  Stream: http://localhost:{args.port}/stream")
    logger.info(f"  Info:   http://localhost:{args.port}/info")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        capture_running = False
        server.shutdown()


if __name__ == "__main__":
    main()
