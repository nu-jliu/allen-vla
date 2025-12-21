import logging
from typing import Any


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
            fmt=f"[%(asctime)s] [%(name)s] [{log_color}%(levelname)s{self.RESET}] -- %(message)s{self.RESET}",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return formatter.format(record)


def setup_logging() -> None:
    """Setup colored logging for the application."""
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
