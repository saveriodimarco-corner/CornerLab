import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cornerlab")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(LOG_DIR / "cornerlab.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
