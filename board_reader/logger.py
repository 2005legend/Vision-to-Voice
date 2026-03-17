"""Logger factory for IntelliAgent Board Reader."""

import logging
from logging.handlers import RotatingFileHandler

from board_reader.config import Config

_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 3


def get_logger(name: str, config: Config) -> logging.Logger:
    """Configure and return a named logger backed by a RotatingFileHandler.

    Args:
        name: Logger name (used as the stage identifier in log entries).
        config: Config instance providing log_level and log_file.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times with the same name
    if logger.handlers:
        return logger

    level = getattr(logging, config.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    handler = RotatingFileHandler(
        config.log_file,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    logger.addHandler(handler)
    return logger
