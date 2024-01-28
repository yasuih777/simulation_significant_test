# !/usr/bin/env python3

import logging
from typing import Literal


def set_logger(level: Literal["debug", "info", "warning", "error"]) -> logging.Logger:
    log_level = get_level(level)

    logger = logging.getLogger(__name__)
    logger.handlers.clear()

    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s]"
        "(%(filename)s | %(funcName)s | %(lineno)s): %(message)s"
    )

    stream_handler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    stream_handler.setLevel(log_level)

    logger.addHandler(stream_handler)

    return logger


def get_level(log_level: Literal["debug", "info", "warning", "error"]) -> int:
    """get logger level

    Args:
        log_level (LogLevel): output log level on console

    Returns:
        int: logger level
    """
    level: int
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warning":
        level = logging.WARNING
    elif log_level == "error":
        level = logging.ERROR

    return level
