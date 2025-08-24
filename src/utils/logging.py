# -*- coding: utf-8 -*-

import logging
import sys
from pathlib import Path

def setup_logger(
    name: str,
    log_file: str = None,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> logging.Logger:
    """
    Setup a logger with optional file output.

    Parameters
    ----------
    name : str
        Name of the logger (usually __name__ of the module).
    log_file : str, optional
        Path to a file to save logs.
    level : int
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    fmt : str
        Logging format.

    Returns
    -------
    logging.Logger
        Configured logger object.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if setup_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(fmt)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
