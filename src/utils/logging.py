"""Logging utilities for the project.

This module provides standardized logging functionality for the entire codebase,
including functions to create consistent loggers and control external library
logging verbosity.
"""

import os
import logging
import sys
from pathlib import Path
from typing import Optional, Union, List


def setup_logger(
    name: str = "glioma",
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Setup logger with console and file handlers.

    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (if None, file logging is disabled)
        console: Whether to enable console logging
        format_string: Custom format string for logging

    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    formatter = logging.Formatter(format_string)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(str(log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent logging from propagating to the root logger
    logger.propagate = False

    return logger


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger with standard formatting.

    Args:
        name: Name of the logger (typically __name__)
        level: Logging level to set

    Returns:
        Configured logger instance
    """
    return setup_logger(name=name, level=level)


def disable_external_loggers(loggers: List[str] = None) -> None:
    """Disable verbose logging from external libraries.

    Args:
        loggers: List of logger names to disable
    """
    if loggers is None:
        loggers = ["transformers", "sentence_transformers", "openai", "httpx"]

    for logger_name in loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
