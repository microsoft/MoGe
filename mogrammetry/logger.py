"""
Logging utilities for MoGrammetry.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name: str = 'mogrammetry',
    level: str = 'INFO',
    log_file: Optional[str] = None,
    console: bool = True,
    colorize: bool = True
) -> logging.Logger:
    """
    Setup logger with console and file handlers.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console: Enable console logging
        colorize: Use colored output for console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []  # Clear existing handlers

    # Create formatters
    detailed_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    simple_format = '%(levelname)s: %(message)s'

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        if colorize and sys.stdout.isatty():
            console_formatter = ColoredFormatter(simple_format)
        else:
            console_formatter = logging.Formatter(simple_format)

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(detailed_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class ProgressLogger:
    """Logger with progress tracking capabilities."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_time = datetime.now()
        self.task_times = {}

    def start_task(self, task_name: str):
        """Start timing a task."""
        self.task_times[task_name] = {'start': datetime.now()}
        self.logger.info(f"Starting: {task_name}")

    def end_task(self, task_name: str, extra_info: str = ""):
        """End timing a task and log duration."""
        if task_name in self.task_times:
            start = self.task_times[task_name]['start']
            end = datetime.now()
            duration = (end - start).total_seconds()
            self.task_times[task_name]['end'] = end
            self.task_times[task_name]['duration'] = duration

            info = f"Completed: {task_name} (took {duration:.2f}s)"
            if extra_info:
                info += f" - {extra_info}"
            self.logger.info(info)
        else:
            self.logger.warning(f"Task '{task_name}' was never started")

    def log_stats(self, stats: dict, prefix: str = ""):
        """Log statistics in a formatted way."""
        if prefix:
            self.logger.info(f"{prefix}:")
        for key, value in stats.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")

    def get_total_time(self) -> float:
        """Get total elapsed time since logger creation."""
        return (datetime.now() - self.start_time).total_seconds()

    def get_summary(self) -> dict:
        """Get summary of all timed tasks."""
        summary = {
            'total_time': self.get_total_time(),
            'tasks': {}
        }
        for task_name, times in self.task_times.items():
            if 'duration' in times:
                summary['tasks'][task_name] = {
                    'duration': times['duration'],
                    'start': times['start'].isoformat(),
                    'end': times['end'].isoformat()
                }
        return summary


# Global logger instance
_global_logger = None


def get_logger() -> logging.Logger:
    """Get the global MoGrammetry logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logger()
    return _global_logger


def set_log_level(level: str):
    """Set the logging level for the global logger."""
    logger = get_logger()
    logger.setLevel(getattr(logging, level.upper()))
    for handler in logger.handlers:
        handler.setLevel(getattr(logging, level.upper()))
