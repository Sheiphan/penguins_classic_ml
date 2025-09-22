"""Logging configuration using Loguru for structured logging."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from .config import LoggingConfig


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    Set up logging configuration using Loguru.
    
    Args:
        config: LoggingConfig instance. If None, uses default configuration.
    """
    if config is None:
        config = LoggingConfig()
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        level=config.level,
        format=config.format,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # Add file handler if log_file is specified
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            config.log_file,
            level=config.level,
            format=config.format,
            rotation=config.rotation,
            retention=config.retention,
            backtrace=True,
            diagnose=True,
        )
    
    logger.info("Logging configured successfully")


def get_logger(name: str):
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Convenience function for quick setup
def setup_default_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Quick setup for default logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    config = LoggingConfig(level=level, log_file=log_file)
    setup_logging(config)