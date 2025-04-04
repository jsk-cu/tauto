"""
Logging utilities for TAuto.
"""

import logging
import os
import sys
from typing import Dict, Any, Optional

# Global dictionary to store logger instances
_loggers = {}


def configure_logging(config: Dict[str, Any]) -> None:
    """
    Configure the logging system based on configuration.
    
    Args:
        config: Logging configuration dictionary
    """
    log_level = config.get("level", "INFO")
    log_format = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = config.get("log_file", None)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str, config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger instance with specified name.
    
    Args:
        name: Name of the logger
        config: Optional logging configuration
        
    Returns:
        logging.Logger: Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    
    # Configure logging if not done yet and config is provided
    if config is not None:
        # Configure global logging
        configure_logging(config)
        
        # Also set the level specifically for this logger
        log_level = config.get("level", "INFO")
        logger.setLevel(getattr(logging, log_level))
    
    _loggers[name] = logger
    return logger