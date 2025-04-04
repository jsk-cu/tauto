"""
Tests for the logging module.
"""

import pytest
import logging
import os
from pathlib import Path

from tauto.utils.logging import get_logger, configure_logging


def test_get_logger():
    """Test getting a logger without configuration."""
    logger = get_logger("test_logger")
    
    assert logger.name == "test_logger"
    assert isinstance(logger, logging.Logger)


def test_get_logger_with_config():
    """Test getting a logger with configuration."""
    config = {
        "level": "DEBUG",
        "format": "%(levelname)s - %(message)s",
    }
    
    logger = get_logger("test_logger_config", config)
    
    assert logger.name == "test_logger_config"
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.DEBUG


def test_configure_logging(temp_dir):
    """Test configuring the logging system."""
    log_file = temp_dir / "test.log"
    
    config = {
        "level": "DEBUG",
        "format": "%(levelname)s - %(message)s",
        "log_file": str(log_file),
    }
    
    configure_logging(config)
    root_logger = logging.getLogger()
    
    # Check log level
    assert root_logger.level == logging.DEBUG
    
    # Check handlers
    assert len(root_logger.handlers) == 2
    assert any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
    assert any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
    
    # Check file handler
    file_handler = next(h for h in root_logger.handlers if isinstance(h, logging.FileHandler))
    assert file_handler.baseFilename == str(log_file)
    
    # Log a message
    root_logger.debug("Test message")
    
    # Check that the message was written to the file
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        content = f.read()
        assert "DEBUG - Test message" in content


def test_get_logger_caching():
    """Test that the same logger instance is returned for the same name."""
    logger1 = get_logger("test_logger_cache")
    logger2 = get_logger("test_logger_cache")
    
    assert logger1 is logger2