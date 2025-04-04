"""
Tests for the configuration module.
"""

import os
import pytest
import yaml
from pathlib import Path

from tauto.config import (
    load_config,
    save_config,
    update_config,
    ConfigManager,
    TAutoConfig,
    get_default_config,
)


def test_get_default_config():
    """Test that get_default_config returns a valid configuration."""
    config = get_default_config()
    
    # Check that all required sections are present
    assert "data" in config
    assert "training" in config
    assert "optimization" in config
    assert "profiling" in config
    assert "wandb" in config
    assert "logging" in config


def test_tauto_config_from_dict():
    """Test creating a TAutoConfig from a dictionary."""
    config_dict = get_default_config()
    config = TAutoConfig.from_dict(config_dict)
    
    # Check that all sections were loaded correctly
    assert config.data == config_dict["data"]
    assert config.training == config_dict["training"]
    assert config.optimization == config_dict["optimization"]
    assert config.profiling == config_dict["profiling"]
    assert config.wandb == config_dict["wandb"]
    assert config.logging == config_dict["logging"]


def test_tauto_config_to_dict():
    """Test converting a TAutoConfig to a dictionary."""
    config_dict = get_default_config()
    config = TAutoConfig.from_dict(config_dict)
    
    # Convert back to dictionary
    result_dict = config.to_dict()
    
    # Check that the result matches the original
    assert result_dict == config_dict


def test_tauto_config_getitem():
    """Test the __getitem__ method of TAutoConfig."""
    config_dict = get_default_config()
    config = TAutoConfig.from_dict(config_dict)
    
    # Check that sections can be accessed via __getitem__
    assert config["data"] == config_dict["data"]
    assert config["training"] == config_dict["training"]
    assert config["optimization"] == config_dict["optimization"]
    assert config["profiling"] == config_dict["profiling"]
    assert config["wandb"] == config_dict["wandb"]
    assert config["logging"] == config_dict["logging"]


def test_config_manager_init():
    """Test initializing a ConfigManager."""
    # With default config
    manager = ConfigManager()
    assert isinstance(manager.config, TAutoConfig)
    
    # With dictionary
    config_dict = get_default_config()
    manager = ConfigManager(config_dict)
    assert isinstance(manager.config, TAutoConfig)
    
    # With TAutoConfig
    config = TAutoConfig.from_dict(config_dict)
    manager = ConfigManager(config)
    assert isinstance(manager.config, TAutoConfig)
    assert manager.config is config


def test_config_manager_update():
    """Test updating a ConfigManager."""
    manager = ConfigManager()
    
    # Update a single value
    manager.update({"data": {"num_workers": 8}})
    assert manager.config.data["num_workers"] == 8
    
    # Update multiple values
    manager.update({
        "training": {"batch_size": 64},
        "optimization": {"torch_compile": {"enabled": False}},
    })
    assert manager.config.training["batch_size"] == 64
    assert manager.config.optimization["torch_compile"]["enabled"] is False


def test_config_save_load(temp_dir):
    """Test saving and loading a configuration."""
    config_path = temp_dir / "config.yaml"
    
    # Create and save a configuration
    config = TAutoConfig.from_dict(get_default_config())
    save_config(config, config_path)
    
    # Check that the file was created
    assert config_path.exists()
    
    # Load the configuration
    loaded_config = load_config(config_path)
    
    # Check that the loaded configuration matches the original
    assert loaded_config.to_dict() == config.to_dict()


def test_update_config():
    """Test updating a configuration."""
    config = TAutoConfig.from_dict(get_default_config())
    
    # Update the configuration
    updates = {
        "data": {"num_workers": 8},
        "training": {"batch_size": 64},
    }
    updated_config = update_config(config, updates)
    
    # Check that the updates were applied
    assert updated_config.data["num_workers"] == 8
    assert updated_config.training["batch_size"] == 64
    
    # Check that the original configuration was not modified
    assert config.data["num_workers"] != 8
    assert config.training["batch_size"] != 64