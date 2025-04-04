"""
Tests for the command line interface.
"""

import pytest
from click.testing import CliRunner
import os
import yaml
from pathlib import Path

from tauto.cli import cli, init_config, validate_config, show_config


@pytest.fixture
def runner():
    """Provide a CLI runner for testing."""
    return CliRunner()


def test_cli_help(runner):
    """Test that the CLI provides help information."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "TAuto: PyTorch Model Optimization Framework CLI" in result.output


def test_init_config(runner, temp_dir):
    """Test initializing a configuration file."""
    config_path = temp_dir / "test_config.yaml"
    
    # Run the command
    result = runner.invoke(cli, ["init-config", "--output", str(config_path)])
    
    # Check that the command succeeded
    assert result.exit_code == 0
    assert f"Configuration saved to {config_path}" in result.output
    
    # Check that the file was created
    assert os.path.exists(config_path)
    
    # Check that the file contains a valid configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        assert "data" in config
        assert "training" in config
        assert "optimization" in config
        assert "profiling" in config
        assert "wandb" in config
        assert "logging" in config


def test_init_config_existing_file(runner, temp_dir):
    """Test that init-config fails if the file exists."""
    config_path = temp_dir / "existing_config.yaml"
    
    # Create the file
    with open(config_path, "w") as f:
        f.write("test")
    
    # Run the command
    result = runner.invoke(cli, ["init-config", "--output", str(config_path)])
    
    # Check that the command failed
    assert result.exit_code == 1
    assert f"Error: {config_path} already exists" in result.output


def test_init_config_force(runner, temp_dir):
    """Test that init-config overwrites the file with --force."""
    config_path = temp_dir / "existing_config.yaml"
    
    # Create the file
    with open(config_path, "w") as f:
        f.write("test")
    
    # Run the command with --force
    result = runner.invoke(cli, ["init-config", "--output", str(config_path), "--force"])
    
    # Check that the command succeeded
    assert result.exit_code == 0
    assert f"Configuration saved to {config_path}" in result.output
    
    # Check that the file was overwritten
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        assert "data" in config


def test_validate_config(runner, config_file):
    """Test validating a configuration file."""
    # Run the command
    result = runner.invoke(cli, ["validate-config", "--config", str(config_file)])
    
    # Check that the command succeeded
    assert result.exit_code == 0
    assert f"Configuration file {config_file} is valid" in result.output


def test_validate_config_invalid(runner, temp_dir):
    """Test validating an invalid configuration file."""
    # Create an invalid configuration file
    invalid_config_path = temp_dir / "invalid_config.yaml"
    with open(invalid_config_path, "w") as f:
        f.write("invalid: yaml:\n  - missing")
    
    # Run the command
    result = runner.invoke(cli, ["validate-config", "--config", str(invalid_config_path)])
    
    # Check that the command failed
    assert result.exit_code == 1
    assert "Error validating configuration" in result.output


def test_show_config(runner, config_file):
    """Test showing a configuration file."""
    # Run the command
    result = runner.invoke(cli, ["show-config", "--config", str(config_file)])
    
    # Check that the command succeeded
    assert result.exit_code == 0
    
    # Check that the output contains configuration sections
    assert "data:" in result.output
    assert "training:" in result.output
    assert "optimization:" in result.output
    assert "profiling:" in result.output
    assert "wandb:" in result.output
    assert "logging:" in result.output