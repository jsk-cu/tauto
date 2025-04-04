"""
Tests for the Weights & Biases utilities.
Note: These tests use mocking to avoid actual W&B API calls.
"""

import pytest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path

from tauto.utils.wandb_utils import setup_wandb, log_config, log_model, log_metrics


@pytest.fixture
def mock_wandb_run():
    """Provide a mock W&B run."""
    run = MagicMock()
    run.name = "mock-run"
    # Create a MagicMock for config that has an update method which is also a MagicMock
    run.config = MagicMock()
    run.config.update = MagicMock()
    return run


@patch("wandb.init")
def test_setup_wandb_enabled(mock_init, mock_wandb_run):
    """Test setting up W&B when enabled."""
    mock_init.return_value = mock_wandb_run
    
    config = {
        "enabled": True,
        "project": "test-project",
        "entity": "test-entity",
        "name": "test-run",
        "tags": ["test"],
    }
    
    run = setup_wandb(config)
    
    # Check that wandb.init was called with the correct arguments
    mock_init.assert_called_once_with(
        project="test-project",
        entity="test-entity",
        name="test-run",
        tags=["test"],
        config={},
    )
    
    # Check that the run was returned
    assert run is mock_wandb_run


@patch("wandb.init")
def test_setup_wandb_disabled(mock_init):
    """Test setting up W&B when disabled."""
    config = {
        "enabled": False,
    }
    
    run = setup_wandb(config)
    
    # Check that wandb.init was not called
    mock_init.assert_not_called()
    
    # Check that None was returned
    assert run is None


@patch("wandb.init")
def test_setup_wandb_exception(mock_init):
    """Test handling exceptions when setting up W&B."""
    mock_init.side_effect = Exception("Test exception")
    
    config = {
        "enabled": True,
        "project": "test-project",
    }
    
    run = setup_wandb(config)
    
    # Check that wandb.init was called
    mock_init.assert_called_once()
    
    # Check that None was returned
    assert run is None


def test_log_config(mock_wandb_run):
    """Test logging configuration to W&B."""
    config = {
        "data": {
            "num_workers": 4,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
        },
    }
    
    log_config(config, mock_wandb_run)
    
    # Check that the config was updated with flattened values
    assert mock_wandb_run.config.update.call_count == 1
    
    # Get the flattened config that was passed to update
    call_args = mock_wandb_run.config.update.call_args[0][0]
    
    # Check that the flattened config contains the expected values
    assert call_args["data/num_workers"] == 4
    assert call_args["training/batch_size"] == 32
    assert call_args["training/learning_rate"] == 0.001


@patch("wandb.run", None)
def test_log_config_no_run():
    """Test logging configuration when no W&B run is active."""
    config = {
        "data": {
            "num_workers": 4,
        },
    }
    
    # This should not raise an exception
    log_config(config)


@patch("wandb.Artifact")
def test_log_model(mock_artifact, mock_wandb_run, temp_dir):
    """Test logging a model to W&B."""
    # Create a mock artifact
    artifact = MagicMock()
    mock_artifact.return_value = artifact
    
    # Create a model file
    model_path = temp_dir / "model.pt"
    with open(model_path, "w") as f:
        f.write("test")
    
    log_model(model_path, mock_wandb_run, "test-model")
    
    # Check that an artifact was created
    mock_artifact.assert_called_once_with(name="test-model", type="model")
    
    # Check that the model file was added to the artifact
    artifact.add_file.assert_called_once_with(str(model_path))
    
    # Check that the artifact was logged
    mock_wandb_run.log_artifact.assert_called_once_with(artifact)


@patch("wandb.Artifact")
def test_log_model_nonexistent(mock_artifact, mock_wandb_run, temp_dir):
    """Test logging a nonexistent model file."""
    model_path = temp_dir / "nonexistent.pt"
    
    log_model(model_path, mock_wandb_run, "test-model")
    
    # Check that no artifact was created
    mock_artifact.assert_not_called()
    
    # Check that no artifact was logged
    mock_wandb_run.log_artifact.assert_not_called()


def test_log_metrics(mock_wandb_run):
    """Test logging metrics to W&B."""
    metrics = {
        "loss": 0.1,
        "accuracy": 0.9,
    }
    
    log_metrics(metrics, step=10, run=mock_wandb_run)
    
    # Check that the metrics were logged
    mock_wandb_run.log.assert_called_once_with(metrics, step=10)


@patch("wandb.run", None)
def test_log_metrics_no_run():
    """Test logging metrics when no W&B run is active."""
    metrics = {
        "loss": 0.1,
    }
    
    # This should not raise an exception
    log_metrics(metrics)