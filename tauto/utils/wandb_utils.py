"""
Weights & Biases integration utilities for TAuto.
"""

import os
from typing import Dict, Any, Optional, Union
import wandb
from pathlib import Path

from tauto.utils.logging import get_logger

logger = get_logger(__name__)


def setup_wandb(config: Dict[str, Any]) -> Optional[wandb.run]:
    """
    Set up Weights & Biases for experiment tracking.
    
    Args:
        config: W&B configuration dictionary
        
    Returns:
        Optional[wandb.run]: W&B run if enabled, None otherwise
    """
    if not config.get("enabled", True):
        logger.info("Weights & Biases tracking is disabled")
        return None
    
    # Get W&B configuration
    project = config.get("project", "tauto")
    entity = config.get("entity", None)
    run_name = config.get("name", None)
    tags = config.get("tags", ["tauto"])
    
    # Initialize W&B
    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=tags,
            config={},  # We'll log the full config separately
        )
        logger.info(f"Initialized W&B run: {run.name}")
        return run
    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {e}")
        logger.warning("Continuing without W&B tracking")
        return None


def log_config(config: Dict[str, Any], run: Optional[wandb.run] = None) -> None:
    """
    Log configuration to W&B.
    
    Args:
        config: Configuration dictionary
        run: W&B run instance
    """
    if run is None:
        run = wandb.run
    
    if run is None:
        logger.warning("No active W&B run, skipping config logging")
        return
    
    # Log all configuration as a flat dictionary
    flattened_config = {}
    
    def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                _flatten_dict(value, f"{prefix}{key}/")
            else:
                flattened_config[f"{prefix}{key}"] = value
    
    _flatten_dict(config)
    run.config.update(flattened_config)
    logger.info("Logged configuration to W&B")


def log_model(model_path: Union[str, Path], run: Optional[wandb.run] = None, name: str = "model") -> None:
    """
    Log a model to W&B.
    
    Args:
        model_path: Path to the model file
        run: W&B run instance
        name: Name for the artifact
    """
    if run is None:
        run = wandb.run
    
    if run is None:
        logger.warning("No active W&B run, skipping model logging")
        return
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        logger.warning(f"Model file {model_path} does not exist, skipping logging")
        return
    
    artifact = wandb.Artifact(name=name, type="model")
    artifact.add_file(str(model_path))
    run.log_artifact(artifact)
    logger.info(f"Logged model {model_path} to W&B as {name}")


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None, run: Optional[wandb.run] = None) -> None:
    """
    Log metrics to W&B.
    
    Args:
        metrics: Metrics dictionary
        step: Optional step number
        run: W&B run instance
    """
    if run is None:
        run = wandb.run
    
    if run is None:
        logger.warning("No active W&B run, skipping metrics logging")
        return
    
    run.log(metrics, step=step)