"""
Basic usage example for TAuto.

This script demonstrates how to use the core functionality
implemented in Phase 1 of the TAuto project.
"""

import os
import sys
import yaml
from pathlib import Path

# Add parent directory to path to import tauto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tauto
from tauto.config import get_default_config, save_config, load_config, ConfigManager
from tauto.utils import get_logger, configure_logging, setup_wandb, log_config


def main():
    # Create a configuration directory
    config_dir = Path("config")
    os.makedirs(config_dir, exist_ok=True)
    
    # Get the default configuration
    default_config = get_default_config()
    
    # Save the default configuration
    default_config_path = config_dir / "default_config.yaml"
    with open(default_config_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False)
    print(f"Default configuration saved to {default_config_path}")
    
    # Create a custom configuration
    config_manager = ConfigManager(default_config)
    custom_config = config_manager.update({
        "data": {"num_workers": 2},
        "training": {"batch_size": 16},
        "wandb": {"project": "tauto-demo"},
    }).config
    
    # Save the custom configuration
    custom_config_path = config_dir / "custom_config.yaml"
    save_config(custom_config, custom_config_path)
    print(f"Custom configuration saved to {custom_config_path}")
    
    # Load the custom configuration
    loaded_config = load_config(custom_config_path)
    print(f"Loaded configuration: {loaded_config.data['num_workers']} workers, {loaded_config.training['batch_size']} batch size")
    
    # Configure logging
    configure_logging(loaded_config.logging)
    logger = get_logger("tauto.example")
    
    logger.info("TAuto example script running")
    logger.debug("This is a debug message")
    
    # Set up W&B (disabled by default)
    loaded_config.wandb["enabled"] = False
    run = setup_wandb(loaded_config.wandb)
    
    if run:
        logger.info("W&B run initialized")
        log_config(loaded_config.to_dict(), run)
    else:
        logger.info("W&B tracking disabled")
    
    logger.info("Example script completed successfully")


if __name__ == "__main__":
    main()