"""
Configuration management module for TAuto.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy
import yaml
import os

from tauto.config.defaults import get_default_config


@dataclass
class TAutoConfig:
    """Configuration class for TAuto."""
    
    data: Dict[str, Any]
    training: Dict[str, Any]
    optimization: Dict[str, Any]
    profiling: Dict[str, Any]
    wandb: Dict[str, Any]
    logging: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TAutoConfig':
        """
        Create a TAutoConfig instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            TAutoConfig: Configuration object
        """
        # Ensure all required sections are present
        default_config = get_default_config()
        for section in default_config:
            if section not in config_dict:
                config_dict[section] = {}
            
            # Fill in missing values with defaults
            for key, value in default_config[section].items():
                if key not in config_dict[section]:
                    config_dict[section][key] = value
        
        return cls(
            data=config_dict.get("data", {}),
            training=config_dict.get("training", {}),
            optimization=config_dict.get("optimization", {}),
            profiling=config_dict.get("profiling", {}),
            wandb=config_dict.get("wandb", {}),
            logging=config_dict.get("logging", {}),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            "data": self.data,
            "training": self.training,
            "optimization": self.optimization,
            "profiling": self.profiling,
            "wandb": self.wandb,
            "logging": self.logging,
        }
    
    def __getitem__(self, key: str) -> Any:
        """
        Get a configuration section.
        
        Args:
            key: Section name
            
        Returns:
            Any: Configuration section
        """
        return getattr(self, key)


class ConfigManager:
    """Manager for TAuto configurations."""
    
    def __init__(self, config: Optional[Union[Dict[str, Any], TAutoConfig]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config: Initial configuration (dictionary or TAutoConfig)
        """
        if config is None:
            self.config = TAutoConfig.from_dict(get_default_config())
        elif isinstance(config, dict):
            self.config = TAutoConfig.from_dict(config)
        else:
            self.config = config
    
    def update(self, updates: Dict[str, Any]) -> 'ConfigManager':
        """
        Update the configuration with new values.
        
        Args:
            updates: Dictionary with updates
            
        Returns:
            ConfigManager: Self for chaining
        """
        config_dict = self.config.to_dict()
        
        for section, section_updates in updates.items():
            if section in config_dict:
                config_dict[section].update(section_updates)
        
        self.config = TAutoConfig.from_dict(config_dict)
        return self
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the configuration to a YAML file.
        
        Args:
            path: Path to save the configuration
        """
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        
        with open(path, "w") as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ConfigManager':
        """
        Load a configuration from a YAML file.
        
        Args:
            path: Path to the configuration file
            
        Returns:
            ConfigManager: Configuration manager with loaded configuration
        """
        path = Path(path)
        
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)


def load_config(path: Union[str, Path]) -> TAutoConfig:
    """
    Load a configuration from a file.
    
    Args:
        path: Path to the configuration file
        
    Returns:
        TAutoConfig: Loaded configuration
    """
    return ConfigManager.load(path).config


def save_config(config: TAutoConfig, path: Union[str, Path]) -> None:
    """
    Save a configuration to a file.
    
    Args:
        config: Configuration to save
        path: Path to save the configuration
    """
    ConfigManager(config).save(path)


def update_config(config: TAutoConfig, updates: Dict[str, Any]) -> TAutoConfig:
    """
    Update a configuration with new values without modifying the original.
    
    Args:
        config: Configuration to update
        updates: Dictionary with updates
        
    Returns:
        TAutoConfig: Updated configuration
    """
    # Create a deep copy of the config dictionary to avoid modifying the original
    config_dict = copy.deepcopy(config.to_dict())
    
    # Apply updates to the copy
    for section, section_updates in updates.items():
        if section in config_dict:
            config_dict[section].update(section_updates)
    
    # Create a new TAutoConfig instance from the updated dictionary
    return TAutoConfig.from_dict(config_dict)