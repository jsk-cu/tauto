"""
Model registry implementation for TAuto.

This module implements a registry for machine learning models,
allowing models to be registered, queried, and instantiated
by name, architecture, task, or other attributes.
"""

import inspect
from typing import Dict, Any, Optional, List, Callable, Type, Union, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass

from tauto.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Information about a registered model."""
    
    name: str
    architecture: str
    description: str
    task: str
    input_size: Optional[Union[int, Tuple[int, ...]]] = None
    output_size: Optional[Union[int, Tuple[int, ...]]] = None
    paper_url: Optional[str] = None
    model_cls: Optional[Type[nn.Module]] = None
    factory_fn: Optional[Callable] = None
    default_args: Optional[Dict[str, Any]] = None
    pretrained_available: bool = False
    reference_speed: Optional[Dict[str, float]] = None
    reference_memory: Optional[Dict[str, float]] = None


class ModelRegistry:
    """Registry for machine learning models."""
    
    def __init__(self):
        """Initialize the model registry."""
        self._registry: Dict[str, ModelInfo] = {}
    
    def register(self, model_info: ModelInfo) -> None:
        """
        Register a model with the registry.
        
        Args:
            model_info: Information about the model
        """
        if model_info.name in self._registry:
            logger.warning(f"Model {model_info.name} already registered. Overwriting.")
        
        self._registry[model_info.name] = model_info
        logger.info(f"Registered model {model_info.name} for task {model_info.task}")
    
    def create(self, name: str, **kwargs) -> nn.Module:
        """
        Create a model instance by name.
        
        Args:
            name: Name of the model to create
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            nn.Module: Model instance
        """
        if name not in self._registry:
            raise ValueError(f"Model {name} not found in registry. Available models: {', '.join(self._registry.keys())}")
        
        model_info = self._registry[name]
        
        # Merge default args with provided kwargs
        args = {}
        if model_info.default_args:
            args.update(model_info.default_args)
        args.update(kwargs)
        
        # Create the model
        if model_info.factory_fn is not None:
            # Use factory function
            model = model_info.factory_fn(**args)
        elif model_info.model_cls is not None:
            # Use model class
            model = model_info.model_cls(**args)
        else:
            raise ValueError(f"Model {name} has no factory function or model class")
        
        logger.info(f"Created model {name} with {sum(p.numel() for p in model.parameters())} parameters")
        return model
    
    def list_available(self) -> List[str]:
        """
        List all available models.
        
        Returns:
            List[str]: Names of available models
        """
        return list(self._registry.keys())
    
    def get_info(self, name: str) -> Optional[ModelInfo]:
        """
        Get information about a registered model.
        
        Args:
            name: Name of the model
            
        Returns:
            Optional[ModelInfo]: Model information, or None if not found
        """
        return self._registry.get(name)
    
    def search(
        self,
        task: Optional[str] = None,
        architecture: Optional[str] = None,
        min_input_size: Optional[int] = None,
        max_input_size: Optional[int] = None,
        pretrained: bool = False,
    ) -> List[str]:
        """
        Search for models matching the given criteria.
        
        Args:
            task: Task the model is designed for
            architecture: Model architecture
            min_input_size: Minimum input size
            max_input_size: Maximum input size
            pretrained: Whether pretrained weights are available
            
        Returns:
            List[str]: Names of matching models
        """
        results = []
        
        for name, info in self._registry.items():
            # Check task
            if task is not None and info.task != task:
                continue
            
            # Check architecture
            if architecture is not None and info.architecture != architecture:
                continue
            
            # Check input size if it's an integer
            if isinstance(info.input_size, int):
                if min_input_size is not None and info.input_size < min_input_size:
                    continue
                if max_input_size is not None and info.input_size > max_input_size:
                    continue
            
            # Check pretrained availability
            if pretrained and not info.pretrained_available:
                continue
            
            results.append(name)
        
        return results


# Global registry instance
_REGISTRY = ModelRegistry()


def register_model(
    name: str,
    architecture: str,
    description: str,
    task: str,
    model_cls: Optional[Type[nn.Module]] = None,
    factory_fn: Optional[Callable] = None,
    input_size: Optional[Union[int, Tuple[int, ...]]] = None,
    output_size: Optional[Union[int, Tuple[int, ...]]] = None,
    paper_url: Optional[str] = None,
    default_args: Optional[Dict[str, Any]] = None,
    pretrained_available: bool = False,
    reference_speed: Optional[Dict[str, float]] = None,
    reference_memory: Optional[Dict[str, float]] = None,
) -> Callable:
    """
    Register a model with the global registry.
    
    Can be used as a decorator on a model class or factory function.
    
    Args:
        name: Unique name for the model
        architecture: Model architecture family
        description: Brief description of the model
        task: Task the model is designed for
        model_cls: Model class (optional if factory_fn is provided)
        factory_fn: Factory function for creating the model (optional if model_cls is provided)
        input_size: Expected input size (e.g., image dimensions)
        output_size: Expected output size (e.g., number of classes)
        paper_url: URL to the paper describing the model
        default_args: Default arguments for model creation
        pretrained_available: Whether pretrained weights are available
        reference_speed: Reference performance metrics (inference speed)
        reference_memory: Reference memory usage
        
    Returns:
        Callable: Original function or class
    """
    def _register(obj):
        nonlocal model_cls, factory_fn
        
        # If obj is a class that inherits from nn.Module, it's a model class
        if inspect.isclass(obj) and issubclass(obj, nn.Module):
            model_cls = obj
        # If obj is a function, it's a factory function
        elif callable(obj):
            factory_fn = obj
        
        # Create model info
        model_info = ModelInfo(
            name=name,
            architecture=architecture,
            description=description,
            task=task,
            input_size=input_size,
            output_size=output_size,
            paper_url=paper_url,
            model_cls=model_cls,
            factory_fn=factory_fn,
            default_args=default_args or {},
            pretrained_available=pretrained_available,
            reference_speed=reference_speed,
            reference_memory=reference_memory,
        )
        
        # Register with global registry
        _REGISTRY.register(model_info)
        
        return obj
    
    # If model_cls or factory_fn is already provided, register directly
    if model_cls is not None or factory_fn is not None:
        model_info = ModelInfo(
            name=name,
            architecture=architecture,
            description=description,
            task=task,
            input_size=input_size,
            output_size=output_size,
            paper_url=paper_url,
            model_cls=model_cls,
            factory_fn=factory_fn,
            default_args=default_args or {},
            pretrained_available=pretrained_available,
            reference_speed=reference_speed,
            reference_memory=reference_memory,
        )
        _REGISTRY.register(model_info)
        
        # If factory_fn was provided, return it
        if factory_fn is not None:
            return factory_fn
        # If model_cls was provided, return it
        if model_cls is not None:
            return model_cls
    
    # Return the decorator
    return _register


def create_model(name: str, **kwargs) -> nn.Module:
    """
    Create a model instance from the global registry.
    
    Args:
        name: Name of the model to create
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        nn.Module: Model instance
    """
    return _REGISTRY.create(name, **kwargs)


def list_available_models() -> List[str]:
    """
    List all available models in the global registry.
    
    Returns:
        List[str]: Names of available models
    """
    return _REGISTRY.list_available()


def get_model_info(name: str) -> Optional[ModelInfo]:
    """
    Get information about a registered model.
    
    Args:
        name: Name of the model
        
    Returns:
        Optional[ModelInfo]: Model information, or None if not found
    """
    return _REGISTRY.get_info(name)