"""
Model registry and factory module for TAuto.
"""

from tauto.models.registry import (
    ModelRegistry,
    register_model,
    create_model,
    list_available_models,
    get_model_info,
)

# Set up model registry
registry = ModelRegistry()

# Import model zoo modules
try:
    from tauto.models.zoo import vision, nlp
    _FULL_MODEL_ZOO_AVAILABLE = True
except ImportError:
    _FULL_MODEL_ZOO_AVAILABLE = False

__all__ = [
    "ModelRegistry",
    "register_model",
    "create_model",
    "list_available_models",
    "get_model_info",
    "registry",
    "_FULL_MODEL_ZOO_AVAILABLE",
]