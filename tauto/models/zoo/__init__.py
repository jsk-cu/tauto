"""
Model zoo module for TAuto.

This module contains predefined model architectures that can be used
with the TAuto framework. The models are organized by task type.
"""

# Import model types when available
try:
    from . import vision
    _VISION_MODELS_AVAILABLE = True
except ImportError:
    _VISION_MODELS_AVAILABLE = False

try:
    from . import nlp
    _NLP_MODELS_AVAILABLE = True
except ImportError:
    _NLP_MODELS_AVAILABLE = False

__all__ = [
    "_VISION_MODELS_AVAILABLE",
    "_NLP_MODELS_AVAILABLE",
]

if _VISION_MODELS_AVAILABLE:
    __all__.append("vision")

if _NLP_MODELS_AVAILABLE:
    __all__.append("nlp")