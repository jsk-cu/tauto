"""
TAuto: PyTorch Model Optimization Framework
"""

__version__ = "0.1.0"

from tauto import config
from tauto import models
from tauto import profile
from tauto import utils

# These will be implemented in future phases
# from tauto import data
# from tauto import optimize
# from tauto import serve

__all__ = [
    "config",
    "models",
    "profile",
    "utils",
]