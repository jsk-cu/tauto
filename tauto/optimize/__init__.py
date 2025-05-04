"""
Optimization module for TAuto.

This module provides tools for optimizing various aspects of model training and inference.
"""

from tauto.optimize.training import (
    MixedPrecisionTraining,
    GradientAccumulation,
    ModelCheckpointing,
    OptimizerFactory,
    train_with_optimization,
)

__all__ = [
    "MixedPrecisionTraining",
    "GradientAccumulation",
    "ModelCheckpointing",
    "OptimizerFactory",
    "train_with_optimization",
]