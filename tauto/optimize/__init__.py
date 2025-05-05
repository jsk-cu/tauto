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

from tauto.optimize.inference import (
    ModelQuantization,
    ModelPruning,
    optimize_for_inference,
    apply_inference_optimizations,
)

from tauto.optimize.distillation import (
    KnowledgeDistillation,
    distill_model,
)

__all__ = [
    # Training optimizations
    "MixedPrecisionTraining",
    "GradientAccumulation",
    "ModelCheckpointing",
    "OptimizerFactory",
    "train_with_optimization",
    
    # Inference optimizations
    "ModelQuantization",
    "ModelPruning",
    "optimize_for_inference",
    "apply_inference_optimizations",
    
    # Knowledge distillation
    "KnowledgeDistillation",
    "distill_model",
]