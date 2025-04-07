"""
Data loading and preprocessing utilities for TAuto.
"""

# Import core functionality that doesn't depend on optional packages
from tauto.data.loader import (
    create_optimized_loader,
    DataLoaderConfig, 
    calculate_optimal_workers,
    DataDebugger
)

# Try to import preprocessing modules, with fallbacks for missing dependencies
try:
    from tauto.data.preprocessing import (
        create_transform_pipeline,
        CachedTransform,
        MemoryEfficientTransform,
        TransformConfig
    )
    _PREPROCESSING_AVAILABLE = True
except ImportError:
    _PREPROCESSING_AVAILABLE = False
    # Define empty placeholder classes if imports fail
    class TransformConfig: pass
    class CachedTransform: pass
    class MemoryEfficientTransform: pass
    def create_transform_pipeline(*args, **kwargs): 
        raise ImportError("Preprocessing modules unavailable. Check torchvision installation.")

__all__ = [
    "create_optimized_loader",
    "DataLoaderConfig",
    "calculate_optimal_workers",
    "DataDebugger",
    "create_transform_pipeline",
    "CachedTransform", 
    "MemoryEfficientTransform",
    "TransformConfig",
    "_PREPROCESSING_AVAILABLE"
]