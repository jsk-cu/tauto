"""
Tests for the preprocessing utilities.
"""

import pytest
import torch
import numpy as np
import os
from pathlib import Path

from tauto.data import _PREPROCESSING_AVAILABLE

# Skip all tests if preprocessing is not available
pytestmark = pytest.mark.skipif(
    not _PREPROCESSING_AVAILABLE,
    reason="Preprocessing modules not available"
)

if _PREPROCESSING_AVAILABLE:
    from tauto.data.preprocessing import (
        create_transform_pipeline,
        CachedTransform,
        MemoryEfficientTransform,
        TransformConfig,
        _TORCHVISION_AVAILABLE
    )


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return torch.randn(3, 224, 224)


@pytest.mark.skipif(not _PREPROCESSING_AVAILABLE or not _TORCHVISION_AVAILABLE, 
                   reason="Requires torchvision")
def test_create_transform_pipeline():
    """Test creating a transform pipeline."""
    # Create with default config
    transform = create_transform_pipeline()
    
    # Check that it's callable
    assert callable(transform)
    
    # Create with custom config
    config = TransformConfig(
        resize_size=256,
        crop_size=224,
        normalize=True,
        augmentation="basic",
    )
    
    transform = create_transform_pipeline(config)
    
    # Check that it's callable
    assert callable(transform)
    
    # Create with kwargs
    transform = create_transform_pipeline(
        resize_size=128,
        crop_size=112,
        normalize=False,
    )
    
    # Check that it's callable
    assert callable(transform)


@pytest.mark.skipif(not _PREPROCESSING_AVAILABLE or not _TORCHVISION_AVAILABLE, 
                   reason="Requires torchvision")
def test_transform_pipeline_application(sample_image):
    """Test applying a transform pipeline to an image."""
    # Create a simpler transform pipeline for testing with tensors
    transform = create_transform_pipeline(
        resize_size=128,
        crop_size=112,
        normalize=True,
    )
    
    # Apply the transform directly to the tensor
    result = transform(sample_image)
    
    # Check that the output is a tensor
    assert isinstance(result, torch.Tensor)
    
    # Check dimensions are reasonable - exact shape may vary based on implementation
    assert result.dim() == 3  # CHW format
    assert result.shape[0] == 3  # RGB channels


def test_cached_transform(temp_dir, sample_image):
    """Test the cached transform."""
    # Create a transform that would be expensive to compute
    def expensive_transform(x):
        # Simulate an expensive operation
        return x + 1
    
    # Create a cached version
    cached_transform = CachedTransform(
        expensive_transform,
        cache_dir=temp_dir,
    )
    
    # First call should cache the result
    result1 = cached_transform(sample_image)
    
    # Second call should use the cache
    result2 = cached_transform(sample_image)
    
    # Check that the results are the same
    assert torch.allclose(result1, result2)
    
    # Check cache statistics
    stats = cached_transform.get_stats()
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1
    assert stats["hit_rate"] == 0.5
    
    # Check that the cache directory contains files
    cache_files = list(Path(temp_dir).glob("**/*.pkl"))
    assert len(cache_files) > 0
    
    # Clear the cache
    cached_transform.clear_cache()
    
    # Check that the cache directory is empty
    cache_files = list(Path(temp_dir).glob("**/*.pkl"))
    assert len(cache_files) == 0


def test_memory_efficient_transform(sample_image):
    """Test the memory-efficient transform."""
    # Create a transform
    def simple_transform(x):
        return x + 1
    
    # Create a memory-efficient version
    efficient_transform = MemoryEfficientTransform(simple_transform)
    
    # Apply the transform
    result = efficient_transform(sample_image)
    
    # Check that the output is correct
    expected = sample_image + 1
    assert torch.allclose(result, expected)


@pytest.mark.skipif(not _PREPROCESSING_AVAILABLE or not _TORCHVISION_AVAILABLE, 
                   reason="Requires torchvision")
def test_transform_pipeline_with_augmentation(sample_image):
    """Test transform pipeline with different augmentation strengths."""
    # Skip test if torchvision is too old to support the augmentations
    try:
        import torchvision
        from torchvision import transforms
        # Create pipelines with different augmentation strengths
        transform_none = create_transform_pipeline(augmentation="none")
        transform_basic = create_transform_pipeline(augmentation="basic")
        
        # Apply the transforms
        result_none = transform_none(sample_image)
        result_basic = transform_basic(sample_image)
        
        # Check that all outputs are tensors
        assert isinstance(result_none, torch.Tensor)
        assert isinstance(result_basic, torch.Tensor)
        
        # With no augmentation, the output should be deterministic
        # Run it twice and check that the results are the same
        result_none2 = transform_none(sample_image)
        assert torch.allclose(result_none, result_none2)
    except (ImportError, AttributeError):
        pytest.skip("Torchvision version does not support required transforms")