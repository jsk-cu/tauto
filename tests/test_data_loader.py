"""
Tests for the data loading utilities.
"""

import pytest
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset

from tauto.data.loader import (
    create_optimized_loader,
    DataLoaderConfig,
    calculate_optimal_workers,
    DataDebugger
)


@pytest.fixture
def random_dataset():
    """Create a random dataset for testing."""
    class RandomDataset(Dataset):
        def __init__(self, size=100, dim=10):
            self.size = size
            self.dim = dim
            self.data = torch.randn(size, dim)
            self.targets = torch.randint(0, 2, (size,))
        
        def __getitem__(self, index):
            return self.data[index], self.targets[index]
        
        def __len__(self):
            return self.size
    
    return RandomDataset()


def test_create_optimized_loader(random_dataset):
    """Test creating an optimized data loader."""
    # Create with default config
    loader = create_optimized_loader(random_dataset)
    
    # Check basic properties
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 32  # Default batch size
    
    # Create with custom config
    config = DataLoaderConfig(
        batch_size=16,
        num_workers=2,
        pin_memory=True,
    )
    
    loader = create_optimized_loader(random_dataset, config)
    
    # Check properties match configuration
    assert loader.batch_size == 16
    assert loader.num_workers == 2
    assert loader.pin_memory is True
    
    # Create with kwargs
    loader = create_optimized_loader(
        random_dataset,
        batch_size=8,
        num_workers=1,
        drop_last=True,
    )
    
    # Check properties match kwargs
    assert loader.batch_size == 8
    assert loader.num_workers == 1
    assert loader.drop_last is True


def test_calculate_optimal_workers():
    """Test calculating optimal number of workers."""
    # Basic test
    workers = calculate_optimal_workers()
    
    # Check that the result is reasonable
    assert isinstance(workers, int)
    assert workers >= 1
    assert workers <= max(1, os.cpu_count() - 1)
    
    # Test with small dataset
    workers_small = calculate_optimal_workers(dataset_size=50)
    
    # Check that small datasets use fewer workers
    assert workers_small <= workers


def test_data_debugger_benchmark(random_dataset):
    """Test benchmarking data loading speed."""
    loader = create_optimized_loader(
        random_dataset,
        batch_size=10,
        num_workers=0,  # Use no workers for testing
        prefetch_factor=None,  # Set to None for num_workers=0
    )
    
    debugger = DataDebugger(loader)
    results = debugger.benchmark_loading_speed(num_batches=5)
    
    # Check that results contain expected keys
    assert "avg_time" in results
    assert "min_time" in results
    assert "max_time" in results
    assert "std_time" in results
    assert "batches_per_second" in results
    assert "samples_per_second" in results
    
    # Values should be reasonable
    assert results["avg_time"] > 0
    assert results["batches_per_second"] > 0
    assert results["samples_per_second"] > 0


def test_data_debugger_memory(random_dataset):
    """Test monitoring memory usage."""
    loader = create_optimized_loader(
        random_dataset,
        batch_size=10,
        num_workers=0,  # Use no workers for testing
    )
    
    debugger = DataDebugger(loader)
    results = debugger.monitor_memory_usage(num_batches=3)
    
    # Check that results contain expected keys
    assert "baseline_memory_mb" in results
    assert "avg_memory_mb" in results
    assert "peak_memory_mb" in results
    assert "memory_increase_mb" in results
    
    # Values should be reasonable
    assert results["baseline_memory_mb"] > 0
    assert results["avg_memory_mb"] >= results["baseline_memory_mb"]


def test_data_debugger_io_bottlenecks(random_dataset):
    """Test detecting I/O bottlenecks."""
    loader = create_optimized_loader(
        random_dataset,
        batch_size=10,
        num_workers=0,  # Use no workers for testing
    )
    
    debugger = DataDebugger(loader)
    # Use only 0 and 1 worker for faster testing
    results = debugger.detect_io_bottlenecks(num_batches=2, worker_range=[0, 1])
    
    # Check that results contain expected keys
    assert "worker_times" in results
    assert "optimal_workers" in results
    assert "bottleneck_type" in results
    
    # Check worker times
    assert 0 in results["worker_times"]
    assert 1 in results["worker_times"]
    
    # Optimal workers should be in the tested range
    assert results["optimal_workers"] in [0, 1]
    
    # Bottleneck type should be either CPU bound or I/O bound
    assert results["bottleneck_type"] in ["CPU bound", "I/O bound"]


def test_data_debugger_full_analysis(random_dataset):
    """Test running a full analysis."""
    loader = create_optimized_loader(
        random_dataset,
        batch_size=10,
        num_workers=0,  # Use no workers for testing
        prefetch_factor=None,  # Set to None for num_workers=0
    )
    
    debugger = DataDebugger(loader)
    
    # Run the full analysis with parameters directly
    results = debugger.run_full_analysis(
        num_batches=2, 
        worker_range=[0, 1]
    )
    
    # Check that results contain all analysis sections
    assert "loading_speed" in results
    assert "memory_usage" in results
    assert "io_bottlenecks" in results