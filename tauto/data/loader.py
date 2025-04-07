"""
Optimized data loading utilities for PyTorch models.
"""

import os
import torch
import psutil
import platform
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, List, Callable, Tuple
from torch.utils.data import DataLoader, Dataset, Sampler, IterableDataset
import time
import numpy as np
from pathlib import Path
import warnings

from tauto.utils import get_logger

logger = get_logger(__name__)


@dataclass
class DataLoaderConfig:
    """Configuration for optimized data loaders."""
    
    batch_size: int = 32
    num_workers: Optional[int] = None
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = False
    timeout: float = 0
    collate_fn: Optional[Callable] = None
    shuffle: bool = True
    sampler: Optional[Sampler] = None
    auto_tune_workers: bool = True


def calculate_optimal_workers(dataset_size: int = None) -> int:
    """
    Calculate the optimal number of workers based on system resources.
    
    Args:
        dataset_size: Size of the dataset (optional)
        
    Returns:
        int: Recommended number of workers
    """
    # Get CPU cores
    cpu_count = os.cpu_count() or 1
    
    # Get available memory
    mem = psutil.virtual_memory()
    available_mem_gb = mem.available / (1024 ** 3)
    
    # Determine memory factor (each worker might use ~1-2GB)
    # Conservative estimate: 1 worker per 2GB available
    mem_workers = max(1, int(available_mem_gb / 2))
    
    # Start with CPU-based estimate, with a maximum of cpu_count - 1
    # to leave one core for the main process
    cpu_workers = max(1, cpu_count - 1)
    
    # Take the minimum of CPU and memory-based estimates
    workers = min(cpu_workers, mem_workers)
    
    # Consider dataset size if provided (small datasets may not benefit from many workers)
    if dataset_size is not None and dataset_size < 1000:
        # For small datasets, fewer workers may be more efficient
        size_factor = max(1, int(dataset_size / 250))
        workers = min(workers, size_factor)
    
    logger.debug(f"Optimal workers calculation: CPU={cpu_count}, Memory={available_mem_gb:.2f}GB")
    logger.debug(f"Recommended workers: {workers}")
    
    return workers


def create_optimized_loader(
    dataset: Dataset,
    config: Optional[Union[Dict[str, Any], DataLoaderConfig]] = None,
    **kwargs
) -> DataLoader:
    """
    Create an optimized DataLoader with best practices for performance.
    
    Args:
        dataset: PyTorch Dataset
        config: DataLoader configuration
        **kwargs: Additional arguments to pass to DataLoader
        
    Returns:
        DataLoader: Optimized data loader
    """
    # Process configuration
    if config is None:
        config = DataLoaderConfig()
    elif isinstance(config, dict):
        # Override defaults with provided config
        default_config = DataLoaderConfig()
        for key, value in config.items():
            if hasattr(default_config, key):
                setattr(default_config, key, value)
        config = default_config
    
    # Update config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Auto-tune workers if requested
    if config.auto_tune_workers and config.num_workers is None:
        config.num_workers = calculate_optimal_workers(len(dataset) if hasattr(dataset, "__len__") else None)
    elif config.num_workers is None:
        # Default to 4 workers if not auto-tuning and not specified
        config.num_workers = 4
    
    # Create DataLoader with optimized settings
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "drop_last": config.drop_last,
        "timeout": config.timeout,
        "collate_fn": config.collate_fn,
        "shuffle": config.shuffle if config.sampler is None else False,
        "sampler": config.sampler,
    }
    
    # Only add prefetch_factor and persistent_workers if num_workers > 0
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor
        loader_kwargs["persistent_workers"] = config.persistent_workers
    
    loader = DataLoader(dataset, **loader_kwargs)
    
    logger.info(f"Created optimized DataLoader with {config.num_workers} workers, "
                f"batch size {config.batch_size}")
    
    return loader


class DataDebugger:
    """
    Utility for debugging and profiling data loading performance.
    """
    
    def __init__(self, dataloader: DataLoader):
        """
        Initialize the data debugger.
        
        Args:
            dataloader: DataLoader to debug
        """
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.results = {}
    
    def benchmark_loading_speed(self, num_batches: int = 50) -> Dict[str, float]:
        """
        Benchmark data loading speed.
        
        Args:
            num_batches: Number of batches to load for benchmarking
            
        Returns:
            Dict[str, float]: Benchmark results
        """
        logger.info(f"Benchmarking data loading speed over {num_batches} batches...")
        
        # Measure time to load batches
        times = []
        start_time = time.time()
        
        for i, batch in enumerate(self.dataloader):
            if i >= num_batches:
                break
            
            batch_time = time.time() - start_time
            times.append(batch_time)
            start_time = time.time()
        
        # Calculate statistics
        times = times[1:]  # Exclude first batch (initialization overhead)
        
        if not times:
            logger.warning("Not enough batches to calculate statistics")
            return {"avg_time": 0, "min_time": 0, "max_time": 0, "std_time": 0}
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        results = {
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_time": std_time,
            "batches_per_second": 1.0 / avg_time,
            "samples_per_second": self.dataloader.batch_size / avg_time,
        }
        
        logger.info(f"Average batch loading time: {avg_time:.4f}s")
        logger.info(f"Batches per second: {results['batches_per_second']:.2f}")
        logger.info(f"Samples per second: {results['samples_per_second']:.2f}")
        
        self.results["loading_speed"] = results
        return results
    
    def monitor_memory_usage(self, num_batches: int = 10) -> Dict[str, float]:
        """
        Monitor memory usage during data loading.
        
        Args:
            num_batches: Number of batches to load for monitoring
            
        Returns:
            Dict[str, float]: Memory usage statistics
        """
        logger.info(f"Monitoring memory usage over {num_batches} batches...")
        
        # Set up memory tracking
        memory_usage = []
        process = psutil.Process(os.getpid())
        
        # Baseline memory usage
        baseline = process.memory_info().rss / (1024 ** 2)  # MB
        
        # Load batches and track memory
        for i, batch in enumerate(self.dataloader):
            if i >= num_batches:
                break
            
            # Get current memory usage
            memory = process.memory_info().rss / (1024 ** 2)  # MB
            memory_usage.append(memory)
        
        # Calculate statistics
        avg_memory = np.mean(memory_usage)
        peak_memory = np.max(memory_usage)
        memory_increase = peak_memory - baseline
        
        results = {
            "baseline_memory_mb": baseline,
            "avg_memory_mb": avg_memory,
            "peak_memory_mb": peak_memory,
            "memory_increase_mb": memory_increase,
        }
        
        logger.info(f"Baseline memory usage: {baseline:.2f} MB")
        logger.info(f"Average memory usage: {avg_memory:.2f} MB")
        logger.info(f"Peak memory usage: {peak_memory:.2f} MB")
        logger.info(f"Memory increase: {memory_increase:.2f} MB")
        
        self.results["memory_usage"] = results
        return results
    
    def detect_io_bottlenecks(self, num_batches: int = 10, worker_range: List[int] = None) -> Dict[str, Any]:
        """
        Detect I/O bottlenecks by testing different numbers of workers.
        
        Args:
            num_batches: Number of batches to load for each test
            worker_range: List of worker counts to test
            
        Returns:
            Dict[str, Any]: Bottleneck analysis results
        """
        if worker_range is None:
            max_workers = os.cpu_count() or 4
            worker_range = [0, 1, 2, 4, max(4, max_workers // 2), max_workers]
            # Remove duplicates and sort
            worker_range = sorted(list(set(worker_range)))
        
        logger.info(f"Detecting I/O bottlenecks with worker counts: {worker_range}")
        
        results = {"worker_times": {}, "optimal_workers": None}
        
        for num_workers in worker_range:
            # Create a new dataloader with the current worker count
            test_loader = DataLoader(
                self.dataset,
                batch_size=self.dataloader.batch_size,
                num_workers=num_workers,
                pin_memory=self.dataloader.pin_memory,
                shuffle=False,  # Disable shuffling for consistent comparison
            )
            
            # Measure loading time
            times = []
            start_time = time.time()
            
            for i, _ in enumerate(test_loader):
                if i >= num_batches:
                    break
                batch_time = time.time() - start_time
                times.append(batch_time)
                start_time = time.time()
            
            # Calculate average time (excluding first batch)
            avg_time = np.mean(times[1:]) if len(times) > 1 else times[0]
            results["worker_times"][num_workers] = avg_time
        
        # Find optimal number of workers
        optimal_workers = min(results["worker_times"], key=results["worker_times"].get)
        results["optimal_workers"] = optimal_workers
        
        # Check if I/O bound or CPU bound
        # If increasing workers consistently decreases time, likely CPU bound
        # If increasing workers doesn't help much, likely I/O bound
        cpu_bound = True
        times = [results["worker_times"][w] for w in worker_range]
        
        # Check if times decrease by at least 10% when doubling workers
        for i in range(1, len(worker_range)):
            if worker_range[i] > worker_range[i-1] * 1.5:  # Only compare when workers increase significantly
                improvement = (times[i-1] - times[i]) / times[i-1]
                if improvement < 0.1:  # Less than 10% improvement
                    cpu_bound = False
                    break
        
        results["bottleneck_type"] = "CPU bound" if cpu_bound else "I/O bound"
        
        logger.info(f"Optimal number of workers: {optimal_workers}")
        logger.info(f"Bottleneck type: {results['bottleneck_type']}")
        
        self.results["io_bottlenecks"] = results
        return results
    
    def run_full_analysis(self, num_batches=50, worker_range=None):
        """
        Run a full analysis of the data loading pipeline.
        
        Args:
            num_batches: Number of batches to load for benchmarking
            worker_range: List of worker counts to test for I/O bottleneck detection
            
        Returns:
            Dict[str, Any]: Complete analysis results
        """
        logger.info("Running full data loading analysis...")
        
        self.benchmark_loading_speed(num_batches=num_batches)
        self.monitor_memory_usage(num_batches=num_batches)
        self.detect_io_bottlenecks(num_batches=num_batches, worker_range=worker_range)
        
        logger.info("Data loading analysis complete")
        return self.results