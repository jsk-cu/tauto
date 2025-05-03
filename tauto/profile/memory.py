"""
Memory profiling utilities for TAuto.

This module provides tools for analyzing memory usage of PyTorch models,
including peak memory usage and memory allocation tracking.
"""

import torch
import time
import gc
import threading
import os
import psutil
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from pathlib import Path
import platform
import numpy as np
import contextlib

from tauto.utils import get_logger

logger = get_logger(__name__)


@contextlib.contextmanager
def _patch_memory_stats():
    """
    Context manager to patch CUDA memory stats.
    
    On PyTorch 1.10+, this enables more detailed memory stats.
    """
    try:
        old_memory_stats = torch.cuda.memory_stats
        
        def patched_memory_stats(*args, **kwargs):
            stats = old_memory_stats(*args, **kwargs)
            if "active_bytes.all.allocated" in stats and "active_bytes.all.freed" in stats:
                stats["active_bytes.all.current"] = (
                    stats["active_bytes.all.allocated"] - stats["active_bytes.all.freed"]
                )
            return stats
        
        torch.cuda.memory_stats = patched_memory_stats
        yield
    except (AttributeError, RuntimeError):
        # If memory_stats is not available or CUDA is not available, skip patching
        yield
    finally:
        # Restore original memory_stats if patched
        if 'old_memory_stats' in locals():
            torch.cuda.memory_stats = old_memory_stats


def track_memory_usage(
    fn: Callable,
    interval: float = 0.01,
    device: Optional[torch.device] = None,
    max_time: float = 10.0,
) -> Dict[str, Any]:
    """
    Track memory usage during function execution.
    
    Args:
        fn: Function to track
        interval: Sampling interval in seconds
        device: Device to track memory for
        max_time: Maximum tracking time in seconds
        
    Returns:
        Dict[str, Any]: Memory usage statistics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize memory tracking
    memory_data = {
        "timestamps": [],
        "cpu_memory_mb": [],
        "gpu_memory_mb": [] if device.type == "cuda" else None,
        "baseline_mb": 0,
        "peak_mb": 0,
        "used_mb": 0,
    }
    
    # Used to signal thread termination
    stop_event = threading.Event()
    
    # Get baseline memory usage
    process = psutil.Process(os.getpid())
    baseline_cpu_memory = process.memory_info().rss / (1024 ** 2)  # MB
    memory_data["baseline_mb"] = baseline_cpu_memory
    
    if device.type == "cuda":
        with _patch_memory_stats():
            try:
                # Try to get detailed stats first
                stats = torch.cuda.memory_stats(device)
                baseline_gpu_memory = stats.get("active_bytes.all.current", 0) / (1024 ** 2)  # MB
            except (RuntimeError, AttributeError):
                # Fall back to simpler API
                baseline_gpu_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
            
            memory_data["baseline_gpu_mb"] = baseline_gpu_memory
    
    # Function to sample memory usage
    def sample_memory():
        start_time = time.time()
        max_cpu_memory = baseline_cpu_memory
        max_gpu_memory = baseline_gpu_memory if device.type == "cuda" else 0
        
        while not stop_event.is_set() and (time.time() - start_time) < max_time:
            # Sample CPU memory
            current_cpu_memory = process.memory_info().rss / (1024 ** 2)  # MB
            max_cpu_memory = max(max_cpu_memory, current_cpu_memory)
            
            # Sample GPU memory if CUDA is available
            if device.type == "cuda":
                with _patch_memory_stats():
                    try:
                        # Try to get detailed stats first
                        stats = torch.cuda.memory_stats(device)
                        current_gpu_memory = stats.get("active_bytes.all.current", 0) / (1024 ** 2)  # MB
                    except (RuntimeError, AttributeError):
                        # Fall back to simpler API
                        current_gpu_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
                    
                    max_gpu_memory = max(max_gpu_memory, current_gpu_memory)
            
            # Record data points
            timestamp = time.time() - start_time
            memory_data["timestamps"].append(timestamp)
            memory_data["cpu_memory_mb"].append(current_cpu_memory)
            
            if device.type == "cuda":
                memory_data["gpu_memory_mb"].append(current_gpu_memory)
            
            # Sleep for the specified interval
            time.sleep(interval)
        
        # Update peak memory
        memory_data["peak_mb"] = max_cpu_memory
        if device.type == "cuda":
            memory_data["peak_gpu_mb"] = max_gpu_memory
    
    # Start memory sampling thread
    thread = threading.Thread(target=sample_memory)
    thread.daemon = True
    thread.start()
    
    try:
        # Run the function to profile
        result = fn()
        
        # Signal thread to stop
        stop_event.set()
        thread.join(timeout=1.0)
        
        # Get final memory usage
        current_cpu_memory = process.memory_info().rss / (1024 ** 2)  # MB
        memory_data["used_mb"] = current_cpu_memory - baseline_cpu_memory
        
        if device.type == "cuda":
            with _patch_memory_stats():
                try:
                    stats = torch.cuda.memory_stats(device)
                    current_gpu_memory = stats.get("active_bytes.all.current", 0) / (1024 ** 2)  # MB
                except (RuntimeError, AttributeError):
                    current_gpu_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
                
                memory_data["used_gpu_mb"] = current_gpu_memory - baseline_gpu_memory
        
        return memory_data
    
    except Exception as e:
        # Signal thread to stop on error
        stop_event.set()
        if thread.is_alive():
            thread.join(timeout=1.0)
        raise e


def measure_peak_memory(
    fn: Callable,
    device: Optional[torch.device] = None,
    num_trials: int = 3,
    reset_after_trial: bool = True,
) -> Dict[str, float]:
    """
    Measure peak memory usage during function execution.
    
    Args:
        fn: Function to measure
        device: Device to measure memory for
        num_trials: Number of trials to run
        reset_after_trial: Whether to reset GPU memory after each trial
        
    Returns:
        Dict[str, float]: Peak memory usage statistics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize measurement
    results = {
        "peak_cpu_mb": 0.0,
        "peak_gpu_mb": 0.0 if device.type == "cuda" else None,
        "avg_cpu_mb": 0.0,
        "avg_gpu_mb": 0.0 if device.type == "cuda" else None,
    }
    
    # Get process for CPU memory tracking
    process = psutil.Process(os.getpid())
    
    # Run measurements
    cpu_peaks = []
    gpu_peaks = [] if device.type == "cuda" else None
    
    for i in range(num_trials):
        if reset_after_trial and i > 0:
            # Collect garbage to reset memory state
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        # Baseline CPU memory
        baseline_cpu = process.memory_info().rss / (1024 ** 2)  # MB
        
        # Baseline GPU memory
        if device.type == "cuda":
            with _patch_memory_stats():
                try:
                    stats = torch.cuda.memory_stats(device)
                    baseline_gpu = stats.get("active_bytes.all.current", 0) / (1024 ** 2)  # MB
                except (RuntimeError, AttributeError):
                    baseline_gpu = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
        
        # Execute function
        fn()
        
        # Peak CPU memory
        peak_cpu = process.memory_info().rss / (1024 ** 2) - baseline_cpu  # MB
        cpu_peaks.append(peak_cpu)
        
        # Peak GPU memory
        if device.type == "cuda":
            with _patch_memory_stats():
                try:
                    stats = torch.cuda.memory_stats(device)
                    peak_gpu = stats.get("active_bytes.all.current", 0) / (1024 ** 2) - baseline_gpu  # MB
                except (RuntimeError, AttributeError):
                    peak_gpu = torch.cuda.memory_allocated(device) / (1024 ** 2) - baseline_gpu  # MB
                
                gpu_peaks.append(peak_gpu)
    
    # Calculate statistics
    results["peak_cpu_mb"] = max(cpu_peaks)
    results["avg_cpu_mb"] = sum(cpu_peaks) / len(cpu_peaks)
    
    if device.type == "cuda":
        results["peak_gpu_mb"] = max(gpu_peaks)
        results["avg_gpu_mb"] = sum(gpu_peaks) / len(gpu_peaks)
    
    return results


def estimate_model_memory(
    model: torch.nn.Module,
    input_size: Optional[Union[Tuple[int, ...], torch.Size]] = None,
    include_activations: bool = True,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Estimate memory usage of a PyTorch model.
    
    Args:
        model: PyTorch model
        input_size: Model input size (excluding batch dimension)
        include_activations: Whether to include activations in the estimate
        batch_size: Batch size for activation estimation
        
    Returns:
        Dict[str, float]: Memory usage estimation
    """
    # Initialize results
    results = {
        "params_mb": 0.0,
        "buffers_mb": 0.0,
        "activations_mb": 0.0 if include_activations else None,
        "total_mb": 0.0,
    }
    
    # Get device
    device = next(model.parameters()).device
    
    # Estimate parameters and buffers
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    results["params_mb"] = param_size / (1024 ** 2)  # MB
    results["buffers_mb"] = buffer_size / (1024 ** 2)  # MB
    
    # Estimate activations if requested
    if include_activations and input_size is not None:
        # Create a dummy input
        input_shape = (batch_size,) + input_size if isinstance(input_size, tuple) else (batch_size,) + tuple(input_size)
        dummy_input = torch.zeros(input_shape, device=device)
        
        # Track memory usage during forward pass
        activations_size = measure_peak_memory(
            lambda: model(dummy_input),
            device=device,
            num_trials=1,
        )
        
        results["activations_mb"] = activations_size["peak_gpu_mb"] if device.type == "cuda" else activations_size["peak_cpu_mb"]
    
    # Calculate total
    results["total_mb"] = results["params_mb"] + results["buffers_mb"]
    if include_activations and results["activations_mb"] is not None:
        results["total_mb"] += results["activations_mb"]
    
    return results