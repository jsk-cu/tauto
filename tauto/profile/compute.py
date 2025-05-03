"""
Compute utilization profiling utilities for TAuto.

This module provides tools for analyzing compute utilization of PyTorch models,
including GPU utilization, FLOPs counting, and performance estimation.
"""

import torch
import time
import threading
import os
import psutil
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from pathlib import Path
import numpy as np
import platform
import contextlib

from tauto.utils import get_logger

logger = get_logger(__name__)


def measure_compute_utilization(
    fn: Callable,
    interval: float = 0.1,
    device: Optional[torch.device] = None,
    max_time: float = 30.0,
) -> Dict[str, Any]:
    """
    Measure compute utilization during function execution.
    
    Args:
        fn: Function to measure
        interval: Sampling interval in seconds
        device: Device to measure utilization for
        max_time: Maximum measurement time in seconds
        
    Returns:
        Dict[str, Any]: Compute utilization statistics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize utilization tracking
    util_data = {
        "timestamps": [],
        "cpu_percent": [],
        "gpu_percent": [] if device.type == "cuda" else None,
        "avg_cpu_percent": 0.0,
        "avg_gpu_percent": 0.0 if device.type == "cuda" else None,
        "peak_cpu_percent": 0.0,
        "peak_gpu_percent": 0.0 if device.type == "cuda" else None,
    }
    
    # Used to signal thread termination
    stop_event = threading.Event()
    
    # Get process
    process = psutil.Process(os.getpid())
    
    # Initialize GPU monitoring if needed
    gpu_handle = None
    if device.type == "cuda":
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device.index if device.index is not None else 0)
            gpu_available = True
        except (ImportError, pynvml.NVMLError):
            # Fall back to torch.cuda functions
            gpu_available = False
            logger.warning("pynvml not available, falling back to basic GPU monitoring")
    else:
        gpu_available = False
    
    # Function to sample utilization
    def sample_utilization():
        start_time = time.time()
        
        # For averaging
        cpu_samples = []
        gpu_samples = [] if device.type == "cuda" else None
        
        while not stop_event.is_set() and (time.time() - start_time) < max_time:
            # Sample CPU utilization
            current_cpu_percent = process.cpu_percent(interval=0.05)
            cpu_samples.append(current_cpu_percent)
            
            # Sample GPU utilization if available
            current_gpu_percent = None
            if device.type == "cuda":
                if gpu_available and gpu_handle is not None:
                    try:
                        # Get GPU utilization using pynvml
                        util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                        current_gpu_percent = util.gpu
                    except pynvml.NVMLError:
                        # Fall back to a simplistic estimate based on memory usage
                        current_gpu_percent = 0.0
                else:
                    # Fall back to a simplistic estimate based on memory usage
                    current_gpu_percent = 0.0
                
                gpu_samples.append(current_gpu_percent)
            
            # Record data points
            timestamp = time.time() - start_time
            util_data["timestamps"].append(timestamp)
            util_data["cpu_percent"].append(current_cpu_percent)
            
            if device.type == "cuda":
                util_data["gpu_percent"].append(current_gpu_percent)
            
            # Sleep for the specified interval
            time.sleep(interval)
        
        # Update statistics
        if cpu_samples:
            util_data["avg_cpu_percent"] = sum(cpu_samples) / len(cpu_samples)
            util_data["peak_cpu_percent"] = max(cpu_samples)
        
        if device.type == "cuda" and gpu_samples:
            util_data["avg_gpu_percent"] = sum(gpu_samples) / len(gpu_samples)
            util_data["peak_gpu_percent"] = max(gpu_samples)
    
    # Start utilization sampling thread
    thread = threading.Thread(target=sample_utilization)
    thread.daemon = True
    thread.start()
    
    try:
        # Run the function to profile
        result = fn()
        
        # Signal thread to stop
        stop_event.set()
        thread.join(timeout=1.0)
        
        return util_data
    
    except Exception as e:
        # Signal thread to stop on error
        stop_event.set()
        if thread.is_alive():
            thread.join(timeout=1.0)
        raise e
    
    finally:
        # Clean up pynvml if initialized
        if device.type == "cuda" and gpu_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except (ImportError, pynvml.NVMLError):
                pass


def measure_flops(
    model: torch.nn.Module,
    input_size: Tuple[int, ...],
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Measure floating-point operations (FLOPs) for a model.
    
    Args:
        model: PyTorch model
        input_size: Model input size (including batch dimension)
        device: Device to measure FLOPs for
        
    Returns:
        Dict[str, float]: FLOPs statistics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = {
        "total_flops": 0.0,
        "total_params": 0,
        "forward_flops": 0.0,
        "backward_flops": 0.0,
        "flops_per_param": 0.0,
    }
    
    # Check if we can use fvcore for FLOPs counting
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table
        
        # Create dummy input
        dummy_input = torch.zeros(input_size, device=device)
        
        # Analyze FLOPs
        flops = FlopCountAnalysis(model, dummy_input)
        results["total_flops"] = flops.total()
        results["flops_table"] = flop_count_table(flops)
        results["flops_str"] = flop_count_str(flops)
        
        # Calculate statistics
        results["total_params"] = sum(p.numel() for p in model.parameters())
        if results["total_params"] > 0:
            results["flops_per_param"] = results["total_flops"] / results["total_params"]
        
        return results
    
    except ImportError:
        logger.warning("fvcore not available for FLOPs counting. Using a basic estimate.")
    
    # Fall back to a basic estimate
    total_params = sum(p.numel() for p in model.parameters())
    results["total_params"] = total_params
    
    # Create dummy input
    dummy_input = torch.zeros(input_size, device=device)
    
    # Estimate FLOPs based on model size and time
    # This is a very rough estimate
    from tauto.profile.memory import measure_peak_memory
    
    # Measure forward pass time
    start_time = time.time()
    for _ in range(10):  # Average over multiple runs
        output = model(dummy_input)
    forward_time = (time.time() - start_time) / 10
    
    # Assuming a relationship between params, time, and FLOPs
    # This is a very rough approximation
    if device.type == "cuda":
        # GPU theoretical performance in GFLOPS
        gpu_name = torch.cuda.get_device_name(device)
        
        # Rough estimates based on GPU name
        if "A100" in gpu_name:
            gpu_tflops = 19.5
        elif "V100" in gpu_name:
            gpu_tflops = 14.0
        elif "T4" in gpu_name:
            gpu_tflops = 8.1
        elif "P100" in gpu_name:
            gpu_tflops = 10.0
        elif "2080" in gpu_name:
            gpu_tflops = 10.0
        elif "3090" in gpu_name:
            gpu_tflops = 35.6
        elif "1080" in gpu_name:
            gpu_tflops = 8.9
        else:
            # Default for unknown GPUs
            gpu_tflops = 5.0
        
        # Convert to FLOPS
        gpu_flops = gpu_tflops * (10 ** 12)
        
        # Estimate FLOPs based on execution time and GPU performance
        # Assuming the GPU is utilized at ~30% efficiency
        forward_flops = forward_time * gpu_flops * 0.3
    else:
        # For CPU, use a simpler heuristic based on parameters
        # Assuming each parameter is used in ~100 ops
        forward_flops = total_params * 100
    
    results["forward_flops"] = forward_flops
    results["backward_flops"] = forward_flops * 2  # Backward is typically 2-3x more expensive
    results["total_flops"] = results["forward_flops"] + results["backward_flops"]
    
    if total_params > 0:
        results["flops_per_param"] = results["total_flops"] / total_params
    
    return results


def estimate_device_capabilities(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Estimate compute capabilities of the current device.
    
    Args:
        device: Device to estimate capabilities for
        
    Returns:
        Dict[str, Any]: Device capability statistics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    capabilities = {
        "device_type": device.type,
        "device_name": None,
        "theoretical_gflops": None,
        "memory_bandwidth_gbps": None,
        "memory_size_gb": None,
        "cores": None,
    }
    
    if device.type == "cuda":
        # Get CUDA device properties
        props = torch.cuda.get_device_properties(device)
        
        capabilities["device_name"] = props.name
        capabilities["memory_size_gb"] = props.total_memory / (1024 ** 3)
        capabilities["cores"] = props.multi_processor_count
        
        # Calculate memory bandwidth
        if hasattr(props, "memory_clock_rate") and hasattr(props, "memory_bus_width"):
            # Memory clock is in kHz, bus width in bits
            memory_clock = props.memory_clock_rate * 1000  # Hz
            bus_width = props.memory_bus_width
            
            # Bandwidth = clock * bus_width * 2 (DDR) / 8 (bytes)
            bandwidth = memory_clock * bus_width * 2 / 8
            capabilities["memory_bandwidth_gbps"] = bandwidth / (10 ** 9)
        
        # Estimate theoretical GFLOPs based on device name
        device_name = props.name
        compute_capability = props.major + props.minor / 10
        
        if compute_capability >= 7.0:  # Volta, Turing, Ampere, etc.
            # Rough estimate based on CUDA cores and clock rate
            if hasattr(props, "clock_rate"):
                clock_rate = props.clock_rate * 1000  # Hz
                # Assuming tensor cores provide 2x the performance of CUDA cores
                capabilities["theoretical_gflops"] = (
                    props.multi_processor_count * 64 * 2 * clock_rate / (10 ** 9)
                )
    else:
        # For CPU
        capabilities["device_name"] = platform.processor()
        capabilities["cores"] = psutil.cpu_count(logical=False)
        capabilities["threads"] = psutil.cpu_count(logical=True)
        
        # Estimate memory info
        mem_info = psutil.virtual_memory()
        capabilities["memory_size_gb"] = mem_info.total / (1024 ** 3)
        
        # Rough estimate of memory bandwidth
        capabilities["memory_bandwidth_gbps"] = 50.0  # Typical DDR4 system
        
        # Rough estimate of theoretical GFLOPs
        # Assuming 4 FLOPs per core per clock cycle on average
        if capabilities["cores"] is not None:
            # Rough estimate of clock speed from processor name
            # This is very imprecise
            import re
            clock_match = re.search(r'(\d+\.\d+)GHz', platform.processor())
            if clock_match:
                clock_ghz = float(clock_match.group(1))
            else:
                # Default to 3 GHz if not found
                clock_ghz = 3.0
            
            capabilities["clock_ghz"] = clock_ghz
            capabilities["theoretical_gflops"] = capabilities["cores"] * clock_ghz * 4
    
    return capabilities