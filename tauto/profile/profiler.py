"""
Main profiler implementation for TAuto.

This module implements the Profiler class which integrates and coordinates
various profiling utilities for comprehensive performance analysis.
"""

import os
import torch
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from pathlib import Path
import json
import tempfile
import functools
import contextlib
import inspect

from tauto.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ProfilerConfig:
    """Configuration for the PyTorch profiler."""
    
    enabled: bool = True
    use_cuda: bool = True
    profile_memory: bool = True
    record_shapes: bool = True
    with_stack: bool = False
    with_flops: bool = False
    profile_kineto: bool = True
    activities: Optional[List[str]] = None
    schedule: Optional[Dict[str, Any]] = None
    on_trace_ready: Optional[Callable] = None
    record_module_hierarchy: bool = False
    profile_dir: str = ".tauto_profile"
    num_warmup_steps: int = 5
    num_active_steps: int = 10
    num_repeat: int = 3
    max_batch_size: Optional[int] = None
    skip_first: bool = True
    collect_baseline: bool = True
    
    def __post_init__(self):
        """Initialize default values after dataclass initialization."""
        if self.activities is None:
            self.activities = ["cpu", "cuda"] if self.use_cuda else ["cpu"]
        
        if self.schedule is None:
            self.schedule = {
                "wait": self.num_warmup_steps,
                "warmup": 0,
                "active": self.num_active_steps,
                "repeat": self.num_repeat,
            }
        
        # Create profile directory if it doesn't exist
        if self.enabled:
            os.makedirs(self.profile_dir, exist_ok=True)


@dataclass
class ProfileResult:
    """Results from profiling."""
    
    name: str
    device: str
    timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S"))
    duration_ms: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    compute_utilization: Dict[str, float] = field(default_factory=dict)
    flops: Optional[Dict[str, float]] = None
    parameters: Optional[Dict[str, float]] = None
    traces: Optional[List[str]] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the profile result to a dictionary.
        
        Returns:
            Dict[str, Any]: Profile result as a dictionary
        """
        return {
            "name": self.name,
            "device": self.device,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "memory_usage": self.memory_usage,
            "compute_utilization": self.compute_utilization,
            "flops": self.flops,
            "parameters": self.parameters,
            "traces": self.traces,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the profile result to a file.
        
        Args:
            path: Path to save the profile result
        """
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        
        # Convert to JSON-serializable dict
        result_dict = self.to_dict()
        
        # Convert Path objects to strings
        if self.traces:
            result_dict["traces"] = [str(trace) for trace in self.traces]
        
        with open(path, "w") as f:
            json.dump(result_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ProfileResult":
        """
        Load a profile result from a file.
        
        Args:
            path: Path to the profile result file
            
        Returns:
            ProfileResult: Loaded profile result
        """
        path = Path(path)
        
        with open(path, "r") as f:
            result_dict = json.load(f)
        
        # Convert traces back to Path objects if they exist
        if "traces" in result_dict and result_dict["traces"]:
            result_dict["traces"] = [Path(trace) for trace in result_dict["traces"]]
        
        return cls(**result_dict)


class Profiler:
    """
    Main profiler class for TAuto.
    
    This class integrates various profiling utilities to provide comprehensive
    performance analysis for PyTorch models.
    """
    
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        config: Optional[Union[Dict[str, Any], ProfilerConfig]] = None,
    ):
        """
        Initialize the profiler.
        
        Args:
            model: PyTorch model to profile
            config: Profiler configuration
        """
        self.model = model
        
        # Process configuration
        if config is None:
            self.config = ProfilerConfig()
        elif isinstance(config, dict):
            # Override defaults with provided config
            default_config = ProfilerConfig()
            for key, value in config.items():
                if hasattr(default_config, key):
                    setattr(default_config, key, value)
            self.config = default_config
        else:
            self.config = config
        
        # Get device information
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.use_cuda else "cpu")
        self.device_info = self._get_device_info()
        
        logger.info(f"Initialized profiler with device: {self.device}")
    
    def _get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the current device.
        
        Returns:
            Dict[str, Any]: Device information
        """
        info = {
            "device_type": self.device.type,
            "device_index": self.device.index if self.device.index is not None else 0,
        }
        
        # Add CUDA-specific information if available
        if self.device.type == "cuda" and torch.cuda.is_available():
            info.update({
                "device_name": torch.cuda.get_device_name(self.device),
                "device_capability": torch.cuda.get_device_capability(self.device),
                "device_memory": torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3),  # GB
            })
        
        return info
    
    def profile_training(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        num_steps: int = 10,
        name: Optional[str] = None,
    ) -> ProfileResult:
        """
        Profile the training phase of a model.
        
        Args:
            dataloader: DataLoader for training data
            criterion: Loss function
            optimizer: Optimizer for training
            num_steps: Number of training steps to profile
            name: Name for the profile result
            
        Returns:
            ProfileResult: Profile results
        """
        if not self.config.enabled:
            logger.warning("Profiling is disabled. Returning empty profile result.")
            return ProfileResult(name=name or "training", device=str(self.device))
        
        if self.model is None:
            raise ValueError("Model must be set to profile training")
        
        if criterion is None:
            # Use dummy criterion if none provided
            criterion = torch.nn.MSELoss()
        
        if optimizer is None:
            # Use dummy optimizer if none provided
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        logger.info(f"Profiling training for {num_steps} steps...")
        
        # Setup profiling results
        result = ProfileResult(
            name=name or "training",
            device=str(self.device),
        )
        
        # Move model to device
        self.model.to(self.device)
        self.model.train()
        
        # Create PyTorch profiler
        profile_path = Path(self.config.profile_dir) / f"{result.name}_{result.timestamp}"
        os.makedirs(profile_path, exist_ok=True)
        
        # Setup tracing for PyTorch profiler
        def trace_handler(prof, steps=None):
            trace_path = profile_path / f"trace_{steps}.json"
            prof.export_chrome_trace(str(trace_path))
            if result.traces is None:
                result.traces = []
            result.traces.append(trace_path)
        
        # Create profiler context
        try:
            # Try to import torch.profiler (PyTorch 1.9+)
            import torch.profiler as torch_profiler
            profiler_cls = torch_profiler.profile
            schedule_cls = torch_profiler.schedule
            activities_map = {
                "cpu": torch_profiler.ProfilerActivity.CPU,
                "cuda": torch_profiler.ProfilerActivity.CUDA,
            }
            activities = [activities_map[act] for act in self.config.activities if act in activities_map]
            
            # Create schedule
            schedule = schedule_cls(
                wait=self.config.schedule["wait"],
                warmup=self.config.schedule["warmup"],
                active=self.config.schedule["active"],
                repeat=self.config.schedule["repeat"],
            )
            
            # Create profiler with compatible parameters
            profiler_kwargs = {
                "activities": activities,
                "schedule": schedule,
                "on_trace_ready": trace_handler,
                "record_shapes": self.config.record_shapes,
                "profile_memory": self.config.profile_memory,
                "with_stack": self.config.with_stack,
            }
            
            # Add optional parameters if supported
            if hasattr(torch.profiler.ProfilerActivity, "CUDA"):
                profiler_kwargs["with_flops"] = self.config.with_flops
                
            # Check if the profiler supports record_module_hierarchy
            if "record_module_hierarchy" in inspect.signature(profiler_cls.__init__).parameters:
                profiler_kwargs["record_module_hierarchy"] = self.config.record_module_hierarchy
            
            # Create profiler with compatible parameters
            profiler_ctx = profiler_cls(**profiler_kwargs)
            
        except ImportError:
            # Fallback for older PyTorch versions
            logger.warning("torch.profiler not available. Using legacy profiler.")
            profiler_ctx = torch.autograd.profiler.profile(
                use_cuda=self.config.use_cuda,
                record_shapes=self.config.record_shapes,
                profile_memory=self.config.profile_memory,
                with_stack=self.config.with_stack,
            )
        
        # Run training loop with profiling
        data_iter = iter(dataloader)
        step = 0
        
        try:
            with profiler_ctx as prof:
                while step < num_steps:
                    # Get batch (with wraparound)
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        batch = next(data_iter)
                    
                    # Unpack batch (handles different formats)
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        inputs, targets = batch
                    else:
                        inputs, targets = batch, batch
                    
                    # Move data to device
                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(self.device)
                    if isinstance(targets, torch.Tensor):
                        targets = targets.to(self.device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    
                    # Handle different output formats
                    if isinstance(outputs, dict) and "loss" in outputs:
                        loss = outputs["loss"]
                    else:
                        loss = criterion(outputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Update step and profiler
                    step += 1
                    
                    # Handle profiler step() for newer versions
                    if hasattr(prof, "step"):
                        prof.step()
        
        except Exception as e:
            logger.error(f"Error during profiling: {e}")
            raise
        
        # Process profiling results
        if hasattr(prof, "key_averages"):
            events = prof.key_averages()
            
            # Total duration
            total_duration = sum(event.cpu_time_total for event in events)
            result.duration_ms["total"] = total_duration / 1000  # Convert to ms
            
            # Self CPU time by operator
            for event in events:
                if event.key not in result.duration_ms:
                    result.duration_ms[f"op.{event.key}"] = event.cpu_time_total / 1000  # Convert to ms
                
                # Add memory metrics if available
                if hasattr(event, "cpu_memory_usage") and event.cpu_memory_usage > 0:
                    result.memory_usage[f"cpu.{event.key}"] = event.cpu_memory_usage / (1024 ** 2)  # Convert to MB
                
                if hasattr(event, "cuda_memory_usage") and event.cuda_memory_usage > 0:
                    result.memory_usage[f"cuda.{event.key}"] = event.cuda_memory_usage / (1024 ** 2)  # Convert to MB
            
            # Total memory usage
            result.memory_usage["cpu_total"] = sum(
                getattr(event, "cpu_memory_usage", 0) for event in events
            ) / (1024 ** 2)  # Convert to MB
            
            if self.device.type == "cuda":
                result.memory_usage["cuda_total"] = sum(
                    getattr(event, "cuda_memory_usage", 0) for event in events
                ) / (1024 ** 2)  # Convert to MB
        
        # Save raw profiler output for later analysis
        try:
            if hasattr(prof, "table"):
                result.raw_data["profiler_output"] = str(prof.table(sort_by="cpu_time_total"))
            else:
                result.raw_data["profiler_output"] = str(prof)
        except Exception as e:
            logger.warning(f"Could not generate profiler output table: {e}")
            result.raw_data["profiler_output"] = "Profiler output not available"
        
        # Add model parameters
        result.parameters = self._get_model_parameters()
        
        logger.info(f"Training profile complete. Duration: {result.duration_ms.get('total', 0):.2f} ms")
        
        return result
    
    def profile_inference(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_steps: int = 10,
        with_grad: bool = False,
        batch_size: Optional[int] = None,
        name: Optional[str] = None,
    ) -> ProfileResult:
        """
        Profile the inference phase of a model.
        
        Args:
            dataloader: DataLoader for inference data
            num_steps: Number of inference steps to profile
            with_grad: Whether to enable gradients during inference
            batch_size: Batch size to use (overrides dataloader batch size)
            name: Name for the profile result
            
        Returns:
            ProfileResult: Profile results
        """
        if not self.config.enabled:
            logger.warning("Profiling is disabled. Returning empty profile result.")
            return ProfileResult(name=name or "inference", device=str(self.device))
        
        if self.model is None:
            raise ValueError("Model must be set to profile inference")
        
        logger.info(f"Profiling inference for {num_steps} steps...")
        
        # Setup profiling results
        result = ProfileResult(
            name=name or "inference",
            device=str(self.device),
        )
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Create PyTorch profiler
        profile_path = Path(self.config.profile_dir) / f"{result.name}_{result.timestamp}"
        os.makedirs(profile_path, exist_ok=True)
        
        # Setup tracing for PyTorch profiler
        def trace_handler(prof, steps=None):
            trace_path = profile_path / f"trace_{steps}.json"
            prof.export_chrome_trace(str(trace_path))
            if result.traces is None:
                result.traces = []
            result.traces.append(trace_path)
        
        # Create profiler context
        try:
            # Try to import torch.profiler (PyTorch 1.9+)
            import torch.profiler as torch_profiler
            profiler_cls = torch_profiler.profile
            schedule_cls = torch_profiler.schedule
            activities_map = {
                "cpu": torch_profiler.ProfilerActivity.CPU,
                "cuda": torch_profiler.ProfilerActivity.CUDA,
            }
            activities = [activities_map[act] for act in self.config.activities if act in activities_map]
            
            # Create schedule
            schedule = schedule_cls(
                wait=self.config.schedule["wait"],
                warmup=self.config.schedule["warmup"],
                active=self.config.schedule["active"],
                repeat=self.config.schedule["repeat"],
            )
            
            # Create profiler with compatible parameters
            profiler_kwargs = {
                "activities": activities,
                "schedule": schedule,
                "on_trace_ready": trace_handler,
                "record_shapes": self.config.record_shapes,
                "profile_memory": self.config.profile_memory,
                "with_stack": self.config.with_stack,
            }
            
            # Add optional parameters if supported
            if hasattr(torch.profiler.ProfilerActivity, "CUDA"):
                profiler_kwargs["with_flops"] = self.config.with_flops
                
            # Check if the profiler supports record_module_hierarchy
            if "record_module_hierarchy" in inspect.signature(profiler_cls.__init__).parameters:
                profiler_kwargs["record_module_hierarchy"] = self.config.record_module_hierarchy
            
            # Create profiler with compatible parameters
            profiler_ctx = profiler_cls(**profiler_kwargs)
            
        except ImportError:
            # Fallback for older PyTorch versions
            logger.warning("torch.profiler not available. Using legacy profiler.")
            profiler_ctx = torch.autograd.profiler.profile(
                use_cuda=self.config.use_cuda,
                record_shapes=self.config.record_shapes,
                profile_memory=self.config.profile_memory,
                with_stack=self.config.with_stack,
            )
        
        # Run inference loop with profiling
        data_iter = iter(dataloader)
        step = 0
        
        try:
            with profiler_ctx as prof:
                # Set gradients based on config
                torch.set_grad_enabled(with_grad)
                
                while step < num_steps:
                    # Get batch (with wraparound)
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        batch = next(data_iter)
                    
                    # Unpack batch (handles different formats)
                    if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                        inputs = batch[0]
                    else:
                        inputs = batch
                    
                    # Move data to device
                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    # Update step and profiler
                    step += 1
                    
                    # Handle profiler step() for newer versions
                    if hasattr(prof, "step"):
                        prof.step()
                
                # Reset grad setting
                torch.set_grad_enabled(True)
        
        except Exception as e:
            logger.error(f"Error during profiling: {e}")
            raise
        
        # Process profiling results
        if hasattr(prof, "key_averages"):
            events = prof.key_averages()
            
            # Total duration
            total_duration = sum(event.cpu_time_total for event in events)
            result.duration_ms["total"] = total_duration / 1000  # Convert to ms
            
            # Per-batch duration
            result.duration_ms["per_batch"] = total_duration / (1000 * num_steps)  # Convert to ms
            
            # Self CPU time by operator
            for event in events:
                if event.key not in result.duration_ms:
                    result.duration_ms[f"op.{event.key}"] = event.cpu_time_total / 1000  # Convert to ms
                
                # Add memory metrics if available
                if hasattr(event, "cpu_memory_usage") and event.cpu_memory_usage > 0:
                    result.memory_usage[f"cpu.{event.key}"] = event.cpu_memory_usage / (1024 ** 2)  # Convert to MB
                
                if hasattr(event, "cuda_memory_usage") and event.cuda_memory_usage > 0:
                    result.memory_usage[f"cuda.{event.key}"] = event.cuda_memory_usage / (1024 ** 2)  # Convert to MB
            
            # Total memory usage
            result.memory_usage["cpu_total"] = sum(
                getattr(event, "cpu_memory_usage", 0) for event in events
            ) / (1024 ** 2)  # Convert to MB
            
            if self.device.type == "cuda":
                result.memory_usage["cuda_total"] = sum(
                    getattr(event, "cuda_memory_usage", 0) for event in events
                ) / (1024 ** 2)  # Convert to MB
        
        # Save raw profiler output for later analysis
        try:
            if hasattr(prof, "table"):
                result.raw_data["profiler_output"] = str(prof.table(sort_by="cpu_time_total"))
            else:
                result.raw_data["profiler_output"] = str(prof)
        except Exception as e:
            logger.warning(f"Could not generate profiler output table: {e}")
            result.raw_data["profiler_output"] = "Profiler output not available"
        
        # Add model parameters
        result.parameters = self._get_model_parameters()
        
        logger.info(f"Inference profile complete. Duration per batch: {result.duration_ms.get('per_batch', 0):.2f} ms")
        
        return result
    
    def profile_memory_usage(
        self,
        dataloader: torch.utils.data.DataLoader,
        name: Optional[str] = None,
    ) -> ProfileResult:
        """
        Profile memory usage during model execution.
        
        Args:
            dataloader: DataLoader for input data
            name: Name for the profile result
            
        Returns:
            ProfileResult: Profile results
        """
        if not self.config.enabled:
            logger.warning("Profiling is disabled. Returning empty profile result.")
            return ProfileResult(name=name or "memory_usage", device=str(self.device))
        
        if self.model is None:
            raise ValueError("Model must be set to profile memory usage")
        
        from tauto.profile.memory import track_memory_usage, estimate_model_memory
        
        logger.info("Profiling memory usage...")
        
        # Setup profiling results
        result = ProfileResult(
            name=name or "memory_usage",
            device=str(self.device),
        )
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Estimate model memory
        model_memory = estimate_model_memory(self.model)
        result.memory_usage.update(model_memory)
        
        # Track memory during execution
        data_iter = iter(dataloader)
        batch = next(data_iter)
        
        # Unpack batch (handles different formats)
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            inputs = batch[0]
        else:
            inputs = batch
        
        # Move data to device
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        
        # Track memory usage
        memory_data = track_memory_usage(
            lambda: self.model(inputs),
            device=self.device,
        )
        
        # Update result
        result.memory_usage.update(memory_data)
        
        # Add model parameters
        result.parameters = self._get_model_parameters()
        
        logger.info(f"Memory profiling complete. Peak usage: {memory_data.get('peak_mb', 0):.2f} MB")
        
        return result
    
    def _get_model_parameters(self) -> Dict[str, float]:
        """
        Get information about model parameters.
        
        Returns:
            Dict[str, float]: Parameter information
        """
        if self.model is None:
            return {}
        
        parameters = {}
        
        # Total parameters
        parameters["total"] = sum(p.numel() for p in self.model.parameters())
        
        # Trainable parameters
        parameters["trainable"] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Non-trainable parameters
        parameters["non_trainable"] = parameters["total"] - parameters["trainable"]
        
        # Parameter sizes in MB
        parameters["size_mb"] = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 2)
        
        return parameters


def profile_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: Optional[Union[Dict[str, Any], ProfilerConfig]] = None,
    mode: str = "all",
    name: Optional[str] = None,
) -> Dict[str, ProfileResult]:
    """
    Profile a model with the given dataloader.
    
    Args:
        model: PyTorch model to profile
        dataloader: DataLoader for input data
        config: Profiler configuration
        mode: Profiling mode ('training', 'inference', 'memory', or 'all')
        name: Base name for the profile results
        
    Returns:
        Dict[str, ProfileResult]: Profile results for each mode
    """
    profiler = Profiler(model, config)
    results = {}
    
    base_name = name or model.__class__.__name__.lower()
    
    if mode in ["all", "training"]:
        results["training"] = profiler.profile_training(
            dataloader=dataloader,
            name=f"{base_name}_training",
        )
    
    if mode in ["all", "inference"]:
        results["inference"] = profiler.profile_inference(
            dataloader=dataloader,
            name=f"{base_name}_inference",
        )
    
    if mode in ["all", "memory"]:
        results["memory"] = profiler.profile_memory_usage(
            dataloader=dataloader,
            name=f"{base_name}_memory",
        )
    
    return results