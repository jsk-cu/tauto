"""
Compiler optimization utilities for TAuto.

This module provides tools for optimizing PyTorch models using various
compilation techniques like torch.compile and TorchScript.
"""

import os
import torch
import time
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import copy
import warnings
from pathlib import Path
import functools
import inspect
import logging

from tauto.utils import get_logger

logger = get_logger(__name__)


class TorchCompile:
    """
    Torch.compile optimization utilities.
    
    This class provides utilities for applying torch.compile optimizations
    to PyTorch models for faster execution.
    """
    
    def __init__(
        self,
        backend: str = "inductor",
        mode: str = "default",
        fullgraph: bool = True,
        options: Optional[Dict[str, Any]] = None,
        disabled: bool = False,
    ):
        """
        Initialize torch.compile utilities.
        
        Args:
            backend: Compilation backend ('inductor', 'aot_eager', 'cudagraphs', etc.)
            mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
            fullgraph: Whether to capture the entire graph or allow fallbacks
            options: Additional options for the compiler
            disabled: Whether compilation should be disabled
        """
        self.backend = backend
        self.mode = mode
        self.fullgraph = fullgraph
        self.options = options or {}
        self.disabled = disabled
        
        # Check if torch.compile is available
        self.available = hasattr(torch, "compile")
        if not self.available and not self.disabled:
            logger.warning("torch.compile is not available in your PyTorch version. "
                         "Compilation will be disabled.")
            self.disabled = True
        
        # Validate backend
        self._validate_backend()
        
        logger.info(f"Initialized torch.compile utilities: backend={backend}, mode={mode}, "
                  f"disabled={disabled}")
    
    def _validate_backend(self):
        """Validate the selected backend."""
        if self.disabled or not self.available:
            return
        
        # List of recommended backends
        recommended_backends = ["inductor", "aot_eager", "cudagraphs", "onnxrt", "tvm"]
        
        # Check if the selected backend is in the recommended list
        if self.backend not in recommended_backends:
            logger.warning(f"Backend '{self.backend}' is not in the list of recommended backends: "
                         f"{recommended_backends}. Compilation might fail.")
    
    def compile_model(
        self,
        model: torch.nn.Module,
        example_inputs: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
        inplace: bool = False,
    ) -> torch.nn.Module:
        """
        Apply torch.compile to a model.
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs for tracing (future proofing)
            inplace: Whether to modify the model in-place
            
        Returns:
            torch.nn.Module: Compiled model
        """
        if self.disabled or not self.available:
            logger.warning("Compilation is disabled. Returning the original model.")
            return model
        
        if not inplace:
            model = copy.deepcopy(model)
        
        # Set model to eval mode for inference optimization
        was_training = model.training
        if not was_training:
            model.eval()
        
        try:
            # Apply torch.compile
            compile_args = {
                "backend": self.backend,
                "mode": self.mode,
                "fullgraph": self.fullgraph,
                **self.options
            }
            
            # Only include parameters that are accepted by torch.compile
            signature = inspect.signature(torch.compile)
            compile_args = {k: v for k, v in compile_args.items() if k in signature.parameters}
            
            # Apply compilation
            compiled_model = torch.compile(model, **compile_args)
            
            # If example inputs are provided, run a sample inference to trigger compilation
            if example_inputs is not None:
                with torch.no_grad():
                    if isinstance(example_inputs, torch.Tensor):
                        compiled_model(example_inputs)
                    else:
                        compiled_model(*example_inputs)
            
            logger.info(f"Successfully compiled model with {self.backend} backend")
            
            # Restore training mode if it was training
            if was_training:
                compiled_model.train()
            
            return compiled_model
            
        except Exception as e:
            logger.error(f"Error during compilation: {e}")
            logger.warning("Returning original model due to compilation error")
            
            # Restore training mode if it was training
            if was_training:
                model.train()
            
            return model
    
    def benchmark_compilation(
        self,
        model: torch.nn.Module,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        num_warmup: int = 5,
        num_runs: int = 20,
    ) -> Dict[str, Any]:
        """
        Benchmark the performance improvement from compilation.
        
        Args:
            model: Original PyTorch model
            inputs: Input data for benchmarking
            num_warmup: Number of warmup runs
            num_runs: Number of benchmark runs
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        if self.disabled or not self.available:
            logger.warning("Compilation is disabled. Benchmark will compare the same model.")
        
        # Create a copy of the original model
        original_model = copy.deepcopy(model).eval()
        
        # Compile the model
        compiled_model = self.compile_model(model, example_inputs=inputs, inplace=False).eval()
        
        # Move to the same device as inputs
        device = inputs.device if isinstance(inputs, torch.Tensor) else inputs[0].device
        original_model = original_model.to(device)
        compiled_model = compiled_model.to(device)
        
        # Benchmark original model
        original_times = []
        with torch.no_grad():
            # Warmup
            for _ in range(num_warmup):
                if isinstance(inputs, torch.Tensor):
                    original_model(inputs)
                else:
                    original_model(*inputs)
            
            # Benchmark
            for _ in range(num_runs):
                start_time = time.time()
                if isinstance(inputs, torch.Tensor):
                    original_model(inputs)
                else:
                    original_model(*inputs)
                end_time = time.time()
                original_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Benchmark compiled model
        compiled_times = []
        with torch.no_grad():
            # Warmup
            for _ in range(num_warmup):
                if isinstance(inputs, torch.Tensor):
                    compiled_model(inputs)
                else:
                    compiled_model(*inputs)
            
            # Benchmark
            for _ in range(num_runs):
                start_time = time.time()
                if isinstance(inputs, torch.Tensor):
                    compiled_model(inputs)
                else:
                    compiled_model(*inputs)
                end_time = time.time()
                compiled_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        orig_avg = sum(original_times) / len(original_times)
        orig_min = min(original_times)
        orig_max = max(original_times)
        orig_std = (sum((t - orig_avg) ** 2 for t in original_times) / len(original_times)) ** 0.5
        
        comp_avg = sum(compiled_times) / len(compiled_times)
        comp_min = min(compiled_times)
        comp_max = max(compiled_times)
        comp_std = (sum((t - comp_avg) ** 2 for t in compiled_times) / len(compiled_times)) ** 0.5
        
        speedup = orig_avg / comp_avg if comp_avg > 0 else 0
        
        return {
            "original_avg_ms": orig_avg,
            "original_min_ms": orig_min,
            "original_max_ms": orig_max,
            "original_std_ms": orig_std,
            "compiled_avg_ms": comp_avg,
            "compiled_min_ms": comp_min,
            "compiled_max_ms": comp_max,
            "compiled_std_ms": comp_std,
            "speedup": speedup,
            "backend": self.backend,
            "mode": self.mode,
        }
    
    @staticmethod
    def debug_model(
        model: torch.nn.Module,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        backend: str = "inductor",
        mode: str = "default",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Debug compilation issues with a model.
        
        Args:
            model: PyTorch model
            inputs: Input data for testing
            backend: Compilation backend to test
            mode: Compilation mode to test
            verbose: Whether to print debug information
            
        Returns:
            Dict[str, Any]: Debug information
        """
        debug_info = {
            "success": False,
            "error": None,
            "error_type": None,
            "backend": backend,
            "mode": mode,
            "torch_version": torch.__version__,
            "device": str(next(model.parameters()).device),
            "model_type": type(model).__name__,
        }
        
        # Check if torch.compile is available
        if not hasattr(torch, "compile"):
            debug_info["error"] = "torch.compile is not available in your PyTorch version"
            debug_info["error_type"] = "UnavailableFeature"
            if verbose:
                logger.error(debug_info["error"])
            return debug_info
        
        # Try compiling with different backends and modes if the specified one fails
        try:
            # Try the specified backend and mode
            compile_args = {
                "backend": backend,
                "mode": mode,
            }
            
            # Only include parameters that are accepted by torch.compile
            signature = inspect.signature(torch.compile)
            compile_args = {k: v for k, v in compile_args.items() if k in signature.parameters}
            
            # Apply compilation
            compiled_model = torch.compile(model, **compile_args)
            
            # Test the compiled model
            with torch.no_grad():
                if isinstance(inputs, torch.Tensor):
                    outputs = compiled_model(inputs)
                else:
                    outputs = compiled_model(*inputs)
            
            debug_info["success"] = True
            
            if verbose:
                logger.info(f"Compilation successful with backend={backend}, mode={mode}")
            
        except Exception as e:
            debug_info["error"] = str(e)
            debug_info["error_type"] = type(e).__name__
            
            if verbose:
                logger.error(f"Compilation failed with backend={backend}, mode={mode}: {e}")
            
            # Try alternative backends if the specified one failed
            alternative_backends = ["aot_eager", "cudagraphs", "inductor", "onnxrt"]
            if backend in alternative_backends:
                alternative_backends.remove(backend)
            
            if verbose:
                logger.info(f"Trying alternative backends: {alternative_backends}")
            
            for alt_backend in alternative_backends:
                try:
                    compiled_model = torch.compile(model, backend=alt_backend)
                    
                    # Test the compiled model
                    with torch.no_grad():
                        if isinstance(inputs, torch.Tensor):
                            outputs = compiled_model(inputs)
                        else:
                            outputs = compiled_model(*inputs)
                    
                    debug_info["alternative_backend"] = alt_backend
                    debug_info["alternative_success"] = True
                    
                    if verbose:
                        logger.info(f"Compilation successful with alternative backend={alt_backend}")
                    
                    break
                    
                except Exception as alt_e:
                    if verbose:
                        logger.warning(f"Alternative backend {alt_backend} also failed: {alt_e}")
        
        return debug_info


class TorchScriptExport:
    """
    TorchScript export and optimization utilities.
    
    This class provides utilities for exporting PyTorch models to TorchScript
    and applying optimizations to the exported models.
    """
    
    def __init__(
        self,
        optimization_level: int = 3,
        strict: bool = False,
        method: str = "trace",
    ):
        """
        Initialize TorchScript export utilities.
        
        Args:
            optimization_level: TorchScript optimization level (0-3)
            strict: Whether to use strict mode for tracing
            method: Export method ('trace' or 'script')
        """
        self.optimization_level = optimization_level
        self.strict = strict
        self.method = method
        
        # Validate method
        if self.method not in ["trace", "script"]:
            raise ValueError(f"Method must be 'trace' or 'script', got {self.method}")
        
        logger.info(f"Initialized TorchScript export utilities: method={method}, "
                  f"optimization_level={optimization_level}")
    
    def export_model(
        self,
        model: torch.nn.Module,
        example_inputs: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> torch.jit.ScriptModule:
        """
        Export a model to TorchScript.
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs for tracing (required for trace method)
            save_path: Path to save the exported model
            
        Returns:
            torch.jit.ScriptModule: Exported model
        """
        # Set model to eval mode
        model.eval()
        
        # Export model based on method
        if self.method == "trace":
            if example_inputs is None:
                raise ValueError("example_inputs is required for tracing")
            
            with torch.no_grad():
                if isinstance(example_inputs, torch.Tensor):
                    ts_model = torch.jit.trace(model, example_inputs, check_trace=self.strict)
                else:
                    ts_model = torch.jit.trace(model, example_inputs, check_trace=self.strict)
            
            logger.info("Model exported using tracing")
            
        else:  # self.method == "script"
            ts_model = torch.jit.script(model)
            logger.info("Model exported using scripting")
        
        # Apply optimizations
        if self.optimization_level > 0:
            ts_model = self._optimize_torchscript(ts_model, self.optimization_level)
        
        # Save the model if a path is provided
        if save_path is not None:
            save_path = Path(save_path)
            os.makedirs(save_path.parent, exist_ok=True)
            torch.jit.save(ts_model, save_path)
            logger.info(f"Exported model saved to {save_path}")
        
        return ts_model
    
    def _optimize_torchscript(
        self,
        ts_model: torch.jit.ScriptModule,
        level: int,
    ) -> torch.jit.ScriptModule:
        """
        Apply optimizations to a TorchScript model.
        
        Args:
            ts_model: TorchScript model
            level: Optimization level (0-3)
            
        Returns:
            torch.jit.ScriptModule: Optimized model
        """
        # Apply different optimization levels
        if level <= 0:
            return ts_model
        
        # Check if _jit_pass_optimize is available
        if not hasattr(torch._C, "_jit_pass_optimize"):
            logger.warning("TorchScript optimization passes not available in this PyTorch version")
            return ts_model
        
        try:
            # Level 1: Basic optimizations
            if level >= 1:
                torch._C._jit_pass_peephole(ts_model.graph)
                torch._C._jit_pass_constant_propagation(ts_model.graph)
            
            # Level 2: More aggressive optimizations
            if level >= 2:
                # These passes might not be available in all PyTorch versions
                if hasattr(torch._C, "_jit_pass_remove_redundant_guards"):
                    torch._C._jit_pass_remove_redundant_guards(ts_model.graph)
                if hasattr(torch._C, "_jit_pass_eliminate_dead_code"):
                    torch._C._jit_pass_eliminate_dead_code(ts_model.graph)
            
            # Level 3: Most aggressive optimizations
            if level >= 3:
                torch._C._jit_pass_inline(ts_model.graph)
                if hasattr(torch._C, "_jit_pass_fuse_linear"):
                    torch._C._jit_pass_fuse_linear(ts_model.graph)
            
            logger.info(f"Applied TorchScript optimizations at level {level}")
            return ts_model
            
        except Exception as e:
            logger.warning(f"Error during TorchScript optimization: {e}")
            logger.warning("Returning unoptimized model")
            return ts_model
    
    def benchmark_export(
        self,
        model: torch.nn.Module,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        num_warmup: int = 5,
        num_runs: int = 20,
    ) -> Dict[str, Any]:
        """
        Benchmark the performance improvement from TorchScript export.
        
        Args:
            model: Original PyTorch model
            inputs: Input data for benchmarking
            num_warmup: Number of warmup runs
            num_runs: Number of benchmark runs
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        # Create a copy of the original model
        original_model = copy.deepcopy(model).eval()
        
        # Export the model to TorchScript
        ts_model = self.export_model(model, example_inputs=inputs)
        
        # Move to the same device as inputs
        device = inputs.device if isinstance(inputs, torch.Tensor) else inputs[0].device
        original_model = original_model.to(device)
        
        # Benchmark original model
        original_times = []
        with torch.no_grad():
            # Warmup
            for _ in range(num_warmup):
                if isinstance(inputs, torch.Tensor):
                    original_model(inputs)
                else:
                    original_model(*inputs)
            
            # Benchmark
            for _ in range(num_runs):
                start_time = time.time()
                if isinstance(inputs, torch.Tensor):
                    original_model(inputs)
                else:
                    original_model(*inputs)
                end_time = time.time()
                original_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Benchmark TorchScript model
        ts_times = []
        with torch.no_grad():
            # Warmup
            for _ in range(num_warmup):
                if isinstance(inputs, torch.Tensor):
                    ts_model(inputs)
                else:
                    ts_model(*inputs)
            
            # Benchmark
            for _ in range(num_runs):
                start_time = time.time()
                if isinstance(inputs, torch.Tensor):
                    ts_model(inputs)
                else:
                    ts_model(*inputs)
                end_time = time.time()
                ts_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        orig_avg = sum(original_times) / len(original_times)
        orig_min = min(original_times)
        orig_max = max(original_times)
        orig_std = (sum((t - orig_avg) ** 2 for t in original_times) / len(original_times)) ** 0.5
        
        ts_avg = sum(ts_times) / len(ts_times)
        ts_min = min(ts_times)
        ts_max = max(ts_times)
        ts_std = (sum((t - ts_avg) ** 2 for t in ts_times) / len(ts_times)) ** 0.5
        
        speedup = orig_avg / ts_avg if ts_avg > 0 else 0
        
        return {
            "original_avg_ms": orig_avg,
            "original_min_ms": orig_min,
            "original_max_ms": orig_max,
            "original_std_ms": orig_std,
            "torchscript_avg_ms": ts_avg,
            "torchscript_min_ms": ts_min,
            "torchscript_max_ms": ts_max,
            "torchscript_std_ms": ts_std,
            "speedup": speedup,
            "method": self.method,
            "optimization_level": self.optimization_level,
        }


def apply_compiler_optimization(
    model: torch.nn.Module,
    optimization: str = "torch_compile",
    example_inputs: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
    save_path: Optional[Union[str, Path]] = None,
    backend: str = "inductor",
    mode: str = "default",
    optimization_level: int = 3,
    inplace: bool = False,
) -> torch.nn.Module:
    """
    Apply compiler optimizations to a model.
    
    Args:
        model: PyTorch model
        optimization: Type of optimization ('torch_compile' or 'torchscript')
        example_inputs: Example inputs for tracing/compiling
        save_path: Path to save the exported model (for TorchScript)
        backend: Backend for torch.compile
        mode: Mode for torch.compile
        optimization_level: Optimization level for TorchScript
        inplace: Whether to modify the model in-place
        
    Returns:
        torch.nn.Module: Optimized model
    """
    if not inplace:
        model = copy.deepcopy(model)
    
    model.eval()
    
    if optimization.lower() == "torch_compile":
        # Apply torch.compile optimization
        compiler = TorchCompile(backend=backend, mode=mode)
        optimized_model = compiler.compile_model(model, example_inputs=example_inputs, inplace=True)
        
    elif optimization.lower() == "torchscript":
        # Export to TorchScript
        exporter = TorchScriptExport(optimization_level=optimization_level)
        optimized_model = exporter.export_model(model, example_inputs=example_inputs, save_path=save_path)
        
    else:
        logger.warning(f"Unknown optimization type: {optimization}. Returning original model.")
        optimized_model = model
    
    return optimized_model