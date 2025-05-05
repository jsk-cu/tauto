"""
Inference optimization utilities for TAuto.

This module provides tools for optimizing the inference process of PyTorch models,
including quantization, pruning, and other optimization techniques.
"""

import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization
from torch.quantization import quantize_dynamic, quantize, prepare, convert, QConfig
from torch.quantization.observer import MinMaxObserver
import time
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import copy
import warnings
import numpy as np
from pathlib import Path

from tauto.utils import get_logger

logger = get_logger(__name__)


class ModelQuantization:
    """
    Model quantization utilities.
    
    This class provides utilities for quantizing PyTorch models,
    which can reduce memory usage and improve inference speed.
    """
    
    def __init__(
        self,
        quantization_type: str = "dynamic",
        dtype: torch.dtype = torch.qint8,
        qconfig: Optional[QConfig] = None,
        modules_to_fuse: Optional[List[List[str]]] = None,
    ):
        """
        Initialize model quantization.
        
        Args:
            quantization_type: Type of quantization ('dynamic', 'static', or 'qat')
            dtype: Data type to quantize to (torch.qint8, torch.quint8)
            qconfig: Custom quantization configuration
            modules_to_fuse: List of module names to fuse before quantization
        """
        self.quantization_type = quantization_type.lower()
        self.dtype = dtype
        self.qconfig = qconfig
        self.modules_to_fuse = modules_to_fuse or []
        
        # Validate quantization type
        valid_types = ["dynamic", "static", "qat"]
        if self.quantization_type not in valid_types:
            raise ValueError(f"Quantization type must be one of {valid_types}, got {self.quantization_type}")
        
        logger.info(f"Initialized model quantization: type={quantization_type}, dtype={dtype}")
    
    def prepare_model(
        self,
        model: nn.Module,
        example_inputs: Optional[torch.Tensor] = None,
        backend: str = "fbgemm",
        inplace: bool = False,
    ) -> nn.Module:
        """
        Prepare a model for quantization.
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs for tracing/calibration
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
            inplace: Whether to modify the model in-place
            
        Returns:
            nn.Module: Prepared model
        """
        if not inplace:
            model = copy.deepcopy(model)
        
        # Set model to eval mode for inference optimization
        model.eval()
        
        # Fuse modules if specified
        if self.modules_to_fuse:
            model = torch.quantization.fuse_modules(model, self.modules_to_fuse)
        
        # Set quantization backend
        if backend == "fbgemm":
            torch.backends.quantized.engine = "fbgemm"
        elif backend == "qnnpack":
            torch.backends.quantized.engine = "qnnpack"
        else:
            raise ValueError(f"Unsupported quantization backend: {backend}")
        
        # Prepare model based on quantization type
        if self.quantization_type == "dynamic":
            # Dynamic quantization doesn't need preparation
            logger.info("Dynamic quantization doesn't require explicit preparation")
            return model
        elif self.quantization_type == "static":
            # Set default qconfig if not provided
            if self.qconfig is None:
                self.qconfig = torch.quantization.get_default_qconfig(backend)
            
            # Set qconfig for the model
            model.qconfig = self.qconfig
            
            # Prepare the model for static quantization
            prepared_model = prepare(model)
            logger.info("Model prepared for static quantization")
            return prepared_model
        elif self.quantization_type == "qat":
            # Set default qat qconfig if not provided
            if self.qconfig is None:
                self.qconfig = torch.quantization.get_default_qat_qconfig(backend)
            
            # Set qconfig for the model
            model.qconfig = self.qconfig
            
            # Prepare the model for quantization aware training
            prepared_model = torch.quantization.prepare_qat(model)
            logger.info("Model prepared for quantization aware training")
            return prepared_model
    
    def calibrate_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 10,
    ) -> nn.Module:
        """
        Calibrate a model for static quantization.
        
        Args:
            model: Prepared PyTorch model
            dataloader: DataLoader with calibration data
            num_batches: Number of batches to use for calibration
            
        Returns:
            nn.Module: Calibrated model
        """
        if self.quantization_type != "static":
            logger.warning(f"Calibration is only needed for static quantization, not {self.quantization_type}")
            return model
        
        # Run calibration
        logger.info(f"Calibrating model with {num_batches} batches")
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                # Forward pass only
                model(inputs)
        
        logger.info("Model calibration complete")
        return model
    
    def quantize_model(
        self,
        model: nn.Module,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        num_calibration_batches: int = 10,
        inplace: bool = False,
    ) -> nn.Module:
        """
        Quantize a model based on the specified quantization type.
        
        Args:
            model: PyTorch model
            dataloader: DataLoader with calibration data (for static quantization)
            num_calibration_batches: Number of batches to use for calibration
            inplace: Whether to modify the model in-place
            
        Returns:
            nn.Module: Quantized model
        """
        if not inplace:
            model = copy.deepcopy(model)
        
        model.eval()
        
        if self.quantization_type == "dynamic":
            # Apply dynamic quantization
            quantized_model = quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU, nn.RNN},  # Quantize linear and RNN layers
                dtype=self.dtype
            )
            # Add is_quantized attribute
            quantized_model.is_quantized = True
            logger.info("Model quantized with dynamic quantization")
            return quantized_model
        elif self.quantization_type == "static":
            if dataloader is None:
                raise ValueError("DataLoader is required for static quantization calibration")
            
            # Prepare the model
            prepared_model = self.prepare_model(model, inplace=True)
            
            # Calibrate the model
            calibrated_model = self.calibrate_model(
                prepared_model,
                dataloader,
                num_batches=num_calibration_batches
            )
            
            # Convert the model to quantized version
            quantized_model = convert(calibrated_model)
            logger.info("Model quantized with static quantization")
            return quantized_model
        elif self.quantization_type == "qat":
            # For QAT, the model should have been trained with quantization awareness
            # We simply convert the trained model
            if not hasattr(model, "qconfig"):
                logger.warning("Model doesn't have qconfig attribute. Did you train with quantization awareness?")
                # Prepare the model for QAT as a fallback
                model = self.prepare_model(model, inplace=True)
            
            # Convert the model to quantized version
            quantized_model = convert(model)
            logger.info("Model quantized with quantization aware training")
            return quantized_model
    
    def evaluate_quantization(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cpu"),
        criterion: Optional[Callable] = None,
        num_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the performance impact of quantization.
        
        Args:
            original_model: Original PyTorch model
            quantized_model: Quantized PyTorch model
            dataloader: DataLoader with evaluation data
            device: Device to run evaluation on
            criterion: Loss function
            num_batches: Number of batches to use for evaluation
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        # Ensure models are in eval mode
        original_model.eval()
        quantized_model.eval()
        
        # Move original model to appropriate device
        original_model = original_model.to(device)
        
        # Note: Quantized models only run on CPU
        if device.type != "cpu" and hasattr(quantized_model, "is_quantized") and quantized_model.is_quantized:
            logger.warning("Quantized models can only run on CPU, ignoring device")
        
        results = {
            "original_size_mb": self._get_model_size(original_model),
            "quantized_size_mb": self._get_model_size(quantized_model),
            "original_latency_ms": 0.0,
            "quantized_latency_ms": 0.0,
            "original_accuracy": 0.0,
            "quantized_accuracy": 0.0,
            "speedup": 0.0,
            "memory_reduction": 0.0,
        }
        
        # Evaluate original model
        original_latency, original_accuracy = self._evaluate_model(
            original_model,
            dataloader,
            device,
            criterion,
            num_batches
        )
        
        # Evaluate quantized model (always on CPU)
        quantized_latency, quantized_accuracy = self._evaluate_model(
            quantized_model,
            dataloader,
            torch.device("cpu"),
            criterion,
            num_batches
        )
        
        # Update results
        results["original_latency_ms"] = original_latency
        results["quantized_latency_ms"] = quantized_latency
        results["original_accuracy"] = original_accuracy
        results["quantized_accuracy"] = quantized_accuracy
        
        # Calculate improvements
        if original_latency > 0:
            results["speedup"] = original_latency / quantized_latency
        
        if results["original_size_mb"] > 0:
            results["memory_reduction"] = 1.0 - (results["quantized_size_mb"] / results["original_size_mb"])
        
        return results
    
    def _get_model_size(self, model: nn.Module) -> float:
        """
        Get the size of a model in megabytes.
        
        Args:
            model: PyTorch model
            
        Returns:
            float: Model size in MB
        """
        # Get size using PyTorch state_dict
        state_dict = model.state_dict()
        size_bytes = 0
        
        for name, param in state_dict.items():
            # Skip if param is not a tensor or doesn't have numel() method
            if not hasattr(param, 'numel') or not hasattr(param, 'element_size'):
                continue
            size_bytes += param.numel() * param.element_size()
        
        return size_bytes / (1024 ** 2)  # Convert to MB
    
    def _evaluate_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: Optional[Callable] = None,
        num_batches: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Evaluate a model's latency and accuracy.
        
        Args:
            model: PyTorch model
            dataloader: DataLoader with evaluation data
            device: Device to run evaluation on
            criterion: Loss function
            num_batches: Number of batches to use for evaluation
            
        Returns:
            Tuple[float, float]: Latency (ms) and accuracy
        """
        # Move model to device
        model = model.to(device)
        
        total_latency = 0.0
        total_samples = 0
        correct = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(dataloader):
                if num_batches is not None and i >= num_batches:
                    break
                
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Measure latency
                start_time = time.time()
                outputs = model(inputs)
                end_time = time.time()
                
                batch_latency = (end_time - start_time) * 1000  # Convert to ms
                total_latency += batch_latency
                
                # Compute loss if criterion is provided
                if criterion is not None:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)
                
                # Compute accuracy
                if isinstance(outputs, torch.Tensor) and outputs.dim() > 1 and outputs.size(1) > 1:
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                
                total_samples += inputs.size(0)
        
        # Calculate averages
        avg_latency = total_latency / (i + 1)  # Average latency per batch
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        
        return avg_latency, accuracy


class ModelPruning:
    """
    Model pruning utilities.
    
    This class provides utilities for pruning PyTorch models,
    which can reduce model size and potentially improve inference speed.
    """
    
    def __init__(
        self,
        pruning_type: str = "unstructured",
        criteria: str = "l1",
        parameters_to_prune: Optional[List[Tuple[nn.Module, str]]] = None,
    ):
        """
        Initialize model pruning.
        
        Args:
            pruning_type: Type of pruning ('unstructured', 'structured', 'global')
            criteria: Pruning criteria ('random', 'l1', 'l2')
            parameters_to_prune: List of (module, parameter_name) tuples to prune
                                 If None, all parameters will be pruned
        """
        self.pruning_type = pruning_type.lower()
        self.criteria = criteria.lower()
        self.parameters_to_prune = parameters_to_prune
        
        # Validate pruning type
        valid_types = ["unstructured", "structured", "global"]
        if self.pruning_type not in valid_types:
            raise ValueError(f"Pruning type must be one of {valid_types}, got {self.pruning_type}")
        
        # Validate criteria
        valid_criteria = ["random", "l1", "l2"]
        if self.criteria not in valid_criteria:
            raise ValueError(f"Pruning criteria must be one of {valid_criteria}, got {self.criteria}")
        
        logger.info(f"Initialized model pruning: type={pruning_type}, criteria={criteria}")
    
    def _get_pruning_method(self) -> Callable:
        """
        Get the appropriate pruning method based on type and criteria.
        
        Returns:
            Callable: Pruning method
        """
        if self.pruning_type == "unstructured":
            if self.criteria == "random":
                return prune.random_unstructured
            elif self.criteria == "l1":
                return prune.l1_unstructured
            elif self.criteria == "l2":
                return prune.ln_structured
        elif self.pruning_type == "structured":
            if self.criteria == "random":
                return prune.random_structured
            elif self.criteria == "l1":
                return prune.ln_structured
            elif self.criteria == "l2":
                return prune.ln_structured
        elif self.pruning_type == "global":
            if self.criteria == "random":
                return prune.global_unstructured
            elif self.criteria == "l1":
                return lambda m, n, a: prune.global_unstructured(
                    m, n, pruning_method=prune.L1Unstructured, amount=a
                )
            elif self.criteria == "l2":
                return lambda m, n, a: prune.global_unstructured(
                    m, n, pruning_method=prune.LnStructured, amount=a, n=2
                )
        
        # Default to L1 unstructured
        return prune.l1_unstructured
    
    def prune_model(
        self,
        model: nn.Module,
        amount: float = 0.2,
        inplace: bool = False,
    ) -> nn.Module:
        """
        Prune a model based on the specified pruning type.
        
        Args:
            model: PyTorch model
            amount: Amount to prune (fraction between 0 and 1)
            inplace: Whether to modify the model in-place
            
        Returns:
            nn.Module: Pruned model
        """
        if not inplace:
            model = copy.deepcopy(model)
        
        # Get all parameters to prune if not specified
        if self.parameters_to_prune is None:
            parameters_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    parameters_to_prune.append((module, "weight"))
        else:
            parameters_to_prune = self.parameters_to_prune
        
        # Get pruning method
        pruning_method = self._get_pruning_method()
        
        # Apply pruning based on type
        if self.pruning_type == "global":
            # Global pruning applies to all parameters at once
            if isinstance(pruning_method, Callable):
                pruning_method(
                    parameters=parameters_to_prune,
                    amount=amount,
                )
            else:
                # Use prune.global_unstructured for other criteria
                import torch.nn.utils.prune as prune
                prune.global_unstructured(
                    parameters=parameters_to_prune,
                    pruning_method=getattr(prune, f"{self.criteria.capitalize()}Unstructured"),
                    amount=amount,
                )
        else:
            # Apply pruning to each parameter individually
            for module, param_name in parameters_to_prune:
                if param_name == "weight" and hasattr(module, param_name):
                    # Make sure we have the original parameter before pruning
                    if not hasattr(module, f"{param_name}_orig"):
                        # Create a forward pre-hook to ensure the mask exists
                        import torch.nn.utils.prune as prune
                        prune.identity(module, param_name)
                    
                    # Now apply the actual pruning
                    pruning_method(module, name=param_name, amount=amount)
        
        logger.info(f"Applied {self.pruning_type} pruning with {amount:.2%} sparsity")
        return model
    
    def iterative_pruning(
        self,
        model: nn.Module,
        train_fn: Callable[[nn.Module], nn.Module],
        initial_amount: float = 0.2,
        final_amount: float = 0.8,
        steps: int = 5,
        inplace: bool = False,
    ) -> nn.Module:
        """
        Apply iterative pruning to a model with fine-tuning between steps.
        
        Args:
            model: PyTorch model
            train_fn: Function that fine-tunes the model after pruning
            initial_amount: Initial pruning amount
            final_amount: Final pruning amount
            steps: Number of pruning steps
            inplace: Whether to modify the model in-place
            
        Returns:
            nn.Module: Pruned model
        """
        if not inplace:
            model = copy.deepcopy(model)
        
        # Calculate pruning schedule
        if steps <= 1:
            pruning_schedule = [final_amount]
        else:
            # Use a geometric schedule for pruning
            ratio = (final_amount / initial_amount) ** (1.0 / (steps - 1))
            pruning_schedule = [initial_amount * (ratio ** i) for i in range(steps)]
        
        logger.info(f"Starting iterative pruning with {steps} steps: {pruning_schedule}")
        
        # Get all parameters to prune if not specified
        if self.parameters_to_prune is None:
            parameters_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    parameters_to_prune.append((module, "weight"))
        else:
            parameters_to_prune = self.parameters_to_prune
        
        # Apply pruning iteratively
        for i, amount in enumerate(pruning_schedule):
            logger.info(f"Pruning step {i+1}/{len(pruning_schedule)}: {amount:.2%} sparsity")
            
            # Remove existing pruning reparameterizations
            for module, param_name in parameters_to_prune:
                if hasattr(module, f"{param_name}_mask") and hasattr(module, f"orig_{param_name}"):
                    prune.remove(module, param_name)
            
            # Apply pruning
            self.prune_model(model, amount=amount, inplace=True)
            
            # Fine-tune the model
            model = train_fn(model)
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            if hasattr(module, f"{param_name}_mask") and hasattr(module, f"orig_{param_name}"):
                prune.remove(module, param_name)
        
        logger.info(f"Completed iterative pruning with final sparsity {pruning_schedule[-1]:.2%}")
        return model
    
    def get_model_sparsity(self, model: nn.Module) -> Dict[str, float]:
        """
        Calculate the sparsity of a model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dict[str, float]: Sparsity metrics
        """
        total_params = 0
        zero_params = 0
        layer_sparsity = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight
                
                # Count parameters
                layer_total = weight.numel()
                layer_zero = (weight == 0).sum().item()
                
                # Update counts
                total_params += layer_total
                zero_params += layer_zero
                
                # Calculate layer sparsity
                if layer_total > 0:
                    layer_sparsity[name] = layer_zero / layer_total
        
        # Calculate overall sparsity
        overall_sparsity = zero_params / total_params if total_params > 0 else 0.0
        
        return {
            "overall_sparsity": overall_sparsity,
            "layer_sparsity": layer_sparsity,
            "total_params": total_params,
            "zero_params": zero_params,
        }
    
    def evaluate_pruning(
        self,
        original_model: nn.Module,
        pruned_model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cpu"),
        criterion: Optional[Callable] = None,
        num_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the performance impact of pruning.
        
        Args:
            original_model: Original PyTorch model
            pruned_model: Pruned PyTorch model
            dataloader: DataLoader with evaluation data
            device: Device to run evaluation on
            criterion: Loss function
            num_batches: Number of batches to use for evaluation
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        # Ensure models are in eval mode
        original_model.eval()
        pruned_model.eval()
        
        # Move models to device
        original_model = original_model.to(device)
        pruned_model = pruned_model.to(device)
        
        # Get model sparsity
        sparsity_metrics = self.get_model_sparsity(pruned_model)
        
        results = {
            "original_size_mb": self._get_model_size(original_model),
            "pruned_size_mb": self._get_model_size(pruned_model),
            "original_latency_ms": 0.0,
            "pruned_latency_ms": 0.0,
            "original_accuracy": 0.0,
            "pruned_accuracy": 0.0,
            "speedup": 0.0,
            "memory_reduction": 0.0,
            "sparsity": sparsity_metrics["overall_sparsity"],
        }
        
        # Evaluate original model
        original_latency, original_accuracy = self._evaluate_model(
            original_model,
            dataloader,
            device,
            criterion,
            num_batches
        )
        
        # Evaluate pruned model
        pruned_latency, pruned_accuracy = self._evaluate_model(
            pruned_model,
            dataloader,
            device,
            criterion,
            num_batches
        )
        
        # Update results
        results["original_latency_ms"] = original_latency
        results["pruned_latency_ms"] = pruned_latency
        results["original_accuracy"] = original_accuracy
        results["pruned_accuracy"] = pruned_accuracy
        
        # Calculate improvements
        if original_latency > 0:
            results["speedup"] = original_latency / pruned_latency
        
        if results["original_size_mb"] > 0:
            results["memory_reduction"] = 1.0 - (results["pruned_size_mb"] / results["original_size_mb"])
        
        return results
    
    def _get_model_size(self, model: nn.Module) -> float:
        """
        Get the size of a model in megabytes.
        
        Args:
            model: PyTorch model
            
        Returns:
            float: Model size in MB
        """
        # Get size using PyTorch state_dict
        state_dict = model.state_dict()
        size_bytes = 0
        
        for name, param in state_dict.items():
            # Skip if param is not a tensor or doesn't have numel() method
            if not hasattr(param, 'numel') or not hasattr(param, 'element_size'):
                continue
            size_bytes += param.numel() * param.element_size()
        
        return size_bytes / (1024 ** 2)  # Convert to MB
    
    def _evaluate_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: Optional[Callable] = None,
        num_batches: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Evaluate a model's latency and accuracy.
        
        Args:
            model: PyTorch model
            dataloader: DataLoader with evaluation data
            device: Device to run evaluation on
            criterion: Loss function
            num_batches: Number of batches to use for evaluation
            
        Returns:
            Tuple[float, float]: Latency (ms) and accuracy
        """
        # Move model to device
        model = model.to(device)
        
        total_latency = 0.0
        total_samples = 0
        correct = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(dataloader):
                if num_batches is not None and i >= num_batches:
                    break
                
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Measure latency
                start_time = time.time()
                outputs = model(inputs)
                end_time = time.time()
                
                batch_latency = (end_time - start_time) * 1000  # Convert to ms
                total_latency += batch_latency
                
                # Compute loss if criterion is provided
                if criterion is not None:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)
                
                # Compute accuracy
                if isinstance(outputs, torch.Tensor) and outputs.dim() > 1 and outputs.size(1) > 1:
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                
                total_samples += inputs.size(0)
        
        # Calculate averages
        avg_latency = total_latency / (i + 1)  # Average latency per batch
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        
        return avg_latency, accuracy


def optimize_for_inference(
    model: nn.Module,
    optimizations: List[str] = ["trace", "fuse"],
    example_inputs: Optional[torch.Tensor] = None,
    inplace: bool = False,
) -> nn.Module:
    """
    Apply general optimizations to make a model more efficient for inference.
    
    Args:
        model: PyTorch model
        optimizations: List of optimizations to apply
        example_inputs: Example inputs for tracing
        inplace: Whether to modify the model in-place
        
    Returns:
        nn.Module: Optimized model
    """
    if not inplace:
        model = copy.deepcopy(model)
    
    # Set model to eval mode
    model.eval()
    
    # Apply optimizations
    if "trace" in optimizations and example_inputs is not None:
        # Trace the model with JIT
        with torch.no_grad():
            trace_model = torch.jit.trace(model, example_inputs)
            # Convert back to regular PyTorch module if possible, or keep as ScriptModule
            if hasattr(trace_model, "module"):
                model = trace_model.module
            else:
                model = trace_model
        logger.info("Applied JIT tracing optimization")
    
    if "fuse" in optimizations:
        # Fuse batch normalization layers with preceding convolutions
        for module in model.modules():
            if hasattr(module, "fuse_model"):
                module.fuse_model()
        logger.info("Applied layer fusion optimization")
    
    if "freeze" in optimizations:
        # Freeze weights to avoid unnecessary gradient computations
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Froze model parameters")
    
    return model


def apply_inference_optimizations(
    model: nn.Module,
    optimizations: List[str],
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    device: torch.device = torch.device("cpu"),
    example_inputs: Optional[torch.Tensor] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Apply selected inference optimizations to a model.
    
    Args:
        model: PyTorch model
        optimizations: List of optimizations to apply
        dataloader: DataLoader for calibration/evaluation
        device: Device to run optimizations on
        example_inputs: Example inputs for tracing/scripting
        config: Configuration for optimizations
        
    Returns:
        Tuple[nn.Module, Dict[str, Any]]: Optimized model and results
    """
    # Initialize results
    results = {}
    original_model = copy.deepcopy(model)
    
    # Set default config if not provided
    if config is None:
        config = {}
    
    # Create a copy of the model to optimize
    optimized_model = copy.deepcopy(model)
    
    # Apply quantization if requested
    if "quantization" in optimizations or "dynamic_quantization" in optimizations:
        # Get quantization config
        quant_config = config.get("quantization", {})
        quant_type = quant_config.get("type", "dynamic")
        dtype = quant_config.get("dtype", torch.qint8)
        
        # Create quantizer
        quantizer = ModelQuantization(
            quantization_type=quant_type,
            dtype=dtype
        )
        
        # Quantize the model
        optimized_model = quantizer.quantize_model(
            optimized_model,
            dataloader=dataloader,
            inplace=True
        )
        
        # Evaluate quantization impact
        if dataloader is not None:
            quant_results = quantizer.evaluate_quantization(
                original_model=original_model,
                quantized_model=optimized_model,
                dataloader=dataloader,
                device=device
            )
            results["quantization"] = quant_results
    
    # Apply pruning if requested
    if "pruning" in optimizations:
        # Get pruning config
        prune_config = config.get("pruning", {})
        prune_type = prune_config.get("type", "unstructured")
        criteria = prune_config.get("criteria", "l1")
        amount = prune_config.get("amount", 0.2)
        
        # Create pruner
        pruner = ModelPruning(
            pruning_type=prune_type,
            criteria=criteria
        )
        
        # Create a fresh copy before pruning
        optimized_model = copy.deepcopy(model)
        
        # Prune the model
        optimized_model = pruner.prune_model(
            optimized_model,
            amount=amount,
            inplace=True
        )
        
        # Make pruning permanent immediately to avoid issues with hooks
        for module in optimized_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight_mask') and hasattr(module, 'weight_orig'):
                    # Apply the mask and remove the pruning reparameterization
                    prune.remove(module, 'weight')
        
        # Evaluate pruning impact
        if dataloader is not None:
            try:
                prune_results = pruner.evaluate_pruning(
                    original_model=original_model,
                    pruned_model=optimized_model,
                    dataloader=dataloader,
                    device=device
                )
                results["pruning"] = prune_results
            except Exception as e:
                logger.warning(f"Error evaluating pruning impact: {e}")
                # Provide basic pruning results without evaluation
                results["pruning"] = {
                    "sparsity": pruner.get_model_sparsity(optimized_model)["overall_sparsity"]
                }
    
    # Apply general inference optimizations
    if "optimize" in optimizations or "trace" in optimizations or "fuse" in optimizations:
        # Get general optimization config
        general_optimizations = []
        if "trace" in optimizations:
            general_optimizations.append("trace")
        if "fuse" in optimizations:
            general_optimizations.append("fuse")
        if "freeze" in optimizations:
            general_optimizations.append("freeze")
        
        # Apply optimizations
        if example_inputs is not None or "trace" not in general_optimizations:
            optimized_model = optimize_for_inference(
                optimized_model,
                optimizations=general_optimizations,
                example_inputs=example_inputs,
                inplace=True
            )
    
    # Return the optimized model and results
    return optimized_model, results