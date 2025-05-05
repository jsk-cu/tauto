"""
Tests for the inference optimization utilities.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import os
from pathlib import Path
import tempfile
import shutil

from tauto.optimize.inference import (
    ModelQuantization,
    ModelPruning,
    optimize_for_inference,
    apply_inference_optimizations,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ConvModel(nn.Module):
    """Simple CNN model for testing with BatchNorm."""
    
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 16 * 16, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def model():
    """Provide a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def conv_model():
    """Provide a conv model for testing."""
    return ConvModel()


@pytest.fixture
def dataloader():
    """Provide a dataloader for testing."""
    # Create random data
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=16)


@pytest.fixture
def conv_dataloader():
    """Provide a dataloader for conv model testing."""
    # Create random image data
    x = torch.randn(100, 3, 32, 32)
    y = torch.randint(0, 10, (100,))
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=16)


def test_model_quantization_init():
    """Test initializing model quantization."""
    # Test with default parameters
    quantizer = ModelQuantization()
    assert quantizer.quantization_type == "dynamic"
    assert quantizer.dtype == torch.qint8
    
    # Test with custom parameters
    quantizer = ModelQuantization(
        quantization_type="static",
        dtype=torch.quint8,
    )
    assert quantizer.quantization_type == "static"
    assert quantizer.dtype == torch.quint8
    
    # Test with invalid quantization type
    with pytest.raises(ValueError):
        ModelQuantization(quantization_type="invalid")


def test_model_quantization_dynamic(model, dataloader):
    """Test dynamic quantization."""
    # Skip if torch version doesn't support quantization
    try:
        import torch.quantization
    except ImportError:
        pytest.skip("PyTorch version doesn't support quantization")
    
    quantizer = ModelQuantization(quantization_type="dynamic")
    
    # Test preparing model (no-op for dynamic quantization)
    prepared_model = quantizer.prepare_model(model)
    
    # Test quantizing model
    quantized_model = quantizer.quantize_model(prepared_model)
    
    # Check the model is actually quantized
    assert hasattr(quantized_model, "is_quantized") and quantized_model.is_quantized


def test_model_quantization_evaluate(model, dataloader):
    """Test evaluating quantization impact."""
    # Skip if torch version doesn't support quantization
    try:
        import torch.quantization
    except ImportError:
        pytest.skip("PyTorch version doesn't support quantization")
    
    quantizer = ModelQuantization(quantization_type="dynamic")
    
    # Quantize the model
    quantized_model = quantizer.quantize_model(model)
    
    # Evaluate the impact
    results = quantizer.evaluate_quantization(
        original_model=model,
        quantized_model=quantized_model,
        dataloader=dataloader,
        device=torch.device("cpu"),
    )
    
    # Check the results
    assert "original_size_mb" in results
    assert "quantized_size_mb" in results
    assert "original_latency_ms" in results
    assert "quantized_latency_ms" in results
    assert "speedup" in results
    assert "memory_reduction" in results


def test_model_pruning_init():
    """Test initializing model pruning."""
    # Test with default parameters
    pruner = ModelPruning()
    assert pruner.pruning_type == "unstructured"
    assert pruner.criteria == "l1"
    
    # Test with custom parameters
    pruner = ModelPruning(
        pruning_type="structured",
        criteria="l2",
    )
    assert pruner.pruning_type == "structured"
    assert pruner.criteria == "l2"
    
    # Test with invalid pruning type
    with pytest.raises(ValueError):
        ModelPruning(pruning_type="invalid")
    
    # Test with invalid criteria
    with pytest.raises(ValueError):
        ModelPruning(criteria="invalid")


def test_model_pruning_unstructured(model):
    """Test unstructured pruning."""
    # Skip if torch version doesn't support pruning
    try:
        import torch.nn.utils.prune as prune
    except ImportError:
        pytest.skip("PyTorch version doesn't support pruning")
    
    pruner = ModelPruning(pruning_type="unstructured")
    
    # Test pruning the model
    pruned_model = pruner.prune_model(model, amount=0.3)
    
    # Check sparsity
    sparsity_info = pruner.get_model_sparsity(pruned_model)
    
    # Check the results
    assert "overall_sparsity" in sparsity_info
    assert "layer_sparsity" in sparsity_info
    assert "total_params" in sparsity_info
    assert "zero_params" in sparsity_info
    
    # Sparsity should be close to 0.3 (not exactly due to rounding)
    assert abs(sparsity_info["overall_sparsity"] - 0.3) < 0.05


def test_model_pruning_evaluate(model, dataloader):
    """Test evaluating pruning impact."""
    # Skip if torch version doesn't support pruning
    try:
        import torch.nn.utils.prune as prune
    except ImportError:
        pytest.skip("PyTorch version doesn't support pruning")
    
    pruner = ModelPruning(pruning_type="unstructured")
    
    # Prune the model
    pruned_model = pruner.prune_model(model, amount=0.3)
    
    # Evaluate the impact
    results = pruner.evaluate_pruning(
        original_model=model,
        pruned_model=pruned_model,
        dataloader=dataloader,
        device=torch.device("cpu"),
    )
    
    # Check the results
    assert "original_size_mb" in results
    assert "pruned_size_mb" in results
    assert "original_latency_ms" in results
    assert "pruned_latency_ms" in results
    assert "speedup" in results
    assert "memory_reduction" in results
    assert "sparsity" in results
    
    # Sparsity should be close to 0.3
    assert abs(results["sparsity"] - 0.3) < 0.05


def test_optimize_for_inference(conv_model, conv_dataloader):
    """Test optimize_for_inference function."""
    # Skip if using CPU-only PyTorch that doesn't support fusing
    try:
        import torch.nn.utils.fusion
    except ImportError:
        try:
            # Check if we can fuse models another way
            hasattr(conv_model, "fuse_model")
        except:
            pytest.skip("PyTorch version doesn't support fusion")
    
    # Get example inputs
    example_inputs = next(iter(conv_dataloader))[0]
    
    # Test optimize_for_inference with tracing
    optimized_model = optimize_for_inference(
        model=conv_model,
        optimizations=["trace"],
        example_inputs=example_inputs,
    )
    
    # Check that the optimized model can be used for inference
    with torch.no_grad():
        output = optimized_model(example_inputs)
        assert output.shape == (example_inputs.shape[0], 10)


def test_apply_inference_optimizations(model, dataloader):
    """Test apply_inference_optimizations function."""
    # Skip if torch version doesn't support quantization
    try:
        import torch.quantization
    except ImportError:
        pytest.skip("PyTorch version doesn't support quantization")
    
    # Test applying quantization
    optimized_model, results = apply_inference_optimizations(
        model=model,
        optimizations=["quantization"],
        dataloader=dataloader,
        device=torch.device("cpu"),
    )
    
    # Check that results contains quantization info
    assert "quantization" in results
    
    # Check that the optimized model can be used for inference
    with torch.no_grad():
        example_inputs = next(iter(dataloader))[0]
        output = optimized_model(example_inputs)
        assert output.shape == (example_inputs.shape[0], 2)


def test_apply_inference_optimizations_pruning(model, dataloader):
    """Test apply_inference_optimizations with pruning."""
    # Skip if torch version doesn't support pruning
    try:
        import torch.nn.utils.prune as prune
    except ImportError:
        pytest.skip("PyTorch version doesn't support pruning")
    
    # Test applying pruning
    config = {
        "pruning": {
            "amount": 0.3,
            "type": "unstructured",
            "criteria": "l1",
        }
    }
    
    optimized_model, results = apply_inference_optimizations(
        model=model,
        optimizations=["pruning"],
        dataloader=dataloader,
        device=torch.device("cpu"),
        config=config,
    )
    
    # Check that results contains pruning info
    assert "pruning" in results
    
    # Check that the pruned model can be used for inference
    with torch.no_grad():
        example_inputs = next(iter(dataloader))[0]
        output = optimized_model(example_inputs)
        assert output.shape == (example_inputs.shape[0], 2)


def test_apply_inference_optimizations_multiple(model, dataloader):
    """Test apply_inference_optimizations with multiple optimizations."""
    # Skip if torch version doesn't support quantization or pruning
    try:
        import torch.quantization
        import torch.nn.utils.prune as prune
    except ImportError:
        pytest.skip("PyTorch version doesn't support quantization or pruning")
    
    # Test applying multiple optimizations
    config = {
        "pruning": {
            "amount": 0.3,
        },
        "quantization": {
            "type": "dynamic",
        }
    }
    
    optimized_model, results = apply_inference_optimizations(
        model=model,
        optimizations=["pruning", "quantization"],
        dataloader=dataloader,
        device=torch.device("cpu"),
        config=config,
    )
    
    # Check that results contains both pruning and quantization info
    assert "pruning" in results
    assert "quantization" in results
    
    # Check that the optimized model can be used for inference
    with torch.no_grad():
        example_inputs = next(iter(dataloader))[0]
        output = optimized_model(example_inputs)
        assert output.shape == (example_inputs.shape[0], 2)