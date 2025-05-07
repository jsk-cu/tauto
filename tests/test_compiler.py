"""
Tests for the compiler optimization utilities.
"""

import pytest
import torch
import torch.nn as nn
import os
from pathlib import Path
import tempfile
import shutil

from tauto.optimize.compiler import (
    TorchCompile,
    TorchScriptExport,
    apply_compiler_optimization,
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
    """Simple CNN model for testing."""
    
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 8 * 8, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x


@pytest.fixture
def model():
    """Provide a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def conv_model():
    """Provide a convolutional model for testing."""
    return ConvModel()


@pytest.fixture
def inputs():
    """Provide inputs for the simple model."""
    return torch.randn(16, 10)


@pytest.fixture
def conv_inputs():
    """Provide inputs for the conv model."""
    return torch.randn(16, 3, 32, 32)


def test_torch_compile_init():
    """Test initializing TorchCompile."""
    # Test with default parameters
    compiler = TorchCompile()
    assert compiler.backend == "inductor"
    assert compiler.mode == "default"
    assert compiler.fullgraph is True
    
    # Test with custom parameters
    compiler = TorchCompile(
        backend="aot_eager",
        mode="reduce-overhead",
        fullgraph=False,
    )
    assert compiler.backend == "aot_eager"
    assert compiler.mode == "reduce-overhead"
    assert compiler.fullgraph is False
    
    # Test disabled
    compiler = TorchCompile(disabled=True)
    assert compiler.disabled is True


def test_torch_compile_model(model, inputs):
    """Test compiling a model with torch.compile."""
    # Skip if torch.compile is not available
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available in this PyTorch version")
    
    compiler = TorchCompile()
    
    # Test compiling the model
    compiled_model = compiler.compile_model(model, example_inputs=inputs)
    
    # Check that the model can still be used for inference
    with torch.no_grad():
        output = model(inputs)
        compiled_output = compiled_model(inputs)
    
    # Check that outputs are similar
    assert output.shape == compiled_output.shape


def test_torch_compile_benchmark(model, inputs):
    """Test benchmarking compilation."""
    # Skip if torch.compile is not available
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available in this PyTorch version")
    
    compiler = TorchCompile()
    
    # Test benchmarking with minimal runs to keep test fast
    results = compiler.benchmark_compilation(
        model=model,
        inputs=inputs,
        num_warmup=1,
        num_runs=3,
    )
    
    # Check that results contain expected keys
    assert "original_avg_ms" in results
    assert "compiled_avg_ms" in results
    assert "speedup" in results


def test_torch_compile_debug(model, inputs):
    """Test debugging compilation issues."""
    # Skip if torch.compile is not available
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available in this PyTorch version")
    
    # Test debugging
    debug_info = TorchCompile.debug_model(
        model=model,
        inputs=inputs,
        backend="inductor",
        verbose=False,
    )
    
    # Check that debug info contains expected keys
    assert "success" in debug_info
    assert "backend" in debug_info
    assert "torch_version" in debug_info


def test_torchscript_export_init():
    """Test initializing TorchScriptExport."""
    # Test with default parameters
    exporter = TorchScriptExport()
    assert exporter.optimization_level == 3
    assert exporter.strict is False
    assert exporter.method == "trace"
    
    # Test with custom parameters
    exporter = TorchScriptExport(
        optimization_level=1,
        strict=True,
        method="script",
    )
    assert exporter.optimization_level == 1
    assert exporter.strict is True
    assert exporter.method == "script"


def test_torchscript_export_model(model, inputs):
    """Test exporting a model to TorchScript."""
    exporter = TorchScriptExport()
    
    # Test exporting the model
    ts_model = exporter.export_model(model, example_inputs=inputs)
    
    # Check that the model can still be used for inference
    with torch.no_grad():
        output = model(inputs)
        ts_output = ts_model(inputs)
    
    # Check that outputs are similar
    assert output.shape == ts_output.shape


def test_torchscript_save_load(model, inputs, tmp_path):
    """Test saving and loading a TorchScript model."""
    exporter = TorchScriptExport()
    
    # Save path
    save_path = tmp_path / "model.pt"
    
    # Export and save the model
    ts_model = exporter.export_model(model, example_inputs=inputs, save_path=save_path)
    
    # Check that the file was created
    assert os.path.exists(save_path)
    
    # Load the model
    loaded_model = torch.jit.load(save_path)
    
    # Check that the loaded model can be used for inference
    with torch.no_grad():
        output = model(inputs)
        loaded_output = loaded_model(inputs)
    
    # Check that outputs are similar
    assert output.shape == loaded_output.shape


def test_torchscript_benchmark(model, inputs):
    """Test benchmarking TorchScript export."""
    exporter = TorchScriptExport()
    
    # Test benchmarking with minimal runs to keep test fast
    results = exporter.benchmark_export(
        model=model,
        inputs=inputs,
        num_warmup=1,
        num_runs=3,
    )
    
    # Check that results contain expected keys
    assert "original_avg_ms" in results
    assert "torchscript_avg_ms" in results
    assert "speedup" in results


def test_apply_compiler_optimization(model, inputs, tmp_path):
    """Test the apply_compiler_optimization utility function."""
    # Skip if torch.compile is not available
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available in this PyTorch version")
    
    # Test torch.compile optimization
    compiled_model = apply_compiler_optimization(
        model=model,
        optimization="torch_compile",
        example_inputs=inputs,
    )
    
    # Check that the model can still be used for inference
    with torch.no_grad():
        output = model(inputs)
        compiled_output = compiled_model(inputs)
    
    # Check that outputs are similar
    assert output.shape == compiled_output.shape
    
    # Test TorchScript optimization
    save_path = tmp_path / "model_ts.pt"
    ts_model = apply_compiler_optimization(
        model=model,
        optimization="torchscript",
        example_inputs=inputs,
        save_path=save_path,
    )
    
    # Check that the file was created
    assert os.path.exists(save_path)
    
    # Check that the model can still be used for inference
    with torch.no_grad():
        output = model(inputs)
        ts_output = ts_model(inputs)
    
    # Check that outputs are similar
    assert output.shape == ts_output.shape