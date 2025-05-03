"""
Tests for the profiler module.
"""

import pytest
import torch
import torch.nn as nn
import os
from pathlib import Path
import time
import tempfile

from tauto.profile.profiler import (
    Profiler,
    ProfilerConfig,
    ProfileResult,
    profile_model,
)


class SimpleModel(nn.Module):
    """Simple model for testing the profiler."""
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1):  # Changed output_dim to 1
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Now outputs a single value
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze(-1)  # Squeeze to match target shape


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for testing the profiler."""
    
    def __init__(self, size=100, dim=10):
        self.data = torch.randn(size, dim)
        self.targets = torch.randint(0, 2, (size,)).float()  # Use float for MSE loss
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]


@pytest.fixture
def model():
    """Provide a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def dataloader():
    """Provide a dataloader for testing."""
    dataset = SimpleDataset()
    return torch.utils.data.DataLoader(dataset, batch_size=16)


@pytest.fixture
def profile_dir(temp_dir):
    """Provide a directory for profile outputs."""
    profile_dir = temp_dir / "profile"
    os.makedirs(profile_dir, exist_ok=True)
    return profile_dir


def test_profiler_config():
    """Test creating and accessing ProfilerConfig."""
    # Default config
    config = ProfilerConfig()
    assert config.enabled is True
    assert config.use_cuda is True
    assert config.profile_memory is True
    
    # Custom config
    config = ProfilerConfig(
        enabled=False,
        use_cuda=False,
        profile_memory=False,
        profile_dir="custom_dir",
    )
    
    assert config.enabled is False
    assert config.use_cuda is False
    assert config.profile_memory is False
    assert config.profile_dir == "custom_dir"
    
    # Post-init directory creation
    assert os.path.exists(config.profile_dir) is False
    
    # Directory should be created when enabled
    config = ProfilerConfig(enabled=True, profile_dir="test_profile_dir")
    assert os.path.exists("test_profile_dir")
    
    # Clean up
    os.rmdir("test_profile_dir")


def test_profile_result():
    """Test creating and manipulating ProfileResult."""
    result = ProfileResult(
        name="test_profile",
        device="cpu",
    )
    
    # Add some data
    result.duration_ms["total"] = 100.0
    result.memory_usage["cpu_total"] = 50.0
    
    # Test to_dict
    result_dict = result.to_dict()
    assert result_dict["name"] == "test_profile"
    assert result_dict["device"] == "cpu"
    assert result_dict["duration_ms"]["total"] == 100.0
    assert result_dict["memory_usage"]["cpu_total"] == 50.0
    
    # Test save and load
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "result.json"
        result.save(save_path)
        
        # Check that the file exists
        assert os.path.exists(save_path)
        
        # Load the result
        loaded_result = ProfileResult.load(save_path)
        
        # Check that it matches the original
        assert loaded_result.name == result.name
        assert loaded_result.device == result.device
        assert loaded_result.duration_ms == result.duration_ms
        assert loaded_result.memory_usage == result.memory_usage


def test_profiler_initialization(model):
    """Test initializing the profiler."""
    # Test with default config
    profiler = Profiler(model)
    assert profiler.model is model
    assert profiler.config.enabled is True
    
    # Test with custom config dict
    config_dict = {
        "enabled": False,
        "use_cuda": False,
        "profile_memory": False,
    }
    
    profiler = Profiler(model, config_dict)
    assert profiler.config.enabled is False
    assert profiler.config.use_cuda is False
    assert profiler.config.profile_memory is False
    
    # Test with ProfilerConfig
    config = ProfilerConfig(enabled=False, use_cuda=False)
    profiler = Profiler(model, config)
    assert profiler.config.enabled is False
    assert profiler.config.use_cuda is False


def test_profiler_training(model, dataloader, profile_dir):
    """Test profiling training."""
    # Skip test if CUDA is not available and requested
    if torch.cuda.is_available() is False:
        device = "cpu"
        use_cuda = False
    else:
        device = "cuda"
        use_cuda = True
    
    config = ProfilerConfig(
        enabled=True,
        use_cuda=use_cuda,
        profile_dir=str(profile_dir),
        num_warmup_steps=1,
        num_active_steps=2,
        num_repeat=1,
    )
    
    profiler = Profiler(model, config)
    
    # Profile training
    result = profiler.profile_training(
        dataloader=dataloader,
        num_steps=3,
        name="test_training",
    )
    
    # Check basic result properties
    assert result.name == "test_training"
    assert result.device == device
    assert "total" in result.duration_ms
    assert result.duration_ms["total"] > 0
    
    # Check that trace files were created
    if result.traces:
        for trace_path in result.traces:
            assert os.path.exists(trace_path)


def test_profiler_inference(model, dataloader, profile_dir):
    """Test profiling inference."""
    # Skip test if CUDA is not available and requested
    if torch.cuda.is_available() is False:
        device = "cpu"
        use_cuda = False
    else:
        device = "cuda"
        use_cuda = True
    
    config = ProfilerConfig(
        enabled=True,
        use_cuda=use_cuda,
        profile_dir=str(profile_dir),
        num_warmup_steps=1,
        num_active_steps=2,
        num_repeat=1,
    )
    
    profiler = Profiler(model, config)
    
    # Profile inference
    result = profiler.profile_inference(
        dataloader=dataloader,
        num_steps=3,
        with_grad=False,
        name="test_inference",
    )
    
    # Check basic result properties
    assert result.name == "test_inference"
    assert result.device == device
    assert "total" in result.duration_ms
    assert result.duration_ms["total"] > 0
    assert "per_batch" in result.duration_ms
    
    # Check parameters
    assert result.parameters is not None
    assert "total" in result.parameters
    assert result.parameters["total"] > 0
    
    # Check that trace files were created
    if result.traces:
        for trace_path in result.traces:
            assert os.path.exists(trace_path)


def test_profiler_memory_usage(model, dataloader, profile_dir):
    """Test profiling memory usage."""
    config = ProfilerConfig(
        enabled=True,
        use_cuda=False,  # Use CPU for consistent testing
        profile_dir=str(profile_dir),
    )
    
    profiler = Profiler(model, config)
    
    # Profile memory usage
    result = profiler.profile_memory_usage(
        dataloader=dataloader,
        name="test_memory",
    )
    
    # Check basic result properties
    assert result.name == "test_memory"
    assert result.device == "cpu"
    
    # Check memory metrics
    assert len(result.memory_usage) > 0
    
    # Check parameters
    assert result.parameters is not None
    assert "total" in result.parameters
    assert "size_mb" in result.parameters


def test_profile_model_function(model, dataloader, profile_dir):
    """Test the profile_model utility function."""
    config = ProfilerConfig(
        enabled=True,
        use_cuda=False,  # Use CPU for consistent testing
        profile_dir=str(profile_dir),
        num_warmup_steps=1,
        num_active_steps=2,
        num_repeat=1,
    )
    
    # Profile in all modes
    results = profile_model(
        model=model,
        dataloader=dataloader,
        config=config,
        mode="all",
        name="test_all",
    )
    
    # Check that all modes were profiled
    assert "training" in results
    assert "inference" in results
    assert "memory" in results
    
    # Check that each result has the correct name
    assert results["training"].name == "test_all_training"
    assert results["inference"].name == "test_all_inference"
    assert results["memory"].name == "test_all_memory"
    
    # Test with specific mode
    results = profile_model(
        model=model,
        dataloader=dataloader,
        config=config,
        mode="inference",
        name="test_inference_only",
    )
    
    # Check that only inference was profiled
    assert "inference" in results
    assert "training" not in results
    assert "memory" not in results
    
    # Check result name
    assert results["inference"].name == "test_inference_only_inference"