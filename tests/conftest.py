"""
Pytest configuration and fixtures for TAuto tests.
"""

import os
import sys
import tempfile
from pathlib import Path
import pytest
import yaml
import torch
import numpy as np

# Add parent directory to path to import tauto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tauto
from tauto.config import get_default_config


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def default_config():
    """Provide the default configuration."""
    return get_default_config()


@pytest.fixture
def config_file(temp_dir, default_config):
    """Provide a temporary configuration file."""
    config_path = temp_dir / "test_config.yaml"
    
    with open(config_path, "w") as f:
        yaml.dump(default_config, f)
    
    return config_path


@pytest.fixture
def small_tensor():
    """Provide a small tensor for testing."""
    return torch.randn(10, 10)


@pytest.fixture
def medium_tensor():
    """Provide a medium-sized tensor for testing."""
    return torch.randn(100, 100)


@pytest.fixture
def small_dataset():
    """Provide a small dataset for testing."""
    class RandomDataset(torch.utils.data.Dataset):
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


@pytest.fixture
def small_model():
    """Provide a small model for testing."""
    class SmallModel(torch.nn.Module):
        def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    return SmallModel()