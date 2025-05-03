"""
Tests for the model registry module.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from tauto.models.registry import (
    ModelRegistry,
    ModelInfo,
    register_model,
    create_model,
    list_available_models,
    get_model_info,
)


class SimpleModel(nn.Module):
    """Simple model for testing the registry."""
    
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


def test_model_info():
    """Test creating and accessing ModelInfo."""
    info = ModelInfo(
        name="test_model",
        architecture="MLP",
        description="A test model",
        task="classification",
        input_size=10,
        output_size=2,
        model_cls=SimpleModel,
    )
    
    assert info.name == "test_model"
    assert info.architecture == "MLP"
    assert info.description == "A test model"
    assert info.task == "classification"
    assert info.input_size == 10
    assert info.output_size == 2
    assert info.model_cls == SimpleModel


def test_model_registry():
    """Test the ModelRegistry class."""
    registry = ModelRegistry()
    
    # Register a model
    info = ModelInfo(
        name="test_model",
        architecture="MLP",
        description="A test model",
        task="classification",
        input_size=10,
        output_size=2,
        model_cls=SimpleModel,
    )
    
    registry.register(info)
    
    # Check that the model is registered
    assert "test_model" in registry.list_available()
    
    # Get model info
    retrieved_info = registry.get_info("test_model")
    assert retrieved_info is info
    
    # Create model
    model = registry.create("test_model")
    assert isinstance(model, SimpleModel)
    
    # Test search
    results = registry.search(task="classification")
    assert "test_model" in results
    
    results = registry.search(task="segmentation")
    assert "test_model" not in results


def test_register_model_decorator():
    """Test the register_model decorator."""
    
    @register_model(
        name="decorated_model",
        architecture="CNN",
        description="A model registered with a decorator",
        task="image_classification",
        input_size=(3, 32, 32),
        output_size=10,
    )
    class TestModel(nn.Module):
        def __init__(self, channels=3, num_classes=10):
            super().__init__()
            self.conv = nn.Conv2d(channels, 16, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(16 * 16 * 16, num_classes)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Check that the model is registered
    assert "decorated_model" in list_available_models()
    
    # Get model info
    info = get_model_info("decorated_model")
    assert info.name == "decorated_model"
    assert info.architecture == "CNN"
    assert info.task == "image_classification"
    
    # Create model
    model = create_model("decorated_model")
    assert isinstance(model, TestModel)


def test_register_factory_function():
    """Test registering a factory function."""
    
    def create_test_model(input_dim=10, hidden_dim=20, output_dim=2):
        return SimpleModel(input_dim, hidden_dim, output_dim)
    
    register_model(
        name="factory_model",
        architecture="MLP",
        description="A model created with a factory function",
        task="classification",
        factory_fn=create_test_model,
        default_args={"input_dim": 10, "hidden_dim": 30, "output_dim": 3},
    )
    
    # Check that the model is registered
    assert "factory_model" in list_available_models()
    
    # Create model with default args
    model = create_model("factory_model")
    assert isinstance(model, SimpleModel)
    assert model.fc1.in_features == 10
    assert model.fc1.out_features == 30
    assert model.fc2.out_features == 3
    
    # Create model with custom args
    model = create_model("factory_model", input_dim=5, output_dim=4)
    assert model.fc1.in_features == 5
    assert model.fc2.out_features == 4


def test_model_registry_error_handling():
    """Test error handling in the ModelRegistry."""
    registry = ModelRegistry()
    
    # Test creating a non-existent model
    with pytest.raises(ValueError):
        registry.create("non_existent_model")
    
    # Register a model without model_cls or factory_fn
    info = ModelInfo(
        name="invalid_model",
        architecture="MLP",
        description="An invalid model",
        task="classification",
    )
    
    registry.register(info)
    
    # Test creating the invalid model
    with pytest.raises(ValueError):
        registry.create("invalid_model")


def test_create_model_global_registry():
    """Test creating a model from the global registry."""
    # Register a model with the global registry
    register_model(
        name="global_test_model",
        architecture="MLP",
        description="A test model in the global registry",
        task="classification",
        model_cls=SimpleModel,
    )
    
    # Create the model
    model = create_model("global_test_model")
    assert isinstance(model, SimpleModel)


def test_model_zoo_imports():
    """Test importing model zoo modules."""
    from tauto.models import _FULL_MODEL_ZOO_AVAILABLE
    
    # This test is mainly to check that the imports work,
    # not to test the actual models
    try:
        from tauto.models.zoo import vision
        has_vision = True
    except ImportError:
        has_vision = False
    
    try:
        from tauto.models.zoo import nlp
        has_nlp = True
    except ImportError:
        has_nlp = False
    
    # At least one of the model zoo modules should be available
    assert has_vision or has_nlp or not _FULL_MODEL_ZOO_AVAILABLE