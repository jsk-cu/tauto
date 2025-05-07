"""
Tests for the hyperparameter optimization utilities.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import os
from pathlib import Path
import tempfile
import shutil
import numpy as np

from tauto.optimize.hyperparams import (
    HyperparameterOptimization,
    create_search_space,
    optimize_hyperparameters,
    PruningCallback,
)

# Skip all tests if Optuna is not available
try:
    import optuna
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _OPTUNA_AVAILABLE, reason="Optuna not available")


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


@pytest.fixture
def dataset():
    """Provide a simple dataset for testing."""
    # Create random data
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    return torch.utils.data.TensorDataset(x, y)


@pytest.fixture
def dataloader(dataset):
    """Provide a dataloader for testing."""
    return torch.utils.data.DataLoader(dataset, batch_size=16)


@pytest.fixture
def search_space():
    """Provide a search space for testing."""
    return {
        "hidden_dim": {
            "type": "int",
            "low": 10,
            "high": 50,
            "step": 10,
        },
        "learning_rate": {
            "type": "float",
            "low": 1e-4,
            "high": 1e-2,
            "log": True,
        },
        "optimizer": {
            "type": "categorical",
            "choices": ["sgd", "adam", "rmsprop"],
        },
        "dropout": {
            "type": "float",
            "low": 0.0,
            "high": 0.5,
            "condition": ("optimizer", "adam"),
        },
    }


def test_create_search_space():
    """Test creating a search space."""
    # Test with numerical parameters
    parameter_ranges = {
        "hidden_dim": (10, 50, 10),  # (low, high, step)
        "learning_rate": (1e-4, 1e-2, True),  # (low, high, log)
        "weight_decay": (1e-5, 1e-3),  # (low, high)
        "batch_size": 32,  # Fixed value
    }
    
    # Test with categorical parameters
    categorical_params = {
        "optimizer": ["sgd", "adam", "rmsprop"],
        "activation": ["relu", "tanh", "sigmoid"],
    }
    
    # Test with conditional parameters
    conditional_params = {
        "dropout": ("optimizer", "adam"),
        "momentum": ("optimizer", "sgd"),
    }
    
    # Test with nested parameters
    nested_params = {
        "scheduler": {
            "type": {
                "type": "categorical",
                "choices": ["cosine", "step", "plateau"],
            },
            "patience": {
                "type": "int",
                "low": 5,
                "high": 20,
                "condition": ("type", "plateau"),
            },
        },
    }
    
    # Create search space
    search_space = create_search_space(
        parameter_ranges=parameter_ranges,
        categorical_params=categorical_params,
        conditional_params=conditional_params,
        nested_params=nested_params,
    )
    
    # Check that the search space is created correctly
    assert "hidden_dim" in search_space
    assert search_space["hidden_dim"]["type"] == "int"
    assert search_space["hidden_dim"]["low"] == 10
    assert search_space["hidden_dim"]["high"] == 50
    assert search_space["hidden_dim"]["step"] == 10
    
    assert "learning_rate" in search_space
    assert search_space["learning_rate"]["type"] == "float"
    assert search_space["learning_rate"]["low"] == 1e-4
    assert search_space["learning_rate"]["high"] == 1e-2
    assert search_space["learning_rate"]["log"] is True
    
    assert "weight_decay" in search_space
    assert search_space["weight_decay"]["type"] == "float"
    assert search_space["weight_decay"]["low"] == 1e-5
    assert search_space["weight_decay"]["high"] == 1e-3
    
    assert "batch_size" in search_space
    assert search_space["batch_size"] == 32
    
    assert "optimizer" in search_space
    assert search_space["optimizer"]["type"] == "categorical"
    assert search_space["optimizer"]["choices"] == ["sgd", "adam", "rmsprop"]
    
    assert "dropout" in search_space
    assert "condition" in search_space["dropout"]
    assert search_space["dropout"]["condition"] == ("optimizer", "adam")
    
    assert "scheduler" in search_space
    assert "search_space" in search_space["scheduler"]
    assert "type" in search_space["scheduler"]["search_space"]
    assert "patience" in search_space["scheduler"]["search_space"]
    assert "condition" in search_space["scheduler"]["search_space"]["patience"]


def test_hyperparameter_optimization_init(search_space):
    """Test initializing hyperparameter optimization."""
    # Define a simple objective function
    def objective(trial, params):
        return params["learning_rate"] * 10
    
    # Create optimizer
    optimizer = HyperparameterOptimization(
        search_space=search_space,
        objective_fn=objective,
        direction="minimize",
        search_algorithm="tpe",
        n_trials=5,
    )
    
    # Check that the optimizer is initialized correctly
    assert optimizer.search_space == search_space
    assert optimizer.objective_fn == objective
    assert optimizer.direction == "minimize"
    assert optimizer.search_algorithm == "tpe"
    assert optimizer.n_trials == 5


def test_suggest_params(search_space):
    """Test suggesting parameters from a search space."""
    # Create optimizer
    def objective(trial, params):
        return 0.0
    
    optimizer = HyperparameterOptimization(
        search_space=search_space,
        objective_fn=objective,
        n_trials=1,
    )
    
    # Create a trial
    study = optuna.create_study()
    trial = study.ask()
    
    # Suggest parameters
    params = optimizer.suggest_params(trial, search_space)
    
    # Check that parameters are suggested correctly
    assert "hidden_dim" in params
    assert isinstance(params["hidden_dim"], int)
    assert 10 <= params["hidden_dim"] <= 50
    assert params["hidden_dim"] % 10 == 0  # Step size of 10
    
    assert "learning_rate" in params
    assert isinstance(params["learning_rate"], float)
    assert 1e-4 <= params["learning_rate"] <= 1e-2
    
    assert "optimizer" in params
    assert params["optimizer"] in ["sgd", "adam", "rmsprop"]
    
    # Dropout is conditional and may not be present
    if params["optimizer"] == "adam":
        assert "dropout" in params
        assert isinstance(params["dropout"], float)
        assert 0.0 <= params["dropout"] <= 0.5
    else:
        assert "dropout" not in params


def test_optimize(search_space):
    """Test running hyperparameter optimization."""
    # Define a simple objective function
    def objective(trial, params):
        # Simple quadratic function
        x = params["hidden_dim"]
        return (x - 30) ** 2
    
    # Create optimizer
    optimizer = HyperparameterOptimization(
        search_space=search_space,
        objective_fn=objective,
        direction="minimize",
        search_algorithm="tpe",
        n_trials=10,
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Check that optimization results are returned
    assert "best_params" in results
    assert "best_value" in results
    assert "best_trial" in results
    assert "study" in results
    
    # Check that the best value is close to the minimum (0 at x=30)
    assert results["best_value"] < 100.0
    
    # Check that the best hidden_dim is close to 30
    assert abs(results["best_params"]["hidden_dim"] - 30) <= 10


def test_save_load_study(search_space, tmp_path):
    """Test saving and loading a study."""
    # Define a simple objective function
    def objective(trial, params):
        return params["learning_rate"] * 10
    
    # Create optimizer
    optimizer = HyperparameterOptimization(
        search_space=search_space,
        objective_fn=objective,
        direction="minimize",
        search_algorithm="tpe",
        n_trials=3,
    )
    
    # Run optimization
    optimizer.optimize()
    
    # Save the study
    save_path = tmp_path / "study.pkl"
    optimizer.save_study(save_path)
    
    # Check that the file was created
    assert os.path.exists(save_path)
    
    # Load the study
    loaded_study = HyperparameterOptimization.load_study(save_path)
    
    # Check that the study was loaded correctly
    assert len(loaded_study.trials) == len(optimizer.study.trials)
    assert loaded_study.best_value == optimizer.study.best_value


def test_optimize_hyperparameters(dataset):
    """Test optimize_hyperparameters function."""
    # Define model creation and evaluation functions
    def train_fn(params):
        # Create model with given hyperparameters
        model = SimpleModel(hidden_dim=params["hidden_dim"])
        
        # Create optimizer
        optimizer_cls = {
            "sgd": optim.SGD,
            "adam": optim.Adam,
            "rmsprop": optim.RMSprop,
        }.get(params["optimizer"], optim.Adam)
        
        optimizer = optimizer_cls(model.parameters(), lr=params["learning_rate"])
        
        # "Train" model (simulate training)
        x = torch.randn(10, 10)
        y = torch.randint(0, 2, (10,))
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        return model
    
    def eval_fn(model):
        # Evaluate model (simulate evaluation)
        x = torch.randn(10, 10)
        model.eval()
        with torch.no_grad():
            outputs = model(x)
        
        # Return a random metric (for testing)
        hidden_dim = model.fc1.out_features
        
        # Simulate a metric that is minimized at hidden_dim=30
        return (hidden_dim - 30) ** 2
    
    # Define search space
    search_space = {
        "hidden_dim": {
            "type": "int",
            "low": 10,
            "high": 50,
            "step": 10,
        },
        "learning_rate": {
            "type": "float",
            "low": 1e-4,
            "high": 1e-2,
            "log": True,
        },
        "optimizer": {
            "type": "categorical",
            "choices": ["sgd", "adam", "rmsprop"],
        },
    }
    
    # Run hyperparameter optimization
    results = optimize_hyperparameters(
        train_fn=train_fn,
        eval_fn=eval_fn,
        search_space=search_space,
        direction="minimize",
        n_trials=5,
        verbose=False,
    )
    
    # Check that optimization results are returned
    assert "best_params" in results
    assert "best_value" in results
    assert "best_model" in results
    assert "study" in results
    assert "optimizer" in results
    
    # Check that the best model is returned
    assert isinstance(results["best_model"], SimpleModel)
    
    # Check that the best value is close to the minimum (0 at hidden_dim=30)
    assert results["best_value"] < 500.0


def test_pruning_callback():
    """Test pruning callback."""
    # Create a trial
    study = optuna.create_study()
    trial = study.ask()
    
    # Create a pruning callback
    callback = PruningCallback(
        trial=trial,
        monitor="val_loss",
        direction="minimize",
    )
    
    # Test callback with improving metrics (should not prune)
    metrics = {"val_loss": 0.5}
    should_stop = callback(epoch=0, metrics=metrics)
    assert not should_stop
    
    # Test callback with worsening metrics
    # Note: We can't easily test pruning in a unit test because it depends on
    # Optuna's internal pruning logic. We just test that the callback doesn't
    # raise exceptions.
    metrics = {"val_loss": 0.6}
    should_stop = callback(epoch=1, metrics=metrics)
    assert not should_stop  # May be True in some cases, but unlikely with just 2 epochs
    
    # Test with missing monitor metric
    metrics = {"train_loss": 0.5}
    should_stop = callback(epoch=2, metrics=metrics)
    assert not should_stop