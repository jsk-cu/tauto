"""
Tests for the training optimization utilities.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import os
from pathlib import Path
import tempfile
import shutil

from tauto.optimize.training import (
    MixedPrecisionTraining,
    GradientAccumulation,
    ModelCheckpointing,
    OptimizerFactory,
    train_with_optimization,
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


@pytest.fixture
def model():
    """Provide a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def optimizer(model):
    """Provide an optimizer for testing."""
    return optim.SGD(model.parameters(), lr=0.01)


@pytest.fixture
def criterion():
    """Provide a loss function for testing."""
    return nn.CrossEntropyLoss()


@pytest.fixture
def scheduler(optimizer):
    """Provide a scheduler for testing."""
    return optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


@pytest.fixture
def dataloader():
    """Provide a dataloader for testing."""
    # Create random data
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=16)


@pytest.fixture
def checkpoint_dir():
    """Provide a temporary directory for checkpoints."""
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir)


def test_mixed_precision_training(model, optimizer, criterion):
    """Test mixed precision training."""
    # Skip test if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping mixed precision test")
    
    device = torch.device("cuda")
    model.to(device)
    
    # Create inputs and targets
    inputs = torch.randn(16, 10, device=device)
    targets = torch.randint(0, 2, (16,), device=device)
    
    # Create mixed precision trainer
    mp_trainer = MixedPrecisionTraining(enabled=True)
    
    # Test step function
    metrics = mp_trainer.step(model, inputs, targets, criterion, optimizer)
    
    # Check outputs
    assert "loss" in metrics
    assert metrics["loss"] > 0.0
    
    # Test state dict
    state_dict = mp_trainer.state_dict()
    assert "enabled" in state_dict
    assert "dtype" in state_dict
    assert "scaler" in state_dict
    
    # Test loading state dict
    new_mp_trainer = MixedPrecisionTraining(enabled=True)
    new_mp_trainer.load_state_dict(state_dict)
    
    assert new_mp_trainer.enabled == mp_trainer.enabled
    assert new_mp_trainer.dtype == mp_trainer.dtype
    
    # Test disabled mode
    mp_trainer_disabled = MixedPrecisionTraining(enabled=False)
    metrics_disabled = mp_trainer_disabled.step(model, inputs, targets, criterion, optimizer)
    
    assert "loss" in metrics_disabled
    assert metrics_disabled["loss"] > 0.0


def test_gradient_accumulation(model, optimizer, criterion):
    """Test gradient accumulation."""
    # Create inputs and targets
    inputs = torch.randn(16, 10)
    targets = torch.randint(0, 2, (16,))
    
    # Create gradient accumulator
    accumulator = GradientAccumulation(accumulation_steps=2)
    
    # Test first step (no update)
    metrics1 = accumulator.step(model, inputs, targets, criterion, optimizer)
    
    # Check outputs
    assert "loss" in metrics1
    assert metrics1["loss"] > 0.0
    assert metrics1["weights_updated"] == False
    
    # Test second step (with update)
    metrics2 = accumulator.step(model, inputs, targets, criterion, optimizer)
    
    # Check outputs
    assert "loss" in metrics2
    assert metrics2["loss"] > 0.0
    assert metrics2["weights_updated"] == True
    
    # Test state dict
    state_dict = accumulator.state_dict()
    assert "accumulation_steps" in state_dict
    assert "current_step" in state_dict
    
    # Test loading state dict
    new_accumulator = GradientAccumulation(accumulation_steps=4)
    new_accumulator.load_state_dict(state_dict)
    
    assert new_accumulator.accumulation_steps == accumulator.accumulation_steps
    assert new_accumulator.current_step == accumulator.current_step


def test_model_checkpointing(model, optimizer, scheduler, checkpoint_dir):
    """Test model checkpointing."""
    # Create checkpointer
    checkpointer = ModelCheckpointing(
        checkpoint_dir=checkpoint_dir,
        save_freq=1,
        save_optimizer=True,
        save_scheduler=True,
    )
    
    # Test saving checkpoint
    checkpoint_path = checkpointer.save_checkpoint(
        model=model,
        epoch=1,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics={"val_loss": 0.5, "val_acc": 0.8},
    )
    
    # Check that file was created
    assert os.path.exists(checkpoint_path)
    
    # Test loading checkpoint
    model2 = SimpleModel()
    optimizer2 = optim.SGD(model2.parameters(), lr=0.01)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.9)
    
    checkpoint_data = checkpointer.load_checkpoint(
        model=model2,
        checkpoint_path=checkpoint_path,
        optimizer=optimizer2,
        scheduler=scheduler2,
    )
    
    # Check loaded data
    assert checkpoint_data["epoch"] == 1
    assert "metrics" in checkpoint_data
    assert checkpoint_data["metrics"]["val_loss"] == 0.5
    
    # Test finding latest checkpoint
    latest_checkpoint = checkpointer.find_latest_checkpoint()
    assert latest_checkpoint == checkpoint_path
    
    # Create a new checkpoint directory for testing max_to_keep
    new_checkpoint_dir = Path(checkpoint_dir) / "max_to_keep_test"
    os.makedirs(new_checkpoint_dir, exist_ok=True)
    
    # Test max_to_keep with a fresh directory
    checkpointer = ModelCheckpointing(
        checkpoint_dir=new_checkpoint_dir,
        save_freq=1,
        max_to_keep=2,
    )
    
    # Save multiple checkpoints
    for i in range(3):
        checkpointer.save_checkpoint(model, epoch=i+2)
    
    # Check that only the latest 2 are kept
    assert len(list(Path(new_checkpoint_dir).glob("*.pt"))) == 2
    
    # Verify which files are kept (should be the latest ones)
    checkpoint_files = sorted(list(Path(new_checkpoint_dir).glob("*.pt")))
    expected_files = [
        new_checkpoint_dir / "checkpoint_epoch_003.pt",
        new_checkpoint_dir / "checkpoint_epoch_004.pt"
    ]
    assert {str(f) for f in checkpoint_files} == {str(f) for f in expected_files}


def test_optimizer_factory(model):
    """Test optimizer factory."""
    # Test creating different optimizers
    optimizer_types = ["sgd", "adam", "adamw", "rmsprop", "adagrad"]
    
    # Map of optimizer types to their actual class names in torch.optim
    optimizer_classes = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad
    }
    
    for opt_type in optimizer_types:
        optimizer = OptimizerFactory.create_optimizer(
            model=model,
            optimizer_type=opt_type,
            learning_rate=0.01,
            weight_decay=1e-5,
        )
        
        # Check that optimizer was created with correct type
        assert isinstance(optimizer, optimizer_classes[opt_type])
        
        # Check learning rate
        assert optimizer.param_groups[0]["lr"] == 0.01
    
    # Test creating different schedulers
    optimizer = OptimizerFactory.create_optimizer(model, "adam")
    scheduler_types = ["cosine", "step", "plateau", "onecycle"]
    
    for sched_type in scheduler_types:
        # For OneCycle, steps_per_epoch is required
        kwargs = {"steps_per_epoch": 100} if sched_type == "onecycle" else {}
        
        scheduler = OptimizerFactory.create_scheduler(
            optimizer=optimizer,
            scheduler_type=sched_type,
            epochs=10,
            **kwargs
        )
        
        # Check that scheduler was created
        assert scheduler is not None


def test_train_with_optimization(model, dataloader, checkpoint_dir):
    """Test train_with_optimization function."""
    # Create a copy of the dataloader for validation
    val_loader = torch.utils.data.DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
    )
    
    # Train for a few epochs
    metrics = train_with_optimization(
        model=model,
        train_loader=dataloader,
        val_loader=val_loader,
        epochs=2,
        device="cpu",
        mixed_precision=False,
        gradient_accumulation_steps=2,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=1,
        verbose=False,
    )
    
    # Check metrics
    assert "train_loss" in metrics
    assert len(metrics["train_loss"]) == 2
    assert "val_loss" in metrics
    assert len(metrics["val_loss"]) == 2
    
    # Check that checkpoints were created
    assert len(list(Path(checkpoint_dir).glob("*.pt"))) > 0
    
    # Test early stopping
    # Reset the model to make it converge less quickly for early stopping test
    for param in model.parameters():
        nn.init.zeros_(param)
    
    # Create a custom validation dataset that will cause early stopping
    # First data point has label 0, rest have label 1
    x = torch.randn(100, 10)
    y = torch.ones(100, dtype=torch.long)
    y[0] = 0  # First example is different class
    
    # Create a dataset where validation loss won't improve
    test_dataset = torch.utils.data.TensorDataset(x, y)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
    
    # Train with early stopping
    metrics = train_with_optimization(
        model=model,
        train_loader=dataloader,
        val_loader=test_loader,  # Use our dataset that won't show improvement
        epochs=10,
        device="cpu",
        early_stopping=True,
        early_stopping_patience=2,  # Stop after 2 epochs without improvement
        verbose=False,
    )
    
    # Check that training stopped early
    assert len(metrics["train_loss"]) < 10
    
    # Test callbacks
    called = [False]
    
    def callback(model, optimizer, scheduler, epoch, metrics):
        called[0] = True
    
    metrics = train_with_optimization(
        model=model,
        train_loader=dataloader,
        epochs=1,
        device="cpu",
        callbacks=[callback],
        verbose=False,
    )
    
    assert called[0] == True