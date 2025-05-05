"""
Tests for the knowledge distillation utilities.
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

from tauto.optimize.distillation import (
    KnowledgeDistillation,
    distill_model,
)


class TeacherModel(nn.Module):
    """Teacher model for testing knowledge distillation."""
    
    def __init__(self, input_dim=10, hidden_dim=100, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class StudentModel(nn.Module):
    """Student model for testing knowledge distillation."""
    
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


class ConvTeacherModel(nn.Module):
    """Conv teacher model for testing feature distillation."""
    
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvStudentModel(nn.Module):
    """Conv student model for testing feature distillation."""
    
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 16 * 16, 128)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.fixture
def teacher_model():
    """Provide a teacher model for testing."""
    return TeacherModel()


@pytest.fixture
def student_model():
    """Provide a student model for testing."""
    return StudentModel()


@pytest.fixture
def conv_teacher_model():
    """Provide a conv teacher model for testing."""
    return ConvTeacherModel()


@pytest.fixture
def conv_student_model():
    """Provide a conv student model for testing."""
    return ConvStudentModel()


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


@pytest.fixture
def criterion():
    """Provide a loss function for testing."""
    return nn.CrossEntropyLoss()


def test_knowledge_distillation_init(teacher_model, student_model):
    """Test initializing knowledge distillation."""
    # Test with default parameters
    distiller = KnowledgeDistillation(
        teacher_model=teacher_model,
        student_model=student_model,
    )
    assert distiller.temperature == 2.0
    assert distiller.alpha == 0.5
    assert distiller.feature_layers is None
    assert distiller.feature_weight == 0.0
    
    # Test with custom parameters
    distiller = KnowledgeDistillation(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=4.0,
        alpha=0.7,
        feature_layers={"fc2": "fc1"},
        feature_weight=0.3,
    )
    assert distiller.temperature == 4.0
    assert distiller.alpha == 0.7
    assert distiller.feature_layers == {"fc2": "fc1"}
    assert distiller.feature_weight == 0.3


def test_feature_hooks(conv_teacher_model, conv_student_model):
    """Test feature hooks for feature distillation."""
    # Test with feature layers
    feature_layers = {
        "conv2": "conv1",
        "fc1": "fc1",
    }
    
    distiller = KnowledgeDistillation(
        teacher_model=conv_teacher_model,
        student_model=conv_student_model,
        feature_layers=feature_layers,
        feature_weight=0.3,
    )
    
    # Get feature hooks
    teacher_features, student_features = distiller.get_feature_hooks()
    
    # Ensure hooks are set up correctly
    assert isinstance(teacher_features, dict)
    assert isinstance(student_features, dict)
    
    # Pass inputs through models to fill features
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        conv_teacher_model(inputs)
        conv_student_model(inputs)
    
    # Check that features are captured
    assert "conv2" in teacher_features
    assert "fc1" in teacher_features
    assert "conv1" in student_features
    assert "fc1" in student_features


def test_distillation_loss(teacher_model, student_model, criterion):
    """Test computing distillation loss."""
    # Create distiller
    distiller = KnowledgeDistillation(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=2.0,
        alpha=0.5,
    )
    
    # Create inputs and targets
    inputs = torch.randn(16, 10)
    targets = torch.randint(0, 2, (16,))
    
    # Get logits
    with torch.no_grad():
        teacher_logits = teacher_model(inputs)
    student_logits = student_model(inputs)
    
    # Compute distillation loss
    loss, components = distiller.distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        targets=targets,
        criterion=criterion,
    )
    
    # Check the loss components
    assert "task_loss" in components
    assert "distillation_loss" in components
    assert "feature_loss" in components
    assert "total_loss" in components
    
    # Task loss should be greater than 0
    assert components["task_loss"] > 0
    
    # Distillation loss should be greater than 0
    assert components["distillation_loss"] > 0
    
    # Feature loss should be 0 (no feature distillation)
    assert components["feature_loss"] == 0


def test_feature_distillation_loss(conv_teacher_model, conv_student_model, criterion):
    """Test computing feature distillation loss."""
    # Create distiller with feature layers
    feature_layers = {
        "conv2": "conv1",
        "fc1": "fc1",
    }
    
    distiller = KnowledgeDistillation(
        teacher_model=conv_teacher_model,
        student_model=conv_student_model,
        temperature=2.0,
        alpha=0.5,
        feature_layers=feature_layers,
        feature_weight=0.3,
    )
    
    # Create inputs and targets
    inputs = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))
    
    # Get features
    teacher_features, student_features = distiller.get_feature_hooks()
    
    # Get logits
    with torch.no_grad():
        teacher_logits = conv_teacher_model(inputs)
    student_logits = conv_student_model(inputs)
    
    # Compute distillation loss
    loss, components = distiller.distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        targets=targets,
        criterion=criterion,
        teacher_features=teacher_features,
        student_features=student_features,
    )
    
    # Check the loss components
    assert "task_loss" in components
    assert "distillation_loss" in components
    assert "feature_loss" in components
    assert "total_loss" in components
    
    # Task loss should be greater than 0
    assert components["task_loss"] > 0
    
    # Distillation loss should be greater than 0
    assert components["distillation_loss"] > 0
    
    # Feature loss should be greater than 0
    assert components["feature_loss"] > 0


def test_train_step(teacher_model, student_model, criterion):
    """Test training step with knowledge distillation."""
    # Create distiller
    distiller = KnowledgeDistillation(
        teacher_model=teacher_model,
        student_model=student_model,
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.01)
    
    # Create inputs and targets
    inputs = torch.randn(16, 10)
    targets = torch.randint(0, 2, (16,))
    
    # Perform training step
    metrics = distiller.train_step(
        inputs=inputs,
        targets=targets,
        optimizer=optimizer,
        criterion=criterion,
    )
    
    # Check the metrics
    assert "total_loss" in metrics
    assert "task_loss" in metrics
    assert "distillation_loss" in metrics
    assert "feature_loss" in metrics
    assert "accuracy" in metrics


def test_evaluate(teacher_model, student_model, dataloader, criterion):
    """Test evaluating a distilled model."""
    # Create distiller
    distiller = KnowledgeDistillation(
        teacher_model=teacher_model,
        student_model=student_model,
    )
    
    # Evaluate the model
    metrics = distiller.evaluate(
        val_loader=dataloader,
        criterion=criterion,
        device=torch.device("cpu"),
    )
    
    # Check the metrics
    assert "val_loss" in metrics
    assert "val_accuracy" in metrics
    
    # Loss should be greater than 0
    assert metrics["val_loss"] > 0
    
    # Accuracy should be between 0 and 1
    assert 0 <= metrics["val_accuracy"] <= 1


def test_train_epoch(teacher_model, student_model, dataloader, criterion):
    """Test training for one epoch with knowledge distillation."""
    # Create distiller
    distiller = KnowledgeDistillation(
        teacher_model=teacher_model,
        student_model=student_model,
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.01)
    
    # Train for one epoch
    metrics = distiller.train_epoch(
        train_loader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        val_loader=dataloader,
    )
    
    # Check the metrics
    assert "train_loss" in metrics
    assert "train_accuracy" in metrics
    assert "val_loss" in metrics
    assert "val_accuracy" in metrics


def test_distill_model_end_to_end(teacher_model, student_model, dataloader, criterion):
    """Test end-to-end knowledge distillation."""
    # Create a temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        # Distill the model
        trained_student, metrics_history = distill_model(
            teacher_model=teacher_model,
            student_model=student_model,
            train_loader=dataloader,
            val_loader=dataloader,
            criterion=criterion,
            device=torch.device("cpu"),
            epochs=2,  # Use a small number of epochs for testing
            checkpoint_dir=tmpdir,
            verbose=False,
        )
        
        # Check that the model was trained
        assert isinstance(trained_student, nn.Module)
        
        # Check the metrics history
        assert "train_loss" in metrics_history
        assert "train_accuracy" in metrics_history
        assert "val_loss" in metrics_history
        assert "val_accuracy" in metrics_history
        
        # Check that metrics were recorded for each epoch
        assert len(metrics_history["train_loss"]) == 2
        
        # Check that a checkpoint was created
        assert os.path.exists(os.path.join(tmpdir, "student_model_best.pt"))


def test_feature_distillation_end_to_end(conv_teacher_model, conv_student_model, conv_dataloader):
    """Test end-to-end feature distillation."""
    # Create feature layers mapping
    feature_layers = {
        "conv2": "conv1",
        "fc1": "fc1",
    }
    
    # Create a temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        # Distill the model with feature distillation
        trained_student, metrics_history = distill_model(
            teacher_model=conv_teacher_model,
            student_model=conv_student_model,
            train_loader=conv_dataloader,
            val_loader=conv_dataloader,
            device=torch.device("cpu"),
            epochs=2,  # Use a small number of epochs for testing
            feature_layers=feature_layers,
            feature_weight=0.3,
            checkpoint_dir=tmpdir,
            verbose=False,
        )
        
        # Check that the model was trained
        assert isinstance(trained_student, nn.Module)
        
        # Check the metrics history
        assert "train_loss" in metrics_history
        assert "train_accuracy" in metrics_history
        assert "val_loss" in metrics_history
        assert "val_accuracy" in metrics_history
        
        # Check that metrics were recorded for each epoch
        assert len(metrics_history["train_loss"]) == 2
        
        # Check that a checkpoint was created
        assert os.path.exists(os.path.join(tmpdir, "student_model_best.pt"))