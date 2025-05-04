"""
Example usage of the TAuto training optimization utilities.

This script demonstrates how to use the training optimization utilities to:
1. Set up mixed precision training
2. Use gradient accumulation for larger effective batch sizes
3. Implement model checkpointing
4. Use the optimizer factory for best practices
5. Train a model with all optimizations enabled
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add parent directory to path to import tauto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tauto.optimize.training import (
    MixedPrecisionTraining,
    GradientAccumulation,
    ModelCheckpointing,
    OptimizerFactory,
    train_with_optimization,
)


class ConvNet(nn.Module):
    """Simple CNN for CIFAR-10 classification."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x


def create_dummy_cifar_data(num_samples=1000, batch_size=32):
    """Create dummy CIFAR-10-like data for testing."""
    # Generate random images and labels
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (num_samples,))
    
    # Create dataset and dataloader
    dataset = TensorDataset(images, labels)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def example_mixed_precision():
    """Example of using mixed precision training."""
    print("\n1. Mixed Precision Training Example")
    print("----------------------------------")
    
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping mixed precision example")
        return
    
    # Create model, optimizer, and data
    model = ConvNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_loader, _ = create_dummy_cifar_data(num_samples=500, batch_size=64)
    
    # Set up mixed precision training
    mp_trainer = MixedPrecisionTraining(enabled=True)
    
    # Train for a few steps
    model.train()
    start_time = time.time()
    
    for i, (inputs, targets) in enumerate(train_loader):
        if i >= 10:
            break
        
        # Move data to GPU
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        # Perform training step with mixed precision
        metrics = mp_trainer.step(
            model=model,
            inputs=inputs,
            targets=targets,
            criterion=criterion,
            optimizer=optimizer,
        )
        
        print(f"Step {i+1}, Loss: {metrics['loss']:.4f}, Accuracy: {metrics.get('accuracy', 0):.4f}")
    
    print(f"Training took {time.time() - start_time:.2f}s with mixed precision")
    
    # Compare with standard precision
    mp_trainer.enabled = False
    model = ConvNet().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    start_time = time.time()
    
    for i, (inputs, targets) in enumerate(train_loader):
        if i >= 10:
            break
        
        # Move data to GPU
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        # Perform training step without mixed precision
        metrics = mp_trainer.step(
            model=model,
            inputs=inputs,
            targets=targets,
            criterion=criterion,
            optimizer=optimizer,
        )
        
        print(f"Step {i+1}, Loss: {metrics['loss']:.4f}, Accuracy: {metrics.get('accuracy', 0):.4f}")
    
    print(f"Training took {time.time() - start_time:.2f}s without mixed precision")


def example_gradient_accumulation():
    """Example of using gradient accumulation."""
    print("\n2. Gradient Accumulation Example")
    print("-------------------------------")
    
    # Create model, optimizer, and data
    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Small batch size
    train_loader, _ = create_dummy_cifar_data(num_samples=500, batch_size=16)
    
    # Set up gradient accumulation
    accumulator = GradientAccumulation(accumulation_steps=4)
    
    # Train for a few steps
    model.train()
    start_time = time.time()
    
    for i, (inputs, targets) in enumerate(train_loader):
        if i >= 10:
            break
        
        # Perform training step with gradient accumulation
        metrics = accumulator.step(
            model=model,
            inputs=inputs,
            targets=targets,
            criterion=criterion,
            optimizer=optimizer,
        )
        
        print(f"Step {i+1}, Loss: {metrics['loss']:.4f}, Weights Updated: {metrics['weights_updated']}")
    
    print(f"Training with gradient accumulation (effective batch size: {16 * 4}) took {time.time() - start_time:.2f}s")


def example_model_checkpointing():
    """Example of using model checkpointing."""
    print("\n3. Model Checkpointing Example")
    print("-----------------------------")
    
    # Create model, optimizer, and data
    model = ConvNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    train_loader, val_loader = create_dummy_cifar_data(num_samples=500, batch_size=32)
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoint_example")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set up checkpointing
    checkpointer = ModelCheckpointing(
        checkpoint_dir=checkpoint_dir,
        save_freq=1,
        save_optimizer=True,
        save_scheduler=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )
    
    # Simulate training for a few epochs
    for epoch in range(3):
        # Simulated metrics
        train_loss = 1.0 / (epoch + 1)
        val_loss = 1.2 / (epoch + 1)
        
        metrics = {
            "train_loss": train_loss,
            "train_acc": 0.7 + epoch * 0.1,
            "val_loss": val_loss,
            "val_acc": 0.6 + epoch * 0.1,
        }
        
        # Save checkpoint
        checkpoint_path = checkpointer.save_checkpoint(
            model=model,
            epoch=epoch + 1,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
        )
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Checkpoint saved: {checkpoint_path is not None}")
    
    # Find best checkpoint
    best_checkpoint = checkpointer.find_best_checkpoint()
    print(f"Best checkpoint: {best_checkpoint}")
    
    # Load the best checkpoint
    if best_checkpoint is not None:
        model2 = ConvNet()
        optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
        
        checkpoint_data = checkpointer.load_checkpoint(
            model=model2,
            checkpoint_path=best_checkpoint,
            optimizer=optimizer2,
        )
        
        print(f"Loaded checkpoint from epoch {checkpoint_data['epoch']}")
        print(f"  Metrics: {checkpoint_data['metrics']}")


def example_optimizer_factory():
    """Example of using the optimizer factory."""
    print("\n4. Optimizer Factory Example")
    print("---------------------------")
    
    # Create model
    model = ConvNet()
    
    # Create different optimizers
    optimizers = {}
    
    for opt_type in ["sgd", "adam", "adamw", "rmsprop"]:
        # Create optimizer
        optimizer = OptimizerFactory.create_optimizer(
            model=model,
            optimizer_type=opt_type,
            learning_rate=0.01,
            weight_decay=1e-5,
        )
        
        optimizers[opt_type] = optimizer
        
        print(f"Created {opt_type.upper()} optimizer with LR: {optimizer.param_groups[0]['lr']}")
    
    # Create different schedulers
    schedulers = {}
    
    for sched_type in ["cosine", "step", "plateau"]:
        # Create scheduler
        scheduler = OptimizerFactory.create_scheduler(
            optimizer=optimizers["adam"],
            scheduler_type=sched_type,
            epochs=10,
        )
        
        schedulers[sched_type] = scheduler
        
        print(f"Created {sched_type} scheduler")
    
    # Simulate learning rate changes
    print("\nLearning rate schedule for 10 epochs:")
    
    for scheduler_name, scheduler in schedulers.items():
        if scheduler_name == "plateau":
            # ReduceLROnPlateau needs a metric
            continue
        
        lrs = []
        for _ in range(10):
            lrs.append(optimizers["adam"].param_groups[0]["lr"])
            scheduler.step()
        
        print(f"  {scheduler_name.capitalize()}: {[f'{lr:.4f}' for lr in lrs]}")


def example_train_with_optimization():
    """Example of using the train_with_optimization function."""
    print("\n5. Complete Training with Optimization Example")
    print("--------------------------------------------")
    
    # Create model, criterion, and data
    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    
    train_loader, val_loader = create_dummy_cifar_data(num_samples=1000, batch_size=32)
    
    # Create checkpoint directory
    checkpoint_dir = Path("full_training_example")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Create optimizer and scheduler using factory
    optimizer = OptimizerFactory.create_optimizer(
        model=model,
        optimizer_type="adam",
        learning_rate=0.001,
        weight_decay=1e-5,
    )
    
    scheduler = OptimizerFactory.create_scheduler(
        optimizer=optimizer,
        scheduler_type="cosine",
        epochs=5,
    )
    
    # Train with all optimizations
    start_time = time.time()
    
    metrics = train_with_optimization(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=5,
        device=device,
        mixed_precision=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        clip_grad_norm=1.0,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=1,
        early_stopping=True,
        early_stopping_patience=3,
        verbose=True,
    )
    
    print(f"Training completed in {time.time() - start_time:.2f}s")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics["train_acc"], label="Train Accuracy")
    plt.plot(metrics["val_acc"], label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curves")
    
    plt.tight_layout()
    plt.savefig(checkpoint_dir / "training_curves.png")
    print(f"Training curves saved to {checkpoint_dir / 'training_curves.png'}")


def main():
    print("TAuto Training Optimization Examples")
    print("===================================")
    
    # Run examples
    example_mixed_precision()
    example_gradient_accumulation()
    example_model_checkpointing()
    example_optimizer_factory()
    example_train_with_optimization()
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main()