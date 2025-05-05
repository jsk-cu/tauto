"""
Example usage of the TAuto inference optimization utilities.

This script demonstrates how to use the inference optimization utilities to:
1. Apply quantization to a model
2. Apply pruning to a model
3. Apply knowledge distillation to transfer knowledge from a large model to a small one
4. Apply general optimizations for inference
5. Benchmark and compare different optimizations
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
from pathlib import Path
import numpy as np

# Add parent directory to path to import tauto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tauto.optimize.inference import (
    ModelQuantization,
    ModelPruning,
    optimize_for_inference,
    apply_inference_optimizations,
)
from tauto.optimize.distillation import (
    KnowledgeDistillation,
    distill_model,
)
from tauto.profile.profiler import Profiler


class TeacherModel(nn.Module):
    """Larger model for knowledge distillation."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x


class StudentModel(nn.Module):
    """Smaller model for knowledge distillation."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 14 * 14, 128)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return x


def create_mnist_dataset(num_samples=1000, batch_size=32):
    """Create a synthetic MNIST-like dataset."""
    # Generate random 28x28 images and random labels
    images = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, 10, (num_samples,))
    
    # Create dataset and split into train/val
    dataset = TensorDataset(images, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, epochs=5, device=None):
    """Train a model on the given data."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model = model.to(device)
    
    # Set up loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        
        for inputs, targets in train_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_acc += (predicted == targets).sum().item()
        
        # Calculate epoch metrics
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        
        # Validate the model
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return model


def evaluate_model(model, dataloader, criterion=None, device=None):
    """Evaluate a model on the given data."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Move model to device
    model = model.to(device)
    
    # Evaluate the model
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update metrics
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_acc += (predicted == targets).sum().item()
    
    # Calculate metrics
    val_loss /= len(dataloader.dataset)
    val_acc /= len(dataloader.dataset)
    
    return val_loss, val_acc


def benchmark_inference(model, dataloader, device=None, num_runs=10):
    """Benchmark inference time for a model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Warm up
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)
            break
    
    # Benchmark
    latencies = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            batch_latencies = []
            
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                
                # Measure inference time
                start_time = time.time()
                outputs = model(inputs)
                end_time = time.time()
                
                batch_latencies.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Record average latency for this run
            latencies.append(sum(batch_latencies) / len(batch_latencies))
    
    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    return {
        "avg_latency_ms": avg_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "throughput_samples_per_sec": dataloader.batch_size * (1000 / avg_latency),
    }


def example_quantization():
    """Example of quantizing a model."""
    print("\n1. Model Quantization Example")
    print("---------------------------")
    
    # Create model and data
    model = TeacherModel()
    train_loader, val_loader = create_mnist_dataset()
    
    # Train the model
    print("Training the model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train_model(model, train_loader, val_loader, epochs=1, device=device)
    
    # Evaluate the original model
    print("\nEvaluating the original model...")
    eval_device = torch.device("cpu")  # Always evaluate on CPU for fair comparison
    val_loss, val_acc = evaluate_model(trained_model, val_loader, device=eval_device)
    print(f"Original Model - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Benchmark the original model
    original_benchmark = benchmark_inference(trained_model, val_loader, device=eval_device)
    print(f"Original Model - Avg Latency: {original_benchmark['avg_latency_ms']:.2f} ms, "
          f"Throughput: {original_benchmark['throughput_samples_per_sec']:.2f} samples/sec")
    
    # Create a quantizer (dynamic quantization)
    print("\nApplying dynamic quantization...")
    quantizer = ModelQuantization(quantization_type="dynamic")
    
    # Quantize the model
    quantized_model = quantizer.quantize_model(trained_model.cpu())
    
    # Evaluate the quantized model
    val_loss, val_acc = evaluate_model(quantized_model, val_loader, device=eval_device)
    print(f"Quantized Model - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Benchmark the quantized model
    quantized_benchmark = benchmark_inference(quantized_model, val_loader, device=eval_device)
    print(f"Quantized Model - Avg Latency: {quantized_benchmark['avg_latency_ms']:.2f} ms, "
          f"Throughput: {quantized_benchmark['throughput_samples_per_sec']:.2f} samples/sec")
    
    # Measure improvement
    speedup = original_benchmark['avg_latency_ms'] / quantized_benchmark['avg_latency_ms']
    print(f"\nSpeedup from quantization: {speedup:.2f}x")
    
    # Get model sizes
    original_size = quantizer._get_model_size(trained_model)
    quantized_size = quantizer._get_model_size(quantized_model)
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size / original_size) * 100:.2f}%")


def example_pruning():
    """Example of pruning a model."""
    print("\n2. Model Pruning Example")
    print("----------------------")
    
    # Create model and data
    model = TeacherModel()
    train_loader, val_loader = create_mnist_dataset()
    
    # Train the model
    print("Training the model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train_model(model, train_loader, val_loader, epochs=1, device=device)
    
    # Evaluate the original model
    print("\nEvaluating the original model...")
    eval_device = torch.device("cpu")  # Always evaluate on CPU for fair comparison
    val_loss, val_acc = evaluate_model(trained_model, val_loader, device=eval_device)
    print(f"Original Model - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Create a pruner
    print("\nApplying unstructured pruning...")
    pruner = ModelPruning(pruning_type="unstructured", criteria="l1")
    
    # Prune the model
    pruned_model = pruner.prune_model(trained_model, amount=0.5)
    
    # Evaluate the pruned model
    val_loss, val_acc = evaluate_model(pruned_model, val_loader, device=eval_device)
    print(f"Pruned Model - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Get sparsity info
    sparsity_info = pruner.get_model_sparsity(pruned_model)
    print(f"Overall sparsity: {sparsity_info['overall_sparsity'] * 100:.2f}%")
    
    # Print layer-wise sparsity
    print("\nLayer-wise sparsity:")
    for layer, sparsity in sparsity_info["layer_sparsity"].items():
        print(f"  {layer}: {sparsity * 100:.2f}%")
    
    # Example of iterative pruning with fine-tuning
    print("\nApplying iterative pruning with fine-tuning...")
    
    # Define a simple fine-tuning function
    def finetune_model(model):
        # Fine-tune for just 1 epoch for demonstration
        return train_model(model, train_loader, val_loader, epochs=1, device=device)
    
    # Reset the model
    iterative_model = TeacherModel()
    iterative_model = train_model(iterative_model, train_loader, val_loader, epochs=1, device=device)
    
    # Apply iterative pruning
    pruned_iterative_model = pruner.iterative_pruning(
        model=iterative_model,
        train_fn=finetune_model,
        initial_amount=0.2,
        final_amount=0.5,
        steps=3,
    )
    
    # Evaluate the iteratively pruned model
    val_loss, val_acc = evaluate_model(pruned_iterative_model, val_loader, device=eval_device)
    print(f"Iteratively Pruned Model - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Get sparsity info
    sparsity_info = pruner.get_model_sparsity(pruned_iterative_model)
    print(f"Overall sparsity: {sparsity_info['overall_sparsity'] * 100:.2f}%")


def example_distillation():
    """Example of knowledge distillation."""
    print("\n3. Knowledge Distillation Example")
    print("------------------------------")
    
    # Create teacher and student models
    teacher_model = TeacherModel()
    student_model = StudentModel()
    
    # Create data
    train_loader, val_loader = create_mnist_dataset()
    
    # Train the teacher model
    print("Training the teacher model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_teacher = train_model(teacher_model, train_loader, val_loader, epochs=1, device=device)
    
    # Evaluate the teacher model
    print("\nEvaluating the teacher model...")
    val_loss, val_acc = evaluate_model(trained_teacher, val_loader, device=device)
    print(f"Teacher Model - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Train a student model from scratch
    print("\nTraining the student model from scratch...")
    trained_student_scratch = train_model(student_model, train_loader, val_loader, epochs=1, device=device)
    
    # Evaluate the student model
    val_loss, val_acc = evaluate_model(trained_student_scratch, val_loader, device=device)
    print(f"Student Model (from scratch) - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Create a new student model for distillation
    student_model_distill = StudentModel()
    
    # Apply knowledge distillation
    print("\nTraining the student model with knowledge distillation...")
    
    # Define feature layers for feature distillation
    feature_layers = {
        "fc1": "fc1",  # Map teacher's fc1 to student's fc1
    }
    
    # Distill the model
    trained_student_distill, metrics = distill_model(
        teacher_model=trained_teacher,
        student_model=student_model_distill,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=3,
        temperature=4.0,  # Higher temperature for smoother logits
        alpha=0.5,  # Equal weight to task and distillation loss
        feature_layers=feature_layers,
        feature_weight=0.1,
    )
    
    # Evaluate the distilled student model
    val_loss, val_acc = evaluate_model(trained_student_distill, val_loader, device=device)
    print(f"Student Model (distilled) - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Compare model sizes
    teacher_params = sum(p.numel() for p in trained_teacher.parameters())
    student_params = sum(p.numel() for p in trained_student_distill.parameters())
    
    print(f"\nTeacher model parameters: {teacher_params:,}")
    print(f"Student model parameters: {student_params:,}")
    print(f"Parameter reduction: {(1 - student_params / teacher_params) * 100:.2f}%")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Distillation Loss Curves")
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics["train_accuracy"], label="Train Accuracy")
    plt.plot(metrics["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Distillation Accuracy Curves")
    
    plt.tight_layout()
    plt.savefig("distillation_curves.png")
    print("Training curves saved to distillation_curves.png")
    
    # Benchmark the models
    eval_device = torch.device("cpu")
    teacher_benchmark = benchmark_inference(trained_teacher, val_loader, device=eval_device)
    student_scratch_benchmark = benchmark_inference(trained_student_scratch, val_loader, device=eval_device)
    student_distill_benchmark = benchmark_inference(trained_student_distill, val_loader, device=eval_device)
    
    print("\nInference Benchmarks:")
    print(f"Teacher Model - Avg Latency: {teacher_benchmark['avg_latency_ms']:.2f} ms, "
          f"Throughput: {teacher_benchmark['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"Student Model (scratch) - Avg Latency: {student_scratch_benchmark['avg_latency_ms']:.2f} ms, "
          f"Throughput: {student_scratch_benchmark['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"Student Model (distilled) - Avg Latency: {student_distill_benchmark['avg_latency_ms']:.2f} ms, "
          f"Throughput: {student_distill_benchmark['throughput_samples_per_sec']:.2f} samples/sec")
    
    # Calculate speedup
    speedup = teacher_benchmark['avg_latency_ms'] / student_distill_benchmark['avg_latency_ms']
    print(f"Speedup from distillation: {speedup:.2f}x")


def example_general_optimizations():
    """Example of applying general inference optimizations."""
    print("\n4. General Inference Optimizations Example")
    print("---------------------------------------")
    
    # Create model and data
    model = TeacherModel()
    train_loader, val_loader = create_mnist_dataset()
    
    # Train the model
    print("Training the model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train_model(model, train_loader, val_loader, epochs=1, device=device)
    
    # Evaluate the original model
    print("\nEvaluating the original model...")
    eval_device = torch.device("cpu")  # Always evaluate on CPU for fair comparison
    val_loss, val_acc = evaluate_model(trained_model, val_loader, device=eval_device)
    print(f"Original Model - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Benchmark the original model
    original_benchmark = benchmark_inference(trained_model, val_loader, device=eval_device)
    print(f"Original Model - Avg Latency: {original_benchmark['avg_latency_ms']:.2f} ms, "
          f"Throughput: {original_benchmark['throughput_samples_per_sec']:.2f} samples/sec")
    
    # Get example inputs for tracing
    example_inputs = next(iter(val_loader))[0]
    
    # Apply general optimizations
    print("\nApplying general optimizations (freeze, fuse)...")
    optimized_model = optimize_for_inference(
        model=trained_model,
        optimizations=["freeze", "fuse"],
        example_inputs=example_inputs,
    )
    
    # Evaluate the optimized model
    val_loss, val_acc = evaluate_model(optimized_model, val_loader, device=eval_device)
    print(f"Optimized Model - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Benchmark the optimized model
    optimized_benchmark = benchmark_inference(optimized_model, val_loader, device=eval_device)
    print(f"Optimized Model - Avg Latency: {optimized_benchmark['avg_latency_ms']:.2f} ms, "
          f"Throughput: {optimized_benchmark['throughput_samples_per_sec']:.2f} samples/sec")
    
    # Measure improvement
    speedup = original_benchmark['avg_latency_ms'] / optimized_benchmark['avg_latency_ms']
    print(f"\nSpeedup from general optimizations: {speedup:.2f}x")
    
    # Try JIT tracing
    print("\nApplying JIT tracing...")
    traced_model = torch.jit.trace(trained_model.cpu(), example_inputs)
    
    # Benchmark the traced model
    traced_benchmark = benchmark_inference(traced_model, val_loader, device=eval_device)
    print(f"Traced Model - Avg Latency: {traced_benchmark['avg_latency_ms']:.2f} ms, "
          f"Throughput: {traced_benchmark['throughput_samples_per_sec']:.2f} samples/sec")
    
    # Measure improvement
    speedup = original_benchmark['avg_latency_ms'] / traced_benchmark['avg_latency_ms']
    print(f"Speedup from JIT tracing: {speedup:.2f}x")


def example_combined_optimizations():
    """Example of combining multiple optimizations."""
    print("\n5. Combined Optimizations Example")
    print("------------------------------")
    
    # Create model and data
    model = TeacherModel()
    train_loader, val_loader = create_mnist_dataset()
    
    # Train the model
    print("Training the model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train_model(model, train_loader, val_loader, epochs=1, device=device)
    
    # Get model size
    num_params = sum(p.numel() for p in trained_model.parameters())
    print(f"Original model parameters: {num_params:,}")
    
    # Evaluate and benchmark the original model
    print("\nEvaluating the original model...")
    eval_device = torch.device("cpu")  # Always evaluate on CPU for fair comparison
    val_loss, val_acc = evaluate_model(trained_model, val_loader, device=eval_device)
    print(f"Original Model - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    original_benchmark = benchmark_inference(trained_model, val_loader, device=eval_device)
    print(f"Original Model - Avg Latency: {original_benchmark['avg_latency_ms']:.2f} ms, "
          f"Throughput: {original_benchmark['throughput_samples_per_sec']:.2f} samples/sec")
    
    # Apply different optimization combinations and profile them
    combinations = [
        ["quantization"],
        ["pruning"],
        ["quantization", "pruning"],
        ["fuse", "freeze"],
        ["quantization", "fuse", "freeze"],
        ["pruning", "fuse", "freeze"],
        ["quantization", "pruning", "fuse", "freeze"],
    ]
    
    results = {}
    
    for optims in combinations:
        name = "+".join(optims)
        print(f"\nApplying {name}...")
        
        # Apply optimizations
        config = {
            "quantization": {
                "type": "dynamic",
            },
            "pruning": {
                "amount": 0.5,
                "type": "unstructured",
            },
        }
        
        # Get example inputs
        example_inputs = next(iter(val_loader))[0]
        
        optimized_model, _ = apply_inference_optimizations(
            model=trained_model,
            optimizations=optims,
            dataloader=val_loader,
            device=eval_device,
            example_inputs=example_inputs,
            config=config,
        )
        
        # Evaluate the optimized model
        val_loss, val_acc = evaluate_model(optimized_model, val_loader, device=eval_device)
        
        # Benchmark the optimized model
        benchmark = benchmark_inference(optimized_model, val_loader, device=eval_device)
        
        # Record results
        results[name] = {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "latency_ms": benchmark["avg_latency_ms"],
            "throughput": benchmark["throughput_samples_per_sec"],
            "speedup": original_benchmark["avg_latency_ms"] / benchmark["avg_latency_ms"],
        }
        
        # Print results
        print(f"{name} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"{name} - Avg Latency: {benchmark['avg_latency_ms']:.2f} ms, "
              f"Throughput: {benchmark['throughput_samples_per_sec']:.2f} samples/sec")
        print(f"{name} - Speedup: {results[name]['speedup']:.2f}x")
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot latency comparison
    plt.subplot(2, 2, 1)
    names = list(results.keys())
    latencies = [results[name]["latency_ms"] for name in names]
    plt.bar(names, latencies)
    plt.axhline(y=original_benchmark["avg_latency_ms"], color='r', linestyle='-', label="Original")
    plt.ylabel("Latency (ms)")
    plt.title("Inference Latency Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    
    # Plot throughput comparison
    plt.subplot(2, 2, 2)
    throughputs = [results[name]["throughput"] for name in names]
    plt.bar(names, throughputs)
    plt.axhline(y=original_benchmark["throughput_samples_per_sec"], color='r', linestyle='-', label="Original")
    plt.ylabel("Throughput (samples/sec)")
    plt.title("Inference Throughput Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    
    # Plot accuracy comparison
    plt.subplot(2, 2, 3)
    accuracies = [results[name]["val_acc"] for name in names]
    plt.bar(names, accuracies)
    plt.axhline(y=val_acc, color='r', linestyle='-', label="Original")
    plt.ylabel("Validation Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    
    # Plot speedup comparison
    plt.subplot(2, 2, 4)
    speedups = [results[name]["speedup"] for name in names]
    plt.bar(names, speedups)
    plt.axhline(y=1.0, color='r', linestyle='-', label="Original")
    plt.ylabel("Speedup Factor")
    plt.title("Inference Speedup Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("optimization_comparison.png")
    print("\nComparison plot saved to optimization_comparison.png")
    
    # Print summary
    print("\nOptimization Summary:")
    print("-" * 90)
    print(f"{'Optimization':<30} {'Accuracy':<10} {'Latency (ms)':<15} {'Throughput':<15} {'Speedup':<10}")
    print("-" * 90)
    print(f"{'Original':<30} {val_acc:<10.4f} {original_benchmark['avg_latency_ms']:<15.2f} "
          f"{original_benchmark['throughput_samples_per_sec']:<15.2f} {'1.00':<10}")
    
    for name in results:
        print(f"{name:<30} {results[name]['val_acc']:<10.4f} {results[name]['latency_ms']:<15.2f} "
              f"{results[name]['throughput']:<15.2f} {results[name]['speedup']:<10.2f}")


def main():
    print("TAuto Inference Optimization Examples")
    print("====================================")
    
    # Run examples
    example_quantization()
    example_pruning()
    example_distillation()
    example_general_optimizations()
    example_combined_optimizations()
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main()