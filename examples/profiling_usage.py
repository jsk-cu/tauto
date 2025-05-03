"""
Example usage of the TAuto profiling utilities.

This script demonstrates how to use the profiling utilities to:
1. Profile model training
2. Profile model inference
3. Analyze memory usage
4. Visualize and report profiling results
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# Add parent directory to path to import tauto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tauto.profile import (
    Profiler,
    ProfilerConfig,
    profile_model,
    visualize_profile_results,
    create_profile_report,
)


class SimpleConvNet(nn.Module):
    """Simple CNN for MNIST classification."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def create_dummy_mnist_data(num_samples=1000, batch_size=32):
    """Create dummy MNIST-like data for testing."""
    # Generate random images and labels
    images = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, 10, (num_samples,))
    
    # Create dataset and dataloader
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def main():
    print("TAuto Profiling Example")
    print("======================")
    
    # Create a model and dataloader
    print("\nCreating model and data...")
    model = SimpleConvNet()
    dataloader = create_dummy_mnist_data()
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Create a profiler configuration
    profile_dir = os.path.join(os.path.dirname(__file__), "profile_results")
    os.makedirs(profile_dir, exist_ok=True)
    
    config = ProfilerConfig(
        enabled=True,
        use_cuda=device.type == "cuda",
        profile_memory=True,
        record_shapes=True,
        with_stack=False,
        with_flops=False,
        profile_dir=profile_dir,
        num_warmup_steps=1,
        num_active_steps=5,
        num_repeat=1,
    )
    
    # 1. Profile training
    print("\n1. Profiling Training")
    print("-------------------")
    
    # Create optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Create profiler and profile training
    profiler = Profiler(model, config)
    
    print("Running training profiling...")
    training_result = profiler.profile_training(
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        num_steps=10,
        name="mnist_training",
    )
    
    # Print profiling results
    print("\nTraining Profile Results:")
    print(f"Total duration: {training_result.duration_ms.get('total', 0):.2f} ms")
    print(f"CPU memory: {training_result.memory_usage.get('cpu_total', 0):.2f} MB")
    
    if device.type == "cuda":
        print(f"GPU memory: {training_result.memory_usage.get('cuda_total', 0):.2f} MB")
    
    # 2. Profile inference
    print("\n2. Profiling Inference")
    print("--------------------")
    
    # Set model to evaluation mode
    model.eval()
    
    print("Running inference profiling...")
    inference_result = profiler.profile_inference(
        dataloader=dataloader,
        num_steps=10,
        with_grad=False,
        name="mnist_inference",
    )
    
    # Print profiling results
    print("\nInference Profile Results:")
    print(f"Total duration: {inference_result.duration_ms.get('total', 0):.2f} ms")
    print(f"Per batch: {inference_result.duration_ms.get('per_batch', 0):.2f} ms")
    print(f"CPU memory: {inference_result.memory_usage.get('cpu_total', 0):.2f} MB")
    
    if device.type == "cuda":
        print(f"GPU memory: {inference_result.memory_usage.get('cuda_total', 0):.2f} MB")
    
    # 3. Profile memory usage
    print("\n3. Profiling Memory Usage")
    print("-----------------------")
    
    print("Running memory profiling...")
    memory_result = profiler.profile_memory_usage(
        dataloader=dataloader,
        name="mnist_memory",
    )
    
    # Print profiling results
    print("\nMemory Profile Results:")
    for key, value in memory_result.memory_usage.items():
        print(f"{key}: {value:.2f} MB")
    
    # 4. Run comprehensive profiling with the utility function
    print("\n4. Comprehensive Profiling")
    print("------------------------")
    
    print("Running comprehensive profiling...")
    # Use the utility function to profile in all modes
    results = profile_model(
        model=model,
        dataloader=dataloader,
        config=config,
        mode="all",
        name="mnist_comprehensive",
    )
    
    # 5. Visualize and create report
    print("\n5. Visualization and Reporting")
    print("----------------------------")
    
    # Combine all results
    all_results = {
        "training": training_result,
        "inference": inference_result,
        "memory": memory_result,
    }
    
    # Visualize results
    print("Creating visualizations...")
    try:
        import matplotlib
        visualization_dir = os.path.join(profile_dir, "visualizations")
        os.makedirs(visualization_dir, exist_ok=True)
        
        plot_paths = visualize_profile_results(
            all_results,
            output_dir=visualization_dir,
            save_format="png",
        )
        
        print(f"Created {len(plot_paths)} visualization plots in {visualization_dir}")
        
        # Create HTML report
        report_path = os.path.join(profile_dir, "profile_report.html")
        report = create_profile_report(
            all_results,
            output_path=report_path,
            include_plots=True,
        )
        
        print(f"Created profile report at {report_path}")
    except ImportError:
        print("Matplotlib not available, skipping visualizations and report")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()