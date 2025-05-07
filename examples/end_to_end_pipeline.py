"""
End-to-end optimization pipeline for TAuto.

This script demonstrates the complete workflow of using TAuto for model optimization,
from loading a model to applying various optimizations and generating reports.
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
import argparse
import yaml

# Add parent directory to path to import tauto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tauto.config import ConfigManager, load_config, get_default_config
from tauto.utils import get_logger, configure_logging, setup_wandb
from tauto.profile import Profiler, ProfilerConfig, profile_model, create_profile_report
from tauto.models import register_model, create_model
from tauto.optimize.training import train_with_optimization
from tauto.optimize.inference import ModelQuantization, ModelPruning, apply_inference_optimizations
from tauto.optimize.compiler import TorchCompile, apply_compiler_optimization
from tauto.report import generate_optimization_report

# Configure logger
logger = get_logger("tauto.pipeline")


class ResNet18(nn.Module):
    """Simple implementation of ResNet-18 for demonstration."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Final classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block might have a stride
        layers.append(self._make_block(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(self._make_block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _make_block(self, in_channels, out_channels, stride=1):
        """Create a residual block."""
        # Main path
        main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            shortcut = nn.Identity()
        
        # Combine main path and shortcut with ReLU
        return ResidualBlock(main_path, shortcut)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class ResidualBlock(nn.Module):
    """A residual block for ResNet."""
    
    def __init__(self, main_path, shortcut):
        super().__init__()
        self.main_path = main_path
        self.shortcut = shortcut
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.main_path(x)
        identity = self.shortcut(identity)
        
        out += identity
        out = self.relu(out)
        
        return out


def create_cifar_dataset(num_samples=1000, batch_size=32, config=None):
    """Create a synthetic CIFAR-like dataset with optimization from config."""
    # Generate random 32x32 images and random labels
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (num_samples,))
    
    # Create dataset
    dataset = TensorDataset(images, labels)
    
    # Get data config if provided
    if config and hasattr(config, "data"):
        data_config = config.data
        
        # Extract optimized dataloader settings
        num_workers = data_config.get("num_workers", 4)
        prefetch_factor = data_config.get("prefetch_factor", 2)
        pin_memory = data_config.get("pin_memory", True)
        persistent_workers = data_config.get("persistent_workers", True if num_workers > 0 else False)
        drop_last = data_config.get("drop_last", False)
    else:
        # Default settings
        num_workers = 4
        prefetch_factor = 2
        pin_memory = True
        persistent_workers = True
        drop_last = False
    
    # Create optimized dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory, 
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=drop_last,
    )
    
    return dataloader


def profile_original_model(model, dataloader, device, output_dir):
    """
    Profile the original model to identify optimization opportunities.
    
    Args:
        model: PyTorch model to profile
        dataloader: DataLoader for inputs
        device: Device to profile on
        output_dir: Directory to save profiling results
        
    Returns:
        Dict: Profiling results
    """
    logger.info("Profiling original model...")
    
    # Configure profiler
    profiler_config = ProfilerConfig(
        enabled=True,
        use_cuda=(device.type == "cuda"),
        profile_memory=True,
        profile_dir=str(output_dir / "profile_results"),
        num_warmup_steps=2,
        num_active_steps=5,
    )
    
    # Profile model
    profile_results = profile_model(
        model=model,
        dataloader=dataloader,
        config=profiler_config,
        mode="all",
        name="original_model",
    )
    
    # Generate profiling report
    report_path = create_profile_report(
        profile_results, 
        output_path=output_dir / "profile_report.html"
    )
    
    logger.info(f"Profiling report saved to {report_path}")
    
    return profile_results


def apply_optimizations(model, dataloader, device, profile_results, config, output_dir):
    """
    Apply optimizations based on profiling results and config.
    
    Args:
        model: PyTorch model to optimize
        dataloader: DataLoader for inputs
        device: Device to optimize for
        profile_results: Results from profiling
        config: Configuration with optimization settings
        output_dir: Directory to save optimized models
        
    Returns:
        Dict: Optimized models and results
    """
    logger.info("Applying optimizations based on profiling results...")
    
    optimized_models = {}
    optimization_results = {}
    
    # Get optimization config
    optimization_config = config.optimization if hasattr(config, "optimization") else {}
    
    # Get example inputs for optimization
    example_inputs, _ = next(iter(dataloader))
    example_inputs = example_inputs.to(device)
    
    # 1. Compiler optimizations
    if optimization_config.get("torch_compile", {}).get("enabled", False):
        logger.info("Applying torch.compile optimization...")
        
        # Get compiler settings
        compiler_config = optimization_config.get("torch_compile", {})
        backend = compiler_config.get("backend", "inductor")
        mode = compiler_config.get("mode", "default")
        
        # Apply optimization
        compiler = TorchCompile(backend=backend, mode=mode)
        compiled_model = compiler.compile_model(model, example_inputs=example_inputs)
        
        # Benchmark
        benchmark_results = compiler.benchmark_compilation(
            model=model,
            inputs=example_inputs,
        )
        
        # Save model and results
        optimized_models["compiled"] = compiled_model
        optimization_results["compiled"] = benchmark_results
        
        # Save model
        torch.save(compiled_model, output_dir / "compiled_model.pt")
        logger.info(f"Compiled model saved to {output_dir / 'compiled_model.pt'}")
    
    # 2. Quantization
    if optimization_config.get("quantization", {}).get("enabled", False):
        logger.info("Applying quantization optimization...")
        
        # Get quantization settings
        quant_config = optimization_config.get("quantization", {})
        precision = quant_config.get("precision", "int8")
        approach = quant_config.get("approach", "post_training")
        
        # Map settings to quantizer parameters
        quant_type = "dynamic" if approach == "post_training" else "static"
        dtype = torch.qint8 if precision == "int8" else torch.quint8
        
        # Create quantizer
        quantizer = ModelQuantization(
            quantization_type=quant_type, 
            dtype=dtype
        )
        
        # Apply quantization
        quantized_model = quantizer.quantize_model(
            model=model,
            dataloader=dataloader,
        )
        
        # Evaluate quantization impact
        quant_results = quantizer.evaluate_quantization(
            original_model=model,
            quantized_model=quantized_model,
            dataloader=dataloader,
            device=torch.device("cpu"),  # Quantized models run on CPU
        )
        
        # Save model and results
        optimized_models["quantized"] = quantized_model
        optimization_results["quantized"] = quant_results
        
        # Save model
        torch.save(quantized_model, output_dir / "quantized_model.pt")
        logger.info(f"Quantized model saved to {output_dir / 'quantized_model.pt'}")
    
    # 3. Pruning
    if optimization_config.get("pruning", {}).get("enabled", False):
        logger.info("Applying pruning optimization...")
        
        # Get pruning settings
        pruning_config = optimization_config.get("pruning", {})
        method = pruning_config.get("method", "magnitude")
        sparsity = pruning_config.get("sparsity", 0.5)
        
        # Map settings to pruner parameters
        pruning_type = "unstructured" if method == "magnitude" else "structured"
        
        # Create pruner
        pruner = ModelPruning(
            pruning_type=pruning_type,
            criteria="l1",
        )
        
        # Apply pruning
        pruned_model = pruner.prune_model(
            model=model,
            amount=sparsity,
        )
        
        # Evaluate pruning impact
        pruning_results = pruner.evaluate_pruning(
            original_model=model,
            pruned_model=pruned_model,
            dataloader=dataloader,
            device=device,
        )
        
        # Save model and results
        optimized_models["pruned"] = pruned_model
        optimization_results["pruned"] = pruning_results
        
        # Save model
        torch.save(pruned_model, output_dir / "pruned_model.pt")
        logger.info(f"Pruned model saved to {output_dir / 'pruned_model.pt'}")
    
    # 4. Combined optimizations
    if len(optimized_models) > 1:
        logger.info("Applying combined optimizations...")
        
        # Start with a fresh copy of the model
        combined_model = copy.deepcopy(model)
        
        # Apply multiple optimizations
        optimizations = []
        if optimization_config.get("pruning", {}).get("enabled", False):
            optimizations.append("pruning")
        if optimization_config.get("quantization", {}).get("enabled", False):
            optimizations.append("quantization")
        if optimization_config.get("torch_compile", {}).get("enabled", False):
            optimizations.append("trace")
        
        # Apply combined optimizations
        combined_model, combined_results = apply_inference_optimizations(
            model=combined_model,
            optimizations=optimizations,
            dataloader=dataloader,
            device=device,
            example_inputs=example_inputs,
            config={
                "pruning": {
                    "amount": sparsity if optimization_config.get("pruning", {}).get("enabled", False) else 0.0,
                },
                "quantization": {
                    "type": quant_type if optimization_config.get("quantization", {}).get("enabled", False) else "dynamic",
                },
            }
        )
        
        # Save model and results
        optimized_models["combined"] = combined_model
        optimization_results["combined"] = combined_results
        
        # Save model
        torch.save(combined_model, output_dir / "combined_model.pt")
        logger.info(f"Combined optimized model saved to {output_dir / 'combined_model.pt'}")
    
    return {"models": optimized_models, "results": optimization_results}


def evaluate_models(original_model, optimized_models, dataloader, device):
    """
    Evaluate original and optimized models for comparison.
    
    Args:
        original_model: Original PyTorch model
        optimized_models: Dictionary of optimized models
        dataloader: DataLoader for evaluation
        device: Device to evaluate on
        
    Returns:
        Dict: Evaluation results
    """
    logger.info("Evaluating and comparing models...")
    
    results = {}
    
    # Define evaluation function
    def evaluate_model(model, name):
        model.eval()
        correct = 0
        total = 0
        total_time = 0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                # For quantized models, use CPU
                if name == "quantized":
                    inputs = inputs.cpu()
                    targets = targets.cpu()
                else:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                
                # Measure inference time
                start_time = time.time()
                outputs = model(inputs)
                end_time = time.time()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                if name == "quantized":
                    correct += (predicted == targets.cpu()).sum().item()
                else:
                    correct += (predicted == targets).sum().item()
                total += targets.size(0)
                
                # Record time
                batch_time = (end_time - start_time) * 1000  # ms
                total_time += batch_time
                num_batches += 1
        
        # Calculate metrics
        accuracy = correct / total
        avg_latency = total_time / num_batches
        throughput = dataloader.batch_size * (1000 / avg_latency)  # samples/sec
        
        return {
            "accuracy": accuracy,
            "latency_ms": avg_latency,
            "throughput": throughput,
        }
    
    # Evaluate original model
    results["original"] = evaluate_model(original_model, "original")
    
    # Evaluate each optimized model
    for name, model in optimized_models.items():
        results[name] = evaluate_model(model, name)
    
    # Calculate improvements relative to original
    for name, result in results.items():
        if name != "original":
            original_latency = results["original"]["latency_ms"]
            result["speedup"] = original_latency / result["latency_ms"] if result["latency_ms"] > 0 else 0
            result["accuracy_delta"] = result["accuracy"] - results["original"]["accuracy"]
    
    return results


def generate_reports(original_model, optimized_models, evaluation_results, 
                     optimization_results, profile_results, config, output_dir):
    """
    Generate comprehensive reports on the optimization process.
    
    Args:
        original_model: Original PyTorch model
        optimized_models: Dictionary of optimized models
        evaluation_results: Results from model evaluation
        optimization_results: Results from individual optimizations
        profile_results: Results from profiling
        config: TAuto configuration
        output_dir: Directory to save reports
    """
    logger.info("Generating optimization reports...")
    
    # Create report output directory
    report_dir = output_dir / "reports"
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate main optimization report
    report_path = generate_optimization_report(
        original_model=original_model,
        optimized_models=optimized_models,
        evaluation_results=evaluation_results,
        optimization_results=optimization_results,
        profile_results=profile_results,
        config=config,
        output_path=report_dir / "optimization_report.html",
    )
    
    logger.info(f"Optimization report saved to {report_path}")
    
    # Generate visualizations
    try:
        # Visualization directory
        viz_dir = report_dir / "visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Performance comparison plot
        plt.figure(figsize=(10, 6))
        
        # Get model names and metrics
        models = list(evaluation_results.keys())
        latencies = [evaluation_results[m]["latency_ms"] for m in models]
        accuracies = [evaluation_results[m]["accuracy"] * 100 for m in models]
        throughputs = [evaluation_results[m]["throughput"] for m in models]
        
        # Plot metrics
        plt.subplot(2, 1, 1)
        plt.bar(models, latencies)
        plt.ylabel("Latency (ms)")
        plt.title("Performance Comparison")
        
        plt.subplot(2, 1, 2)
        plt.bar(models, accuracies)
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "performance_comparison.png")
        
        # Throughput comparison
        plt.figure(figsize=(10, 4))
        plt.bar(models, throughputs)
        plt.ylabel("Throughput (samples/sec)")
        plt.title("Throughput Comparison")
        plt.savefig(viz_dir / "throughput_comparison.png")
        
        # If there are multiple optimized models, plot speedup comparison
        if len(optimized_models) > 0:
            plt.figure(figsize=(10, 4))
            speedups = [evaluation_results[m].get("speedup", 1.0) for m in models if m != "original"]
            model_names = [m for m in models if m != "original"]
            plt.bar(model_names, speedups)
            plt.ylabel("Speedup Factor")
            plt.title("Optimization Speedup Comparison")
            plt.axhline(y=1.0, color='r', linestyle='-')
            plt.savefig(viz_dir / "speedup_comparison.png")
        
        logger.info(f"Visualization plots saved to {viz_dir}")
    
    except Exception as e:
        logger.warning(f"Error generating visualization plots: {e}")


def main():
    """Main pipeline function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="TAuto End-to-End Optimization Pipeline")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--output", "-o", type=str, default="tauto_results", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        # Create default configuration
        config = ConfigManager(get_default_config()).config
        logger.info("Using default configuration")
        
        # Save default configuration
        with open(output_dir / "default_config.yaml", "w") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
        logger.info(f"Default configuration saved to {output_dir / 'default_config.yaml'}")
    
    # Configure logging
    configure_logging(config.logging)
    
    # Set up W&B if enabled
    if config.wandb.get("enabled", False):
        run = setup_wandb(config.wandb)
        if run:
            run.config.update(config.to_dict())
            logger.info("Initialized W&B tracking")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() and 
                         config.profiling.get("use_cuda", True) else "cpu")
    logger.info(f"Using device: {device}")
    
    # Step 1: Create model
    logger.info("Creating model...")
    model = ResNet18(num_classes=10)
    model = model.to(device)
    
    # Register model with TAuto registry
    register_model(
        name="resnet18",
        architecture="CNN",
        description="ResNet-18 for CIFAR classification",
        task="image_classification",
        model_cls=ResNet18,
        default_args={"num_classes": 10},
    )
    
    # Step 2: Create optimized data loaders
    logger.info("Creating optimized data loaders...")
    batch_size = config.training.get("batch_size", 32)
    train_loader = create_cifar_dataset(num_samples=1000, batch_size=batch_size, config=config)
    val_loader = create_cifar_dataset(num_samples=200, batch_size=batch_size, config=config)
    
    # Step 3: Profile original model
    profile_results = profile_original_model(model, val_loader, device, output_dir)
    
    # Step 4: Apply optimizations
    optimization_output = apply_optimizations(
        model=model,
        dataloader=val_loader,
        device=device,
        profile_results=profile_results,
        config=config,
        output_dir=output_dir,
    )
    
    optimized_models = optimization_output["models"]
    optimization_results = optimization_output["results"]
    
    # Step 5: Evaluate and compare models
    evaluation_results = evaluate_models(
        original_model=model,
        optimized_models=optimized_models,
        dataloader=val_loader,
        device=device,
    )
    
    # Print evaluation summary
    logger.info("\nModel Evaluation Summary:")
    logger.info(f"{'Model':<15} {'Accuracy':<10} {'Latency (ms)':<15} {'Throughput':<15} {'Speedup':<10}")
    logger.info("-" * 65)
    
    for name, result in evaluation_results.items():
        speedup = result.get("speedup", 1.0) if name != "original" else 1.0
        logger.info(f"{name:<15} {result['accuracy']:<10.4f} {result['latency_ms']:<15.2f} "
                  f"{result['throughput']:<15.2f} {speedup:<10.2f}")
    
    # Step 6: Generate reports
    generate_reports(
        original_model=model,
        optimized_models=optimized_models,
        evaluation_results=evaluation_results,
        optimization_results=optimization_results,
        profile_results=profile_results,
        config=config,
        output_dir=output_dir,
    )
    
    logger.info(f"\nOptimization pipeline completed successfully. Results saved to {output_dir}")


if __name__ == "__main__":
    import copy  # Import here to avoid potential issues with function definition order
    main()