"""
Example usage of the TAuto compiler optimization utilities.

This script demonstrates how to use the compiler optimization utilities to:
1. Apply torch.compile to a model
2. Export a model to TorchScript 
3. Compare performance between different compilation techniques
4. Debug compilation issues
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

from tauto.optimize.compiler import (
    TorchCompile,
    TorchScriptExport,
    apply_compiler_optimization,
)
from tauto.profile import Profiler


class ResNet18(nn.Module):
    """
    Simple implementation of ResNet-18 model for demonstration.
    """
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


class SimpleCNN(nn.Module):
    """A simple CNN model for comparison."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_cifar_dataset(num_samples=1000, batch_size=32):
    """Create a synthetic CIFAR-like dataset."""
    # Generate random 32x32 images and random labels
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (num_samples,))
    
    # Create dataset and data loader
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def benchmark_model(model, dataloader, num_warmup=5, num_runs=20, device=None):
    """Benchmark a model's inference performance."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Warm up
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_warmup:
                break
            
            inputs = inputs.to(device)
            _ = model(inputs)
    
    # Benchmark
    latencies = []
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_runs:
                break
            
            inputs = inputs.to(device)
            
            # Measure inference time
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()
            
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
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


def example_torch_compile():
    """Example of using torch.compile."""
    print("\n1. Torch.compile Optimization Example")
    print("----------------------------------")
    
    # Skip if torch.compile is not available
    if not hasattr(torch, "compile"):
        print("torch.compile is not available in your PyTorch version. Skipping this example.")
        return
    
    # Create a model and data
    print("Creating ResNet18 model and CIFAR-like dataset...")
    model = ResNet18(num_classes=10)
    dataloader = create_cifar_dataset(num_samples=1000, batch_size=32)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get a sample batch
    example_inputs, _ = next(iter(dataloader))
    example_inputs = example_inputs.to(device)
    
    # Benchmark the original model
    model = model.to(device)
    print("\nBenchmarking original model...")
    baseline_results = benchmark_model(model, dataloader, device=device)
    print(f"Original Model - Avg Latency: {baseline_results['avg_latency_ms']:.2f} ms, "
          f"Throughput: {baseline_results['throughput_samples_per_sec']:.2f} samples/sec")
    
    # Apply torch.compile with default settings
    print("\nApplying torch.compile with 'inductor' backend...")
    compiler = TorchCompile(backend="inductor", mode="default")
    compiled_model = compiler.compile_model(model, example_inputs=example_inputs)
    
    # Benchmark the compiled model
    print("Benchmarking compiled model...")
    compiled_results = benchmark_model(compiled_model, dataloader, device=device)
    print(f"Compiled Model - Avg Latency: {compiled_results['avg_latency_ms']:.2f} ms, "
          f"Throughput: {compiled_results['throughput_samples_per_sec']:.2f} samples/sec")
    
    # Calculate speedup
    speedup = baseline_results['avg_latency_ms'] / compiled_results['avg_latency_ms']
    print(f"Speedup with torch.compile: {speedup:.2f}x")
    
    # Try different backends
    backends = ["inductor", "aot_eager", "aot_ts"]
    backend_results = {}
    
    print("\nComparing different backends:")
    for backend in backends:
        try:
            print(f"Testing backend: {backend}")
            compiler = TorchCompile(backend=backend)
            compiled_model = compiler.compile_model(model, example_inputs=example_inputs)
            
            # Benchmark
            results = benchmark_model(compiled_model, dataloader, num_runs=10, device=device)
            backend_results[backend] = results
            
            # Print results
            print(f"  Backend: {backend} - Avg Latency: {results['avg_latency_ms']:.2f} ms, "
                  f"Speedup: {baseline_results['avg_latency_ms'] / results['avg_latency_ms']:.2f}x")
            
        except Exception as e:
            print(f"  Error with backend {backend}: {e}")
    
    # Try different compilation modes
    modes = ["default", "reduce-overhead", "max-autotune"]
    mode_results = {}
    
    print("\nComparing different compilation modes with inductor backend:")
    for mode in modes:
        try:
            print(f"Testing mode: {mode}")
            compiler = TorchCompile(backend="inductor", mode=mode)
            compiled_model = compiler.compile_model(model, example_inputs=example_inputs)
            
            # Benchmark
            results = benchmark_model(compiled_model, dataloader, num_runs=10, device=device)
            mode_results[mode] = results
            
            # Print results
            print(f"  Mode: {mode} - Avg Latency: {results['avg_latency_ms']:.2f} ms, "
                  f"Speedup: {baseline_results['avg_latency_ms'] / results['avg_latency_ms']:.2f}x")
            
        except Exception as e:
            print(f"  Error with mode {mode}: {e}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Backend comparison
    plt.subplot(1, 2, 1)
    if backend_results:
        backends = list(backend_results.keys())
        latencies = [backend_results[b]['avg_latency_ms'] for b in backends]
        baseline = baseline_results['avg_latency_ms']
        
        # Plot
        plt.bar(['Original'] + backends, [baseline] + latencies)
        plt.ylabel('Inference Latency (ms)')
        plt.title('Inference Latency by Backend')
        plt.xticks(rotation=45)
    
    # Mode comparison
    plt.subplot(1, 2, 2)
    if mode_results:
        modes = list(mode_results.keys())
        latencies = [mode_results[m]['avg_latency_ms'] for m in modes]
        baseline = baseline_results['avg_latency_ms']
        
        # Plot
        plt.bar(['Original'] + modes, [baseline] + latencies)
        plt.ylabel('Inference Latency (ms)')
        plt.title('Inference Latency by Compilation Mode')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("torch_compile_comparison.png")
    print("Saved comparison plot to 'torch_compile_comparison.png'")
    
    # Debug model compilation
    print("\nDebugging model compilation:")
    debug_info = TorchCompile.debug_model(model, example_inputs, backend="inductor", verbose=True)
    
    if debug_info["success"]:
        print("Model can be successfully compiled with inductor backend")
    else:
        print(f"Compilation issue: {debug_info['error']}")
        if "alternative_backend" in debug_info:
            print(f"Alternative backend {debug_info['alternative_backend']} succeeded")


def example_torchscript():
    """Example of using TorchScript export."""
    print("\n2. TorchScript Export Example")
    print("---------------------------")
    
    # Create a model and data
    print("Creating SimpleCNN model and CIFAR-like dataset...")
    model = SimpleCNN(num_classes=10)
    dataloader = create_cifar_dataset(num_samples=1000, batch_size=32)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get a sample batch
    example_inputs, _ = next(iter(dataloader))
    example_inputs = example_inputs.to(device)
    
    # Benchmark the original model
    model = model.to(device)
    print("\nBenchmarking original model...")
    baseline_results = benchmark_model(model, dataloader, device=device)
    print(f"Original Model - Avg Latency: {baseline_results['avg_latency_ms']:.2f} ms, "
          f"Throughput: {baseline_results['throughput_samples_per_sec']:.2f} samples/sec")
    
    # Export to TorchScript with tracing
    print("\nExporting model to TorchScript with tracing...")
    exporter = TorchScriptExport(method="trace", optimization_level=3)
    
    # Create a directory for exported models
    os.makedirs("exported_models", exist_ok=True)
    ts_path = Path("exported_models") / "traced_model.pt"
    
    # Export the model
    ts_model = exporter.export_model(model, example_inputs=example_inputs, save_path=ts_path)
    
    # Benchmark the TorchScript model
    print("Benchmarking TorchScript model...")
    ts_results = benchmark_model(ts_model, dataloader, device=device)
    print(f"TorchScript Model - Avg Latency: {ts_results['avg_latency_ms']:.2f} ms, "
          f"Throughput: {ts_results['throughput_samples_per_sec']:.2f} samples/sec")
    
    # Calculate speedup
    speedup = baseline_results['avg_latency_ms'] / ts_results['avg_latency_ms']
    print(f"Speedup with TorchScript tracing: {speedup:.2f}x")
    
    # Try scripting instead of tracing
    try:
        print("\nExporting model to TorchScript with scripting...")
        script_exporter = TorchScriptExport(method="script", optimization_level=3)
        script_path = Path("exported_models") / "scripted_model.pt"
        script_model = script_exporter.export_model(model, save_path=script_path)
        
        # Benchmark the scripted model
        print("Benchmarking scripted model...")
        script_results = benchmark_model(script_model, dataloader, device=device)
        print(f"Scripted Model - Avg Latency: {script_results['avg_latency_ms']:.2f} ms, "
              f"Throughput: {script_results['throughput_samples_per_sec']:.2f} samples/sec")
        
        # Calculate speedup
        script_speedup = baseline_results['avg_latency_ms'] / script_results['avg_latency_ms']
        print(f"Speedup with TorchScript scripting: {script_speedup:.2f}x")
        
    except Exception as e:
        print(f"Error during scripting: {e}")
        print("Scripting may not be supported for this model. Continuing with traced model.")
        script_results = None
    
    # Compare different optimization levels
    optimization_levels = [0, 1, 2, 3]
    opt_results = {}
    
    print("\nComparing different optimization levels:")
    for level in optimization_levels:
        try:
            print(f"Testing optimization level: {level}")
            opt_exporter = TorchScriptExport(method="trace", optimization_level=level)
            opt_path = Path("exported_models") / f"traced_model_opt{level}.pt"
            
            # Export and benchmark
            opt_model = opt_exporter.export_model(model, example_inputs=example_inputs, save_path=opt_path)
            results = benchmark_model(opt_model, dataloader, num_runs=10, device=device)
            opt_results[level] = results
            
            # Print results
            print(f"  Level {level} - Avg Latency: {results['avg_latency_ms']:.2f} ms, "
                  f"Speedup: {baseline_results['avg_latency_ms'] / results['avg_latency_ms']:.2f}x")
            
        except Exception as e:
            print(f"  Error with optimization level {level}: {e}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Optimization level comparison
    plt.subplot(1, 2, 1)
    if opt_results:
        levels = list(opt_results.keys())
        latencies = [opt_results[l]['avg_latency_ms'] for l in levels]
        baseline = baseline_results['avg_latency_ms']
        
        # Plot
        plt.bar(['Original'] + [f"Level {l}" for l in levels], [baseline] + latencies)
        plt.ylabel('Inference Latency (ms)')
        plt.title('Inference Latency by Optimization Level')
        plt.xticks(rotation=45)
    
    # Method comparison
    plt.subplot(1, 2, 2)
    methods = ['Original', 'Traced']
    latencies = [baseline_results['avg_latency_ms'], ts_results['avg_latency_ms']]
    
    if script_results:
        methods.append('Scripted')
        latencies.append(script_results['avg_latency_ms'])
    
    # Plot
    plt.bar(methods, latencies)
    plt.ylabel('Inference Latency (ms)')
    plt.title('Inference Latency by TorchScript Method')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("torchscript_comparison.png")
    print("Saved comparison plot to 'torchscript_comparison.png'")


def example_combined_optimizations():
    """Example of combining compiler optimizations with other techniques."""
    print("\n3. Combined Optimizations Example")
    print("-------------------------------")
    
    # Create a model and data
    print("Creating ResNet18 model and CIFAR-like dataset...")
    model = ResNet18(num_classes=10)
    dataloader = create_cifar_dataset(num_samples=1000, batch_size=32)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get a sample batch
    example_inputs, _ = next(iter(dataloader))
    example_inputs = example_inputs.to(device)
    
    # Benchmark the original model
    model = model.to(device)
    print("\nBenchmarking original model...")
    baseline_results = benchmark_model(model, dataloader, device=device)
    print(f"Original Model - Avg Latency: {baseline_results['avg_latency_ms']:.2f} ms, "
          f"Throughput: {baseline_results['throughput_samples_per_sec']:.2f} samples/sec")
    
    # Define optimization combinations to test
    optimizations = [
        ["torch_compile"],
        ["torchscript"],
    ]
    
    # Profiler available?
    has_profiler = True
    try:
        from tauto.profile import Profiler, ProfilerConfig
    except ImportError:
        has_profiler = False
        print("Profiler not available, skipping detailed profiling")
    
    # Apply and benchmark each optimization
    results = {}
    models = {}
    
    for opt in optimizations:
        name = " + ".join(opt)
        print(f"\nApplying optimization: {name}")
        
        try:
            # Apply the optimization
            if "torch_compile" in opt and hasattr(torch, "compile"):
                # For torch.compile, we want to keep the original model type
                opt_model = apply_compiler_optimization(
                    model=model,
                    optimization="torch_compile",
                    example_inputs=example_inputs,
                    backend="inductor",
                    mode="default",
                )
            elif "torchscript" in opt:
                # For TorchScript, we get a ScriptModule
                os.makedirs("exported_models", exist_ok=True)
                save_path = Path("exported_models") / f"{name.replace(' ', '_')}.pt"
                
                opt_model = apply_compiler_optimization(
                    model=model,
                    optimization="torchscript",
                    example_inputs=example_inputs,
                    save_path=save_path,
                    optimization_level=3,
                )
            else:
                print(f"Unknown optimization: {opt}")
                continue
            
            # Save the model
            models[name] = opt_model
            
            # Benchmark
            print(f"Benchmarking {name}...")
            opt_results = benchmark_model(opt_model, dataloader, device=device)
            
            # Calculate speedup
            speedup = baseline_results['avg_latency_ms'] / opt_results['avg_latency_ms']
            
            # Show results
            print(f"  {name} - Avg Latency: {opt_results['avg_latency_ms']:.2f} ms, "
                  f"Throughput: {opt_results['throughput_samples_per_sec']:.2f} samples/sec, "
                  f"Speedup: {speedup:.2f}x")
            
            # Store results
            results[name] = {
                "latency_ms": opt_results['avg_latency_ms'],
                "throughput": opt_results['throughput_samples_per_sec'],
                "speedup": speedup,
            }
            
            # Profile with Profiler if available
            if has_profiler:
                try:
                    print(f"Profiling {name}...")
                    config = ProfilerConfig(
                        enabled=True,
                        use_cuda=(device.type == "cuda"),
                        profile_memory=True,
                        profile_dir="profile_results",
                    )
                    
                    profiler = Profiler(opt_model, config)
                    profile_result = profiler.profile_inference(
                        dataloader=dataloader,
                        num_steps=5,
                        name=f"{name}_inference",
                    )
                    
                    # Show profiling results
                    print(f"  {name} - Profile:")
                    print(f"    Avg Latency: {profile_result.duration_ms.get('per_batch', 0):.2f} ms")
                    print(f"    Memory Usage: {profile_result.memory_usage.get('peak_mb', 0):.2f} MB")
                    
                except Exception as e:
                    print(f"Error during profiling: {e}")
            
        except Exception as e:
            print(f"Error applying {name}: {e}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Latency comparison
    plt.subplot(1, 2, 1)
    if results:
        names = ['Original'] + list(results.keys())
        latencies = [baseline_results['avg_latency_ms']] + [results[n]['latency_ms'] for n in results]
        
        # Plot
        plt.bar(names, latencies)
        plt.ylabel('Inference Latency (ms)')
        plt.title('Inference Latency by Optimization')
        plt.xticks(rotation=45)
    
    # Speedup comparison
    plt.subplot(1, 2, 2)
    if results:
        names = list(results.keys())
        speedups = [results[n]['speedup'] for n in results]
        
        # Plot
        plt.bar(names, speedups)
        plt.ylabel('Speedup Factor')
        plt.title('Speedup by Optimization')
        plt.axhline(y=1.0, color='r', linestyle='-')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("combined_optimizations.png")
    print("Saved comparison plot to 'combined_optimizations.png'")
    
    # Print summary
    print("\nOptimization Summary:")
    print(f"{'Optimization':<20} {'Latency (ms)':<15} {'Throughput':<15} {'Speedup':<10}")
    print("-" * 60)
    print(f"{'Original':<20} {baseline_results['avg_latency_ms']:<15.2f} "
          f"{baseline_results['throughput_samples_per_sec']:<15.2f} {'1.00':<10}")
    
    for name, result in results.items():
        print(f"{name:<20} {result['latency_ms']:<15.2f} {result['throughput']:<15.2f} "
              f"{result['speedup']:<10.2f}")


def main():
    print("TAuto Compiler Optimization Examples")
    print("===================================")
    
    # Run examples
    example_torch_compile()
    example_torchscript()
    example_combined_optimizations()
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main()