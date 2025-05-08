# TAuto: PyTorch Model Optimization Framework

TAuto is an AutoML optimization suite focused on performance optimization of PyTorch-based machine learning models. The framework provides comprehensive tools for improving training speed, inference performance, and resource utilization.

## Project Overview

TAuto automates the application of numerous optimization techniques to PyTorch models, helping researchers and practitioners achieve better performance without deep expertise in optimization techniques. The framework addresses bottlenecks throughout the ML pipeline:

- **Data Pipeline Optimization**: Improve data loading and preprocessing efficiency
- **Training Optimization**: Accelerate model training with mixed precision, gradient accumulation, and more
- **Inference Optimization**: Enhance inference speed with quantization, pruning, and knowledge distillation
- **Profiling**: Analyze model performance with memory and compute utilization profiling
- **Hyperparameter Optimization**: Automatically tune hyperparameters for optimal performance
- **Compiler Optimization**: Leverage PyTorch compilation features for further speedups

## Repository Structure

```
tauto/
├── tauto/                      # Main package
│   ├── __init__.py
│   ├── config/                 # Configuration management
│   │   ├── __init__.py
│   │   ├── defaults.py         # Default configurations
│   │   └── config_manager.py   # Configuration manager
│   ├── data/                   # Data handling components
│   │   ├── __init__.py
│   │   └── loader.py           # Optimized data loaders
│   ├── models/                 # Model definitions and registry
│   │   ├── __init__.py
│   │   ├── registry.py         # Model registry
│   │   └── zoo/                # Pre-configured models
│   ├── optimize/               # Optimization techniques
│   │   ├── __init__.py
│   │   ├── training.py         # Training optimizations
│   │   ├── inference.py        # Inference optimizations
│   │   ├── distillation.py     # Knowledge distillation
│   │   ├── compiler.py         # Compiler optimizations
│   │   └── hyperparams.py      # Hyperparameter optimization
│   ├── profile/                # Profiling utilities
│   │   ├── __init__.py
│   │   ├── profiler.py         # Main profiler implementation
│   │   ├── memory.py           # Memory profiling
│   │   ├── compute.py          # Compute utilization profiling
│   │   └── visualization.py    # Profiling visualizations
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── wandb_utils.py      # W&B integration utilities
│       └── logging.py          # Logging utilities
├── examples/                   # Example usage scripts
│   ├── basic_usage.py          # Basic framework usage
│   ├── model_registry_usage.py # Model registry examples
│   ├── optimize_training.py    # Training optimization examples
│   ├── optimize_inference.py   # Inference optimization examples
│   ├── optimize_compiler.py    # Compiler optimization examples
│   └── optimize_hyperparams.py # Hyperparameter optimization examples
├── notebooks/                  # Jupyter notebooks
│   ├── getting_started.ipynb   # Getting started tutorial
├── tests/                      # Unit tests
├── setup.py                    # Package setup script
└── requirements.txt            # Dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/jsk-cu/tauto.git
cd tauto

# Install the package
pip install -e .

# For development dependencies
pip install -e ".[dev]"
```

## Basic Usage

Here's a simple example of how to use TAuto:

```python
import tauto

# Load configuration
config = tauto.config.get_default_config()

# Create a model from the registry
model = tauto.models.create_model("resnet18", pretrained=True)

# Profile model performance
profiler = tauto.profile.Profiler(model)
profile_results = profiler.profile_inference(dataloader)

# Apply optimization techniques
optimized_model = tauto.optimize.apply_inference_optimizations(
    model=model,
    optimizations=["quantization", "pruning", "fuse", "freeze"],
    dataloader=dataloader,
    config={
        "quantization": {"type": "dynamic"},
        "pruning": {"amount": 0.3}
    }
)
```

## Feature Examples

### Data Pipeline Optimization

```python
from tauto.data import create_optimized_loader

# Create an optimized data loader
optimized_loader = create_optimized_loader(
    dataset=dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)
```

### Training Optimization

```python
from tauto.optimize.training import train_with_optimization

# Train with all optimizations enabled
metrics = train_with_optimization(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    epochs=10,
    device=torch.device("cuda"),
    mixed_precision=True,
    gradient_accumulation_steps=2,
    clip_grad_norm=1.0,
    checkpoint_dir="checkpoints",
    early_stopping=True
)
```

### Inference Optimization

```python
from tauto.optimize import apply_inference_optimizations

# Apply multiple optimizations at once
optimized_model, results = apply_inference_optimizations(
    model=model,
    optimizations=["quantization", "pruning", "fuse", "freeze"],
    dataloader=val_loader,
    config={
        "quantization": {"type": "dynamic"},
        "pruning": {"amount": 0.3}
    }
)
```

### Compiler Optimization

```python
from tauto.optimize.compiler import TorchCompile

# Use torch.compile for faster execution
compiler = TorchCompile(backend="inductor", mode="max-autotune")
compiled_model = compiler.compile_model(model, example_inputs)

# Benchmark the speedup
results = compiler.benchmark_compilation(
    model=model,
    inputs=example_inputs,
    num_warmup=5,
    num_runs=20
)
print(f"Speedup: {results['speedup']:.2f}x")
```

### Knowledge Distillation

```python
from tauto.optimize import distill_model

# Distill knowledge from large model to small model
student_model = distill_model(
    teacher_model=large_model,
    student_model=small_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    temperature=2.0,
    alpha=0.5
)
```

### Hyperparameter Optimization

```python
from tauto.optimize.hyperparams import optimize_hyperparameters, create_search_space

# Define search space
search_space = create_search_space(
    parameter_ranges={
        'learning_rate': (1e-4, 1e-2, True),  # log scale
        'batch_size': (16, 128, 16),          # step 16
        'hidden_dim': (64, 512, 64),          # step 64
    },
    categorical_params={
        'optimizer': ['sgd', 'adam', 'adamw'],
        'activation': ['relu', 'leaky_relu', 'elu'],
    }
)

# Run optimization
results = optimize_hyperparameters(
    train_fn=train_model_fn,
    eval_fn=evaluate_model_fn,
    search_space=search_space,
    direction='minimize',
    n_trials=50
)
```

### Profiling

```python
from tauto.profile import Profiler, ProfilerConfig

# Configure the profiler
config = ProfilerConfig(
    enabled=True,
    use_cuda=True,
    profile_memory=True,
    record_shapes=True
)

# Profile model
profiler = Profiler(model, config)

# Profile training
train_profile = profiler.profile_training(
    dataloader=train_loader,
    criterion=criterion,
    optimizer=optimizer
)

# Profile inference
inference_profile = profiler.profile_inference(
    dataloader=val_loader,
    num_steps=100
)

# Visualize results
from tauto.profile.visualization import create_profile_report
report_path = create_profile_report(
    results={"training": train_profile, "inference": inference_profile},
    output_path="profile_report.html"
)
```

## Results and Observations

TAuto can significantly improve model performance across different dimensions. Here are some example improvements observed in our tests:

| Optimization Technique | Speed Improvement | Memory Reduction | Accuracy Impact |
|------------------------|-------------------|------------------|-----------------|
| Dynamic Quantization   | 2-4x              | 75%              | Minimal         |
| Post-Training Pruning  | 1.2-2x            | 50% (varies)     | Moderate        |
| Knowledge Distillation | 3-10x             | 70-95%           | Minimal to moderate |
| Torch.compile          | 1.5-3x            | Minimal          | None            |
| Combined               | 4-15x             | 80-95%           | Varies          |

### Example Visualization

When running the examples, you'll get visualization plots showing the performance improvements:

- `optimization_comparison.png`: Comparison of different optimization techniques
- `torch_compile_comparison.png`: Comparison of different compiler backends
- `torchscript_comparison.png`: Comparison of TorchScript optimization levels
- `optimization_history.png`: Hyperparameter optimization history
- `param_importances.png`: Importance of different hyperparameters

## Running the Examples

The `examples/` directory contains several example scripts demonstrating different aspects of TAuto:

```bash
# Basic usage example
python examples/basic_usage.py

# Model registry example
python examples/model_registry_usage.py

# Training optimization example
python examples/optimize_training.py

# Inference optimization example
python examples/optimize_inference.py

# Compiler optimization example
python examples/optimize_compiler.py

# Hyperparameter optimization example
python examples/optimize_hyperparams.py

# Profiling example
python examples/profiling_usage.py
```

## Running Tests

To run the test suite:

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_profiler.py

# Run with coverage
pytest --cov=tauto tests/
```

## Future Work

- Integration with more deployment targets (ONNX, TorchServe, TensorRT)
- More sophisticated autotuning for operator fusion and kernel selection
- Support for distributed training optimization
- Reinforcement learning-based optimization strategy search
- Enhanced visualization and reporting capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.