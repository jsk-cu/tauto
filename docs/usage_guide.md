# TAuto Usage Guide

This guide provides detailed instructions on how to use the TAuto framework for optimizing PyTorch-based machine learning models.

## Table of Contents

1. [Installation](#installation)
2. [Getting Started](#getting-started)
3. [Configuration Management](#configuration-management)
4. [Data Optimization](#data-optimization)
5. [Model Registry](#model-registry)
6. [Profiling](#profiling)
7. [Optimization Techniques](#optimization-techniques)
   - [Training Optimization](#training-optimization)
   - [Inference Optimization](#inference-optimization)
   - [Knowledge Distillation](#knowledge-distillation)
   - [Hyperparameter Optimization](#hyperparameter-optimization)
   - [Compiler Optimization](#compiler-optimization)
8. [End-to-End Workflows](#end-to-end-workflows)
9. [Integration with Weights & Biases](#integration-with-weights--biases)
10. [Best Practices](#best-practices)

---

## Installation

### Prerequisites

- Python 3.8 or later
- PyTorch 2.0 or later

### Installing TAuto

You can install TAuto directly from the repository:

```bash
git clone https://github.com/jsk-cu/tauto.git
cd tauto
pip install -e .
```

For development purposes, install with development dependencies:

```bash
pip install -e ".[dev]"
```

### Verifying Installation

To verify that TAuto is installed correctly:

```python
import tauto
print(tauto.__version__)
```

## Getting Started

### Basic Example

Let's start with a simple example of using TAuto to optimize a PyTorch model:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tauto

# Create a simple model
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 2)
)

# Create some dummy data
x = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32)

# Load default configuration
config = tauto.config.get_default_config()

# Profile the model
profiler = tauto.profile.Profiler(model)
results = profiler.profile_inference(dataloader)

# Apply optimization
optimized_model = tauto.optimize.inference.optimize_for_inference(
    model, optimizations=["trace", "fuse"]
)

# Save the optimized model
torch.save(optimized_model, "optimized_model.pt")
```

## Configuration Management

TAuto uses a configuration system to manage settings across different components. The configuration is hierarchical and can be loaded from YAML files.

### Default Configuration

To get the default configuration:

```python
from tauto.config import get_default_config

config = get_default_config()
```

### Loading and Saving Configurations

```python
from tauto.config import load_config, save_config, ConfigManager

# Load from file
config = load_config("config.yaml")

# Save to file
save_config(config, "new_config.yaml")

# Using ConfigManager
manager = ConfigManager()
manager.update({"training": {"batch_size": 64}})
manager.save("updated_config.yaml")
```

### Configuration Sections

The main configuration sections are:

1. **data**: Settings for data loading and preprocessing
2. **training**: Settings for model training
3. **optimization**: Settings for various optimization techniques
4. **profiling**: Settings for performance profiling
5. **wandb**: Settings for Weights & Biases integration
6. **logging**: Settings for logging

Example of a complete configuration:

```yaml
data:
  num_workers: 4
  prefetch_factor: 2
  pin_memory: true
  persistent_workers: true
  drop_last: false
  cache_dir: .tauto_cache
  use_cache: true

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: adam
  scheduler: cosine
  grad_accumulation_steps: 1
  mixed_precision: true
  gradient_clip_val: 1.0
  checkpoint_interval: 1
  checkpoint_dir: checkpoints
  early_stopping:
    enabled: true
    patience: 5
    min_delta: 0.0001
    metric: val_loss
    mode: min

optimization:
  torch_compile:
    enabled: true
    backend: inductor
    mode: max-autotune
  quantization:
    enabled: false
    precision: int8
    approach: post_training
  pruning:
    enabled: false
    method: magnitude
    sparsity: 0.5
  distillation:
    enabled: false
    teacher_model: null
    temperature: 2.0
    alpha: 0.5

profiling:
  enabled: true
  use_cuda: true
  profile_memory: true
  record_shapes: true
  with_stack: false
  profile_dir: .tauto_profile

wandb:
  enabled: true
  project: tauto
  entity: null
  name: null
  tags:
  - tauto
  log_code: true
  log_artifacts: true

logging:
  level: INFO
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  log_file: null
```

## Data Optimization

TAuto provides tools for optimizing data loading and preprocessing pipelines.

### Optimized Data Loaders

```python
from tauto.data import create_optimized_loader

# Create an optimized data loader
loader = create_optimized_loader(
    dataset,
    batch_size=32,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=True
)
```

### Optimized Transformations

```python
from tauto.data import create_transform_pipeline, CachedTransform

# Create an optimized transform pipeline
transform = create_transform_pipeline([
    ('resize', (224, 224)),
    ('normalize', ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
])

# Cache transform results
cached_transform = CachedTransform(transform, cache_dir='.cache')
```

## Model Registry

TAuto includes a model registry that helps manage model creation and sharing.

### Registering Models

```python
from tauto.models import register_model

@register_model(
    name="simple_cnn",
    architecture="CNN",
    description="Simple CNN for image classification",
    task="image_classification"
)
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 14 * 14, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x
```

### Creating Models from Registry

```python
from tauto.models import create_model, list_available_models

# List available models
models = list_available_models()
print(models)

# Create a model from registry
model = create_model("simple_cnn", in_channels=1, num_classes=10)
```

## Profiling

TAuto provides comprehensive profiling tools to analyze model performance.

### Basic Profiling

```python
from tauto.profile import Profiler, ProfilerConfig

# Configure profiler
config = ProfilerConfig(
    enabled=True,
    use_cuda=True,
    profile_memory=True,
    record_shapes=True
)

# Create profiler and profile model
profiler = Profiler(model, config)
results = profiler.profile_inference(dataloader, num_steps=10)

# Print memory usage
print(f"Memory usage: {results.memory_usage.get('peak_mb', 0):.2f} MB")
print(f"Avg inference time: {results.duration_ms.get('per_batch', 0):.2f} ms")
```

### Comprehensive Profiling

```python
from tauto.profile import profile_model, create_profile_report

# Profile model in all modes
results = profile_model(
    model=model,
    dataloader=dataloader,
    mode="all"  # "training", "inference", "memory"
)

# Generate HTML report
report_path = create_profile_report(
    results=results,
    output_path="profile_report.html"
)
```

### Memory Analysis

```python
from tauto.profile import estimate_model_memory

# Estimate model memory usage
memory_info = estimate_model_memory(
    model=model,
    input_size=(3, 224, 224)
)

print(f"Parameters: {memory_info['params_mb']:.2f} MB")
print(f"Activations: {memory_info['activations_mb']:.2f} MB")
print(f"Total: {memory_info['total_mb']:.2f} MB")
```

## Optimization Techniques

TAuto offers various optimization techniques to improve model performance.

### Training Optimization

#### Mixed Precision Training

```python
from tauto.optimize.training import MixedPrecisionTraining

# Create trainer with mixed precision
mp_trainer = MixedPrecisionTraining(enabled=True)

# Use in training loop
for inputs, targets in dataloader:
    metrics = mp_trainer.step(
        model=model,
        inputs=inputs,
        targets=targets,
        criterion=criterion,
        optimizer=optimizer
    )
```

#### Gradient Accumulation

```python
from tauto.optimize.training import GradientAccumulation

# Create accumulator for effective batch size of 128 (32 * 4)
accumulator = GradientAccumulation(accumulation_steps=4)

# Use in training loop
for inputs, targets in dataloader:
    metrics = accumulator.step(
        model=model,
        inputs=inputs,
        targets=targets,
        criterion=criterion,
        optimizer=optimizer
    )
```

#### Smart Checkpointing

```python
from tauto.optimize.training import ModelCheckpointing

# Create checkpointer that saves best models only
checkpointer = ModelCheckpointing(
    checkpoint_dir="checkpoints",
    save_best_only=True,
    monitor="val_loss",
    mode="min"
)

# In training loop
val_loss = 0.1  # Computed validation loss
checkpointer.save_checkpoint(
    model=model,
    epoch=epoch,
    optimizer=optimizer,
    metrics={"val_loss": val_loss}
)

# Load best checkpoint after training
best_checkpoint = checkpointer.find_best_checkpoint()
loaded_data = checkpointer.load_checkpoint(
    model=model,
    checkpoint_path=best_checkpoint,
    optimizer=optimizer
)
```

#### Optimizer Factory

```python
from tauto.optimize.training import OptimizerFactory

# Create optimizer with recommended settings
optimizer = OptimizerFactory.create_optimizer(
    model=model,
    optimizer_type="adam",
    learning_rate=0.001,
    weight_decay=1e-5
)

# Create learning rate scheduler
scheduler = OptimizerFactory.create_scheduler(
    optimizer=optimizer,
    scheduler_type="cosine",
    epochs=10
)
```

#### Complete Training Pipeline

```python
from tauto.optimize.training import train_with_optimization

# Train with multiple optimizations
metrics = train_with_optimization(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=10,
    device=torch.device("cuda"),
    mixed_precision=True,
    gradient_accumulation_steps=2,
    clip_grad_norm=1.0,
    checkpoint_dir="checkpoints",
    early_stopping=True,
    early_stopping_patience=5
)
```

### Inference Optimization

#### Quantization

```python
from tauto.optimize.inference import ModelQuantization

# Dynamic quantization
quantizer = ModelQuantization(quantization_type="dynamic")
quantized_model = quantizer.quantize_model(model)

# Evaluate quantization impact
results = quantizer.evaluate_quantization(
    original_model=model,
    quantized_model=quantized_model,
    dataloader=val_loader
)

print(f"Speedup: {results['speedup']:.2f}x")
print(f"Memory reduction: {results['memory_reduction'] * 100:.2f}%")
```

#### Pruning

```python
from tauto.optimize.inference import ModelPruning

# Unstructured pruning
pruner = ModelPruning(pruning_type="unstructured", criteria="l1")
pruned_model = pruner.prune_model(model, amount=0.5)

# Evaluate pruning impact
results = pruner.evaluate_pruning(
    original_model=model,
    pruned_model=pruned_model,
    dataloader=val_loader
)

print(f"Sparsity: {results['sparsity'] * 100:.2f}%")
print(f"Speedup: {results['speedup']:.2f}x")
```

#### Fusing, Freezing, and Tracing

```python
from tauto.optimize.inference import optimize_for_inference

# Apply multiple optimizations
# Get example inputs for tracing
example_inputs = next(iter(dataloader))[0]

optimized_model = optimize_for_inference(
    model=model,
    optimizations=["trace", "fuse", "freeze"],
    example_inputs=example_inputs
)
```

#### Combined Optimizations

```python
from tauto.optimize.inference import apply_inference_optimizations

# Apply multiple inference optimizations
optimized_model, results = apply_inference_optimizations(
    model=model,
    optimizations=["quantization", "pruning", "fuse"],
    dataloader=val_loader,
    example_inputs=example_inputs,
    config={
        "quantization": {"type": "dynamic"},
        "pruning": {"amount": 0.3}
    }
)
```

### Knowledge Distillation

```python
from tauto.optimize.distillation import distill_model

# Create teacher and student models
teacher_model = create_model("resnet50", pretrained=True)
student_model = create_model("resnet18")

# Apply knowledge distillation
trained_student, metrics = distill_model(
    teacher_model=teacher_model,
    student_model=student_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    temperature=2.0,
    alpha=0.5,
    feature_layers={"layer4": "layer4"},  # Map teacher to student layers
    feature_weight=0.1
)
```

### Hyperparameter Optimization

```python
from tauto.optimize.hyperparams import create_search_space, optimize_hyperparameters

# Define parameter search space
search_space = create_search_space(
    parameter_ranges={
        "learning_rate": (1e-4, 1e-2, True),  # log scale
        "batch_size": (16, 128, 16),  # step size
        "weight_decay": (1e-5, 1e-3),
    },
    categorical_params={
        "optimizer": ["sgd", "adam", "rmsprop"],
        "activation": ["relu", "leaky_relu", "elu"],
    }
)

# Define model training and evaluation functions
def train_fn(params):
    # Train model with params
    model = create_model("resnet18")
    # ... configure and train model using params ...
    return model

def eval_fn(model):
    # Evaluate model
    # ... evaluate model ...
    return val_loss

# Run hyperparameter optimization
results = optimize_hyperparameters(
    train_fn=train_fn,
    eval_fn=eval_fn,
    search_space=search_space,
    direction="minimize",
    n_trials=100
)

# Get best model and parameters
best_model = results["best_model"]
best_params = results["best_params"]
```

### Compiler Optimization

```python
from tauto.optimize.compiler import TorchCompile, TorchScriptExport

# Apply torch.compile optimization
example_inputs = next(iter(dataloader))[0]

# Using torch.compile
compiler = TorchCompile(backend="inductor", mode="default")
compiled_model = compiler.compile_model(model, example_inputs=example_inputs)

# Benchmark the compiled model
results = compiler.benchmark_compilation(
    model=model,
    inputs=example_inputs
)
print(f"Speedup: {results['speedup']:.2f}x")

# Export to TorchScript
exporter = TorchScriptExport(method="trace", optimization_level=3)
ts_model = exporter.export_model(
    model=model,
    example_inputs=example_inputs,
    save_path="model.pt"
)
```

## End-to-End Workflows

TAuto provides complete end-to-end workflows that combine multiple optimization techniques.

### Complete Optimization Pipeline

The following example demonstrates a complete optimization pipeline:

```python
import torch
import torch.nn as nn
from pathlib import Path
import tauto

# Load configuration
config = tauto.config.load_config("config.yaml")

# Configure logging
tauto.utils.configure_logging(config.logging)
logger = tauto.utils.get_logger("tauto.example")

# Set up W&B tracking
run = tauto.utils.setup_wandb(config.wandb)
if run:
    tauto.utils.log_config(config.to_dict(), run)

# Create model and data loaders
model = tauto.models.create_model("resnet18", num_classes=10)
train_loader = create_dataloader(train_dataset, config)
val_loader = create_dataloader(val_dataset, config)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Step 1: Profile original model
logger.info("Profiling original model...")
profiler = tauto.profile.Profiler(model, config.profiling)
profile_results = profiler.profile_training(train_loader)
profile_report = tauto.profile.create_profile_report(
    results={"training": profile_results},
    output_path="profile_report.html"
)

# Step 2: Train with optimizations
logger.info("Training model with optimizations...")
optimizer = tauto.optimize.training.OptimizerFactory.create_optimizer(
    model=model,
    optimizer_type=config.training.optimizer,
    learning_rate=config.training.learning_rate,
    weight_decay=config.training.weight_decay
)

scheduler = tauto.optimize.training.OptimizerFactory.create_scheduler(
    optimizer=optimizer,
    scheduler_type=config.training.scheduler,
    epochs=config.training.epochs
)

criterion = nn.CrossEntropyLoss()

# Train with optimizations
metrics = tauto.optimize.training.train_with_optimization(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=config.training.epochs,
    device=device,
    mixed_precision=config.training.mixed_precision,
    gradient_accumulation_steps=config.training.grad_accumulation_steps,
    clip_grad_norm=config.training.gradient_clip_val,
    checkpoint_dir=config.training.checkpoint_dir,
    early_stopping=config.training.early_stopping.enabled,
    early_stopping_patience=config.training.early_stopping.patience
)

# Step 3: Apply inference optimizations
logger.info("Applying inference optimizations...")
example_inputs = next(iter(val_loader))[0].to(device)

# Determine which optimizations to apply
optimizations = []
if config.optimization.quantization.enabled:
    optimizations.append("quantization")
if config.optimization.pruning.enabled:
    optimizations.append("pruning")
if config.optimization.torch_compile.enabled:
    optimizations.append("trace")

# Apply optimizations
optimization_config = {
    "quantization": {
        "type": config.optimization.quantization.approach,
    },
    "pruning": {
        "amount": config.optimization.pruning.sparsity,
    },
}

optimized_model, opt_results = tauto.optimize.inference.apply_inference_optimizations(
    model=model,
    optimizations=optimizations,
    dataloader=val_loader,
    device=device,
    example_inputs=example_inputs,
    config=optimization_config
)

# Step 4: Evaluate models
logger.info("Evaluating models...")
original_metrics = evaluate_model(model, val_loader, device)
optimized_metrics = evaluate_model(optimized_model, val_loader, device)

evaluation_results = {
    "original": original_metrics,
    "optimized": optimized_metrics
}

# Step 5: Generate reports
logger.info("Generating reports...")
report_path = tauto.report.generate_optimization_report(
    original_model=model,
    optimized_models={"optimized": optimized_model},
    evaluation_results=evaluation_results,
    optimization_results=opt_results,
    output_path="optimization_report.html"
)

# Export results to JSON
json_path = tauto.report.export_results_to_json(
    evaluation_results=evaluation_results,
    optimization_results=opt_results,
    output_path="results.json"
)

# Step 6: Save models
logger.info("Saving models...")
torch.save(model, "models/original_model.pt")
torch.save(optimized_model, "models/optimized_model.pt")

# Log to W&B if enabled
if run:
    run.log({"original_accuracy": original_metrics["accuracy"]})
    run.log({"optimized_accuracy": optimized_metrics["accuracy"]})
    run.log({"speedup": optimized_metrics["latency"] / original_metrics["latency"]})
    run.log_artifact("models/optimized_model.pt", name="optimized_model")

logger.info("Optimization pipeline completed successfully!")
```

## Integration with Weights & Biases

TAuto provides built-in integration with Weights & Biases for experiment tracking.

### Setting Up W&B

```python
from tauto.utils import setup_wandb, log_config, log_metrics

# Configure W&B
wandb_config = {
    "project": "tauto-demo",
    "name": "experiment-1",
    "tags": ["optimization", "quantization"]
}

# Initialize W&B run
run = setup_wandb(wandb_config)

# Log configuration
if run:
    log_config(config.to_dict(), run)

# Log metrics during training
if run:
    log_metrics({
        "train_loss": 0.1,
        "train_accuracy": 0.95,
        "val_loss": 0.2,
        "val_accuracy": 0.9
    }, step=epoch)

# Log model
if run:
    log_model("models/optimized_model.pt", run, "optimized_model")
```

## Best Practices

### Configuration Management

1. **Start with default configuration**: Begin with the default configuration and modify only the settings you need to change.
2. **Save configurations**: Always save your configurations to make experiments reproducible.
3. **Use hierarchical organization**: Organize complex configurations hierarchically for better readability.

### Data Optimization

1. **Optimize workers**: Use `calculate_optimal_workers()` to determine the optimal number of worker processes.
2. **Enable pinned memory**: Use pinned memory when transferring data to GPU.
3. **Use persistent workers**: Enable persistent workers to avoid the overhead of starting and stopping workers.

### Profiling

1. **Profile before optimizing**: Always profile your model before applying optimizations to identify bottlenecks.
2. **Profile different aspects**: Profile both training and inference to get a complete picture of performance.
3. **Analyze memory usage**: Pay attention to memory usage, especially for large models.

### Training Optimization

1. **Use mixed precision**: Enable mixed precision training for faster training on compatible GPUs.
2. **Gradient accumulation for large batches**: Use gradient accumulation to train with larger effective batch sizes.
3. **Smart checkpointing**: Save only the best models to save disk space.

### Inference Optimization

1. **Start with simple optimizations**: Begin with tracing and fusing before moving to more complex optimizations like quantization.
2. **Evaluate accuracy impact**: Always evaluate the impact of optimizations on model accuracy.
3. **Combine optimizations**: Combine multiple optimization techniques for maximum performance.

### Deployment

1. **Export to TorchScript**: Export optimized models to TorchScript for deployment.
2. **Optimize for target hardware**: Use quantization for memory-constrained devices.
3. **Benchmark on target platform**: Always benchmark optimized models on the target deployment platform.