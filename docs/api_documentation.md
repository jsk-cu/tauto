# TAuto API Documentation

This document provides comprehensive documentation for the TAuto API, detailing each module and its functionality.

## Table of Contents

1. [Configuration Management](#configuration-management)
2. [Data Handling](#data-handling)
3. [Model Registry](#model-registry)
4. [Optimization](#optimization)
   - [Training Optimization](#training-optimization)
   - [Inference Optimization](#inference-optimization)
   - [Compiler Optimization](#compiler-optimization)
   - [Knowledge Distillation](#knowledge-distillation)
   - [Hyperparameter Optimization](#hyperparameter-optimization)
5. [Profiling](#profiling)
6. [Reporting](#reporting)
7. [Utilities](#utilities)

---

## Configuration Management

The `tauto.config` module provides tools for managing configuration settings across TAuto components.

### Core Functions and Classes

#### `get_default_config()`

Returns the default configuration for TAuto as a dictionary.

```python
from tauto.config import get_default_config

config = get_default_config()
```

#### `ConfigManager`

Manages configuration settings with methods for loading, saving, and updating configurations.

```python
from tauto.config import ConfigManager, get_default_config

# Create from default config
config_manager = ConfigManager()

# Update configuration
config_manager.update({
    "data": {"num_workers": 2},
    "training": {"batch_size": 16}
})

# Save configuration
config_manager.save("config.yaml")

# Load configuration
loaded_manager = ConfigManager.load("config.yaml")
```

#### `TAutoConfig`

A dataclass that provides a structured representation of a TAuto configuration.

```python
from tauto.config import TAutoConfig

# Access configuration sections
learning_rate = config.training["learning_rate"]
```

#### `load_config(path)`

Load a configuration from a YAML file.

```python
from tauto.config import load_config

config = load_config("config.yaml")
```

#### `save_config(config, path)`

Save a configuration to a YAML file.

```python
from tauto.config import save_config

save_config(config, "config.yaml")
```

#### `update_config(config, updates)`

Update a configuration with new values without modifying the original.

```python
from tauto.config import update_config

updated_config = update_config(config, {
    "training": {"batch_size": 64}
})
```

---

## Data Handling

The `tauto.data` module provides utilities for optimizing data loading and preprocessing pipelines.

### Core Functions and Classes

#### `create_optimized_loader(dataset, **kwargs)`

Create an optimized DataLoader with performance-tuned settings.

```python
from tauto.data import create_optimized_loader

loader = create_optimized_loader(
    dataset, 
    batch_size=32, 
    num_workers=4
)
```

#### `DataLoaderConfig`

Configuration class for optimized data loaders.

```python
from tauto.data import DataLoaderConfig

config = DataLoaderConfig(
    num_workers=4,
    prefetch_factor=2,
    pin_memory=True
)
```

#### `calculate_optimal_workers()`

Calculate the optimal number of worker processes based on system resources.

```python
from tauto.data import calculate_optimal_workers

num_workers = calculate_optimal_workers()
```

#### `DataDebugger`

Utility class for debugging and profiling data loading pipelines.

```python
from tauto.data import DataDebugger

debugger = DataDebugger(dataloader)
results = debugger.profile_loading(num_batches=10)
```

#### `create_transform_pipeline(transforms, **kwargs)`

Create an optimized data transformation pipeline.

```python
from tauto.data import create_transform_pipeline

transform = create_transform_pipeline([
    ('resize', (224, 224)),
    ('normalize', ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
])
```

#### `CachedTransform`

Transform wrapper that caches results to avoid redundant computations.

```python
from tauto.data import CachedTransform

cached_transform = CachedTransform(transform, cache_dir='.cache')
```

---

## Model Registry

The `tauto.models` module provides a registry for managing and creating PyTorch models.

### Core Functions and Classes

#### `register_model()`

Register a model with the global registry.

```python
from tauto.models import register_model

@register_model(
    name="resnet18",
    architecture="CNN",
    description="ResNet-18 for image classification",
    task="image_classification"
)
class ResNet18(nn.Module):
    # Model definition
    pass
```

#### `create_model()`

Create a model instance from the registry.

```python
from tauto.models import create_model

model = create_model("resnet18", pretrained=True)
```

#### `list_available_models()`

List all available models in the registry.

```python
from tauto.models import list_available_models

models = list_available_models()
```

#### `get_model_info()`

Get information about a registered model.

```python
from tauto.models import get_model_info

info = get_model_info("resnet18")
print(info.description)
```

#### `ModelRegistry`

Registry class for managing model registration and creation.

```python
from tauto.models import ModelRegistry

registry = ModelRegistry()
registry.register(model_info)
model = registry.create("resnet18")
```

---

## Optimization

The `tauto.optimize` module provides various optimization techniques for improving model training and inference.

### Training Optimization

The `tauto.optimize.training` module contains utilities for optimizing the training process.

#### `MixedPrecisionTraining`

Class for applying mixed precision training with PyTorch AMP.

```python
from tauto.optimize.training import MixedPrecisionTraining

mp_trainer = MixedPrecisionTraining(enabled=True)
metrics = mp_trainer.step(
    model=model,
    inputs=inputs,
    targets=targets,
    criterion=criterion,
    optimizer=optimizer
)
```

#### `GradientAccumulation`

Class for implementing gradient accumulation to enable larger effective batch sizes.

```python
from tauto.optimize.training import GradientAccumulation

accumulator = GradientAccumulation(accumulation_steps=4)
metrics = accumulator.step(
    model=model,
    inputs=inputs,
    targets=targets,
    criterion=criterion,
    optimizer=optimizer
)
```

#### `ModelCheckpointing`

Class for managing model checkpoints during training.

```python
from tauto.optimize.training import ModelCheckpointing

checkpointer = ModelCheckpointing(
    checkpoint_dir="checkpoints",
    save_best_only=True,
    monitor="val_loss"
)

# Save checkpoint
checkpointer.save_checkpoint(
    model=model,
    epoch=epoch,
    optimizer=optimizer,
    metrics={"val_loss": val_loss}
)

# Load best checkpoint
best_checkpoint = checkpointer.find_best_checkpoint()
checkpointer.load_checkpoint(
    model=model,
    checkpoint_path=best_checkpoint,
    optimizer=optimizer
)
```

#### `OptimizerFactory`

Factory class for creating optimizers with recommended settings.

```python
from tauto.optimize.training import OptimizerFactory

optimizer = OptimizerFactory.create_optimizer(
    model=model,
    optimizer_type="adam",
    learning_rate=0.001,
    weight_decay=1e-5
)

scheduler = OptimizerFactory.create_scheduler(
    optimizer=optimizer,
    scheduler_type="cosine",
    epochs=10
)
```

#### `train_with_optimization()`

Comprehensive training function that applies multiple optimizations.

```python
from tauto.optimize.training import train_with_optimization

metrics = train_with_optimization(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=10,
    device=torch.device("cuda"),
    mixed_precision=True,
    gradient_accumulation_steps=2,
    checkpoint_dir="checkpoints",
    early_stopping=True
)
```

### Inference Optimization

The `tauto.optimize.inference` module contains utilities for optimizing model inference.

#### `ModelQuantization`

Class for applying quantization to reduce model size and improve inference speed.

```python
from tauto.optimize.inference import ModelQuantization

quantizer = ModelQuantization(quantization_type="dynamic")
quantized_model = quantizer.quantize_model(model)

# Evaluate quantization impact
results = quantizer.evaluate_quantization(
    original_model=model,
    quantized_model=quantized_model,
    dataloader=val_loader
)
```

#### `ModelPruning`

Class for applying pruning to remove redundant weights.

```python
from tauto.optimize.inference import ModelPruning

pruner = ModelPruning(pruning_type="unstructured", criteria="l1")
pruned_model = pruner.prune_model(model, amount=0.5)

# Evaluate pruning impact
results = pruner.evaluate_pruning(
    original_model=model,
    pruned_model=pruned_model,
    dataloader=val_loader
)
```

#### `optimize_for_inference()`

Apply general optimizations for inference.

```python
from tauto.optimize.inference import optimize_for_inference

optimized_model = optimize_for_inference(
    model=model,
    optimizations=["trace", "fuse", "freeze"],
    example_inputs=example_inputs
)
```

#### `apply_inference_optimizations()`

Apply multiple inference optimizations with a single call.

```python
from tauto.optimize.inference import apply_inference_optimizations

optimized_model, results = apply_inference_optimizations(
    model=model,
    optimizations=["quantization", "pruning", "fuse"],
    dataloader=val_loader,
    config={
        "quantization": {"type": "dynamic"},
        "pruning": {"amount": 0.3}
    }
)
```

### Compiler Optimization

The `tauto.optimize.compiler` module contains utilities for applying compiler optimizations to PyTorch models.

#### `TorchCompile`

Class for applying torch.compile optimizations.

```python
from tauto.optimize.compiler import TorchCompile

compiler = TorchCompile(backend="inductor", mode="default")
compiled_model = compiler.compile_model(model, example_inputs=example_inputs)

# Benchmark compilation
results = compiler.benchmark_compilation(
    model=model,
    inputs=example_inputs,
    num_warmup=5,
    num_runs=20
)
```

#### `TorchScriptExport`

Class for exporting models to TorchScript.

```python
from tauto.optimize.compiler import TorchScriptExport

exporter = TorchScriptExport(method="trace", optimization_level=3)
ts_model = exporter.export_model(
    model=model,
    example_inputs=example_inputs,
    save_path="model.pt"
)

# Benchmark export
results = exporter.benchmark_export(
    model=model,
    inputs=example_inputs
)
```

#### `apply_compiler_optimization()`

Apply compiler optimizations with a single call.

```python
from tauto.optimize.compiler import apply_compiler_optimization

optimized_model = apply_compiler_optimization(
    model=model,
    optimization="torch_compile",
    example_inputs=example_inputs,
    backend="inductor",
    mode="default"
)
```

### Knowledge Distillation

The `tauto.optimize.distillation` module provides utilities for knowledge distillation.

#### `KnowledgeDistillation`

Class for applying knowledge distillation to transfer knowledge from a teacher to a student model.

```python
from tauto.optimize.distillation import KnowledgeDistillation

distiller = KnowledgeDistillation(
    teacher_model=teacher_model,
    student_model=student_model,
    temperature=2.0,
    alpha=0.5
)

# Train student with distillation
model, history = distiller.train(
    train_loader=train_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    epochs=10
)
```

#### `distill_model()`

Comprehensive function for distilling knowledge from a teacher to a student model.

```python
from tauto.optimize.distillation import distill_model

trained_student, metrics = distill_model(
    teacher_model=teacher_model,
    student_model=student_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    temperature=2.0,
    alpha=0.5,
    feature_layers={"layer3": "layer1"},
    feature_weight=0.1
)
```

### Hyperparameter Optimization

The `tauto.optimize.hyperparams` module provides utilities for hyperparameter optimization.

#### `HyperparameterOptimization`

Class for optimizing hyperparameters using Optuna.

```python
from tauto.optimize.hyperparams import HyperparameterOptimization

def objective(trial, params):
    # Train model with params
    return val_loss

optimizer = HyperparameterOptimization(
    search_space=search_space,
    objective_fn=objective,
    direction="minimize",
    search_algorithm="tpe",
    n_trials=100
)

results = optimizer.optimize()
```

#### `create_search_space()`

Create a search space definition for hyperparameter optimization.

```python
from tauto.optimize.hyperparams import create_search_space

search_space = create_search_space(
    parameter_ranges={
        "learning_rate": (1e-4, 1e-2, True),  # log scale
        "batch_size": (16, 128, 16),  # step size
    },
    categorical_params={
        "optimizer": ["sgd", "adam", "rmsprop"],
    },
    conditional_params={
        "momentum": ("optimizer", "sgd"),
    }
)
```

#### `optimize_hyperparameters()`

Comprehensive function for hyperparameter optimization.

```python
from tauto.optimize.hyperparams import optimize_hyperparameters

def train_fn(params):
    # Train model with params
    return model

def eval_fn(model):
    # Evaluate model
    return val_loss

results = optimize_hyperparameters(
    train_fn=train_fn,
    eval_fn=eval_fn,
    search_space=search_space,
    direction="minimize",
    n_trials=100
)
```

#### `PruningCallback`

Callback for early stopping and pruning during hyperparameter optimization.

```python
from tauto.optimize.hyperparams import PruningCallback

callback = PruningCallback(
    trial=trial,
    monitor="val_loss",
    direction="minimize"
)
```

---

## Profiling

The `tauto.profile` module provides tools for analyzing model performance.

### Core Functions and Classes

#### `Profiler`

Main profiler class for comprehensive performance analysis.

```python
from tauto.profile import Profiler, ProfilerConfig

config = ProfilerConfig(
    enabled=True,
    use_cuda=True,
    profile_memory=True
)

profiler = Profiler(model, config)
results = profiler.profile_training(
    dataloader=train_loader,
    criterion=criterion,
    optimizer=optimizer
)
```

#### `profile_model()`

Profile a model's performance with a single call.

```python
from tauto.profile import profile_model

results = profile_model(
    model=model,
    dataloader=dataloader,
    mode="all"  # "training", "inference", "memory"
)
```

#### `track_memory_usage()`

Track memory usage during model execution.

```python
from tauto.profile import track_memory_usage

memory_data = track_memory_usage(
    lambda: model(inputs),
    device=torch.device("cuda")
)
```

#### `measure_compute_utilization()`

Measure compute utilization during model execution.

```python
from tauto.profile import measure_compute_utilization

utilization = measure_compute_utilization(
    lambda: model(inputs),
    device=torch.device("cuda")
)
```

#### `estimate_model_memory()`

Estimate the memory usage of a model.

```python
from tauto.profile import estimate_model_memory

memory_usage = estimate_model_memory(
    model=model,
    input_size=(3, 224, 224)
)
```

#### `visualize_profile_results()`

Visualize profiling results.

```python
from tauto.profile import visualize_profile_results

plot_paths = visualize_profile_results(
    results=profile_results,
    output_dir="profile_visualizations"
)
```

#### `create_profile_report()`

Create a comprehensive HTML report of profiling results.

```python
from tauto.profile import create_profile_report

report_path = create_profile_report(
    results=profile_results,
    output_path="profile_report.html"
)
```

---

## Reporting

The `tauto.report` module provides tools for generating reports on optimization results.

### Core Functions and Classes

#### `generate_optimization_report()`

Generate a comprehensive HTML report on optimization results.

```python
from tauto.report import generate_optimization_report

report_path = generate_optimization_report(
    original_model=original_model,
    optimized_models=optimized_models,
    evaluation_results=evaluation_results,
    optimization_results=optimization_results,
    output_path="optimization_report.html"
)
```

#### `export_results_to_csv()`

Export evaluation results to a CSV file.

```python
from tauto.report import export_results_to_csv

csv_path = export_results_to_csv(
    evaluation_results=evaluation_results,
    output_path="results.csv"
)
```

#### `export_results_to_json()`

Export evaluation and optimization results to a JSON file.

```python
from tauto.report import export_results_to_json

json_path = export_results_to_json(
    evaluation_results=evaluation_results,
    optimization_results=optimization_results,
    output_path="results.json"
)
```

---

## Utilities

The `tauto.utils` module provides general utility functions used across TAuto.

### Core Functions and Classes

#### `get_logger()`

Get a logger with the specified name.

```python
from tauto.utils import get_logger

logger = get_logger("tauto.example")
logger.info("Example message")
```

#### `configure_logging()`

Configure the logging system.

```python
from tauto.utils import configure_logging

configure_logging({
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": "tauto.log"
})
```

#### `setup_wandb()`

Set up Weights & Biases for experiment tracking.

```python
from tauto.utils import setup_wandb

run = setup_wandb({
    "project": "tauto-demo",
    "name": "experiment-1",
    "tags": ["demo", "example"]
})
```

#### `log_config()`

Log configuration to Weights & Biases.

```python
from tauto.utils import log_config

log_config(config, run)
```

#### `log_model()`

Log a model to Weights & Biases.

```python
from tauto.utils import log_model

log_model("model.pt", run, "optimized_model")
```

#### `log_metrics()`

Log metrics to Weights & Biases.

```python
from tauto.utils import log_metrics

log_metrics({
    "loss": 0.1,
    "accuracy": 0.95
}, step=10)
```