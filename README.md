# TAuto: PyTorch Model Optimization Framework

TAuto is an AutoML optimization suite focused on performance optimization of PyTorch-based machine learning models. The framework provides automated tools for optimizing data pipelines, training processes, and inference performance.

## Features

- **Data Pipeline Optimization**: Improve data loading and preprocessing performance
- **Training Optimization**: Accelerate model training with mixed precision, gradient accumulation, and more
- **Inference Optimization**: Enhance inference speed with quantization, pruning, and knowledge distillation
- **Profiling**: Analyze model performance with memory and compute utilization profiling
- **Hyperparameter Optimization**: Automatically tune hyperparameters for optimal performance
- **Compiler Optimization**: Leverage PyTorch compilation features for further speedups

## Installation

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

```python
import tauto

# Load configuration
config = tauto.config.load_config("config.yaml")

# Optimize data loading
optimized_loader = tauto.data.create_optimized_loader(
    dataset, batch_size=32, num_workers=4
)

# Profile model performance
profiler = tauto.profile.Profiler(model)
profile_results = profiler.profile_training(optimized_loader)

# Apply optimizations
optimized_model = tauto.optimize.apply_optimizations(
    model, optimizations=["mixed_precision", "torch_compile"]
)

# Train with optimizations
trainer = tauto.optimize.Trainer(
    model=optimized_model,
    dataloader=optimized_loader,
    config=config
)
trainer.train()
```

# Training Optimization in TAuto

As part of TAuto's focus on high-performance machine learning, the framework now includes a suite of training optimization techniques in the `tauto.optimize` module. These optimizations can help you train models faster, with fewer resources, and with better results.

## Key Features

- **Mixed Precision Training**: Train with FP16 precision on supported hardware for up to 3x faster training
- **Gradient Accumulation**: Train with large effective batch sizes even on limited GPU memory
- **Model Checkpointing**: Save and resume training with best-practice checkpointing utilities
- **Optimizer Factory**: Create optimizers with recommended settings based on model architecture
- **Advanced Training Loop**: Simplify training with a feature-rich training loop that integrates all optimizations

## Usage Example

```python
import torch
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

For more detailed examples, see [examples/optimize_training.py](examples/optimize_training.py).

## Available Training Optimizations

### Mixed Precision Training
TAuto leverages PyTorch's Automatic Mixed Precision (AMP) for faster, memory-efficient training on compatible GPUs.

```python
from tauto.optimize.training import MixedPrecisionTraining

# Create a mixed precision trainer
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

### Gradient Accumulation
Accumulate gradients over multiple batches to simulate larger batch sizes without increasing memory usage.

```python
from tauto.optimize.training import GradientAccumulation

# Accumulate gradients over 4 batches
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

### Model Checkpointing
Save and load model checkpoints with smart features like keeping only the best N checkpoints.

```python
from tauto.optimize.training import ModelCheckpointing

# Create checkpointer that saves only when validation loss improves
checkpointer = ModelCheckpointing(
    checkpoint_dir="checkpoints",
    save_best_only=True,
    monitor="val_loss",
    mode="min"
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

## Documentation

For more detailed documentation and examples, see the [examples](examples/) and [notebooks](notebooks/) directories.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.