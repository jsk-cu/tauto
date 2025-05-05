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

# Inference Optimization in TAuto

As part of TAuto's comprehensive machine learning optimization framework, we've implemented a robust suite of inference optimization techniques in Phase 5. These optimizations can significantly improve inference speed, reduce memory footprint, and enable deployment on resource-constrained devices.

## Key Features

- **Model Quantization**: Reduce model size and improve inference speed with various quantization techniques
- **Model Pruning**: Remove redundant weights to create sparse models with minimal accuracy impact
- **Knowledge Distillation**: Transfer knowledge from large teacher models to compact student models
- **General Inference Optimizations**: Apply PyTorch optimizations like fusion and freezing for faster inference

## Quantization

TAuto supports multiple quantization approaches:

```python
from tauto.optimize import ModelQuantization

# Dynamic quantization (post-training, fastest to apply)
quantizer = ModelQuantization(quantization_type="dynamic")
quantized_model = quantizer.quantize_model(model)

# Static quantization (requires calibration)
quantizer = ModelQuantization(quantization_type="static")
prepared_model = quantizer.prepare_model(model)
calibrated_model = quantizer.calibrate_model(prepared_model, calibration_dataloader)
quantized_model = quantizer.quantize_model(calibrated_model)

# Evaluate quantization impact
results = quantizer.evaluate_quantization(
    original_model=model,
    quantized_model=quantized_model,
    dataloader=val_loader
)

# Check speedup and memory reduction
print(f"Speedup: {results['speedup']:.2f}x")
print(f"Memory reduction: {results['memory_reduction'] * 100:.2f}%")
```

## Pruning

TAuto implements various pruning techniques to reduce model complexity:

```python
from tauto.optimize import ModelPruning

# Apply unstructured pruning (L1 norm based)
pruner = ModelPruning(pruning_type="unstructured", criteria="l1")
pruned_model = pruner.prune_model(model, amount=0.5)  # Remove 50% of weights

# Apply structured pruning (channel pruning)
pruner = ModelPruning(pruning_type="structured", criteria="l1")
pruned_model = pruner.prune_model(model, amount=0.3)

# Get sparsity information
sparsity_info = pruner.get_model_sparsity(pruned_model)
print(f"Overall sparsity: {sparsity_info['overall_sparsity'] * 100:.2f}%")

# Apply iterative pruning with fine-tuning
def finetune_fn(model):
    # Custom fine-tuning logic
    return fine_tuned_model

pruned_model = pruner.iterative_pruning(
    model=model,
    train_fn=finetune_fn,
    initial_amount=0.2,
    final_amount=0.8,
    steps=5
)
```

## Knowledge Distillation

Transfer knowledge from large, accurate models to smaller, faster ones:

```python
from tauto.optimize import distill_model

# Basic distillation
student_model = distill_model(
    teacher_model=large_model,
    student_model=small_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    temperature=2.0,
    alpha=0.5
)

# With feature distillation (intermediate layers)
feature_layers = {
    "layer3": "layer1",  # Map teacher's layer3 to student's layer1
    "fc1": "fc1",
}

student_model = distill_model(
    teacher_model=large_model,
    student_model=small_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    temperature=2.0,
    alpha=0.5,
    feature_layers=feature_layers,
    feature_weight=0.1
)
```

## General Inference Optimizations

Apply PyTorch-specific optimizations:

```python
from tauto.optimize import optimize_for_inference

# Apply general optimizations
optimized_model = optimize_for_inference(
    model=model,
    optimizations=["trace", "fuse", "freeze"],
    example_inputs=example_inputs
)

# Or use a single function to apply multiple optimizations
from tauto.optimize import apply_inference_optimizations

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

## Examples

For detailed examples, check out the `examples/optimize_inference.py` file, which demonstrates:

1. Applying quantization to a model
2. Applying pruning to a model
3. Using knowledge distillation to transfer knowledge from a large model to a small one
4. Applying general optimizations for inference
5. Benchmarking and comparing different optimization methods

## Benchmarks

Depending on your model architecture and dataset, you can expect these approximate improvements:

| Optimization Technique | Speed Improvement | Memory Reduction | Accuracy Impact |
|------------------------|-------------------|------------------|-----------------|
| Dynamic Quantization   | 2-4x              | 75%              | Minimal         |
| Post-Training Pruning  | 1.2-2x            | 50% (varies)     | Moderate        |
| Knowledge Distillation | 3-10x             | 70-95%           | Minimal to moderate |
| General Optimizations  | 1.2-1.5x          | Minimal          | None            |
| Combined               | 4-15x             | 80-95%           | Varies          |

## Next Steps

Once you've optimized your model for inference, you may want to:

1. Export it to a different format (ONNX, TorchScript, etc.)
2. Deploy it to a specific target device
3. Integrate it with a serving framework

These capabilities will be added in future phases of the TAuto project.

## Documentation

For more detailed documentation and examples, see the [examples](examples/) and [notebooks](notebooks/) directories.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.