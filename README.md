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

## Documentation

For more detailed documentation and examples, see the [examples](examples/) and [notebooks](notebooks/) directories.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.