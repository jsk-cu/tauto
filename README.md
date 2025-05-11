# HPML Project: TAuto - PyTorch Model Optimization Framework

## Team Information
- **Team Name**: Null Pointer Exception
- **Members**:
  - Jonathan S. Kent (JSK2272)
  - Simay Cural (SC5559)

---

## 1. Problem Statement
TAuto addresses the challenge of optimizing PyTorch-based machine learning models for improved performance, reduced memory usage, and faster training/inference. While PyTorch offers many optimization techniques, applying them correctly requires specialized knowledge and significant manual effort. TAuto automates this process by providing an AutoML optimization suite that analyzes models and applies appropriate optimizations with minimal user intervention.

---

## 2. Model Description
TAuto is not a specific model but rather a framework for optimizing PyTorch models. It supports:

- **Framework**: PyTorch 2.0+
- **Optimization Techniques**:
  - Mixed precision training
  - Gradient accumulation
  - Quantization (dynamic, static, QAT)
  - Pruning (structured and unstructured)
  - Knowledge distillation
  - Compiler optimizations (torch.compile and TorchScript)
  - Hyperparameter optimization
- **Profiling Tools**: Memory, compute utilization, and execution time analysis
- **Model Registry**: Pre-configured models for common tasks

---

## 3. Final Results Summary

Since TAuto is a framework rather than a specific model implementation, traditional metrics like accuracy don't directly apply. Please see the provided W&B report for some example results. Performance varies based on model architecture, dataset, and hardware.

---

## 4. Reproducibility Instructions

### A. Requirements

Install dependencies:
```bash
# Clone the repository
git clone https://github.com/jsk-cu/tauto.git
cd tauto

# Install the package
pip install -e .

# For development dependencies
pip install -e ".[dev]"
```

---

### B. Wandb Dashboard

TAuto includes built-in Weights & Biases integration. Configure in your project with:

```python
from tauto.utils import setup_wandb

run = setup_wandb({
    "project": "tauto-demo",
    "name": "experiment-1",
    "tags": ["optimization"]
})
```

---

### C. Specify for Training or For Inference or if Both 

TAuto supports both training and inference optimizations:

**Training Optimization**:
```python
from tauto.optimize.training import train_with_optimization

metrics = train_with_optimization(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    mixed_precision=True,
    gradient_accumulation_steps=2
)
```

**Inference Optimization**:
```python
from tauto.optimize.inference import apply_inference_optimizations

optimized_model, results = apply_inference_optimizations(
    model=model,
    optimizations=["quantization", "pruning"],
    dataloader=val_loader
)
```

---

### D. Evaluation

To evaluate optimization improvements:

```python
from tauto.profile import Profiler

# Profile original model
profiler = Profiler(model)
original_results = profiler.profile_inference(dataloader)

# Profile optimized model
profiler = Profiler(optimized_model)
optimized_results = profiler.profile_inference(dataloader)

# Generate report
from tauto.report import generate_optimization_report

report_path = generate_optimization_report(
    original_model=model,
    optimized_models={"optimized": optimized_model},
    evaluation_results={
        "original": original_results,
        "optimized": optimized_results
    },
    optimization_results=results
)
```

---

### E. Quickstart: Minimum Reproducible Result

To demonstrate TAuto's basic functionality with quantization optimization:

```python
import torch
import torch.nn as nn
import tauto

# Create a simple model
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 2)
)

# Create dummy data
x = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))
dataset = torch.utils.data.TensorDataset(x, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

# Profile original model
from tauto.profile import Profiler
profiler = Profiler(model)
original_results = profiler.profile_inference(dataloader)

# Apply quantization
from tauto.optimize.inference import ModelQuantization
quantizer = ModelQuantization(quantization_type="dynamic")
quantized_model = quantizer.quantize_model(model)

# Evaluate improvement
results = quantizer.evaluate_quantization(
    original_model=model,
    quantized_model=quantized_model,
    dataloader=dataloader
)

print(f"Speedup: {results['speedup']:.2f}x")
print(f"Memory reduction: {results['memory_reduction']*100:.2f}%")
```

---

## 5. Notes
- TAuto is designed as a framework, so it doesn't produce fixed results like a specific model would
- Performance improvements vary based on model architecture, dataset, and hardware
- The framework includes extensive documentation in the `docs/` directory
- Example notebooks demonstrating various optimization techniques are in the `notebooks/` directory
- For complex end-to-end optimization workflows, refer to the examples in `examples/` directory
- All source code is located in the `tauto/` package directory