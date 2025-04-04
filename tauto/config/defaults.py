"""
Default configurations for TAuto.
"""

from typing import Dict, Any


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration for TAuto.
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        "data": {
            "num_workers": 4,
            "prefetch_factor": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "drop_last": False,
            "cache_dir": ".tauto_cache",
            "use_cache": True,
        },
        "training": {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "optimizer": "adam",
            "scheduler": "cosine",
            "grad_accumulation_steps": 1,
            "mixed_precision": True,
            "gradient_clip_val": 1.0,
            "checkpoint_interval": 1,
            "checkpoint_dir": "checkpoints",
            "early_stopping": {
                "enabled": True,
                "patience": 5,
                "min_delta": 0.0001,
                "metric": "val_loss",
                "mode": "min",
            },
        },
        "optimization": {
            "torch_compile": {
                "enabled": True,
                "backend": "inductor",
                "mode": "max-autotune",
            },
            "quantization": {
                "enabled": False,
                "precision": "int8",
                "approach": "post_training",
            },
            "pruning": {
                "enabled": False,
                "method": "magnitude",
                "sparsity": 0.5,
            },
            "distillation": {
                "enabled": False,
                "teacher_model": None,
                "temperature": 2.0,
                "alpha": 0.5,
            },
        },
        "profiling": {
            "enabled": True,
            "use_cuda": True,
            "profile_memory": True,
            "record_shapes": True,
            "with_stack": False,
            "profile_dir": ".tauto_profile",
        },
        "wandb": {
            "enabled": True,
            "project": "tauto",
            "entity": None,
            "name": None,
            "tags": ["tauto"],
            "log_code": True,
            "log_artifacts": True,
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "log_file": None,
        },
    }