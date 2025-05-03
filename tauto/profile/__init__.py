"""
Profiling utilities for TAuto.

This module provides tools for analyzing the performance of models,
including memory usage, compute utilization, and execution time.
"""

from tauto.profile.profiler import (
    Profiler,
    ProfilerConfig,
    ProfileResult,
    profile_model,
)

from tauto.profile.memory import (
    track_memory_usage,
    measure_peak_memory,
    estimate_model_memory,
)

from tauto.profile.compute import (
    measure_compute_utilization,
    measure_flops,
    estimate_device_capabilities,
)

from tauto.profile.visualization import (
    visualize_profile_results,
    create_profile_report,
    plot_memory_usage,
    plot_compute_utilization,
    plot_training_performance,
    plot_inference_performance,
)

__all__ = [
    "Profiler",
    "ProfilerConfig",
    "ProfileResult",
    "profile_model",
    "track_memory_usage",
    "measure_peak_memory",
    "estimate_model_memory",
    "measure_compute_utilization",
    "measure_flops",
    "estimate_device_capabilities",
    "visualize_profile_results",
    "create_profile_report",
    "plot_memory_usage",
    "plot_compute_utilization",
    "plot_training_performance",
    "plot_inference_performance",
]