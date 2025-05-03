"""
Visualization utilities for profiling results.

This module provides tools for visualizing and reporting profiling results,
including memory usage, compute utilization, and execution time.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import json
import tempfile
import contextlib
import numpy as np

from tauto.utils import get_logger
from tauto.profile.profiler import ProfileResult

logger = get_logger(__name__)

# Try to import matplotlib, but provide fallbacks if not available
try:
    import matplotlib
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Limited visualization functionality will be provided.")

# Try to import pandas, but provide fallbacks if not available
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False
    logger.warning("pandas not available. Limited data processing functionality will be provided.")


def visualize_profile_results(
    results: Dict[str, ProfileResult],
    output_dir: Optional[Union[str, Path]] = None,
    save_format: str = "png",
) -> Dict[str, Path]:
    """
    Visualize profiling results and save plots to files.
    
    Args:
        results: Dictionary of profile results
        output_dir: Directory to save plots (defaults to profile result directory)
        save_format: Format to save plots in ('png', 'pdf', 'svg')
        
    Returns:
        Dict[str, Path]: Paths to saved plots
    """
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot visualize results: matplotlib not available")
        return {}
    
    # Determine output directory
    if output_dir is None:
        # Try to use the directory from the first profile result
        if results and list(results.values())[0].traces:
            output_dir = Path(list(results.values())[0].traces[0]).parent
        else:
            output_dir = Path(".tauto_profile_viz")
    
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots
    plot_paths = {}
    
    # Plot duration comparison if multiple results
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract duration data
        names = []
        durations = []
        
        for name, result in results.items():
            total_duration = result.duration_ms.get("total", 0)
            if total_duration > 0:
                names.append(name)
                durations.append(total_duration)
        
        if names and durations:
            # Create bar plot
            ax.bar(names, durations)
            ax.set_ylabel("Duration (ms)")
            ax.set_title("Execution Duration Comparison")
            
            # Rotate x labels for better readability
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            # Save plot
            path = output_dir / f"duration_comparison.{save_format}"
            plt.savefig(path)
            plt.close(fig)
            
            plot_paths["duration_comparison"] = path
    
    # Plot memory comparison if multiple results
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract memory data
        names = []
        cpu_memory = []
        gpu_memory = []
        
        for name, result in results.items():
            if "cpu_total" in result.memory_usage:
                names.append(name)
                cpu_memory.append(result.memory_usage["cpu_total"])
                
                if "cuda_total" in result.memory_usage:
                    gpu_memory.append(result.memory_usage["cuda_total"])
                else:
                    gpu_memory.append(0)
        
        if names and (cpu_memory or gpu_memory):
            # Set width for bars
            width = 0.35
            x = np.arange(len(names))
            
            # Create grouped bar plot
            if cpu_memory:
                ax.bar(x - width/2, cpu_memory, width, label="CPU Memory")
            
            if gpu_memory and any(gpu_memory):
                ax.bar(x + width/2, gpu_memory, width, label="GPU Memory")
            
            ax.set_ylabel("Memory Usage (MB)")
            ax.set_title("Memory Usage Comparison")
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            ax.legend()
            
            # Rotate x labels for better readability
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            # Save plot
            path = output_dir / f"memory_comparison.{save_format}"
            plt.savefig(path)
            plt.close(fig)
            
            plot_paths["memory_comparison"] = path
    
    # Create individual plots for each result
    for name, result in results.items():
        # Plot memory timeline if available
        if result.raw_data and "memory_timeline" in result.raw_data:
            memory_timeline = result.raw_data["memory_timeline"]
            
            if "timestamps" in memory_timeline and "cpu_memory_mb" in memory_timeline:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.plot(memory_timeline["timestamps"], memory_timeline["cpu_memory_mb"], label="CPU Memory")
                
                if "gpu_memory_mb" in memory_timeline and memory_timeline["gpu_memory_mb"]:
                    ax.plot(memory_timeline["timestamps"], memory_timeline["gpu_memory_mb"], label="GPU Memory")
                
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Memory Usage (MB)")
                ax.set_title(f"Memory Usage Timeline - {name}")
                ax.legend()
                
                plt.tight_layout()
                
                # Save plot
                path = output_dir / f"{name}_memory_timeline.{save_format}"
                plt.savefig(path)
                plt.close(fig)
                
                plot_paths[f"{name}_memory_timeline"] = path
        
        # Plot operator breakdown if available
        if result.duration_ms:
            # Filter out the total duration and get top N operators
            op_durations = {k: v for k, v in result.duration_ms.items() if k.startswith("op.")}
            
            if op_durations:
                # Sort by duration (descending)
                sorted_ops = sorted(op_durations.items(), key=lambda x: x[1], reverse=True)
                
                # Take top 10 operators
                top_ops = sorted_ops[:10]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create horizontal bar plot
                y_pos = np.arange(len(top_ops))
                op_names = [op[0].replace("op.", "") for op in top_ops]
                op_times = [op[1] for op in top_ops]
                
                ax.barh(y_pos, op_times)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(op_names)
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_xlabel("Duration (ms)")
                ax.set_title(f"Top Operators by Duration - {name}")
                
                plt.tight_layout()
                
                # Save plot
                path = output_dir / f"{name}_op_breakdown.{save_format}"
                plt.savefig(path)
                plt.close(fig)
                
                plot_paths[f"{name}_op_breakdown"] = path
    
    logger.info(f"Saved {len(plot_paths)} plots to {output_dir}")
    return plot_paths


def create_profile_report(
    results: Dict[str, ProfileResult],
    output_path: Optional[Union[str, Path]] = None,
    include_plots: bool = True,
    plot_format: str = "png",
) -> Path:
    """
    Create a comprehensive HTML report of profiling results.
    
    Args:
        results: Dictionary of profile results
        output_path: Path to save the report (defaults to auto-generated path)
        include_plots: Whether to include plots in the report
        plot_format: Format for included plots
        
    Returns:
        Path: Path to the saved report
    """
    # Determine output path
    if output_path is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = Path(f"tauto_profile_report_{timestamp}.html")
    
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Generate plots if requested
    plot_paths = {}
    if include_plots and _MATPLOTLIB_AVAILABLE:
        output_dir = output_path.parent / "plots"
        plot_paths = visualize_profile_results(results, output_dir, plot_format)
    
    # Start building the HTML report
    html_content = []
    
    # Add header
    html_content.append("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TAuto Profiling Report</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }
            h1, h2, h3 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            tr:hover { background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .plot-container { margin: 20px 0; text-align: center; }
            .plot-container img { max-width: 100%; height: auto; }
            .section { margin-bottom: 30px; }
            .summary-box { background-color: #f8f9fa; border-radius: 4px; padding: 15px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>TAuto Profiling Report</h1>
            <p>Generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
    """)
    
    # Add summary section
    html_content.append("""
            <div class="section">
                <h2>Summary</h2>
                <div class="summary-box">
                    <p><strong>Number of profiles:</strong> """ + str(len(results)) + """</p>
    """)
    
    # Add device information from the first result
    if results:
        first_result = list(results.values())[0]
        html_content.append(f"""
                    <p><strong>Device:</strong> {first_result.device}</p>
        """)
    
    html_content.append("""
                </div>
            </div>
    """)
    
    # Add comparison plots if available
    if "duration_comparison" in plot_paths or "memory_comparison" in plot_paths:
        html_content.append("""
            <div class="section">
                <h2>Comparison</h2>
        """)
        
        if "duration_comparison" in plot_paths:
            html_content.append(f"""
                <div class="plot-container">
                    <h3>Duration Comparison</h3>
                    <img src="{plot_paths['duration_comparison'].relative_to(output_path.parent)}" alt="Duration Comparison">
                </div>
            """)
        
        if "memory_comparison" in plot_paths:
            html_content.append(f"""
                <div class="plot-container">
                    <h3>Memory Usage Comparison</h3>
                    <img src="{plot_paths['memory_comparison'].relative_to(output_path.parent)}" alt="Memory Usage Comparison">
                </div>
            """)
        
        html_content.append("""
            </div>
        """)
    
    # Add detailed sections for each profile result
    for name, result in results.items():
        html_content.append(f"""
            <div class="section">
                <h2>Profile: {name}</h2>
                
                <h3>Duration</h3>
                <table>
                    <tr><th>Metric</th><th>Value (ms)</th></tr>
        """)
        
        # Add duration metrics
        for metric, value in result.duration_ms.items():
            if not metric.startswith("op."):  # Skip operator-specific metrics for the main table
                html_content.append(f"""
                    <tr><td>{metric}</td><td>{value:.2f}</td></tr>
                """)
        
        html_content.append("""
                </table>
                
                <h3>Memory Usage</h3>
                <table>
                    <tr><th>Metric</th><th>Value (MB)</th></tr>
        """)
        
        # Add memory metrics
        for metric, value in result.memory_usage.items():
            if not metric.startswith("cpu.") and not metric.startswith("cuda."):  # Skip detailed metrics
                html_content.append(f"""
                    <tr><td>{metric}</td><td>{value:.2f}</td></tr>
                """)
        
        html_content.append("""
                </table>
        """)
        
        # Add parameters if available
        if result.parameters:
            html_content.append("""
                <h3>Model Parameters</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
            """)
            
            for metric, value in result.parameters.items():
                if metric.endswith("_mb"):
                    html_content.append(f"""
                        <tr><td>{metric}</td><td>{value:.2f} MB</td></tr>
                    """)
                else:
                    html_content.append(f"""
                        <tr><td>{metric}</td><td>{value:,}</td></tr>
                    """)
            
            html_content.append("""
                </table>
            """)
        
        # Add compute utilization if available
        if result.compute_utilization:
            html_content.append("""
                <h3>Compute Utilization</h3>
                <table>
                    <tr><th>Metric</th><th>Value (%)</th></tr>
            """)
            
            for metric, value in result.compute_utilization.items():
                html_content.append(f"""
                    <tr><td>{metric}</td><td>{value:.2f}</td></tr>
                """)
            
            html_content.append("""
                </table>
            """)
        
        # Add FLOPS information if available
        if result.flops:
            html_content.append("""
                <h3>FLOPs</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
            """)
            
            for metric, value in result.flops.items():
                if value > 1e9:
                    html_content.append(f"""
                        <tr><td>{metric}</td><td>{value/1e9:.2f} GFLOPs</td></tr>
                    """)
                elif value > 1e6:
                    html_content.append(f"""
                        <tr><td>{metric}</td><td>{value/1e6:.2f} MFLOPs</td></tr>
                    """)
                else:
                    html_content.append(f"""
                        <tr><td>{metric}</td><td>{value:.2f} FLOPs</td></tr>
                    """)
            
            html_content.append("""
                </table>
            """)
        
        # Add top operators table
        op_durations = {k: v for k, v in result.duration_ms.items() if k.startswith("op.")}
        if op_durations:
            html_content.append("""
                <h3>Top Operators</h3>
                <table>
                    <tr><th>Operator</th><th>Duration (ms)</th><th>Percentage</th></tr>
            """)
            
            # Calculate total time for percentage
            total_time = result.duration_ms.get("total", sum(op_durations.values()))
            
            # Sort by duration (descending)
            sorted_ops = sorted(op_durations.items(), key=lambda x: x[1], reverse=True)
            
            # Take top 20 operators
            top_ops = sorted_ops[:20]
            
            for op_name, op_time in top_ops:
                # Remove the "op." prefix
                clean_name = op_name.replace("op.", "")
                percentage = (op_time / total_time) * 100 if total_time > 0 else 0
                
                html_content.append(f"""
                    <tr><td>{clean_name}</td><td>{op_time:.2f}</td><td>{percentage:.2f}%</td></tr>
                """)
            
            html_content.append("""
                </table>
            """)
        
        # Add plots if available
        result_plots = {k: v for k, v in plot_paths.items() if k.startswith(f"{name}_")}
        if result_plots:
            html_content.append("""
                <h3>Visualizations</h3>
            """)
            
            for plot_name, plot_path in result_plots.items():
                html_content.append(f"""
                <div class="plot-container">
                    <img src="{plot_path.relative_to(output_path.parent)}" alt="{plot_name}">
                </div>
                """)
        
        # Close the section
        html_content.append("""
            </div>
        """)
    
    # Add footer
    html_content.append("""
        </div>
    </body>
    </html>
    """)
    
    # Write to file
    with open(output_path, "w") as f:
        f.write("".join(html_content))
    
    logger.info(f"Saved profile report to {output_path}")
    return output_path


def plot_memory_usage(
    result: ProfileResult,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    """
    Plot memory usage from a profile result.
    
    Args:
        result: Profile result containing memory data
        output_path: Path to save the plot
        show: Whether to show the plot (in addition to saving)
        
    Returns:
        Optional[Path]: Path to the saved plot, or None if not saved
    """
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot plot memory usage: matplotlib not available")
        return None
    
    # Check if memory timeline data is available
    if not (result.raw_data and "memory_timeline" in result.raw_data):
        logger.warning("No memory timeline data available for plotting")
        return None
    
    memory_timeline = result.raw_data["memory_timeline"]
    
    if not ("timestamps" in memory_timeline and "cpu_memory_mb" in memory_timeline):
        logger.warning("Incomplete memory timeline data")
        return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(memory_timeline["timestamps"], memory_timeline["cpu_memory_mb"], label="CPU Memory")
    
    if "gpu_memory_mb" in memory_timeline and memory_timeline["gpu_memory_mb"]:
        ax.plot(memory_timeline["timestamps"], memory_timeline["gpu_memory_mb"], label="GPU Memory")
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_title(f"Memory Usage Timeline - {result.name}")
    ax.legend()
    
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Saved memory usage plot to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return output_path if output_path else None


def plot_compute_utilization(
    result: ProfileResult,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    """
    Plot compute utilization from a profile result.
    
    Args:
        result: Profile result containing compute utilization data
        output_path: Path to save the plot
        show: Whether to show the plot (in addition to saving)
        
    Returns:
        Optional[Path]: Path to the saved plot, or None if not saved
    """
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot plot compute utilization: matplotlib not available")
        return None
    
    # Check if compute timeline data is available
    if not (result.raw_data and "compute_timeline" in result.raw_data):
        logger.warning("No compute timeline data available for plotting")
        return None
    
    compute_timeline = result.raw_data["compute_timeline"]
    
    if not ("timestamps" in compute_timeline and "cpu_percent" in compute_timeline):
        logger.warning("Incomplete compute timeline data")
        return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(compute_timeline["timestamps"], compute_timeline["cpu_percent"], label="CPU Utilization")
    
    if "gpu_percent" in compute_timeline and compute_timeline["gpu_percent"]:
        ax.plot(compute_timeline["timestamps"], compute_timeline["gpu_percent"], label="GPU Utilization")
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Utilization (%)")
    ax.set_title(f"Compute Utilization - {result.name}")
    ax.legend()
    
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Saved compute utilization plot to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return output_path if output_path else None


def plot_training_performance(
    results: List[ProfileResult],
    metric: str = "duration_ms",
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    """
    Plot training performance comparison from multiple profile results.
    
    Args:
        results: List of profile results to compare
        metric: Metric to plot (e.g., 'duration_ms', 'memory_usage')
        output_path: Path to save the plot
        show: Whether to show the plot (in addition to saving)
        
    Returns:
        Optional[Path]: Path to the saved plot, or None if not saved
    """
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot plot training performance: matplotlib not available")
        return None
    
    if not results:
        logger.warning("No profile results provided")
        return None
    
    # Extract data for the specified metric
    names = []
    values = []
    
    for result in results:
        if metric == "duration_ms" and "total" in result.duration_ms:
            names.append(result.name)
            values.append(result.duration_ms["total"])
        elif metric == "memory_usage" and "peak_mb" in result.memory_usage:
            names.append(result.name)
            values.append(result.memory_usage["peak_mb"])
        elif metric == "compute_utilization" and "avg_gpu_percent" in result.compute_utilization:
            names.append(result.name)
            values.append(result.compute_utilization["avg_gpu_percent"])
    
    if not names or not values:
        logger.warning(f"No data available for metric '{metric}'")
        return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(names, values)
    
    ax.set_xlabel("Profile")
    
    if metric == "duration_ms":
        ax.set_ylabel("Duration (ms)")
        ax.set_title("Training Duration Comparison")
    elif metric == "memory_usage":
        ax.set_ylabel("Memory Usage (MB)")
        ax.set_title("Peak Memory Usage Comparison")
    elif metric == "compute_utilization":
        ax.set_ylabel("GPU Utilization (%)")
        ax.set_title("Average GPU Utilization Comparison")
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Saved training performance plot to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return output_path if output_path else None


def plot_inference_performance(
    results: List[ProfileResult],
    batch_sizes: Optional[List[int]] = None,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    """
    Plot inference performance scaling with batch size.
    
    Args:
        results: List of profile results for different batch sizes
        batch_sizes: List of batch sizes corresponding to results
        output_path: Path to save the plot
        show: Whether to show the plot (in addition to saving)
        
    Returns:
        Optional[Path]: Path to the saved plot, or None if not saved
    """
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot plot inference performance: matplotlib not available")
        return None
    
    if not results:
        logger.warning("No profile results provided")
        return None
    
    # If batch sizes are not provided, try to extract from result names
    if batch_sizes is None:
        batch_sizes = []
        for result in results:
            # Try to extract batch size from name (assuming format like "batch_16")
            import re
            match = re.search(r"batch[_-](\d+)", result.name)
            if match:
                batch_sizes.append(int(match.group(1)))
            else:
                # If batch size can't be extracted, use a default series
                batch_sizes = list(range(1, len(results) + 1))
                break
    
    # Ensure we have the same number of batch sizes and results
    if len(batch_sizes) != len(results):
        logger.warning("Number of batch sizes does not match number of results")
        batch_sizes = list(range(1, len(results) + 1))
    
    # Extract data
    latencies = []
    throughputs = []
    
    for result in results:
        if "per_batch" in result.duration_ms:
            latency = result.duration_ms["per_batch"]
            latencies.append(latency)
            
            # Calculate throughput (samples/sec)
            batch_idx = results.index(result)
            batch_size = batch_sizes[batch_idx]
            throughput = (batch_size / latency) * 1000  # Convert ms to samples/sec
            throughputs.append(throughput)
    
    if not latencies:
        logger.warning("No latency data available in results")
        return None
    
    # Create the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = "tab:blue"
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Latency (ms)", color=color)
    ax1.plot(batch_sizes, latencies, "o-", color=color, label="Latency")
    ax1.tick_params(axis="y", labelcolor=color)
    
    # Create a second y-axis for throughput
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Throughput (samples/sec)", color=color)
    ax2.plot(batch_sizes, throughputs, "s-", color=color, label="Throughput")
    ax2.tick_params(axis="y", labelcolor=color)
    
    # Add a title and legend
    plt.title("Inference Performance Scaling")
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Saved inference performance plot to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return output_path if output_path else None