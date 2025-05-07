"""
Report generator for TAuto optimization results.

This module provides utilities for generating comprehensive reports on the
results of model optimization, including visualizations and comparisons.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import json
import numpy as np
import base64
from io import BytesIO

# Try to import visualization libraries but gracefully handle missing dependencies
try:
    import matplotlib
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

from tauto.utils import get_logger

logger = get_logger(__name__)


def generate_optimization_report(
    original_model: Any,
    optimized_models: Dict[str, Any],
    evaluation_results: Dict[str, Dict[str, float]],
    optimization_results: Dict[str, Dict[str, Any]],
    profile_results: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
    output_path: Optional[Union[str, Path]] = None,
    include_visualizations: bool = True,
) -> Path:
    """
    Generate a comprehensive HTML report on optimization results.
    
    Args:
        original_model: Original model before optimization
        optimized_models: Dictionary of optimized models
        evaluation_results: Results from model evaluation
        optimization_results: Results from individual optimizations
        profile_results: Results from profiling (optional)
        config: TAuto configuration (optional)
        output_path: Path to save the report (defaults to "optimization_report.html")
        include_visualizations: Whether to include visualization plots
        
    Returns:
        Path: Path to the generated report
    """
    # Set default output path if not provided
    if output_path is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = Path(f"optimization_report_{timestamp}.html")
    else:
        output_path = Path(output_path)
    
    # Create output directory if needed
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Create visualizations if requested
    visualization_data = None
    if include_visualizations and _MATPLOTLIB_AVAILABLE:
        visualization_data = _create_visualization_plots(
            evaluation_results=evaluation_results,
            optimization_results=optimization_results,
        )
    
    # Generate the HTML report
    html_content = _generate_html_report(
        original_model=original_model,
        optimized_models=optimized_models,
        evaluation_results=evaluation_results,
        optimization_results=optimization_results,
        profile_results=profile_results,
        config=config,
        visualization_data=visualization_data,
    )
    
    # Write the report to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Optimization report saved to {output_path}")
    return output_path


def _create_visualization_plots(
    evaluation_results: Dict[str, Dict[str, float]],
    optimization_results: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """
    Create visualization plots for the report.
    
    Args:
        evaluation_results: Results from model evaluation
        optimization_results: Results from individual optimizations
        
    Returns:
        Dict[str, str]: Dictionary of plot names and encoded images
    """
    visualization_data = {}
    
    # Performance comparison plot
    plt.figure(figsize=(10, 6))
    
    # Get model names and metrics
    models = list(evaluation_results.keys())
    latencies = [evaluation_results[m]["latency_ms"] for m in models]
    accuracies = [evaluation_results[m]["accuracy"] * 100 for m in models]
    
    # Plot metrics
    plt.subplot(2, 1, 1)
    bars = plt.bar(models, latencies)
    plt.ylabel("Latency (ms)")
    plt.title("Performance Comparison")
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.subplot(2, 1, 2)
    bars = plt.bar(models, accuracies)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot to a base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    visualization_data['performance_comparison'] = img_str
    plt.close()
    
    # Throughput comparison
    plt.figure(figsize=(10, 4))
    throughputs = [evaluation_results[m]["throughput"] for m in models]
    bars = plt.bar(models, throughputs)
    plt.ylabel("Throughput (samples/sec)")
    plt.title("Throughput Comparison")
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Save plot to a base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    visualization_data['throughput_comparison'] = img_str
    plt.close()
    
    # If there are optimized models, create speedup comparison
    if len(evaluation_results) > 1:
        plt.figure(figsize=(10, 4))
        non_original_models = [m for m in models if m != "original"]
        speedups = [evaluation_results[m].get("speedup", 1.0) for m in non_original_models]
        
        bars = plt.bar(non_original_models, speedups)
        plt.ylabel("Speedup Factor")
        plt.title("Optimization Speedup Comparison")
        plt.axhline(y=1.0, color='r', linestyle='-')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}x', ha='center', va='bottom', fontsize=9)
        
        # Save plot to a base64 string
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        visualization_data['speedup_comparison'] = img_str
        plt.close()
    
    # Memory comparison if available
    has_memory_data = False
    for results in optimization_results.values():
        if "memory_reduction" in results or "original_size_mb" in results:
            has_memory_data = True
            break
    
    if has_memory_data:
        plt.figure(figsize=(10, 4))
        
        # Extract memory data
        memory_data = {}
        for model_name, results in optimization_results.items():
            if "original_size_mb" in results and "quantized_size_mb" in results:
                memory_data[model_name] = {
                    "original": results["original_size_mb"],
                    "optimized": results["quantized_size_mb"],
                }
            elif "memory_reduction" in results and model_name in evaluation_results:
                # Estimate memory values from reduction percentage
                if "original" in evaluation_results:
                    memory_data[model_name] = {
                        "reduction": results["memory_reduction"],
                    }
        
        # Plot memory comparison if we have data
        if memory_data:
            model_names = list(memory_data.keys())
            ind = np.arange(len(model_names))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            original_sizes = []
            optimized_sizes = []
            
            for model in model_names:
                if "original" in memory_data[model] and "optimized" in memory_data[model]:
                    original_sizes.append(memory_data[model]["original"])
                    optimized_sizes.append(memory_data[model]["optimized"])
            
            if original_sizes and optimized_sizes:
                bars1 = ax.bar(ind - width/2, original_sizes, width, label='Original')
                bars2 = ax.bar(ind + width/2, optimized_sizes, width, label='Optimized')
                
                # Add values on bars
                for bar in bars1:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{height:.2f} MB', ha='center', va='bottom', fontsize=8)
                
                for bar in bars2:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{height:.2f} MB', ha='center', va='bottom', fontsize=8)
                
                ax.set_ylabel('Model Size (MB)')
                ax.set_title('Memory Usage Comparison')
                ax.set_xticks(ind)
                ax.set_xticklabels(model_names)
                ax.legend()
                
                # Save plot to a base64 string
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                visualization_data['memory_comparison'] = img_str
        
        plt.close()
    
    return visualization_data


def _generate_html_report(
    original_model: Any,
    optimized_models: Dict[str, Any],
    evaluation_results: Dict[str, Dict[str, float]],
    optimization_results: Dict[str, Dict[str, Any]],
    profile_results: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
    visualization_data: Optional[Dict[str, str]] = None,
) -> str:
    """
    Generate the HTML content for the optimization report.
    
    Args:
        original_model: Original model before optimization
        optimized_models: Dictionary of optimized models
        evaluation_results: Results from model evaluation
        optimization_results: Results from individual optimizations
        profile_results: Results from profiling (optional)
        config: TAuto configuration (optional)
        visualization_data: Visualization plots as base64 encoded images
        
    Returns:
        str: HTML content for the report
    """
    # Timestamp for the report
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Start building the HTML
    html_parts = []
    
    # HTML head with styling
    html_parts.append(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TAuto Optimization Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3, h4 {{
                color: #2c3e50;
            }}
            .header {{
                background-color: #3498db;
                color: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 5px;
            }}
            .header h1 {{
                color: white;
                margin: 0;
            }}
            .summary-box {{
                background-color: #f8f9fa;
                border-left: 5px solid #3498db;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 0 5px 5px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .visualization {{
                text-align: center;
                margin: 20px 0;
            }}
            .visualization img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            .highlight {{
                background-color: #e8f4f8;
                padding: 2px 5px;
                border-radius: 3px;
                font-weight: bold;
            }}
            .positive {{
                color: #27ae60;
            }}
            .negative {{
                color: #e74c3c;
            }}
            .section {{
                margin-bottom: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
            }}
            .footer {{
                text-align: center;
                font-size: 0.9em;
                color: #7f8c8d;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>TAuto Optimization Report</h1>
            <p>Generated on: {timestamp}</p>
        </div>
    """)
    
    # Executive Summary
    html_parts.append("""
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="summary-box">
    """)
    
    # Calculate high-level metrics for the summary
    best_model = "original"
    best_speedup = 1.0
    best_accuracy = evaluation_results["original"]["accuracy"]
    
    for model_name, result in evaluation_results.items():
        if model_name != "original":
            speedup = result.get("speedup", 1.0)
            if speedup > best_speedup:
                best_speedup = speedup
                best_model = model_name
            
            if result["accuracy"] > best_accuracy:
                best_accuracy = result["accuracy"]
    
    html_parts.append(f"""
                <p><strong>Models optimized:</strong> {len(optimized_models)}</p>
                <p><strong>Best performing model:</strong> {best_model} (speedup: <span class="highlight">{best_speedup:.2f}x</span>)</p>
                <p><strong>Original model accuracy:</strong> {evaluation_results["original"]["accuracy"]:.4f} ({evaluation_results["original"]["accuracy"]*100:.2f}%)</p>
    """)
    
    if best_model != "original":
        acc_change = evaluation_results[best_model]["accuracy"] - evaluation_results["original"]["accuracy"]
        acc_class = "positive" if acc_change >= 0 else "negative"
        html_parts.append(f"""
                <p><strong>Best model accuracy:</strong> {evaluation_results[best_model]["accuracy"]:.4f} ({evaluation_results[best_model]["accuracy"]*100:.2f}%) <span class="{acc_class}">({acc_change:+.4f})</span></p>
        """)
    
    html_parts.append("""
            </div>
        </div>
    """)
    
    # Optimization Results Table
    html_parts.append("""
        <div class="section">
            <h2>Optimization Results</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Latency (ms)</th>
                    <th>Throughput (samples/sec)</th>
                    <th>Speedup</th>
                    <th>Memory Reduction</th>
                </tr>
    """)
    
    # Original model row
    orig_result = evaluation_results["original"]
    html_parts.append(f"""
                <tr>
                    <td>Original</td>
                    <td>{orig_result['accuracy']:.4f} ({orig_result['accuracy']*100:.2f}%)</td>
                    <td>{orig_result['latency_ms']:.2f}</td>
                    <td>{orig_result['throughput']:.2f}</td>
                    <td>1.00x</td>
                    <td>-</td>
                </tr>
    """)
    
    # Optimized models rows
    for model_name in optimized_models:
        if model_name in evaluation_results:
            result = evaluation_results[model_name]
            speedup = result.get("speedup", 1.0)
            
            # Get memory reduction if available
            memory_reduction = "-"
            if model_name in optimization_results:
                opt_result = optimization_results[model_name]
                if "memory_reduction" in opt_result:
                    memory_reduction = f"{opt_result['memory_reduction']*100:.2f}%"
            
            # Calculate accuracy difference
            acc_diff = result["accuracy"] - orig_result["accuracy"]
            acc_class = "positive" if acc_diff >= 0 else "negative"
            
            html_parts.append(f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{result['accuracy']:.4f} ({result['accuracy']*100:.2f}%) <span class="{acc_class}">({acc_diff:+.4f})</span></td>
                    <td>{result['latency_ms']:.2f}</td>
                    <td>{result['throughput']:.2f}</td>
                    <td><span class="highlight">{speedup:.2f}x</span></td>
                    <td>{memory_reduction}</td>
                </tr>
            """)
    
    html_parts.append("""
            </table>
        </div>
    """)
    
    # Add visualizations if available
    if visualization_data:
        html_parts.append("""
        <div class="section">
            <h2>Performance Visualizations</h2>
        """)
        
        # Add performance comparison plot
        if 'performance_comparison' in visualization_data:
            html_parts.append(f"""
            <div class="visualization">
                <h3>Performance Comparison</h3>
                <img src="data:image/png;base64,{visualization_data['performance_comparison']}" alt="Performance Comparison">
            </div>
            """)
        
        # Add throughput comparison plot
        if 'throughput_comparison' in visualization_data:
            html_parts.append(f"""
            <div class="visualization">
                <h3>Throughput Comparison</h3>
                <img src="data:image/png;base64,{visualization_data['throughput_comparison']}" alt="Throughput Comparison">
            </div>
            """)
        
        # Add speedup comparison plot
        if 'speedup_comparison' in visualization_data:
            html_parts.append(f"""
            <div class="visualization">
                <h3>Speedup Comparison</h3>
                <img src="data:image/png;base64,{visualization_data['speedup_comparison']}" alt="Speedup Comparison">
            </div>
            """)
        
        # Add memory comparison plot
        if 'memory_comparison' in visualization_data:
            html_parts.append(f"""
            <div class="visualization">
                <h3>Memory Usage Comparison</h3>
                <img src="data:image/png;base64,{visualization_data['memory_comparison']}" alt="Memory Comparison">
            </div>
            """)
        
        html_parts.append("""
        </div>
        """)
    
    # Detailed Optimization Results
    html_parts.append("""
        <div class="section">
            <h2>Detailed Optimization Results</h2>
    """)
    
    # Add details for each optimization
    for model_name, opt_results in optimization_results.items():
        html_parts.append(f"""
            <h3>{model_name.capitalize()} Optimization</h3>
            <div class="summary-box">
        """)
        
        # Add common metrics
        if "speedup" in opt_results:
            html_parts.append(f"""
                <p><strong>Speedup:</strong> {opt_results['speedup']:.2f}x</p>
            """)
        
        if "memory_reduction" in opt_results:
            html_parts.append(f"""
                <p><strong>Memory reduction:</strong> {opt_results['memory_reduction']*100:.2f}%</p>
            """)
        
        # Add specific metrics based on optimization type
        if model_name == "quantized" and ("original_size_mb" in opt_results and "quantized_size_mb" in opt_results):
            html_parts.append(f"""
                <p><strong>Original model size:</strong> {opt_results['original_size_mb']:.2f} MB</p>
                <p><strong>Quantized model size:</strong> {opt_results['quantized_size_mb']:.2f} MB</p>
            """)
        
        if model_name == "pruned" and "sparsity" in opt_results:
            html_parts.append(f"""
                <p><strong>Pruning sparsity:</strong> {opt_results['sparsity']*100:.2f}%</p>
            """)
        
        if model_name == "compiled" and "backend" in opt_results:
            html_parts.append(f"""
                <p><strong>Compilation backend:</strong> {opt_results['backend']}</p>
                <p><strong>Compilation mode:</strong> {opt_results.get('mode', 'default')}</p>
            """)
        
        html_parts.append("""
            </div>
        """)
    
    html_parts.append("""
        </div>
    """)
    
    # Configuration
    if config:
        html_parts.append("""
        <div class="section">
            <h2>Configuration</h2>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Setting</th>
                    <th>Value</th>
                </tr>
        """)
        
        # Add training settings
        if hasattr(config, "training"):
            for key, value in config.training.items():
                if isinstance(value, dict):
                    # Skip nested dictionaries for simplicity
                    continue
                html_parts.append(f"""
                <tr>
                    <td>Training</td>
                    <td>{key}</td>
                    <td>{value}</td>
                </tr>
                """)
        
        # Add optimization settings
        if hasattr(config, "optimization"):
            for component, component_config in config.optimization.items():
                if isinstance(component_config, dict):
                    for key, value in component_config.items():
                        html_parts.append(f"""
                        <tr>
                            <td>Optimization/{component}</td>
                            <td>{key}</td>
                            <td>{value}</td>
                        </tr>
                        """)
                else:
                    html_parts.append(f"""
                    <tr>
                        <td>Optimization</td>
                        <td>{component}</td>
                        <td>{component_config}</td>
                    </tr>
                    """)
        
        # Add profiling settings
        if hasattr(config, "profiling"):
            for key, value in config.profiling.items():
                if isinstance(value, dict):
                    # Skip nested dictionaries for simplicity
                    continue
                html_parts.append(f"""
                <tr>
                    <td>Profiling</td>
                    <td>{key}</td>
                    <td>{value}</td>
                </tr>
                """)
        
        html_parts.append("""
            </table>
        </div>
        """)
    
    # Recommendations based on results
    html_parts.append("""
        <div class="section">
            <h2>Recommendations</h2>
            <div class="summary-box">
    """)
    
    # Determine best optimization based on results
    if len(optimization_results) > 0:
        # Find optimization with best speedup
        best_opt = max(evaluation_results.items(), key=lambda x: x[1].get("speedup", 1.0) if x[0] != "original" else 0)
        best_opt_name = best_opt[0] if best_opt[0] != "original" else None
        
        if best_opt_name:
            best_opt_speedup = evaluation_results[best_opt_name].get("speedup", 1.0)
            best_opt_acc_diff = evaluation_results[best_opt_name]["accuracy"] - evaluation_results["original"]["accuracy"]
            
            html_parts.append(f"""
                <p><strong>Best optimization technique:</strong> {best_opt_name.capitalize()}</p>
                <p><strong>Performance improvement:</strong> {best_opt_speedup:.2f}x speedup</p>
                <p><strong>Accuracy impact:</strong> {best_opt_acc_diff:+.4f} ({best_opt_acc_diff*100:+.2f}%)</p>
            """)
            
            # Provide specific recommendations based on the best technique
            if best_opt_name == "quantized":
                html_parts.append("""
                <p><strong>Recommendation:</strong> Consider deploying the quantized model for inference scenarios where memory usage and latency are critical. The minimal accuracy impact makes this a good trade-off.</p>
                """)
            elif best_opt_name == "pruned":
                html_parts.append("""
                <p><strong>Recommendation:</strong> Consider using the pruned model, which offers a good balance between performance and accuracy. For further improvements, you could try iterative pruning with fine-tuning.</p>
                """)
            elif best_opt_name == "compiled":
                html_parts.append("""
                <p><strong>Recommendation:</strong> The compiled model provides significant speedup without any accuracy impact. This is ideal for deployment scenarios where the compilation backend is supported.</p>
                """)
            elif best_opt_name == "combined":
                html_parts.append("""
                <p><strong>Recommendation:</strong> The combined optimization approach provides the best overall results. Consider this approach for deployment, but be aware of any platform-specific limitations for the optimizations used.</p>
                """)
        else:
            html_parts.append("""
                <p><strong>Recommendation:</strong> None of the optimization techniques provided significant improvements. Consider profiling the model more extensively to identify specific bottlenecks.</p>
            """)
    else:
        html_parts.append("""
            <p>No optimization results available for recommendations.</p>
        """)
    
    html_parts.append("""
            </div>
        </div>
    """)
    
    # Footer
    html_parts.append(f"""
        <div class="footer">
            <p>Generated by TAuto Optimization Framework</p>
            <p>{timestamp}</p>
        </div>
    </body>
    </html>
    """)
    
    return "".join(html_parts)


def export_results_to_csv(
    evaluation_results: Dict[str, Dict[str, float]],
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """
    Export evaluation results to a CSV file.
    
    Args:
        evaluation_results: Results from model evaluation
        output_path: Path to save the CSV file
        
    Returns:
        Optional[Path]: Path to the generated CSV file, or None if pandas is not available
    """
    if not _PANDAS_AVAILABLE:
        logger.warning("Pandas not available. Cannot export results to CSV.")
        return None
    
    if output_path is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = Path(f"optimization_results_{timestamp}.csv")
    else:
        output_path = Path(output_path)
    
    # Create output directory if needed
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Convert results to a dataframe
    data = []
    for model_name, results in evaluation_results.items():
        row = {"model": model_name}
        row.update(results)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Results exported to CSV: {output_path}")
    
    return output_path


def export_results_to_json(
    evaluation_results: Dict[str, Dict[str, float]],
    optimization_results: Dict[str, Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Export evaluation and optimization results to a JSON file.
    
    Args:
        evaluation_results: Results from model evaluation
        optimization_results: Results from individual optimizations
        output_path: Path to save the JSON file
        
    Returns:
        Path: Path to the generated JSON file
    """
    if output_path is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = Path(f"optimization_results_{timestamp}.json")
    else:
        output_path = Path(output_path)
    
    # Create output directory if needed
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Combine results
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_results": evaluation_results,
        "optimization_results": {},
    }
    
    # Process optimization results to make them JSON serializable
    for model_name, results in optimization_results.items():
        # Filter out non-serializable objects
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                serializable_results[key] = value
        
        data["optimization_results"][model_name] = serializable_results
    
    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Results exported to JSON: {output_path}")
    
    return output_path