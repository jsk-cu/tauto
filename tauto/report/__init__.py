"""
Reporting module for TAuto.

This module provides tools for generating reports and visualizations
of model optimization results.
"""

from tauto.report.report_generator import (
    generate_optimization_report,
    export_results_to_csv,
    export_results_to_json,
)

__all__ = [
    "generate_optimization_report",
    "export_results_to_csv",
    "export_results_to_json",
]