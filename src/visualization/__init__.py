"""
Visualization utilities package for pediatric pneumonia detection.

This package provides comprehensive visualization tools including:
- Dataset exploration and analysis
- Model interpretability with Grad-CAM
- Performance analysis and comparison
- Clinical reporting and decision analysis
- Interactive visualization components

Key components:
- visualizer: Core visualization utilities and classes
"""

from .visualizer import (
    DatasetVisualizer,
    GradCAMVisualizer,
    PerformanceVisualizer,
    create_clinical_report
)

__all__ = [
    'DatasetVisualizer',
    'GradCAMVisualizer',
    'PerformanceVisualizer',
    'create_clinical_report',
]