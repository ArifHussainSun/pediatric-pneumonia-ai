"""
Evaluation utilities package for pediatric pneumonia detection.

This package provides comprehensive evaluation framework including:
- Model performance evaluation with medical metrics
- ROC and Precision-Recall curve analysis
- Model comparison utilities
- Statistical analysis and visualization
- Clinical interpretation of results

Key components:
- evaluator: Core evaluation utilities and ModelEvaluator class
"""

from .evaluator import (
    ModelEvaluator,
    compare_models
)

__all__ = [
    'ModelEvaluator',
    'compare_models',
]