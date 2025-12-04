"""
Analysis module for model interpretation and preprocessing validation.

Provides tools for:
- GradCAM visualization
- Patch-based quality analysis
- ROI detection and targeted preprocessing
- Comprehensive visualizations
"""

from .gradcam_roi_analysis import (
    GradCAM,
    PatchAnalyzer,
    LungROIDetector,
    ROIBasedPreprocessor,
    create_comprehensive_visualization
)

__all__ = [
    'GradCAM',
    'PatchAnalyzer',
    'LungROIDetector',
    'ROIBasedPreprocessor',
    'create_comprehensive_visualization'
]
