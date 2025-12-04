"""
Data utilities package for pediatric pneumonia detection.

This package provides comprehensive data handling utilities including:
- Image preprocessing and augmentation
- PyTorch datasets and data loaders
- Feature extraction for traditional ML
- Data splitting and organization utilities

Key components:
- data_utils: Core preprocessing and feature extraction functions
- datasets: PyTorch-compatible dataset classes and data loaders
"""

from .data_utils import (
    augment_image_efficient,
    extract_features,
    extract_roi_features,
    extract_all_features,
    DataSplitter,
    load_image_safe,
    get_dataset_stats
)

from .datasets import (
    PneumoniaDataset,
    PneumoniaCSVDataset,
    get_medical_transforms,
    create_data_loaders,
    create_csv_data_loaders
)

__all__ = [
    # Data utilities
    'augment_image_efficient',
    'extract_features',
    'extract_roi_features',
    'extract_all_features',
    'DataSplitter',
    'load_image_safe',
    'get_dataset_stats',

    # Dataset classes
    'PneumoniaDataset',
    'PneumoniaCSVDataset',

    # Data loader utilities
    'get_medical_transforms',
    'create_data_loaders',
    'create_csv_data_loaders',
]