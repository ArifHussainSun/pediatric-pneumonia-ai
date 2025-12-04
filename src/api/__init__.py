"""
REST API Service for Pediatric Pneumonia Detection

This package provides a production-ready REST API server for deploying
pneumonia detection models. Designed for integration with clinical systems
and healthcare applications.

Key Features:
- High-performance inference endpoints
- Medical image preprocessing pipeline
- Batch processing capabilities
- Comprehensive monitoring and logging
- Security and authentication
- Clinical integration support

The API enables remote inference for clinical applications while
maintaining HIPAA compliance and medical-grade performance standards.
"""

from .server import InferenceServer, create_app
from .endpoints import api_blueprint
from .preprocessing import ImagePreprocessor
from .monitoring import APIMonitor

__all__ = [
    'InferenceServer',
    'create_app',
    'api_blueprint',
    'ImagePreprocessor',
    'APIMonitor'
]

__version__ = '1.0.0'