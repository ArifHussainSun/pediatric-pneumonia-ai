"""
Model Export Package for Pediatric Pneumonia Detection

This package provides utilities for exporting trained PyTorch models
to various formats for production deployment:
- ONNX format for cross-platform compatibility
- TorchScript for optimized PyTorch deployment
- Mobile formats (CoreML, TensorFlow Lite) for edge devices

The export functionality enables Tech4Life to deploy models on:
- Cloud platforms (AWS, Azure, Google Cloud)
- Mobile devices (iPads, Android tablets)
- Edge computing devices
- Any platform supporting ONNX runtime
"""

from .model_exporter import (
    ModelExporter,
    export_model,
    export_all_models,
    validate_exported_model
)

__all__ = [
    'ModelExporter',
    'export_model',
    'export_all_models',
    'validate_exported_model'
]

__version__ = "1.0.0"