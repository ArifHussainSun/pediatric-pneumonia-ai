"""
Mobile Deployment Package for Pediatric Pneumonia Detection

This package provides comprehensive mobile deployment capabilities for
pneumonia detection models on iPads, tablets, and mobile devices.

Key Features:
- Model optimization for mobile hardware
- Cross-platform deployment (iOS, Android)
- Offline inference capabilities
- Performance monitoring and benchmarking
- Resource-constrained optimization

Mobile Formats Supported:
- CoreML (iOS/macOS)
- TensorFlow Lite (Android/cross-platform)
- ONNX Mobile (cross-platform)
- Quantized models for efficiency

The mobile package enables Tech4Life to deploy AI models directly on
clinical devices without requiring cloud connectivity.
"""

from .mobile_optimizer import MobileOptimizer, optimize_model_for_mobile
from .quantization import ModelQuantizer, quantize_model
from .benchmarking import MobileBenchmark, benchmark_mobile_model

__all__ = [
    'MobileOptimizer',
    'optimize_model_for_mobile',
    'ModelQuantizer',
    'quantize_model',
    'MobileBenchmark',
    'benchmark_mobile_model'
]

__version__ = '1.0.0'