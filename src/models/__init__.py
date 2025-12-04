"""
Pediatric Pneumonia Detection Models Package

This package contains all model architectures for pediatric pneumonia detection,
including CNN models, fusion models, and experimental CNN-LSTM architectures.

Available Models:
- XceptionFineTune: Pre-trained Xception for medical image classification
- VGG16FineTune: VGG16 with custom classification head
- MobileNetFineTune: Lightweight MobileNet variants for deployment
- XceptionVGGFusion: Multi-backbone fusion architecture
- XceptionLSTM: Experimental CNN-LSTM hybrid model
"""

from .xception import XceptionFineTune, create_xception_model
from .vgg import VGG16FineTune, VGG16Lightweight, create_vgg16_model
from .mobilenet import (
    MobileNetV2Seyon,
    MobileNetFineTune,
    MobileNetV3FineTune,
    MobileNetTiny,
    create_mobilenet_model
)
from .fusion import XceptionVGGFusion, WeightedFusionModel, create_fusion_model
from .xception_lstm import XceptionLSTM, CustomCNNLSTM, create_xception_lstm_model

__all__ = [
    # CNN Models
    'XceptionFineTune',
    'VGG16FineTune',
    'VGG16Lightweight',
    'MobileNetFineTune',
    'MobileNetV3FineTune',
    'MobileNetTiny',

    # Fusion Models
    'XceptionVGGFusion',
    'WeightedFusionModel',

    # Experimental Models
    'XceptionLSTM',
    'CustomCNNLSTM',

    # Factory Functions
    'create_xception_model',
    'create_vgg16_model',
    'create_mobilenet_model',
    'create_fusion_model',
    'create_xception_lstm_model',
]

# Model registry for easy access
MODEL_REGISTRY = {
    'xception': create_xception_model,
    'vgg': create_vgg16_model,
    'mobilenet': create_mobilenet_model,
    'fusion': create_fusion_model,
    'xception_lstm': create_xception_lstm_model,
}

def create_model(model_type: str, **kwargs):
    """
    Factory function to create any model from the registry.

    Args:
        model_type (str): Type of model to create
        **kwargs: Model-specific parameters

    Returns:
        torch.nn.Module: Initialized model

    Example:
        >>> model = create_model('xception', num_classes=2, freeze_layers=100)
        >>> fusion_model = create_model('fusion', model_type='xception_vgg')
    """
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model type '{model_type}'. Available: {available}")

    return MODEL_REGISTRY[model_type](**kwargs)

def list_available_models():
    """List all available model types."""
    return list(MODEL_REGISTRY.keys())

def get_model_info():
    """Get information about all available models."""
    return {
        'xception': {
            'description': 'Pre-trained Xception with custom classification head',
            'parameters': '~22M',
            'use_case': 'High accuracy medical image classification'
        },
        'vgg': {
            'description': 'VGG16 with lightweight and standard variants',
            'parameters': '~15M (standard), ~2M (lightweight)',
            'use_case': 'Balanced performance and interpretability'
        },
        'mobilenet': {
            'description': 'Lightweight MobileNet variants (V2, V3, Tiny)',
            'parameters': '~3.5M (V2), ~2M (V3-small), ~0.5M (Tiny)',
            'use_case': 'Mobile and edge deployment'
        },
        'fusion': {
            'description': 'Multi-backbone fusion combining Xception + VGG16',
            'parameters': '~37M',
            'use_case': 'Maximum accuracy with ensemble-like performance'
        },
        'xception_lstm': {
            'description': 'Experimental CNN-LSTM with spatial tokenization',
            'parameters': '~22M (advanced), ~1M (lightweight)',
            'use_case': 'Research into spatial-temporal medical image analysis'
        }
    }