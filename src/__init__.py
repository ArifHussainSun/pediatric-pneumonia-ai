"""
Pediatric Pneumonia AI Detection System

A comprehensive deep learning framework for automated pediatric pneumonia detection
from chest X-ray images. This package provides state-of-the-art CNN models,
distributed training capabilities, and comprehensive evaluation tools.

Key Features:
- Multiple CNN architectures (Xception, VGG, MobileNet, Fusion models)
- Experimental CNN-LSTM hybrid models
- Multi-GPU distributed training for DGX stations
- Comprehensive evaluation and visualization tools
- Production-ready containerization with Docker
- Clinical decision support utilities

Modules:
- models: Neural network architectures for pneumonia detection
- data: Data loading, preprocessing, and augmentation utilities
- training: Training frameworks and distributed training support
- evaluation: Model evaluation and performance analysis
- visualization: Visualization tools and clinical reporting

Example Usage:
    >>> from src.models import create_model
    >>> from src.data import create_data_loaders
    >>> from src.training import ModelTrainer
    >>>
    >>> # Create model and data loaders
    >>> model = create_model('xception', num_classes=2)
    >>> train_loader, val_loader = create_data_loaders('data/train', 'data/val')
    >>>
    >>> # Train model
    >>> trainer = ModelTrainer(model, criterion, optimizer)
    >>> trainer.train(train_loader, val_loader, epochs=50)

Requirements:
- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA support for GPU training
- See requirements.txt for complete dependencies

For more information, see README.md and documentation.
"""

__version__ = "1.0.0"
__author__ = "Tech4Life Pediatric Pneumonia Detection Team"
__email__ = "contact@tech4life.ai"
__description__ = "Deep learning framework for pediatric pneumonia detection"

# Core imports for easy access
from . import models
from . import data
from . import training
from . import evaluation
from . import visualization

# Version information
VERSION_INFO = {
    'version': __version__,
    'author': __author__,
    'description': __description__,
    'modules': ['models', 'data', 'training', 'evaluation', 'visualization']
}

def get_version_info():
    """Get detailed version information."""
    return VERSION_INFO

def list_available_models():
    """List all available model architectures."""
    from .models import list_available_models
    return list_available_models()

def get_model_info():
    """Get information about all available models."""
    from .models import get_model_info
    return get_model_info()

# Package-level convenience functions
def create_model(model_type: str, **kwargs):
    """
    Create a model instance.

    Args:
        model_type: Type of model to create
        **kwargs: Model-specific parameters

    Returns:
        Initialized model instance
    """
    from .models import create_model
    return create_model(model_type, **kwargs)

def create_data_loaders(train_dir: str, test_dir: str, **kwargs):
    """
    Create training and test data loaders.

    Args:
        train_dir: Training data directory
        test_dir: Test data directory
        **kwargs: DataLoader parameters

    Returns:
        Tuple of (train_loader, test_loader)
    """
    from .data import create_data_loaders
    return create_data_loaders(train_dir, test_dir, **kwargs)

def evaluate_model(model, test_loader, **kwargs):
    """
    Evaluate a model on test data.

    Args:
        model: Trained model
        test_loader: Test data loader
        **kwargs: Evaluation parameters

    Returns:
        Evaluation results dictionary
    """
    from .evaluation import ModelEvaluator
    evaluator = ModelEvaluator(model, **kwargs)
    return evaluator.evaluate(test_loader)

# Package metadata
__all__ = [
    'models',
    'data',
    'training',
    'evaluation',
    'visualization',
    'get_version_info',
    'list_available_models',
    'get_model_info',
    'create_model',
    'create_data_loaders',
    'evaluate_model'
]