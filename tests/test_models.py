"""
Basic tests for pneumonia detection models.
"""
import pytest
import torch
import numpy as np
from PIL import Image
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import create_model

def test_model_creation():
    """Test basic model creation."""
    model = create_model('mobilenet', num_classes=2)
    assert model is not None
    assert hasattr(model, 'forward')

def test_model_inference():
    """Test model inference with dummy data."""
    model = create_model('mobilenet', num_classes=2)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 2)
    assert torch.isfinite(output).all()

def test_image_preprocessing():
    """Test image preprocessing pipeline."""
    from data.datasets import get_medical_transforms

    # Create dummy image
    dummy_image = Image.new('RGB', (256, 256), color='white')

    transform = get_medical_transforms(is_training=False)
    processed = transform(dummy_image)

    assert processed.shape == (3, 224, 224)
    assert processed.dtype == torch.float32

if __name__ == "__main__":
    test_model_creation()
    test_model_inference()
    test_image_preprocessing()
    print("All basic tests passed!")