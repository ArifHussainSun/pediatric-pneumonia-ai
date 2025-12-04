#!/usr/bin/env python3
"""
Debug script to identify and fix the CUDA loss function error.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models import create_model
from src.data import create_data_loaders, get_medical_transforms

def debug_loss_error():
    """Debug the CUDA loss function error."""

    print("Debugging CUDA loss function error...")

    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create a simple model
    try:
        print("Creating model...")
        model = create_model('xception', num_classes=2)
        model = model.to(device)
        print("Model created successfully")
    except Exception as e:
        print(f"Model creation failed: {e}")
        return

    # Create loss function
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # Create dummy data that matches what your dataset produces
    print("Testing with dummy data...")
    batch_size = 2

    # Create dummy inputs (3 channels, 224x224 like medical images)
    dummy_images = torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.float32)
    dummy_labels = torch.tensor([0, 1], device=device, dtype=torch.long)  # PNEUMONIA=0, NORMAL=1

    print(f"Input shape: {dummy_images.shape}, dtype: {dummy_images.dtype}")
    print(f"Labels shape: {dummy_labels.shape}, dtype: {dummy_labels.dtype}")

    try:
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_images)
            print(f"Output shape: {outputs.shape}, dtype: {outputs.dtype}")

            # Check if outputs are logits (should be raw scores, not probabilities)
            print(f"Output range: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}")

            # Test loss computation
            loss = criterion(outputs, dummy_labels)
            print(f"Loss computed successfully: {loss.item():.4f}")

    except Exception as e:
        print(f"Forward pass failed: {e}")
        print(f"Error type: {type(e)}")

        # Additional debugging
        print("\\nDebugging model output...")
        try:
            outputs = model(dummy_images)
            print(f"Raw outputs: {outputs}")

            # Check for NaN or Inf
            if torch.isnan(outputs).any():
                print("NaN detected in outputs!")
            if torch.isinf(outputs).any():
                print("Inf detected in outputs!")

            # Check output values
            if outputs.requires_grad:
                print("Outputs require grad")
            else:
                print("Outputs don't require grad")

        except Exception as e2:
            print(f"Model forward failed: {e2}")

    print("\\n Trying potential fixes...")

    # Fix 1: Ensure proper data types
    try:
        dummy_labels_fixed = dummy_labels.long()  # Explicitly convert to long
        outputs = model(dummy_images)
        loss = criterion(outputs, dummy_labels_fixed)
        print(" Fix 1 (explicit long conversion) works")
    except Exception as e:
        print(f" Fix 1 failed: {e}")

    # Fix 2: Check for proper model output format
    try:
        outputs = model(dummy_images)
        if len(outputs.shape) != 2 or outputs.shape[1] != 2:
            print(f"  Unexpected output shape: {outputs.shape}")
            print("Expected: [batch_size, num_classes] = [2, 2]")

        # Ensure outputs are Float32
        outputs = outputs.float()
        loss = criterion(outputs, dummy_labels.long())
        print(" Fix 2 (explicit float32 conversion) works")
    except Exception as e:
        print(f" Fix 2 failed: {e}")

    # Fix 3: Check device consistency
    try:
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Criterion device: next(criterion.parameters()).device if hasattr(criterion, 'parameters') else 'CPU'")
        print(f"Input device: {dummy_images.device}")
        print(f"Labels device: {dummy_labels.device}")
    except Exception as e:
        print(f"Device check failed: {e}")

def test_with_real_data():
    """Test with actual data loader if possible."""
    print("\\n Testing with real data loader...")

    # This will only work if you have actual data
    try:
        # Try to create data loaders with a small batch
        train_loader, test_loader = create_data_loaders(
            train_dir="data/train",
            test_dir="data/test",
            batch_size=2,
            num_workers=0  # Avoid multiprocessing issues
        )

        print(" Data loaders created successfully")

        # Get one batch
        for images, labels in train_loader:
            print(f"Real data - Images: {images.shape}, {images.dtype}")
            print(f"Real data - Labels: {labels.shape}, {labels.dtype}")
            print(f"Label values: {labels}")
            break

    except Exception as e:
        print(f" Real data test failed (expected if no data): {e}")

if __name__ == "__main__":
    debug_loss_error()
    test_with_real_data()

    print("\\n💡 Common fixes for this error:")
    print("1. Ensure labels are torch.long (int64)")
    print("2. Ensure model outputs are torch.float32")
    print("3. Ensure all tensors are on the same device")
    print("4. Check that model outputs have shape [batch_size, num_classes]")
    print("5. Verify no NaN or Inf values in model outputs")