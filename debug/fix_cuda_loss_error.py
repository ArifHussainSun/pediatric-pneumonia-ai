#!/usr/bin/env python3
"""
Fix for CUDA loss function error in pediatric pneumonia detection.

This script contains patches to fix the common CUDA error:
"nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Float'

The error typically occurs due to:
1. Data type mismatches between model outputs and targets
2. Device placement issues
3. Incorrect tensor shapes

This patch can be applied directly to the training code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def safe_cross_entropy_loss(outputs, targets, **kwargs):
    """
    Safe CrossEntropyLoss that handles common CUDA errors.

    Args:
        outputs: Model outputs (logits) [batch_size, num_classes]
        targets: Ground truth labels [batch_size] as long integers

    Returns:
        Loss tensor
    """
    # Ensure outputs are float32
    if outputs.dtype != torch.float32:
        outputs = outputs.float()

    # Ensure targets are long (int64)
    if targets.dtype != torch.long:
        targets = targets.long()

    # Ensure both tensors are on the same device
    if outputs.device != targets.device:
        targets = targets.to(outputs.device)

    # Check for invalid values
    if torch.isnan(outputs).any():
        print("WARNING: NaN detected in outputs, replacing with zeros")
        outputs = torch.nan_to_num(outputs, 0.0)

    if torch.isinf(outputs).any():
        print("WARNING: Inf detected in outputs, clipping values")
        outputs = torch.clamp(outputs, -50, 50)

    # Verify shapes
    if len(outputs.shape) != 2:
        raise ValueError(f"Expected outputs shape [batch_size, num_classes], got {outputs.shape}")

    if len(targets.shape) != 1:
        if len(targets.shape) == 2 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        else:
            raise ValueError(f"Expected targets shape [batch_size], got {targets.shape}")

    # Verify target values are in valid range
    num_classes = outputs.shape[1]
    if targets.min() < 0 or targets.max() >= num_classes:
        raise ValueError(f"Target values must be in range [0, {num_classes-1}], got min={targets.min()}, max={targets.max()}")

    # Use standard CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(**kwargs)
    return criterion(outputs, targets)


class SafeCrossEntropyLoss(nn.Module):
    """
    A safe wrapper around CrossEntropyLoss that handles CUDA errors.
    """

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, size_average=size_average, ignore_index=ignore_index,
            reduce=reduce, reduction=reduction, label_smoothing=label_smoothing
        )

    def forward(self, outputs, targets):
        return safe_cross_entropy_loss(outputs, targets)


def patch_distributed_trainer():
    """
    Apply patch to distributed trainer to use safe loss function.
    """
    import sys
    from pathlib import Path

    # Read the distributed trainer file
    trainer_path = Path(__file__).parent / "src" / "training" / "distributed_trainer.py"

    if not trainer_path.exists():
        print(f"Trainer file not found: {trainer_path}")
        return False

    # Read current content
    with open(trainer_path, 'r') as f:
        content = f.read()

    # Apply patches
    patches = [
        # Replace standard CrossEntropyLoss with safe version
        ("nn.CrossEntropyLoss()", "SafeCrossEntropyLoss()"),

        # Add import for safe loss function
        ("import torch.nn as nn",
         "import torch.nn as nn\nfrom fix_cuda_loss_error import SafeCrossEntropyLoss"),

        # Add explicit type checking in training loop
        ("loss = self.criterion(outputs, targets)",
         """# Ensure proper data types and device placement
            outputs = outputs.float()
            targets = targets.long()
            targets = targets.to(outputs.device)
            loss = self.criterion(outputs, targets)"""),
    ]

    modified = False
    for old, new in patches:
        if old in content and new not in content:
            content = content.replace(old, new)
            modified = True
            print(f" Applied patch: {old[:50]}...")

    if modified:
        # Backup original file
        backup_path = trainer_path.with_suffix('.py.backup')
        with open(backup_path, 'w') as f:
            with open(trainer_path, 'r') as orig:
                f.write(orig.read())

        # Write patched version
        with open(trainer_path, 'w') as f:
            f.write(content)

        print(f" Patched distributed trainer. Backup saved to {backup_path}")
        return True
    else:
        print("No patches needed or already applied")
        return False


def create_safe_training_script():
    """
    Create a safe training script with error handling.
    """
    safe_script = """#!/usr/bin/env python3
'''
Safe training script with CUDA error handling.
'''

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models import create_model
from src.data import create_data_loaders

class SafeTrainer:
    def __init__(self, model_type='xception'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create model with error handling
        try:
            self.model = create_model(model_type, num_classes=2)
            self.model = self.model.to(self.device)
            print(f" Model created: {model_type}")
        except Exception as e:
            print(f" Model creation failed: {e}")
            raise

        # Use safe loss function
        self.criterion = SafeCrossEntropyLoss()

    def safe_forward(self, images, targets):
        '''Safe forward pass with error handling.'''
        try:
            # Ensure proper data types and device
            images = images.to(self.device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(self.device, dtype=torch.long, non_blocking=True)

            # Forward pass
            outputs = self.model(images)

            # Compute loss with safety checks
            loss = self.criterion(outputs, targets)

            return outputs, loss

        except RuntimeError as e:
            if "nll_loss_forward_reduce_cuda_kernel" in str(e):
                print(f"CUDA loss error detected. Attempting fix...")

                # Force CPU computation as fallback
                images_cpu = images.cpu()
                targets_cpu = targets.cpu()
                model_cpu = self.model.cpu()

                outputs_cpu = model_cpu(images_cpu)
                loss_cpu = self.criterion(outputs_cpu, targets_cpu)

                # Move back to GPU
                self.model = self.model.to(self.device)

                return outputs_cpu.to(self.device), loss_cpu.to(self.device)
            else:
                raise

def test_safe_training():
    '''Test the safe training approach.'''
    print(" Testing safe training...")

    try:
        trainer = SafeTrainer('xception')

        # Create dummy data
        batch_size = 4
        dummy_images = torch.randn(batch_size, 3, 224, 224)
        dummy_targets = torch.tensor([0, 1, 0, 1])  # Binary classification

        outputs, loss = trainer.safe_forward(dummy_images, dummy_targets)

        print(f" Safe training test passed!")
        print(f"   Output shape: {outputs.shape}")
        print(f"   Loss: {loss.item():.4f}")

    except Exception as e:
        print(f" Safe training test failed: {e}")

if __name__ == "__main__":
    test_safe_training()
"""

    script_path = Path(__file__).parent / "safe_training.py"
    with open(script_path, 'w') as f:
        f.write(safe_script)

    print(f" Created safe training script: {script_path}")


def main():
    """Main function to apply all fixes."""
    print(" Applying CUDA loss function fixes...")

    # Apply patches
    patch_distributed_trainer()

    # Create safe training script
    create_safe_training_script()

    print("\\n Fixes applied! Try running your training again.")
    print("\\nIf the error persists, try:")
    print("1. Use python safe_training.py to test")
    print("2. Check PyTorch CUDA installation: python -c 'import torch; print(torch.cuda.is_available())'")
    print("3. Update PyTorch: pip install torch torchvision --upgrade")
    print("4. Use CPU training as fallback: export CUDA_VISIBLE_DEVICES=''")


if __name__ == "__main__":
    main()