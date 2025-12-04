#!/usr/bin/env python3
"""
Train ResNet50 Standalone for Pneumonia Detection

Trains ResNet50 (ImageNet pretrained) with CLAHE augmentation to compare
against MobileNet variants and understand if the larger model performs better.

Purpose:
- Benchmark ResNet50 performance against MobileNet v4 and v5
- Determine if larger model architecture improves pneumonia detection
- Assess if distillation is necessary or if ResNet50 alone is sufficient

Usage:
    python3 scripts/train_resnet50_standalone.py \
        --data_dir ~/pediatric-pneumonia-ai/data \
        --output_dir outputs/resnet50_standalone \
        --epochs 30 \
        --batch_size 64
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import cv2
import numpy as np
import random
from datetime import datetime
from PIL import Image

from src.data.datasets import PneumoniaDataset as BasePneumoniaDataset


class CLAHEAugmentedTransform:
    """Custom transform that applies CLAHE augmentation to grayscale images."""

    def __init__(self, prob=0.5, clip_limit_range=(1.5, 3.0), grid_size=(8, 8)):
        self.prob = prob
        self.clip_limit_min, self.clip_limit_max = clip_limit_range
        self.grid_size = grid_size

    def __call__(self, image):
        """Apply CLAHE augmentation to PIL image."""
        if random.random() > self.prob:
            return image

        # Convert to grayscale numpy array
        gray = np.array(image.convert('L'))

        # Random clip limit within range
        clip_limit = random.uniform(self.clip_limit_min, self.clip_limit_max)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=self.grid_size)
        enhanced = clahe.apply(gray)

        # Convert back to PIL RGB
        return Image.fromarray(enhanced).convert('RGB')


def get_clahe_augmented_transforms(is_training=True, clahe_prob=0.4):
    """Get transforms with CLAHE augmentation."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if is_training:
        return transforms.Compose([
            CLAHEAugmentedTransform(prob=clahe_prob),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])


# Use the proper PneumoniaDataset from src.data.datasets
# No need to redefine it here


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100.0 * correct / total


def validate(model, dataloader, device):
    """Validate model."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train ResNet50 for pneumonia detection')
    parser.add_argument('--data_dir', type=str, default='~/pediatric-pneumonia-ai/data',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='outputs/resnet50_standalone',
                       help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--clahe_prob', type=float, default=0.4,
                       help='Probability of applying CLAHE augmentation')

    args = parser.parse_args()

    # Expand paths
    data_dir = Path(args.data_dir).expanduser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("TRAINING RESNET50 STANDALONE FOR PNEUMONIA DETECTION")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"CLAHE probability: {args.clahe_prob}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print("=" * 80)

    # Load ResNet50
    print("\nLoading ResNet50 (ImageNet pretrained)...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Modify final layer for 2 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)
    print(f"✓ ResNet50 loaded with {num_features} features")

    # Prepare datasets
    print("\nPreparing datasets...")
    train_transform = get_clahe_augmented_transforms(is_training=True, clahe_prob=args.clahe_prob)
    val_transform = get_clahe_augmented_transforms(is_training=False)

    train_dataset = BasePneumoniaDataset(data_dir / 'train', transform=train_transform)
    val_dataset = BasePneumoniaDataset(data_dir / 'test', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=8, pin_memory=True)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("\nStarting training...")
    print(f"Goal: Compare ResNet50 against MobileNet v4 (96.10%, 4 FN) and v5 (95.70%, 1 FN)")
    print()

    best_val_acc = 0.0
    patience = 7
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Validate
        val_acc = validate(model, val_loader, device)
        print(f"Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print(f"✓ New best model saved: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break

        # Step scheduler
        scheduler.step()

    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pth')

    # Save training info
    with open(output_dir / 'training_info.txt', 'w') as f:
        f.write(f"Model: ResNet50 (ImageNet pretrained)\n")
        f.write(f"Training date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Epochs trained: {epoch}\n")
        f.write(f"CLAHE probability: {args.clahe_prob}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Test: python3 validation/scripts/test_1000_images_conservative.py \\")
    print(f"         --model_path {output_dir}/best_model.pth \\")
    print("         --data_dir ~/pediatric-pneumonia-ai/data/test \\")
    print("         --no-preprocessing --model_type resnet50")
    print("=" * 80)


if __name__ == '__main__':
    main()
