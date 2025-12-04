#!/usr/bin/env python3
"""
Retrain MobileNetV1 with CLAHE Preprocessing Augmentation

This script retrains the pneumonia detection model with CLAHE-augmented
training data to improve robustness to varying image quality and handle
blurry/low-detail pneumonia cases better.

Background:
- Current model: 97.60% accuracy, but 21 false negatives
- 13/21 (61.9%) false negatives are blurry/low-sharpness cases
- Post-hoc preprocessing improvements limited by distribution shift
- Solution: Train model to be robust to CLAHE preprocessing levels

Usage:
  Fine-tune existing model:
    python scripts/retrain_with_clahe_augmentation.py --mode finetune

  Train from scratch:
    python scripts/retrain_with_clahe_augmentation.py --mode scratch
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
import cv2
import numpy as np
import random
from datetime import datetime
from PIL import Image

from src.models.mobilenet import MobileNetFineTune
from src.data.datasets import PneumoniaDataset, get_medical_transforms
from src.training.trainer import ModelTrainer


class CLAHEAugmentedTransform:
    """
    Custom transform that applies CLAHE augmentation to grayscale images.

    Randomly applies CLAHE with varying clip limits to simulate different
    preprocessing levels the model might encounter at inference time.
    """

    def __init__(self, prob=0.5, clip_limit_range=(1.5, 3.0), grid_size=(8, 8)):
        """
        Args:
            prob: Probability of applying CLAHE
            clip_limit_range: (min, max) clip limit for CLAHE
            grid_size: Tile grid size for CLAHE
        """
        self.prob = prob
        self.clip_limit_min, self.clip_limit_max = clip_limit_range
        self.grid_size = grid_size

    def __call__(self, image):
        """
        Apply CLAHE augmentation to PIL image.

        Args:
            image: PIL Image in RGB mode

        Returns:
            Augmented PIL Image
        """
        # Only apply with probability prob
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


def get_clahe_augmented_transforms(is_training=True, clahe_prob=0.5,
                                   clahe_clip_min=1.5, clahe_clip_max=3.0):
    """
    Get transforms with CLAHE augmentation for training.

    For training: Applies CLAHE augmentation before standard augmentations
    For validation: No CLAHE, only standard preprocessing
    """
    if is_training:
        transform = transforms.Compose([
            # CLAHE augmentation (applied to PIL image)
            CLAHEAugmentedTransform(
                prob=clahe_prob,
                clip_limit_range=(clahe_clip_min, clahe_clip_max)
            ),
            # Standard medical transforms
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation: NO CLAHE, standard preprocessing only
        transform = get_medical_transforms(is_training=False)

    return transform


def load_base_model(model_path, device):
    """Load existing model as starting point for fine-tuning."""
    print(f"Loading base model from: {model_path}")
    model = MobileNetFineTune(num_classes=2, freeze_layers=0)
    model.load_custom_weights(str(model_path))
    model.to(device)
    print("✓ Base model loaded successfully")
    return model


def create_dataloaders(data_dir, batch_size, num_workers,
                       clahe_prob=0.5, clahe_clip_min=1.5, clahe_clip_max=3.0):
    """Create train and validation dataloaders with CLAHE augmentation."""

    data_dir = Path(data_dir)

    # Training data with CLAHE augmentation
    train_transform = get_clahe_augmented_transforms(
        is_training=True,
        clahe_prob=clahe_prob,
        clahe_clip_min=clahe_clip_min,
        clahe_clip_max=clahe_clip_max
    )

    train_dataset = PneumoniaDataset(
        data_dir / 'train',
        transform=train_transform
    )

    # Validation data WITHOUT CLAHE augmentation
    val_transform = get_clahe_augmented_transforms(is_training=False)

    val_dataset = PneumoniaDataset(
        data_dir / 'test',
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"CLAHE augmentation probability: {clahe_prob}")
    print(f"CLAHE clip limit range: [{clahe_clip_min}, {clahe_clip_max}]")

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Retrain with CLAHE augmentation')

    # Training mode
    parser.add_argument('--mode', type=str, choices=['finetune', 'scratch'],
                       default='finetune',
                       help='Training mode: finetune existing model or train from scratch')

    # Paths
    parser.add_argument('--base_model', type=str,
                       default='outputs/dgx_station_experiment/Best_MobilenetV1.pth',
                       help='Path to base model for fine-tuning')
    parser.add_argument('--data_dir', type=str,
                       default='/workspace/data/chest_xray',
                       help='Path to chest X-ray dataset')
    parser.add_argument('--output_dir', type=str,
                       default='outputs/clahe_augmented_finetune',
                       help='Output directory for trained model')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate (use 0.0001 for finetune, 0.001 for scratch)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')

    # CLAHE augmentation parameters
    parser.add_argument('--clahe_prob', type=float, default=0.5,
                       help='Probability of applying CLAHE augmentation')
    parser.add_argument('--clahe_clip_min', type=float, default=1.5,
                       help='Minimum CLAHE clip limit')
    parser.add_argument('--clahe_clip_max', type=float, default=3.0,
                       help='Maximum CLAHE clip limit')

    # Regularization parameters
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate for regularization (default: 0.5)')
    parser.add_argument('--normal_class_weight', type=float, default=1.3,
                       help='Weight for NORMAL class to reduce false positives (default: 1.3)')

    # System
    parser.add_argument('--gpu', type=str, default='0,1,2,3',
                       help='GPU IDs to use (comma-separated)')
    parser.add_argument('--workers', type=int, default=16,
                       help='Number of data loading workers')
    parser.add_argument('--experiment_name', type=str, default='clahe_aug_v1',
                       help='Experiment name for logging')

    args = parser.parse_args()

    # Set GPU devices
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    # Print configuration
    print("="*80)
    print("RETRAINING WITH CLAHE AUGMENTATION")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Base model: {args.base_model if args.mode == 'finetune' else 'None (training from scratch)'}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"CLAHE probability: {args.clahe_prob}")
    print(f"CLAHE clip range: [{args.clahe_clip_min}, {args.clahe_clip_max}]")
    print("="*80)
    print()

    # Create model
    if args.mode == 'finetune':
        model = load_base_model(args.base_model, device)
        print(f"Note: Fine-tuned model uses original dropout rate from checkpoint")
    else:
        print(f"Creating new MobileNetV1 model with dropout_rate={args.dropout_rate}...")
        model = MobileNetFineTune(num_classes=2, freeze_layers=0, dropout_rate=args.dropout_rate)
        model.to(device)
        print("✓ Model created")

    # Create dataloaders
    print("\nPreparing datasets...")
    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        args.batch_size,
        args.workers,
        args.clahe_prob,
        args.clahe_clip_min,
        args.clahe_clip_max
    )

    # Setup training with class weighting to reduce false positives
    # Class 0 = PNEUMONIA (weight 1.0), Class 1 = NORMAL (weight from args)
    # Higher weight on NORMAL class penalizes false positives more
    class_weights = torch.tensor([1.0, args.normal_class_weight]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Using class weights: PNEUMONIA=1.0, NORMAL={args.normal_class_weight}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3
    )

    # Create trainer
    log_dir = output_dir / 'tensorboard' / args.experiment_name
    trainer = ModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        log_dir=str(log_dir)
    )

    # Train model
    print("\nStarting training...")
    print(f"Target: Improve from 97.60% accuracy, 21 FN to ~98.0%+, <18 FN")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    print()

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=5,  # Reduced from 7 to stop sooner
        save_best=True
    )

    # Save best model
    best_model_path = output_dir / 'best_model.pth'
    if trainer.best_model_state:
        torch.save(trainer.best_model_state, best_model_path)
        print(f"\n✓ Best model saved to: {best_model_path}")

    # Save final model
    final_model_path = output_dir / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"✓ Final model saved to: {final_model_path}")

    # Print summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Model saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Validate: python validation/scripts/test_1000_images_conservative.py \\")
    print(f"            --model_path {output_dir}/best_model.pth")
    print(f"2. Test FN cases: python validation/scripts/test_specific_cases.py \\")
    print(f"                 --model_path {output_dir}/best_model.pth")
    print("="*80)


if __name__ == "__main__":
    main()
