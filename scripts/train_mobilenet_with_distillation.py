#!/usr/bin/env python3
"""
Train MobileNetV1 with CLAHE Augmentation + Knowledge Distillation

Uses ResNet50 (and optionally VGG16) as teacher ensemble to improve MobileNetV1
performance on blurry/low-detail pneumonia cases.

Student: MobileNetV1 v4 (CLAHE-augmented model)
Teacher: ResNet50 (ImageNet pretrained) + optional VGG16

Usage:
  # With ResNet50 teacher (recommended)
  python3 scripts/train_mobilenet_with_distillation.py \
    --use_resnet_teacher \
    --data_dir ~/pediatric-pneumonia-ai/data

  # With both ResNet50 + VGG16 teachers (if you have trained VGG16)
  python3 scripts/train_mobilenet_with_distillation.py \
    --vgg_teacher outputs/vgg16_teacher/best_model.pth \
    --use_resnet_teacher \
    --data_dir ~/pediatric-pneumonia-ai/data
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import cv2
import numpy as np
import random
from PIL import Image
from tqdm import tqdm

from src.models.mobilenet import MobileNetFineTune
from src.data.datasets import PneumoniaDataset, get_medical_transforms

# Optional: only import VGG if we have it
try:
    from src.models.vgg import VGG16FineTune
    HAS_VGG = True
except ImportError:
    HAS_VGG = False


class CLAHEAugmentedTransform:
    """Apply CLAHE augmentation with random clip limits."""

    def __init__(self, prob=0.5, clip_limit_range=(1.5, 3.0), grid_size=(8, 8)):
        self.prob = prob
        self.clip_limit_min, self.clip_limit_max = clip_limit_range
        self.grid_size = grid_size

    def __call__(self, image):
        if random.random() > self.prob:
            return image

        gray = np.array(image.convert('L'))
        clip_limit = random.uniform(self.clip_limit_min, self.clip_limit_max)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=self.grid_size)
        enhanced = clahe.apply(gray)
        return Image.fromarray(enhanced).convert('RGB')


def get_clahe_augmented_transforms(is_training=True, clahe_prob=0.5,
                                   clahe_clip_min=1.5, clahe_clip_max=3.0):
    """Get transforms with CLAHE augmentation."""
    if is_training:
        return transforms.Compose([
            CLAHEAugmentedTransform(prob=clahe_prob,
                                   clip_limit_range=(clahe_clip_min, clahe_clip_max)),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return get_medical_transforms(is_training=False)


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining:
    - Hard loss: Cross-entropy with true labels
    - Soft loss: KL divergence with teacher predictions

    Temperature: Controls softness of probability distributions (higher = softer)
    Alpha: Weight for hard loss (1-alpha for soft loss)
    """

    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        # Hard loss with true labels
        hard_loss = self.ce_loss(student_logits, labels)

        # Soft loss with teacher predictions
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)

        # Combine losses
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


def load_teachers(vgg_path=None, use_resnet=False, device='cuda'):
    """Load teacher models."""
    teachers = []
    teacher_names = []

    # Load VGG16 teacher if provided
    if vgg_path and Path(vgg_path).exists():
        if not HAS_VGG:
            print("Warning: VGG16 path provided but VGG16FineTune not available")
        else:
            print(f"Loading VGG16 teacher from: {vgg_path}")
            vgg = VGG16FineTune(num_classes=2)
            checkpoint = torch.load(vgg_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                vgg.load_state_dict(checkpoint['model_state_dict'])
            else:
                vgg.load_state_dict(checkpoint)
            vgg.to(device)
            vgg.eval()
            teachers.append(vgg)
            teacher_names.append('VGG16')
            print("✓ VGG16 teacher loaded")

    # Load ResNet50 teacher (ImageNet pretrained)
    if use_resnet:
        print("Loading ResNet50 teacher (ImageNet pretrained)")
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Modify final layer for binary classification
        resnet.fc = nn.Linear(resnet.fc.in_features, 2)
        resnet.to(device)
        resnet.eval()
        teachers.append(resnet)
        teacher_names.append('ResNet50')
        print("✓ ResNet50 teacher loaded")

    if not teachers:
        raise ValueError("No teachers loaded! Provide --vgg_teacher or --use_resnet_teacher")

    print(f"\nTeacher Ensemble: {' + '.join(teacher_names)}")
    return teachers, teacher_names


def ensemble_teacher_predictions(teachers, inputs):
    """Get averaged predictions from teacher ensemble."""
    with torch.no_grad():
        teacher_outputs = [teacher(inputs) for teacher in teachers]
        # Average the logits from all teachers
        ensemble_logits = torch.stack(teacher_outputs).mean(dim=0)
    return ensemble_logits


def train_epoch_with_distillation(model, teachers, train_loader, criterion, optimizer, device):
    """Train one epoch with knowledge distillation."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Get teacher predictions (ensemble average)
        teacher_logits = ensemble_teacher_predictions(teachers, inputs)

        # Forward pass
        optimizer.zero_grad()
        student_logits = model(inputs)

        # Calculate distillation loss
        loss = criterion(student_logits, teacher_logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = student_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, device):
    """Validate model."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train MobileNet with CLAHE + Knowledge Distillation')

    # Paths
    parser.add_argument('--student_model', type=str,
                       default='outputs/clahe_augmented_finetune_v4/best_model.pth',
                       help='Path to student MobileNet v4 model')
    parser.add_argument('--vgg_teacher', type=str, default=None,
                       help='Path to trained VGG16 teacher model (optional)')
    parser.add_argument('--use_resnet_teacher', action='store_true',
                       help='Use pretrained ResNet50 as teacher (recommended)')
    parser.add_argument('--data_dir', type=str,
                       default='~/pediatric-pneumonia-ai/data',
                       help='Path to dataset (train/ and test/ subdirectories)')
    parser.add_argument('--output_dir', type=str,
                       default='outputs/mobilenet_distillation_v1',
                       help='Output directory for trained model')

    # Training
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.00005,
                       help='Learning rate (lower than initial training)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')

    # CLAHE augmentation
    parser.add_argument('--clahe_prob', type=float, default=0.4,
                       help='Probability of applying CLAHE augmentation')
    parser.add_argument('--clahe_clip_min', type=float, default=1.5,
                       help='Minimum CLAHE clip limit')
    parser.add_argument('--clahe_clip_max', type=float, default=3.0,
                       help='Maximum CLAHE clip limit')

    # Distillation
    parser.add_argument('--temperature', type=float, default=3.0,
                       help='Temperature for distillation (higher = softer distributions)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Weight for hard loss (1-alpha for soft loss)')

    # System
    parser.add_argument('--gpu', type=str, default='0,1,2,3',
                       help='GPU IDs to use (comma-separated)')
    parser.add_argument('--workers', type=int, default=16,
                       help='Number of data loading workers')

    args = parser.parse_args()

    # Setup
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Expand ~ in paths
    data_dir = Path(args.data_dir).expanduser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("TRAINING MOBILENET V4 WITH KNOWLEDGE DISTILLATION")
    print("="*80)
    print(f"Device: {device}")
    print(f"GPUs: {torch.cuda.device_count()}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Student model: {args.student_model}")
    print(f"CLAHE probability: {args.clahe_prob}")
    print(f"CLAHE clip range: [{args.clahe_clip_min}, {args.clahe_clip_max}]")
    print(f"Distillation temperature: {args.temperature}")
    print(f"Hard/soft loss ratio: {args.alpha}/{1-args.alpha}")
    print("="*80)

    # Load teachers
    teachers, teacher_names = load_teachers(
        vgg_path=args.vgg_teacher,
        use_resnet=args.use_resnet_teacher,
        device=device
    )

    # Load student model (MobileNetV1 v4)
    print(f"\nLoading student model: {args.student_model}")
    student = MobileNetFineTune(num_classes=2, freeze_layers=0)
    if not Path(args.student_model).exists():
        raise FileNotFoundError(f"Student model not found: {args.student_model}")
    student.load_custom_weights(str(args.student_model))
    student.to(device)
    print("✓ Student model loaded (MobileNetV1 v4 - CLAHE-augmented)")

    # Create dataloaders
    print("\nPreparing datasets...")

    train_transform = get_clahe_augmented_transforms(
        is_training=True,
        clahe_prob=args.clahe_prob,
        clahe_clip_min=args.clahe_clip_min,
        clahe_clip_max=args.clahe_clip_max
    )

    train_dataset = PneumoniaDataset(data_dir / 'train', transform=train_transform)
    val_dataset = PneumoniaDataset(data_dir / 'test',
                                   transform=get_clahe_augmented_transforms(is_training=False))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.workers, pin_memory=True)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Setup training
    criterion = DistillationLoss(temperature=args.temperature, alpha=args.alpha)
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate,
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=0.5, patience=3)

    # Training loop
    print("\nStarting training with knowledge distillation...")
    print(f"Goal: Leverage teacher knowledge to further improve v4 model")
    print(f"Teachers: {' + '.join(teacher_names)}")
    print(f"V4 baseline: 96.10% accuracy, 99.20% sensitivity, 4 FN")
    print()

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    early_stopping_patience = 7

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch_with_distillation(
            student, teachers, train_loader, criterion, optimizer, device
        )

        # Validate
        val_acc = validate(student, val_loader, device)

        # Update scheduler
        scheduler.step(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            best_path = output_dir / 'best_model.pth'
            torch.save(student.state_dict(), best_path)
            print(f"✓ New best model saved: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{early_stopping_patience})")

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
            break

    # Save final model
    final_path = output_dir / 'final_model.pth'
    torch.save(student.state_dict(), final_path)

    # Save training info
    info_path = output_dir / 'training_info.txt'
    with open(info_path, 'w') as f:
        f.write(f"Knowledge Distillation Training\n")
        f.write(f"================================\n\n")
        f.write(f"Student: MobileNetV1 v4 (CLAHE-augmented)\n")
        f.write(f"Teachers: {', '.join(teacher_names)}\n\n")
        f.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Total epochs: {epoch+1}\n\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  CLAHE probability: {args.clahe_prob}\n")
        f.write(f"  Temperature: {args.temperature}\n")
        f.write(f"  Alpha (hard/soft): {args.alpha}/{1-args.alpha}\n")
        f.write(f"  Learning rate: {args.learning_rate}\n")
        f.write(f"  Batch size: {args.batch_size}\n\n")
        f.write(f"V4 Baseline Performance:\n")
        f.write(f"  Accuracy: 96.10%\n")
        f.write(f"  Sensitivity: 99.20%\n")
        f.write(f"  False Negatives: 4\n")
        f.write(f"  False Positives: 35\n")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"Teachers used: {' + '.join(teacher_names)}")
    print(f"Model saved to: {output_dir}")
    print("\nV4 Baseline: 96.10% accuracy, 99.20% sensitivity, 4 FN")
    print(f"Distillation: {best_val_acc:.2f}% validation accuracy")
    print("\nNext steps:")
    print(f"1. Test: python3 validation/scripts/test_1000_images_conservative.py \\")
    print(f"         --model_path {output_dir}/best_model.pth \\")
    print(f"         --data_dir ~/pediatric-pneumonia-ai/data/test \\")
    print(f"         --no-preprocessing")
    print("="*80)


if __name__ == "__main__":
    main()
