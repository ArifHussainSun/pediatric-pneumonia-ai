"""
Training utilities for pediatric pneumonia detection models.

This module provides comprehensive training framework including:
- Unified training pipeline for all model architectures
- Early stopping and learning rate scheduling
- Training history tracking and visualization
- Cross-validation support
- Model checkpointing and persistence

Optimized for medical imaging with proper validation techniques
and comprehensive monitoring.
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Unified training framework for all model types.

    Provides comprehensive training functionality including:
    - Training and validation loops
    - Early stopping
    - Learning rate scheduling
    - Training history tracking
    - Model checkpointing
    - TensorBoard logging

    Args:
        model: PyTorch model to train
        criterion: Loss function
        optimizer: Optimizer
        device: Computing device (GPU/CPU)
        scheduler: Learning rate scheduler (optional)
        log_dir: Directory for TensorBoard logs (optional)
    """

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: Optional[torch.device] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 log_dir: Optional[str] = None):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'epoch_times': []
        }

        # TensorBoard logging
        self.writer = None
        if log_dir:
            self.writer = SummaryWriter(log_dir)

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_model_state = None

        print(f"Trainer initialized. Using device: {self.device}")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (epoch_loss, epoch_accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1} - Training')

        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(self.device), labels.long().to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Ensure proper data types for CUDA loss computation
            if outputs.dtype != torch.float32:
                outputs = outputs.float()
            if labels.dtype != torch.long:
                labels = labels.long()

            # Move to same device if needed
            if outputs.device != labels.device:
                labels = labels.to(outputs.device)

            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            current_loss = running_loss / (batch_idx + 1)
            current_acc = correct / total
            progress_bar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{100.*current_acc:.2f}%'
            })

            # Log batch metrics
            if self.writer and batch_idx % 100 == 0:
                global_step = self.current_epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/Batch_Acc', current_acc, global_step)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (epoch_loss, epoch_accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {self.current_epoch + 1} - Validation'):
                inputs, labels = inputs.to(self.device), labels.long().to(self.device)

                outputs = self.model(inputs)
                # Ensure proper data types for CUDA loss computation
                outputs = outputs.float()  # Ensure float32
                labels = labels.long()     # Ensure int64 (long)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             epochs: int,
             early_stopping_patience: Optional[int] = None,
             save_best: bool = True,
             verbose: bool = True) -> Dict[str, List[float]]:
        """
        Complete training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            early_stopping_patience: Early stopping patience (optional)
            save_best: Whether to save best model state
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        if verbose:
            print(f"Starting training for {epochs} epochs...")
            print("-" * 60)

        start_time = time.time()
        patience_counter = 0

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation phase
            val_loss, val_acc = self.validate_epoch(val_loader)

            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            self.history['epoch_times'].append(epoch_time)

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
                self.writer.add_scalar('Train/Accuracy_Epoch', train_acc, epoch)
                self.writer.add_scalar('Val/Loss_Epoch', val_loss, epoch)
                self.writer.add_scalar('Val/Accuracy_Epoch', val_acc, epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            # Print progress
            if verbose:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                      f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")

            # Save best model
            if save_best and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                if verbose:
                    print(f"    → New best validation accuracy: {self.best_val_acc:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break

        # Load best model
        if save_best and self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"Loaded best model with validation accuracy: {self.best_val_acc:.4f}")

        total_time = time.time() - start_time
        if verbose:
            print(f"\nTraining completed in {total_time:.2f} seconds")
            print(f"Average epoch time: {np.mean(self.history['epoch_times']):.2f}s")

        return self.history

    def save_checkpoint(self, filepath: str, include_history: bool = True):
        """
        Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
            include_history: Whether to include training history
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'current_epoch': self.current_epoch,
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if include_history:
            checkpoint['history'] = self.history

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.current_epoch = checkpoint.get('current_epoch', 0)

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        print(f"Checkpoint loaded from {filepath}")

    def plot_training_history(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot training history.

        Args:
            save_path: Path to save plot (optional)
            show: Whether to display plot
        """
        if not self.history['train_loss']:
            print("No training history to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc', color='blue')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc', color='red')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Learning rate plot
        axes[1, 0].plot(self.history['lr'], color='green')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)

        # Epoch time plot
        if self.history['epoch_times']:
            axes[1, 1].plot(self.history['epoch_times'], color='orange')
            axes[1, 1].set_title('Epoch Training Time')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].remove()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

    def cleanup(self):
        """Clean up resources."""
        if self.writer:
            self.writer.close()


class TrainingConfig:
    """Configuration class for training parameters."""

    def __init__(self,
                 batch_size: int = 32,
                 epochs: int = 50,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 early_stopping_patience: Optional[int] = 10,
                 val_split: float = 0.2,
                 scheduler_type: str = 'plateau',
                 scheduler_params: Optional[Dict[str, Any]] = None):

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.val_split = val_split
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'early_stopping_patience': self.early_stopping_patience,
            'val_split': self.val_split,
            'scheduler_type': self.scheduler_type,
            'scheduler_params': self.scheduler_params
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.

    Args:
        model: PyTorch model
        config: Training configuration

    Returns:
        Configured optimizer
    """
    return optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )


def create_scheduler(optimizer: torch.optim.Optimizer,
                    config: TrainingConfig) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler based on configuration.

    Args:
        optimizer: PyTorch optimizer
        config: Training configuration

    Returns:
        Configured scheduler or None
    """
    if config.scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_params.get('factor', 0.5),
            patience=config.scheduler_params.get('patience', 5),
            threshold=config.scheduler_params.get('threshold', 1e-4),
            min_lr=config.scheduler_params.get('min_lr', 1e-7)
        )
    elif config.scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.scheduler_params.get('eta_min', 1e-6)
        )
    elif config.scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_params.get('step_size', 30),
            gamma=config.scheduler_params.get('gamma', 0.1)
        )
    elif config.scheduler_type == 'exponential':
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.scheduler_params.get('gamma', 0.95)
        )
    else:
        return None


def cross_validate_model(model_factory,
                        dataset,
                        config: TrainingConfig,
                        k_folds: int = 5,
                        criterion_factory=None,
                        device: Optional[torch.device] = None) -> pd.DataFrame:
    """
    Perform k-fold cross-validation.

    Args:
        model_factory: Function that returns a new model instance
        dataset: Dataset for cross-validation
        config: Training configuration
        k_folds: Number of folds
        criterion_factory: Function that returns criterion (optional)
        device: Computing device

    Returns:
        DataFrame with cross-validation results
    """
    print(f"Performing {k_folds}-fold cross-validation...")

    if criterion_factory is None:
        criterion_factory = lambda: nn.BCEWithLogitsLoss()

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare stratified k-fold
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        print(f"\nFold {fold + 1}/{k_folds}")
        print("-" * 40)

        # Create fold datasets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)

        # Create model, optimizer, scheduler
        model = model_factory()
        criterion = criterion_factory()
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)

        # Create trainer
        trainer = ModelTrainer(model, criterion, optimizer, device, scheduler)

        # Train with fewer epochs for CV
        cv_epochs = min(config.epochs, 20)  # Limit epochs for CV
        history = trainer.train(
            train_loader, val_loader,
            epochs=cv_epochs,
            early_stopping_patience=config.early_stopping_patience,
            verbose=False
        )

        # Get final validation metrics
        final_val_acc = history['val_acc'][-1]
        final_val_loss = history['val_loss'][-1]
        best_val_acc = max(history['val_acc'])

        fold_results.append({
            'fold': fold + 1,
            'final_val_acc': final_val_acc,
            'final_val_loss': final_val_loss,
            'best_val_acc': best_val_acc,
            'epochs_trained': len(history['val_acc'])
        })

        print(f"Fold {fold + 1} - Final Val Acc: {final_val_acc:.4f}, "
              f"Best Val Acc: {best_val_acc:.4f}")

    # Create results DataFrame
    cv_df = pd.DataFrame(fold_results)

    print(f"\n{'='*60}")
    print(f"{k_folds}-Fold Cross-Validation Results")
    print(f"{'='*60}")

    for metric in ['final_val_acc', 'best_val_acc', 'final_val_loss']:
        mean_val = cv_df[metric].mean()
        std_val = cv_df[metric].std()
        print(f"{metric.replace('_', ' ').title()::<20} {mean_val:.4f} ± {std_val:.4f}")

    return cv_df


if __name__ == "__main__":
    # Test training utilities
    print("Testing training utilities...")

    # Create dummy model for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    # Test configuration
    config = TrainingConfig(
        batch_size=16,
        epochs=5,
        learning_rate=1e-3,
        early_stopping_patience=3
    )

    print("Configuration created:")
    print(json.dumps(config.to_dict(), indent=2))

    # Test optimizer and scheduler creation
    model = DummyModel()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Scheduler: {type(scheduler).__name__ if scheduler else 'None'}")

    print("Training utilities ready!")