"""
Distributed training script for pediatric pneumonia detection models.

This module provides multi-GPU distributed training capabilities optimized
for DGX stations with 4x Tesla V100 GPUs. Supports all model architectures
from the pediatric pneumonia detection project.

Key features:
- PyTorch DistributedDataParallel (DDP) for efficient multi-GPU training
- Automatic mixed precision (AMP) for V100 optimization
- Gradient accumulation for large effective batch sizes
- Model checkpointing and resuming
- TensorBoard logging with distributed metrics
- Memory-efficient data loading with DistributedSampler
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import json
import yaml
from pathlib import Path
import argparse
from typing import Dict, Any, Optional, Tuple

# Add src to Python path for model imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model
from src.data import create_data_loaders


class DistributedTrainer:
    """
    Multi-GPU distributed trainer for pneumonia detection models.

    Optimized for DGX stations with 4x Tesla V100 GPUs (32GB VRAM each).
    Uses PyTorch DistributedDataParallel for efficient scaling.

    Args:
        config (dict): Training configuration dictionary
        rank (int): Process rank for distributed training
        world_size (int): Total number of processes
    """

    def __init__(self, config: Dict[str, Any], rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")

        # Set up distributed training
        self.setup_distributed()

        # Initialize model and move to GPU
        self.model = self.create_model()
        self.model.to(self.device)

        # Wrap model with DDP
        self.model = DDP(self.model, device_ids=[rank], output_device=rank)

        # Initialize training components
        self.criterion = self.setup_criterion()
        self.optimizer = self.setup_optimizer()
        self.scaler = GradScaler(enabled=config.get('use_amp', True))
        self.scheduler = self.setup_scheduler()

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = 0.0

        # Logging (only on rank 0)
        if self.rank == 0:
            self.setup_logging()

    def setup_distributed(self):
        """Initialize distributed training environment."""
        os.environ['MASTER_ADDR'] = self.config.get('master_addr', 'localhost')
        os.environ['MASTER_PORT'] = str(self.config.get('master_port', 12355))

        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=self.rank,
            world_size=self.world_size
        )

        # Set CUDA device
        torch.cuda.set_device(self.rank)

        if self.rank == 0:
            print(f"Initialized distributed training:")
            print(f"  World size: {self.world_size}")
            print(f"  Backend: nccl")
            print(f"  Master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")

    def create_model(self) -> nn.Module:
        """Create model based on configuration."""
        model_config = self.config['model']
        model_type = model_config['type']

        # Get params and remove model_type if it exists to avoid duplicate argument
        params = model_config.get('params', {}).copy()
        params.pop('model_type', None)  # Remove model_type from params
        return create_model(model_type, **params)

    def setup_criterion(self) -> nn.Module:
        """Setup loss function."""
        loss_type = self.config.get('loss', 'cross_entropy')

        if loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_type == 'focal':
            # Custom focal loss for imbalanced medical data
            return FocalLoss(alpha=self.config.get('focal_alpha', 0.25),
                           gamma=self.config.get('focal_gamma', 2.0))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'adamw')

        # Filter parameters (exclude frozen layers)
        params = [p for p in self.model.parameters() if p.requires_grad]

        if opt_type == 'adamw':
            return torch.optim.AdamW(
                params,
                lr=opt_config.get('lr', 1e-4),
                weight_decay=opt_config.get('weight_decay', 0.01),
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        elif opt_type == 'sgd':
            return torch.optim.SGD(
                params,
                lr=opt_config.get('lr', 0.01),
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

    def setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        sched_config = self.config.get('scheduler', {})
        if not sched_config:
            return None

        sched_type = sched_config.get('type', 'cosine')

        if sched_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=sched_config.get('min_lr', 1e-6)
            )
        elif sched_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 10),
                threshold=sched_config.get('threshold', 1e-4)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")

    def setup_logging(self):
        """Setup logging and checkpointing (rank 0 only)."""
        # Create output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup TensorBoard
        log_dir = self.output_dir / 'tensorboard'
        self.writer = SummaryWriter(log_dir)

        # Save config
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        print(f"Logging to: {self.output_dir}")
        print(f"TensorBoard logs: {log_dir}")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        # Metrics tracking
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(train_loader)

        # Set sampler epoch for proper shuffling
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(self.epoch)

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(enabled=self.config.get('use_amp', True)):
                outputs = self.model(images)

                # Ensure proper data types for CUDA loss computation
                outputs = outputs.float()  # Ensure float32
                targets = targets.long()   # Ensure int64 (long)

                loss = self.criterion(outputs, targets)

                # Scale loss for gradient accumulation
                loss = loss / self.config.get('gradient_accumulation_steps', 1)

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                # Gradient clipping
                if self.config.get('max_grad_norm'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['max_grad_norm']
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.global_step += 1

            # Statistics
            total_loss += loss.item() * self.config.get('gradient_accumulation_steps', 1)

            # Calculate accuracy
            if outputs.dim() == 2 and outputs.size(1) > 1:
                # Multi-class classification
                _, predicted = torch.max(outputs.data, 1)
            else:
                # Binary classification
                predicted = (torch.sigmoid(outputs) > 0.5).long().squeeze()

            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            # Log batch metrics (rank 0 only)
            if self.rank == 0 and batch_idx % self.config.get('log_interval', 100) == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {self.epoch}/{self.config['training']['epochs']} "
                      f"[{batch_idx}/{num_batches}] "
                      f"Loss: {loss.item():.4f} "
                      f"Acc: {100.*correct/total:.2f}% "
                      f"LR: {current_lr:.6f}")

                # TensorBoard logging
                self.writer.add_scalar('Train/Loss_Batch', loss.item(), self.global_step)
                self.writer.add_scalar('Train/Learning_Rate', current_lr, self.global_step)

        # Aggregate metrics across all processes
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }

        # Reduce metrics across all ranks
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            metrics[key] = tensor.item() / self.world_size

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Forward pass
                with autocast(enabled=self.config.get('use_amp', True)):
                    outputs = self.model(images)

                    # Ensure proper data types for CUDA loss computation
                    outputs = outputs.float()  # Ensure float32
                    targets = targets.long()   # Ensure int64 (long)

                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()

                # Calculate accuracy
                if outputs.dim() == 2 and outputs.size(1) > 1:
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    predicted = (torch.sigmoid(outputs) > 0.5).long().squeeze()

                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        # Aggregate metrics
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total
        }

        # Reduce metrics across all ranks
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            metrics[key] = tensor.item() / self.world_size

        return metrics

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint (rank 0 only)."""
        if self.rank != 0:
            return

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_metric': self.best_metric
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved with accuracy: {metrics['accuracy']:.4f}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        if self.rank == 0:
            print(f"Starting training for {self.config['training']['epochs']} epochs")
            print(f"Model: {self.config['model']['type']}")
            print(f"Batch size per GPU: {train_loader.batch_size}")
            print(f"Effective batch size: {train_loader.batch_size * self.world_size}")

        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch

            # Training
            train_metrics = self.train_epoch(train_loader)

            # Validation
            val_metrics = self.validate(val_loader)

            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()

            # Check if best model
            is_best = val_metrics['accuracy'] > self.best_metric
            if is_best:
                self.best_metric = val_metrics['accuracy']

            # Save checkpoint
            self.save_checkpoint(val_metrics, is_best)

            # Logging (rank 0 only)
            if self.rank == 0:
                print(f"Epoch {epoch}/{self.config['training']['epochs']} - "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train Acc: {train_metrics['accuracy']:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.4f}")

                # TensorBoard logging
                self.writer.add_scalar('Train/Loss_Epoch', train_metrics['loss'], epoch)
                self.writer.add_scalar('Train/Accuracy_Epoch', train_metrics['accuracy'], epoch)
                self.writer.add_scalar('Val/Loss_Epoch', val_metrics['loss'], epoch)
                self.writer.add_scalar('Val/Accuracy_Epoch', val_metrics['accuracy'], epoch)

        if self.rank == 0:
            print(f"Training completed! Best validation accuracy: {self.best_metric:.4f}")

    def cleanup(self):
        """Clean up distributed training."""
        if self.rank == 0 and hasattr(self, 'writer'):
            self.writer.close()
        dist.destroy_process_group()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in medical datasets."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_distributed_data_loaders(config: Dict[str, Any], rank: int, world_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Create distributed data loaders for training and validation.

    Args:
        config: Configuration dictionary with data settings
        rank: Process rank for distributed training
        world_size: Total number of processes

    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_config = config.get('data', {})

    # Extract data paths from config
    train_dir = data_config.get('train_dir', 'data/train')
    val_dir = data_config.get('val_dir', 'data/val')

    # Create data loaders with distributed sampling
    train_loader, val_loader = create_data_loaders(
        train_dir=train_dir,
        test_dir=val_dir,
        batch_size=config.get('training', {}).get('batch_size', 32),
        num_workers=data_config.get('num_workers', 4),
        image_size=(data_config.get('image_size', 224), data_config.get('image_size', 224)),
        use_weighted_sampling=False,  # Disable for distributed training
        distributed=True,  # Enable distributed sampling
        pin_memory=data_config.get('pin_memory', True)
    )

    return train_loader, val_loader


def run_distributed_training(rank: int, world_size: int, config: Dict[str, Any]):
    """Run distributed training on a single process."""
    trainer = None
    try:
        # Initialize trainer
        trainer = DistributedTrainer(config, rank, world_size)

        # Create data loaders
        train_loader, val_loader = create_distributed_data_loaders(config, rank, world_size)

        if train_loader is None or val_loader is None:
            if rank == 0:
                print("Data loaders not available. Please implement create_data_loaders function.")
            return

        # Start training
        trainer.train(train_loader, val_loader)

    except Exception as e:
        print(f"Rank {rank} failed with error: {e}")
        raise
    finally:
        # Clean up
        if trainer is not None:
            trainer.cleanup()


def main():
    """Main entry point for distributed training."""
    parser = argparse.ArgumentParser(description='Distributed Training for Pneumonia Detection')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration file')
    parser.add_argument('--world-size', type=int, default=4,
                       help='Number of GPUs to use (default: 4 for DGX)')
    parser.add_argument('--master-addr', type=str, default='localhost',
                       help='Master node address')
    parser.add_argument('--master-port', type=str, default='12355',
                       help='Master node port')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Update config with command line args
    config['master_addr'] = args.master_addr
    config['master_port'] = int(args.master_port)

    # Spawn processes for distributed training
    mp.spawn(
        run_distributed_training,
        args=(args.world_size, config),
        nprocs=args.world_size,
        join=True
    )


if __name__ == '__main__':
    main()