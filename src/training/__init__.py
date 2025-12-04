"""
Training utilities package for pediatric pneumonia detection.

This package provides comprehensive training infrastructure including:
- Single-GPU and multi-GPU distributed training
- Training configuration management
- Cross-validation utilities
- Model checkpointing and persistence
- Training history tracking and visualization

Key components:
- trainer: Core training utilities and ModelTrainer class
- distributed_trainer: Multi-GPU distributed training for DGX deployment
"""

from .trainer import (
    ModelTrainer,
    TrainingConfig,
    create_optimizer,
    create_scheduler,
    cross_validate_model
)

from .distributed_trainer import (
    DistributedTrainer,
    FocalLoss,
    create_distributed_data_loaders
)

__all__ = [
    # Core training
    'ModelTrainer',
    'TrainingConfig',
    'create_optimizer',
    'create_scheduler',
    'cross_validate_model',

    # Distributed training
    'DistributedTrainer',
    'FocalLoss',
    'create_distributed_data_loaders',
]