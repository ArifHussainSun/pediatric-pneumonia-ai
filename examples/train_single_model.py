#!/usr/bin/env python3
"""
Example script for training a single pneumonia detection model.

This script demonstrates how to:
- Load and prepare data
- Create and configure a model
- Train the model with proper validation
- Evaluate performance
- Save the trained model

Usage:
    python examples/train_single_model.py --data_dir data/ --model xception --epochs 50
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model
from src.data import create_data_loaders
from src.training import ModelTrainer, TrainingConfig, create_optimizer, create_scheduler
from src.evaluation import ModelEvaluator
from src.visualization import PerformanceVisualizer

import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser(description='Train a single pneumonia detection model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory containing train/ and test/ subdirectories')
    parser.add_argument('--model', type=str, default='xception',
                       choices=['xception', 'vgg', 'mobilenet', 'fusion', 'xception_lstm'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for saving results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Starting training with device: {device}")
    print(f"Model: {args.model}")
    print(f"Data directory: {args.data_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create data loaders
    print("Loading data...")
    train_dir = Path(args.data_dir) / 'train'
    test_dir = Path(args.data_dir) / 'test'

    train_loader, test_loader = create_data_loaders(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=args.batch_size,
        distributed=False
    )

    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")

    # Create model
    print(f"Creating {args.model} model...")
    model = create_model(args.model, num_classes=2)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Create training components
    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=10
    )

    # Fix CUDA data type issues
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Create trainer
    trainer = ModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        log_dir=output_dir / 'tensorboard'
    )

    # Train model
    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,  # Using test as validation for simplicity
        epochs=args.epochs,
        early_stopping_patience=config.early_stopping_patience
    )

    # Plot training history
    trainer.plot_training_history(
        save_path=output_dir / f'{args.model}_training_history.png',
        show=False
    )

    # Evaluate model
    print("Evaluating model...")
    evaluator = ModelEvaluator(model, device)
    results = evaluator.evaluate(test_loader, return_predictions=True)

    # Print results
    evaluator.print_detailed_results(results, args.model.title())

    # Visualize results
    visualizer = PerformanceVisualizer()
    visualizer.plot_comprehensive_analysis(
        y_true=results['labels'],
        y_pred=results['predictions'],
        y_prob=results['probabilities'],
        save_path=output_dir / f'{args.model}_evaluation.png',
        show=False
    )

    # Save model
    model_path = output_dir / f'{args.model}_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'results': {k: v for k, v in results.items() if k not in ['labels', 'predictions', 'probabilities']},
        'history': history
    }, model_path)

    print(f"Training completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"Model saved to: {model_path}")
    print(f"Final test accuracy: {results['accuracy']:.4f}")

    # Cleanup
    trainer.cleanup()


if __name__ == "__main__":
    main()