#!/usr/bin/env python3
"""
Example script for evaluating a trained pneumonia detection model.

This script demonstrates how to:
- Load a trained model
- Evaluate on test data
- Generate comprehensive performance reports
- Create visualizations
- Generate clinical reports

Usage:
    python examples/evaluate_model.py --model_path outputs/xception_model.pth --data_dir data/
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model
from src.data import create_data_loaders
from src.evaluation import ModelEvaluator, compare_models
from src.visualization import PerformanceVisualizer, create_clinical_report

import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained pneumonia detection model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory containing test/ subdirectory')
    parser.add_argument('--model_type', type=str, default='xception',
                       help='Model architecture type')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for saving results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--generate_report', action='store_true',
                       help='Generate clinical report')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Starting evaluation with device: {device}")
    print(f"Model path: {args.model_path}")
    print(f"Data directory: {args.data_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    print("Loading test data...")
    test_dir = Path(args.data_dir)

    # Create placeholder train_loader (not used for evaluation)
    train_loader, test_loader = create_data_loaders(
        train_dir=test_dir,  # Using test as both for simplicity
        test_dir=test_dir,
        batch_size=args.batch_size,
        distributed=False
    )

    print(f"   Test samples: {len(test_loader.dataset)}")

    # Load model
    print(f"Loading {args.model_type} model...")
    model = create_model(args.model_type, num_classes=2)

    # Load trained weights
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("   Loaded model weights from checkpoint")
    else:
        # Assume the file contains just the state dict
        model.load_state_dict(checkpoint)
        print("   Loaded model weights directly")

    model.to(device)
    model.eval()

    # Evaluate model
    print("Evaluating model performance...")
    evaluator = ModelEvaluator(model, device)
    results = evaluator.evaluate(test_loader, return_predictions=True)

    # Print detailed results
    evaluator.print_detailed_results(results, f"{args.model_type.title()} Model")

    # Create comprehensive visualization
    print("Creating performance visualizations...")
    visualizer = PerformanceVisualizer()

    # Comprehensive analysis
    visualizer.plot_comprehensive_analysis(
        y_true=results['labels'],
        y_pred=results['predictions'],
        y_prob=results['probabilities'],
        save_path=output_dir / f'{args.model_type}_comprehensive_analysis.png',
        show=False
    )

    # Threshold analysis
    threshold_df = evaluator.plot_threshold_analysis(
        test_loader,
        save_path=output_dir / f'{args.model_type}_threshold_analysis.png',
        show=False
    )

    # Save threshold analysis
    threshold_df.to_csv(output_dir / f'{args.model_type}_threshold_analysis.csv', index=False)

    # Generate clinical report if requested
    if args.generate_report:
        print("Generating clinical report...")
        report = create_clinical_report(
            y_true=results['labels'],
            y_pred=results['predictions'],
            y_prob=results['probabilities'],
            model_name=f"{args.model_type.title()} Pneumonia Detection Model",
            save_path=output_dir / f'{args.model_type}_clinical_report.txt'
        )

    # Save detailed results
    results_summary = {
        'model_type': args.model_type,
        'model_path': str(args.model_path),
        'test_samples': len(test_loader.dataset),
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'specificity': float(results['specificity']),
        'f1_score': float(results['f1_score']),
        'roc_auc': float(results['roc_auc']),
        'pr_auc': float(results['pr_auc']),
        'optimal_threshold': float(results['optimal_threshold']),
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'tp': int(results['tp']),
        'fp': int(results['fp']),
        'tn': int(results['tn']),
        'fn': int(results['fn'])
    }

    import json
    with open(output_dir / f'{args.model_type}_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"Evaluation completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"Key Metrics:")
    print(f"   Accuracy:  {results['accuracy']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1-Score:  {results['f1_score']:.4f}")
    print(f"   ROC AUC:   {results['roc_auc']:.4f}")


if __name__ == "__main__":
    main()