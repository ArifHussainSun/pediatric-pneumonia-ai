#!/usr/bin/env python3
"""
Example script for comparing multiple trained pneumonia detection models.

This script demonstrates how to:
- Load multiple trained models
- Evaluate all models on the same test set
- Generate comparative analysis
- Create comparison visualizations

Usage:
    python examples/compare_models.py --models xception:outputs/xception_model.pth vgg:outputs/vgg_model.pth --data_dir data/
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model
from src.data import create_data_loaders
from src.evaluation import compare_models
from src.visualization import PerformanceVisualizer

import torch
import json


def main():
    parser = argparse.ArgumentParser(description='Compare multiple pneumonia detection models')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                       help='Model specifications in format "type:path" (e.g., xception:model.pth)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory containing test/ subdirectory')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                       help='Output directory for saving results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Starting model comparison with device: {device}")
    print(f"Data directory: {args.data_dir}")

    # Parse model specifications
    models_dict = {}
    for model_spec in args.models:
        try:
            model_type, model_path = model_spec.split(':')
            models_dict[model_type] = model_path
        except ValueError:
            print(f"Invalid model specification: {model_spec}")
            print("   Format should be 'type:path' (e.g., xception:model.pth)")
            return

    print(f"Loading {len(models_dict)} models:")
    for model_type, model_path in models_dict.items():
        print(f"   {model_type}: {model_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    print("Loading test data...")
    test_dir = Path(args.data_dir) / 'test'

    # Create placeholder train_loader (not used for evaluation)
    train_loader, test_loader = create_data_loaders(
        train_dir=test_dir,  # Using test as both for simplicity
        test_dir=test_dir,
        batch_size=args.batch_size,
        distributed=False
    )

    print(f"   Test samples: {len(test_loader.dataset)}")

    # Load all models
    loaded_models = {}
    for model_type, model_path in models_dict.items():
        print(f"Loading {model_type} model...")

        # Create model
        model = create_model(model_type, num_classes=2)

        # Load trained weights
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.to(device)
            model.eval()
            loaded_models[model_type] = model
            print(f"   Successfully loaded {model_type}")

        except Exception as e:
            print(f"   Failed to load {model_type}: {e}")
            continue

    if not loaded_models:
        print("No models were successfully loaded!")
        return

    print(f"Evaluating and comparing {len(loaded_models)} models...")

    # Compare models
    results, comparison_df = compare_models(
        models_dict=loaded_models,
        test_loader=test_loader,
        save_path=output_dir / 'model_comparison.png',
        device=device
    )

    # Additional comparison visualizations
    visualizer = PerformanceVisualizer()

    # Create simplified results dict for visualization
    viz_results = {}
    for model_name, model_results in results.items():
        viz_results[model_name] = {
            'accuracy': model_results['accuracy'],
            'precision': model_results['precision'],
            'recall': model_results['recall'],
            'f1_score': model_results['f1_score'],
            'roc_auc': model_results['roc_auc'],
            'specificity': model_results['specificity']
        }

    visualizer.plot_model_comparison(
        results_dict=viz_results,
        save_path=output_dir / 'detailed_comparison.png',
        show=False
    )

    # Save detailed comparison results
    comparison_results = {
        'models_compared': list(loaded_models.keys()),
        'test_samples': len(test_loader.dataset),
        'comparison_metrics': comparison_df.to_dict(),
        'detailed_results': {}
    }

    # Add detailed results (excluding arrays for JSON serialization)
    for model_name, model_results in results.items():
        comparison_results['detailed_results'][model_name] = {
            k: float(v) if isinstance(v, (float, int)) else v
            for k, v in model_results.items()
            if k not in ['fpr', 'tpr', 'roc_thresholds', 'precision_curve',
                        'recall_curve', 'pr_thresholds', 'confusion_matrix',
                        'predictions', 'probabilities', 'labels']
        }

    # Save results
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)

    comparison_df.to_csv(output_dir / 'comparison_metrics.csv')

    # Print summary
    print(f"\nMODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(comparison_df.round(4))

    # Find best performers
    print(f"\nBEST PERFORMERS:")
    for metric in comparison_df.index:
        best_model = comparison_df.loc[metric].idxmax()
        best_score = comparison_df.loc[metric].max()
        print(f"   {metric.replace('_', ' ').title():>12}: {best_model} ({best_score:.4f})")

    print(f"\nModel comparison completed successfully!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()