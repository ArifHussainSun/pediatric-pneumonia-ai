"""
Evaluation utilities for pediatric pneumonia detection models.

This module provides comprehensive evaluation framework including:
- Model evaluation with multiple metrics
- ROC and PR curve analysis
- Confusion matrix visualization
- Model comparison utilities
- Statistical analysis and reporting

Optimized for medical imaging applications with focus on
clinically relevant metrics.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from tqdm import tqdm


class ModelEvaluator:
    """
    Comprehensive model evaluation framework for pneumonia detection.

    Provides detailed evaluation including:
    - Classification metrics (accuracy, precision, recall, F1, specificity)
    - ROC and Precision-Recall curves
    - Confusion matrix analysis
    - Statistical significance testing
    - Visualization utilities

    Args:
        model: PyTorch model to evaluate
        device: Computing device (GPU/CPU)
        threshold: Classification threshold (default: 0.5)
    """

    def __init__(self,
                 model: nn.Module,
                 device: Optional[torch.device] = None,
                 threshold: float = 0.5):

        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.model.to(self.device)

        print(f"ModelEvaluator initialized. Using device: {self.device}")

    def evaluate(self,
                test_loader: DataLoader,
                return_predictions: bool = False,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evaluation on test set.

        Args:
            test_loader: Test data loader
            return_predictions: Whether to return raw predictions
            verbose: Whether to show progress bar

        Returns:
            Dictionary containing all evaluation metrics and curves
        """
        self.model.eval()

        all_predictions = []
        all_probabilities = []
        all_labels = []

        desc = 'Evaluating' if verbose else None
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=desc, disable=not verbose):
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                # Handle multi-class outputs (2 classes: NORMAL=1, PNEUMONIA=0)
                if outputs.dim() > 1 and outputs.size(1) > 1:
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1).float()
                    # Use probability of positive class (NORMAL=1) for binary metrics
                    prob_positive = probabilities[:, 1]  # NORMAL class probability
                else:
                    # Fallback for single output (binary)
                    outputs = outputs.squeeze()
                    probabilities = torch.sigmoid(outputs)
                    predictions = (probabilities > self.threshold).float()
                    prob_positive = probabilities

                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(prob_positive.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)

        # Calculate metrics
        results = self._calculate_metrics(all_labels, all_predictions, all_probabilities)

        if return_predictions:
            results.update({
                'predictions': all_predictions,
                'probabilities': all_probabilities,
                'labels': all_labels
            })

        return results

    def _calculate_metrics(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities

        Returns:
            Dictionary with all metrics
        """
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Basic classification metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Additional medical metrics
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        ppv = precision  # Positive Predictive Value (same as precision)

        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        # Precision-Recall curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

        # Optimal threshold using Youden's Index
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = roc_thresholds[optimal_idx]

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'npv': npv,
            'ppv': ppv,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm,
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'fpr': fpr, 'tpr': tpr, 'roc_thresholds': roc_thresholds,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'pr_thresholds': pr_thresholds,
            'optimal_threshold': optimal_threshold,
            'youden_index': youden_index[optimal_idx]
        }

    def evaluate_at_thresholds(self,
                              test_loader: DataLoader,
                              thresholds: List[float] = None) -> pd.DataFrame:
        """
        Evaluate model at multiple thresholds.

        Args:
            test_loader: Test data loader
            thresholds: List of thresholds to evaluate

        Returns:
            DataFrame with metrics at each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)

        # Get predictions once
        results = self.evaluate(test_loader, return_predictions=True, verbose=False)
        y_true = results['labels']
        y_prob = results['probabilities']

        threshold_results = []

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)

            threshold_results.append({
                'threshold': threshold,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'specificity': self._calculate_specificity(y_true, y_pred)
            })

        return pd.DataFrame(threshold_results)

    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    def plot_evaluation(self,
                       results: Dict[str, Any],
                       model_name: str = "Model",
                       save_path: Optional[str] = None,
                       show: bool = True) -> None:
        """
        Plot comprehensive evaluation results.

        Args:
            results: Results from evaluate() method
            model_name: Name for plot titles
            save_path: Path to save plot
            show: Whether to display plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Confusion Matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=results['confusion_matrix'],
            display_labels=['Normal', 'Pneumonia']
        )
        disp.plot(ax=axes[0, 0], cmap='Blues', values_format='d')
        axes[0, 0].set_title(f'{model_name} - Confusion Matrix')

        # ROC Curve
        axes[0, 1].plot(results['fpr'], results['tpr'],
                       label=f'ROC Curve (AUC = {results["roc_auc"]:.3f})',
                       linewidth=2)
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.6)

        # Mark optimal threshold
        optimal_idx = np.argmax(results['tpr'] - results['fpr'])
        axes[0, 1].plot(results['fpr'][optimal_idx], results['tpr'][optimal_idx],
                       'ro', markersize=8, label=f'Optimal (t={results["optimal_threshold"]:.3f})')

        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title(f'{model_name} - ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Precision-Recall Curve
        axes[1, 0].plot(results['recall_curve'], results['precision_curve'],
                       label=f'PR Curve (AUC = {results["pr_auc"]:.3f})',
                       linewidth=2)

        # Baseline (random classifier)
        baseline = np.sum(results['labels']) / len(results['labels']) if 'labels' in results else 0.5
        axes[1, 0].axhline(y=baseline, color='k', linestyle='--',
                          label=f'Baseline ({baseline:.3f})', alpha=0.6)

        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title(f'{model_name} - Precision-Recall Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Metrics Bar Plot
        metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'NPV']
        values = [results['accuracy'], results['precision'], results['recall'],
                 results['specificity'], results['f1_score'], results['npv']]
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']

        bars = axes[1, 1].bar(metrics, values, color=colors)
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title(f'{model_name} - Performance Metrics')
        axes[1, 1].set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

    def plot_threshold_analysis(self,
                               test_loader: DataLoader,
                               save_path: Optional[str] = None,
                               show: bool = True) -> pd.DataFrame:
        """
        Plot threshold analysis showing metrics vs threshold.

        Args:
            test_loader: Test data loader
            save_path: Path to save plot
            show: Whether to display plot

        Returns:
            DataFrame with threshold analysis
        """
        threshold_df = self.evaluate_at_thresholds(test_loader)

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(threshold_df['threshold'], threshold_df['accuracy'], 'o-', label='Accuracy')
        plt.plot(threshold_df['threshold'], threshold_df['f1_score'], 's-', label='F1-Score')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Accuracy and F1-Score vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.plot(threshold_df['threshold'], threshold_df['precision'], 'o-', label='Precision')
        plt.plot(threshold_df['threshold'], threshold_df['recall'], 's-', label='Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Precision and Recall vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        plt.plot(threshold_df['threshold'], threshold_df['specificity'], 'o-', label='Specificity')
        plt.plot(threshold_df['threshold'], threshold_df['recall'], 's-', label='Sensitivity')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Sensitivity and Specificity vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        # Calculate balanced accuracy
        balanced_acc = (threshold_df['recall'] + threshold_df['specificity']) / 2
        plt.plot(threshold_df['threshold'], balanced_acc, 'o-', label='Balanced Accuracy')
        plt.plot(threshold_df['threshold'], threshold_df['f1_score'], 's-', label='F1-Score')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Balanced Accuracy and F1-Score vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        return threshold_df

    def print_detailed_results(self,
                              results: Dict[str, Any],
                              model_name: str = "Model") -> None:
        """
        Print detailed evaluation results.

        Args:
            results: Results from evaluate() method
            model_name: Model name for header
        """
        print(f"\n{'='*70}")
        print(f"{model_name.upper()} - DETAILED EVALUATION RESULTS")
        print(f"{'='*70}")

        print(f"\nCLASSIFICATION METRICS:")
        print(f"   Accuracy:           {results['accuracy']:.4f}")
        print(f"   Precision (PPV):    {results['precision']:.4f}")
        print(f"   Recall (Sensitivity): {results['recall']:.4f}")
        print(f"   Specificity:        {results['specificity']:.4f}")
        print(f"   F1-Score:           {results['f1_score']:.4f}")
        print(f"   NPV:                {results['npv']:.4f}")

        print(f"\nAUC SCORES:")
        print(f"   ROC AUC:            {results['roc_auc']:.4f}")
        print(f"   PR AUC:             {results['pr_auc']:.4f}")

        print(f"\nOPTIMAL THRESHOLD:")
        print(f"   Threshold:          {results['optimal_threshold']:.4f}")
        print(f"   Youden Index:       {results['youden_index']:.4f}")

        print(f"\n🧮 CONFUSION MATRIX:")
        print(f"   True Positives:     {results['tp']:>6d}")
        print(f"   False Positives:    {results['fp']:>6d}")
        print(f"   True Negatives:     {results['tn']:>6d}")
        print(f"   False Negatives:    {results['fn']:>6d}")

        print(f"\n📋 CONFUSION MATRIX:")
        print(f"         Predicted")
        print(f"         Normal  Pneumonia")
        print(f"Actual Normal    {results['tn']:>6d}  {results['fp']:>8d}")
        print(f"       Pneumonia {results['fn']:>6d}  {results['tp']:>8d}")

        # Clinical interpretation
        print(f"\n🏥 CLINICAL INTERPRETATION:")
        if results['recall'] > 0.9:
            print(f"   High Sensitivity: Good at catching pneumonia cases")
        elif results['recall'] > 0.8:
            print(f"   Moderate Sensitivity: May miss some pneumonia cases")
        else:
            print(f"   Low Sensitivity: Missing too many pneumonia cases")

        if results['specificity'] > 0.9:
            print(f"   High Specificity: Low false positive rate")
        elif results['specificity'] > 0.8:
            print(f"   Moderate Specificity: Some false positives")
        else:
            print(f"   Low Specificity: Too many false positives")


def compare_models(models_dict: Dict[str, nn.Module],
                  test_loader: DataLoader,
                  save_path: Optional[str] = None,
                  device: Optional[torch.device] = None) -> Tuple[Dict[str, Dict], pd.DataFrame]:
    """
    Compare multiple models on the same test set.

    Args:
        models_dict: Dictionary of {model_name: model} pairs
        test_loader: Test data loader
        save_path: Path to save comparison results
        device: Computing device

    Returns:
        Tuple of (results_dict, comparison_dataframe)
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    print("Evaluating models...")
    for model_name, model in models_dict.items():
        print(f"\nEvaluating {model_name}...")
        evaluator = ModelEvaluator(model, device)
        model_results = evaluator.evaluate(test_loader, verbose=False)
        results[model_name] = model_results

    # Create comparison DataFrame
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'roc_auc', 'pr_auc']
    comparison_df = pd.DataFrame({
        model_name: [model_results[metric] for metric in metrics]
        for model_name, model_results in results.items()
    }, index=metrics)

    # Plot comparison
    _plot_model_comparison(results, comparison_df, save_path)

    # Print comparison table
    print(f"\n{'='*80}")
    print("MODEL PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    print(comparison_df.round(4))

    # Find best model for each metric
    print(f"\nBEST PERFORMERS:")
    for metric in metrics:
        best_model = comparison_df.loc[metric].idxmax()
        best_score = comparison_df.loc[metric].max()
        print(f"   {metric.replace('_', ' ').title():>15}: {best_model} ({best_score:.4f})")

    # Save results
    if save_path:
        _save_comparison_results(results, comparison_df, save_path)

    return results, comparison_df


def _plot_model_comparison(results: Dict[str, Dict],
                          comparison_df: pd.DataFrame,
                          save_path: Optional[str]) -> None:
    """Plot model comparison visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Metrics comparison
    comparison_df.T.plot(kind='bar', ax=axes[0, 0], rot=45, width=0.8)
    axes[0, 0].set_title('Model Performance Comparison', fontsize=14, pad=20)
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)

    # ROC curves comparison
    for model_name, model_results in results.items():
        axes[0, 1].plot(model_results['fpr'], model_results['tpr'],
                       label=f'{model_name} (AUC = {model_results["roc_auc"]:.3f})',
                       linewidth=2)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.6)
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curves Comparison', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # PR curves comparison
    for model_name, model_results in results.items():
        axes[1, 0].plot(model_results['recall_curve'], model_results['precision_curve'],
                       label=f'{model_name} (AUC = {model_results["pr_auc"]:.3f})',
                       linewidth=2)
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curves Comparison', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Sensitivity vs Specificity scatter
    for model_name, model_results in results.items():
        axes[1, 1].scatter(model_results['specificity'], model_results['recall'],
                          label=model_name, s=120, alpha=0.7)
        axes[1, 1].annotate(model_name,
                           (model_results['specificity'], model_results['recall']),
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
    axes[1, 1].set_xlabel('Specificity')
    axes[1, 1].set_ylabel('Sensitivity (Recall)')
    axes[1, 1].set_title('Sensitivity vs Specificity', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(-0.05, 1.05)
    axes[1, 1].set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def _save_comparison_results(results: Dict[str, Dict],
                           comparison_df: pd.DataFrame,
                           save_path: str) -> None:
    """Save comparison results to files."""
    base_path = Path(save_path).with_suffix('')

    # Save comparison DataFrame
    comparison_df.to_csv(f"{base_path}_comparison.csv")

    # Save detailed results (excluding arrays for JSON)
    json_results = {}
    for model_name, model_results in results.items():
        json_results[model_name] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in model_results.items()
            if k not in ['fpr', 'tpr', 'roc_thresholds', 'precision_curve',
                        'recall_curve', 'pr_thresholds', 'confusion_matrix',
                        'predictions', 'probabilities', 'labels']
        }

    with open(f"{base_path}_results.json", 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to: {base_path}_comparison.csv and {base_path}_results.json")


if __name__ == "__main__":
    # Test evaluation utilities
    print("Testing evaluation utilities...")

    # Create dummy model for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return torch.randn(x.size(0), 1)  # Random outputs for testing

    # Test evaluator creation
    model = DummyModel()
    evaluator = ModelEvaluator(model)

    print(f"ModelEvaluator created successfully")
    print(f"   Device: {evaluator.device}")
    print(f"   Threshold: {evaluator.threshold}")

    print("Evaluation utilities ready for deployment!")