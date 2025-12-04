"""
Visualization utilities for pediatric pneumonia detection.

This module provides comprehensive visualization tools including:
- Dataset exploration and analysis
- Model interpretability with Grad-CAM
- Performance analysis and error visualization
- Clinical decision analysis
- Interactive visualization tools

Designed specifically for medical imaging applications with
focus on clinical interpretability and actionable insights.
"""

import os
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

import cv2
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy import ndimage

# Grad-CAM imports (optional)
try:
    from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, XGradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("pytorch_grad_cam not available. Install with: pip install grad-cam")

# Set style
plt.style.use('default')
sns.set_palette("husl")


class DatasetVisualizer:
    """Visualization tools for dataset exploration and analysis."""

    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)

    def analyze_dataset_statistics(self) -> Dict[str, Dict[str, int]]:
        """Analyze dataset statistics across splits and classes."""
        stats = {}

        for split in ['train', 'test', 'val']:
            split_stats = {'NORMAL': 0, 'PNEUMONIA': 0, 'total': 0}

            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_dir = self.data_dir / split / class_name
                if class_dir.exists():
                    # Count image files
                    count = len([f for f in class_dir.iterdir()
                               if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                    split_stats[class_name] = count
                    split_stats['total'] += count

            if split_stats['total'] > 0:  # Only include splits with data
                stats[split] = split_stats

        return stats

    def plot_dataset_distribution(self,
                                 stats: Optional[Dict] = None,
                                 save_path: Optional[str] = None,
                                 show: bool = True) -> None:
        """Visualize dataset distribution across splits and classes."""
        if stats is None:
            stats = self.analyze_dataset_statistics()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Split distribution
        splits = list(stats.keys())
        split_totals = [stats[split]['total'] for split in splits]

        if split_totals:
            axes[0].pie(split_totals, labels=splits, autopct='%1.1f%%',
                       startangle=90, colors=['lightblue', 'lightcoral', 'lightgreen'][:len(splits)])
            axes[0].set_title('Dataset Split Distribution')

            # Class distribution per split
            normal_counts = [stats[split]['NORMAL'] for split in splits]
            pneumonia_counts = [stats[split]['PNEUMONIA'] for split in splits]

            x = np.arange(len(splits))
            width = 0.35

            axes[1].bar(x - width/2, normal_counts, width, label='Normal', alpha=0.8)
            axes[1].bar(x + width/2, pneumonia_counts, width, label='Pneumonia', alpha=0.8)
            axes[1].set_xlabel('Dataset Split')
            axes[1].set_ylabel('Number of Images')
            axes[1].set_title('Class Distribution by Split')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(splits)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Overall class balance
            total_normal = sum(stats[split]['NORMAL'] for split in stats)
            total_pneumonia = sum(stats[split]['PNEUMONIA'] for split in stats)

            axes[2].pie([total_normal, total_pneumonia],
                       labels=['Normal', 'Pneumonia'],
                       autopct='%1.1f%%',
                       startangle=90,
                       colors=['lightblue', 'lightcoral'])
            axes[2].set_title('Overall Class Distribution')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        # Print statistics
        self._print_dataset_stats(stats)

    def _print_dataset_stats(self, stats: Dict[str, Dict[str, int]]) -> None:
        """Print detailed dataset statistics."""
        print("\nDATASET STATISTICS")
        print("=" * 60)

        total_images = 0
        for split, split_stats in stats.items():
            print(f"\n{split.upper()} SET:")
            print(f"  Normal:    {split_stats['NORMAL']:>6,} images")
            print(f"  Pneumonia: {split_stats['PNEUMONIA']:>6,} images")
            print(f"  Total:     {split_stats['total']:>6,} images")

            if split_stats['total'] > 0:
                balance_ratio = split_stats['PNEUMONIA'] / max(split_stats['NORMAL'], 1)
                print(f"  Balance Ratio: {balance_ratio:.2f} (Pneumonia/Normal)")

            total_images += split_stats['total']

        print(f"\nOVERALL STATISTICS:")
        print(f"  Total Images: {total_images:,}")

    def visualize_sample_images(self,
                               num_samples: int = 8,
                               split: str = 'train',
                               save_path: Optional[str] = None,
                               show: bool = True) -> None:
        """Display sample images from each class."""
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))

        for class_idx, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
            class_dir = self.data_dir / split / class_name
            if not class_dir.exists():
                continue

            image_files = [f for f in class_dir.iterdir()
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png']][:num_samples]

            for idx in range(num_samples):
                if idx < len(image_files):
                    img_path = image_files[idx]
                    img = Image.open(img_path).convert('RGB')
                    axes[class_idx, idx].imshow(img, cmap='gray')
                    axes[class_idx, idx].set_title(f'{class_name}\n{img_path.name}', fontsize=10)
                else:
                    axes[class_idx, idx].text(0.5, 0.5, 'No Image', ha='center', va='center')
                axes[class_idx, idx].axis('off')

        plt.suptitle(f'Sample Images from {split.title()} Set', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()


class GradCAMVisualizer:
    """Grad-CAM visualization for model interpretability."""

    def __init__(self, model: nn.Module, target_layers: Optional[List] = None):
        if not GRADCAM_AVAILABLE:
            raise ImportError("pytorch_grad_cam is required. Install with: pip install grad-cam")

        self.model = model
        self.model.eval()

        # Auto-detect target layers if not provided
        if target_layers is None:
            self.target_layers = self._get_target_layers()
        else:
            self.target_layers = target_layers

        # Initialize Grad-CAM
        self.cam = GradCAM(model=model, target_layers=self.target_layers)

    def _get_target_layers(self) -> List[nn.Module]:
        """Automatically detect appropriate target layers."""
        # Look for common layer patterns
        for name, module in self.model.named_modules():
            if 'features' in name and isinstance(module, nn.Conv2d):
                return [module]
            elif 'layer4' in name and isinstance(module, nn.Conv2d):
                return [module]
            elif 'block' in name and isinstance(module, nn.Conv2d):
                return [module]

        # Fallback: use the last convolutional layer
        conv_layers = []
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(module)

        return [conv_layers[-1]] if conv_layers else []

    def generate_cam(self,
                    input_tensor: torch.Tensor,
                    target_class: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM visualization."""
        if target_class is None:
            # Use the predicted class
            with torch.no_grad():
                output = self.model(input_tensor)
                predicted_class = (torch.sigmoid(output) > 0.5).int().item()
            target_class = predicted_class

        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)

        return grayscale_cam[0]  # Return first batch item

    def visualize_prediction(self,
                            image_path: Union[str, Path],
                            save_path: Optional[str] = None,
                            show: bool = True) -> Dict[str, Any]:
        """Complete visualization pipeline for a single image."""
        # Load and preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Original image for visualization
        original_img = Image.open(image_path).convert('RGB')
        original_img = original_img.resize((224, 224))
        original_array = np.array(original_img) / 255.0

        # Preprocessed image for model
        device = next(self.model.parameters()).device
        input_tensor = transform(original_img).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probability = torch.sigmoid(output).item()
            prediction = "Pneumonia" if probability > 0.5 else "Normal"
            confidence = probability if probability > 0.5 else (1 - probability)

        # Generate Grad-CAM
        cam = self.generate_cam(input_tensor)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Grad-CAM heatmap
        axes[1].imshow(cam, cmap='jet', alpha=0.8)
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')

        # Overlay
        cam_image = show_cam_on_image(original_array, cam, use_rgb=True)
        axes[2].imshow(cam_image)
        axes[2].set_title(f'Grad-CAM Overlay\nPrediction: {prediction}\nConfidence: {confidence:.3f}')
        axes[2].axis('off')

        plt.suptitle(f'Model Interpretation: {Path(image_path).name}', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': confidence,
            'cam': cam
        }


class PerformanceVisualizer:
    """Visualization tools for model performance analysis."""

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or ['Normal', 'Pneumonia']

    def plot_comprehensive_analysis(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  y_prob: np.ndarray,
                                  save_path: Optional[str] = None,
                                  show: bool = True) -> None:
        """Comprehensive performance analysis with multiple visualizations."""
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 4, figure=fig)

        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')

        # 2. ROC Curve
        ax2 = fig.add_subplot(gs[0, 1])
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
        ax2.plot([0, 1], [0, 1], 'k--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Precision-Recall Curve
        ax3 = fig.add_subplot(gs[0, 2])
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ax3.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})', linewidth=2)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Prediction Distribution
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.hist(y_prob[y_true == 0], bins=30, alpha=0.5, label='Normal', density=True)
        ax4.hist(y_prob[y_true == 1], bins=30, alpha=0.5, label='Pneumonia', density=True)
        ax4.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        ax4.set_xlabel('Predicted Probability')
        ax4.set_ylabel('Density')
        ax4.set_title('Prediction Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Metrics by Threshold
        ax5 = fig.add_subplot(gs[1, 0:2])
        thresholds = np.linspace(0, 1, 100)
        precisions, recalls, f1s, accuracies = [], [], [], []

        for threshold in thresholds:
            y_pred_thresh = (y_prob >= threshold).astype(int)
            if len(np.unique(y_pred_thresh)) > 1:
                precisions.append(precision_score(y_true, y_pred_thresh, zero_division=0))
                recalls.append(recall_score(y_true, y_pred_thresh, zero_division=0))
                f1s.append(f1_score(y_true, y_pred_thresh, zero_division=0))
                accuracies.append(accuracy_score(y_true, y_pred_thresh))
            else:
                precisions.append(0)
                recalls.append(0)
                f1s.append(0)
                accuracies.append(0)

        ax5.plot(thresholds, precisions, label='Precision', linewidth=2)
        ax5.plot(thresholds, recalls, label='Recall', linewidth=2)
        ax5.plot(thresholds, f1s, label='F1-Score', linewidth=2)
        ax5.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
        ax5.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Default Threshold')
        ax5.set_xlabel('Threshold')
        ax5.set_ylabel('Metric Value')
        ax5.set_title('Metrics vs Threshold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Class-wise Performance
        ax6 = fig.add_subplot(gs[1, 2:4])
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)

        metrics = ['precision', 'recall', 'f1-score']
        normal_scores = [report[self.class_names[0]][metric] for metric in metrics]
        pneumonia_scores = [report[self.class_names[1]][metric] for metric in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        ax6.bar(x - width/2, normal_scores, width, label=self.class_names[0], alpha=0.8)
        ax6.bar(x + width/2, pneumonia_scores, width, label=self.class_names[1], alpha=0.8)

        ax6.set_xlabel('Metrics')
        ax6.set_ylabel('Score')
        ax6.set_title('Class-wise Performance')
        ax6.set_xticks(x)
        ax6.set_xticklabels([m.title() for m in metrics])
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1)

        # Add value labels on bars
        for i, (normal, pneumonia) in enumerate(zip(normal_scores, pneumonia_scores)):
            ax6.text(i - width/2, normal + 0.01, f'{normal:.3f}', ha='center', va='bottom')
            ax6.text(i + width/2, pneumonia + 0.01, f'{pneumonia:.3f}', ha='center', va='bottom')

        # 7. Summary Statistics
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')

        tn, fp, fn, tp = cm.ravel()

        metrics_text = f"""
        DETAILED PERFORMANCE METRICS

        Basic Metrics:                          Clinical Metrics:
        • Accuracy:     {accuracy_score(y_true, y_pred):.4f}      • Sensitivity:   {tp/(tp+fn):.4f}
        • Precision:    {precision_score(y_true, y_pred):.4f}      • Specificity:   {tn/(tn+fp):.4f}
        • Recall:       {recall_score(y_true, y_pred):.4f}      • PPV:           {tp/(tp+fp):.4f}
        • F1-Score:     {f1_score(y_true, y_pred):.4f}      • NPV:           {tn/(tn+fn):.4f}

        AUC Scores:                             Confusion Matrix:
        • ROC AUC:      {roc_auc:.4f}                 • True Positives:    {tp:>5d}
        • PR AUC:       {pr_auc:.4f}                 • False Positives:   {fp:>5d}
                                                • True Negatives:    {tn:>5d}
        Clinical Impact:                        • False Negatives:   {fn:>5d}
        • Missed Cases: {fn} ({fn/(tp+fn)*100:.1f}% of pneumonia)
        • False Alarms: {fp} ({fp/(tn+fp)*100:.1f}% of normal)
        """

        ax7.text(0.05, 0.95, metrics_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.suptitle('Comprehensive Performance Analysis', fontsize=16, y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

    def plot_model_comparison(self,
                             results_dict: Dict[str, Dict[str, float]],
                             save_path: Optional[str] = None,
                             show: bool = True) -> None:
        """Compare multiple models performance."""
        # Create comparison DataFrame
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        available_metrics = [m for m in metrics if all(m in results for results in results_dict.values())]

        comparison_df = pd.DataFrame({
            model_name: [model_results[metric] for metric in available_metrics]
            for model_name, model_results in results_dict.items()
        }, index=available_metrics)

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Metrics comparison bar plot
        comparison_df.T.plot(kind='bar', ax=axes[0], rot=45, width=0.8)
        axes[0].set_title('Model Performance Comparison')
        axes[0].set_ylabel('Score')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)

        # Radar chart (if we have enough models)
        if len(results_dict) >= 2:
            # Calculate positions for radar chart
            angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))

            ax = axes[1]
            ax = plt.subplot(122, projection='polar')

            for model_name, model_results in results_dict.items():
                values = [model_results[metric] for metric in available_metrics]
                values += values[:1]  # Complete the circle

                ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
                ax.fill(angles, values, alpha=0.25)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics])
            ax.set_ylim(0, 1)
            ax.set_title('Performance Radar Chart')
            ax.legend(bbox_to_anchor=(1.2, 1.0))
        else:
            axes[1].axis('off')
            axes[1].text(0.5, 0.5, 'Radar chart requires\n≥2 models', ha='center', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()


def create_clinical_report(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_prob: np.ndarray,
                          model_name: str = "Pneumonia Detection Model",
                          save_path: Optional[str] = None) -> str:
    """Generate a comprehensive clinical report."""
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    specificity = tn / (tn + fp)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_true, y_prob)

    # Generate report
    report = f"""
    =====================================================================
    CLINICAL PERFORMANCE REPORT: {model_name}
    =====================================================================
    Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

    EXECUTIVE SUMMARY
    -------------------------------------------------------------------
    Overall Accuracy:        {accuracy:.1%}
    Pneumonia Detection:     {recall:.1%} (Sensitivity)
    Normal Recognition:      {specificity:.1%} (Specificity)
    Positive Predictive:     {precision:.1%} (PPV)
    ROC AUC Score:          {roc_auc:.3f}

    DETAILED PERFORMANCE METRICS
    -------------------------------------------------------------------
    Classification Accuracy: {accuracy:.4f}
    Precision (PPV):         {precision:.4f}
    Recall (Sensitivity):    {recall:.4f}
    Specificity:            {specificity:.4f}
    F1-Score:               {f1:.4f}
    ROC AUC:                {roc_auc:.4f}
    PR AUC:                 {pr_auc:.4f}

    📋 CONFUSION MATRIX ANALYSIS
    -------------------------------------------------------------------
    True Positives (TP):    {tp:>5d}  (Correctly identified pneumonia)
    False Positives (FP):   {fp:>5d}  (Normal cases flagged as pneumonia)
    True Negatives (TN):    {tn:>5d}  (Correctly identified normal)
    False Negatives (FN):   {fn:>5d}  (Missed pneumonia cases)

    Total Cases Analyzed:   {len(y_true):>5d}

    🏥 CLINICAL IMPACT ASSESSMENT
    -------------------------------------------------------------------
    Missed Pneumonia Cases: {fn} out of {tp + fn} ({fn/(tp+fn)*100:.1f}%)
    False Alarm Rate:       {fp} out of {tn + fp} ({fp/(tn+fp)*100:.1f}%)

    Positive Predictive Value: {precision:.1%}
    - {precision*100:.1f}% of positive predictions are correct

    Negative Predictive Value: {tn/(tn+fn):.1%}
    - {(tn/(tn+fn))*100:.1f}% of negative predictions are correct

    CLINICAL CONSIDERATIONS
    -------------------------------------------------------------------
    • Sensitivity of {recall:.1%} means {(1-recall)*100:.1f}% of pneumonia cases may be missed
    • Specificity of {specificity:.1%} means {(1-specificity)*100:.1f}% of normal cases may be over-diagnosed
    • Consider clinical context when interpreting results
    • This model should supplement, not replace, clinical judgment

    PERFORMANCE ASSESSMENT
    -------------------------------------------------------------------
    """

    # Add performance assessment
    if roc_auc >= 0.9:
        report += "    Model Performance: EXCELLENT (ROC AUC ≥ 0.90)\n"
    elif roc_auc >= 0.8:
        report += "    Model Performance: GOOD (ROC AUC ≥ 0.80)\n"
    elif roc_auc >= 0.7:
        report += "    Model Performance: FAIR (ROC AUC ≥ 0.70)\n"
    else:
        report += "    Model Performance: POOR (ROC AUC < 0.70)\n"

    # Add recommendations
    report += f"""
    RECOMMENDATIONS
    -------------------------------------------------------------------
    """

    if recall < 0.8:
        report += "    • Consider lowering classification threshold to improve sensitivity\n"
    if specificity < 0.8:
        report += "    • Consider raising classification threshold to reduce false positives\n"
    if precision < 0.7:
        report += "    • High false positive rate may require additional validation\n"

    report += """
    • Validate on diverse patient populations before clinical deployment
    • Implement confidence scoring for uncertain cases
    • Provide radiologist review for borderline cases
    • Regular model performance monitoring recommended

    =====================================================================
    END OF REPORT
    =====================================================================
    """

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"📄 Clinical report saved to: {save_path}")

    return report


if __name__ == "__main__":
    # Test visualization utilities
    print("🎨 Testing visualization utilities...")

    # Create dummy data for testing
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=100, p=[0.6, 0.4])
    y_prob = np.random.beta(2, 5, size=100)
    y_pred = (y_prob > 0.5).astype(int)

    # Test performance visualizer
    perf_viz = PerformanceVisualizer()

    print("PerformanceVisualizer created successfully")

    # Test clinical report
    report = create_clinical_report(y_true, y_pred, y_prob, "Test Model")
    print("Clinical report generated successfully")

    print("Visualization utilities ready for deployment!")