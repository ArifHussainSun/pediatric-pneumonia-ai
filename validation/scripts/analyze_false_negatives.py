#!/usr/bin/env python3
"""
Analyze False Negative Cases with GradCAM, Patch Analysis, and ROI Detection

This script performs comprehensive analysis on the 6 false negative cases
from the 500-image validation test to understand:
- Where the model is looking (GradCAM)
- Image quality distribution (patch analysis)
- Lung region detection (ROI)
- Whether ROI-based preprocessing can help
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import json

from src.models.mobilenet import MobileNetFineTune
from src.analysis import (
    GradCAM,
    PatchAnalyzer,
    LungROIDetector,
    ROIBasedPreprocessor
)
from src.data.datasets import get_medical_transforms


# False negative cases from current validation test (November 10, 2025)
FALSE_NEGATIVES = [
    {
        'path': '/Users/Adam/Downloads/College/Co op/Tech4Life-25Summer/chest_xray/train/PNEUMONIA/person1500_bacteria_3916.jpeg',
        'confidence': 0.603,  # 60.3% wrong
        'quality': 'unknown',
        'notes': 'Moderate confidence mistake'
    },
    {
        'path': '/Users/Adam/Downloads/College/Co op/Tech4Life-25Summer/chest_xray/train/PNEUMONIA/person620_bacteria_2492.jpeg',
        'confidence': 0.795,  # 79.5% wrong - NEW CASE
        'quality': 'unknown',
        'notes': 'NEW false negative - not in Oct 18 test'
    },
    {
        'path': '/Users/Adam/Downloads/College/Co op/Tech4Life-25Summer/chest_xray/train/PNEUMONIA/person1483_virus_2574.jpeg',
        'confidence': 0.986,  # 98.6% WRONG! - DANGEROUS
        'quality': 'unknown',
        'notes': 'CONFIDENTLY WRONG - extremely high confidence'
    },
    {
        'path': '/Users/Adam/Downloads/College/Co op/Tech4Life-25Summer/chest_xray/train/PNEUMONIA/person780_bacteria_2684.jpeg',
        'confidence': 0.740,  # 74% wrong
        'quality': 'unknown',
        'notes': 'High confidence mistake'
    },
    {
        'path': '/Users/Adam/Downloads/College/Co op/Tech4Life-25Summer/chest_xray/train/PNEUMONIA/person435_virus_885.jpeg',
        'confidence': 0.9999,  # 99.99% WRONG! - EXTREMELY DANGEROUS
        'quality': 'unknown',
        'notes': 'MOST DANGEROUS - virtually 100% confident but wrong'
    },
    {
        'path': '/Users/Adam/Downloads/College/Co op/Tech4Life-25Summer/chest_xray/train/PNEUMONIA/person820_virus_1456.jpeg',
        'confidence': 0.9999,  # 99.99% WRONG! - EXTREMELY DANGEROUS
        'quality': 'unknown',
        'notes': 'MOST DANGEROUS - virtually 100% confident but wrong'
    },
    {
        'path': '/Users/Adam/Downloads/College/Co op/Tech4Life-25Summer/chest_xray/train/PNEUMONIA/person454_virus_938.jpeg',
        'confidence': 0.980,  # 98% WRONG! - DANGEROUS
        'quality': 'unknown',
        'notes': 'CONFIDENTLY WRONG - very high confidence'
    }
]


def load_model():
    """Load trained MobileNet model (same as API server)."""
    print("Loading MobileNet model...")
    # Use the same model as API server (Best_MobilenetV1.pth)
    model_path = project_root / "outputs" / "dgx_station_experiment" / "Best_MobilenetV1.pth"

    model = MobileNetFineTune(num_classes=2, freeze_layers=0)

    if model_path.exists():
        model.load_custom_weights(str(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print(f"WARNING: Model not found at {model_path}")
        return None

    model.eval()
    return model


def analyze_single_case(case: dict, model: torch.nn.Module, save_dir: Path):
    """
    Perform comprehensive analysis on a single false negative case.

    Args:
        case: False negative case information
        model: Loaded MobileNet model
        save_dir: Directory to save visualizations

    Returns:
        analysis_results: Dictionary with analysis metrics
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {Path(case['path']).name}")
    print(f"Confidence: {case['confidence']*100:.1f}% NORMAL (WRONG!)")
    print(f"Quality: {case['quality']}")
    print(f"Notes: {case['notes']}")
    print(f"{'='*80}")

    # Load image
    image_path = Path(case['path'])
    if not image_path.exists():
        print(f"ERROR: Image not found at {image_path}")
        return None

    # Load original image
    original_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        print(f"ERROR: Could not load image")
        return None

    # Prepare for model
    transform = get_medical_transforms(is_training=False)
    pil_image = Image.fromarray(original_image).convert('RGB')
    input_tensor = transform(pil_image).unsqueeze(0)

    # Initialize analyzers
    gradcam = GradCAM(model)
    patch_analyzer = PatchAnalyzer(grid_size=(8, 8))
    roi_detector = LungROIDetector()
    roi_preprocessor = ROIBasedPreprocessor(roi_detector)

    # Get model prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        normal_prob = probs[0, 1].item()
        pneumonia_prob = probs[0, 0].item()

    print(f"\nModel Output:")
    print(f"  NORMAL probability: {normal_prob*100:.2f}%")
    print(f"  PNEUMONIA probability: {pneumonia_prob*100:.2f}%")

    # GradCAM - see where model is looking
    gradcam_heatmap = gradcam.generate_cam(input_tensor, target_class=1)  # Class 1 = NORMAL
    gradcam_vis = gradcam.visualize(original_image, input_tensor, target_class=1)

    # Patch analysis
    patch_analysis = patch_analyzer.analyze_patches(original_image)
    patch_heatmaps = patch_analyzer.create_heatmaps(patch_analysis)

    print(f"\nPatch Analysis:")
    print(f"  Avg brightness: {patch_analysis['brightness'].mean():.3f}")
    print(f"  Avg contrast: {patch_analysis['contrast'].mean():.3f}")
    print(f"  Avg sharpness: {patch_analysis['sharpness'].mean():.3f}")
    print(f"  Brightness std: {patch_analysis['brightness'].std():.3f}")
    print(f"  Contrast std: {patch_analysis['contrast'].std():.3f}")

    # ROI detection
    roi_info = roi_detector.detect_lung_roi(original_image)
    roi_vis = roi_detector.visualize_roi(original_image, roi_info)

    print(f"\nROI Analysis:")
    print(f"  Lung area ratio: {roi_info['lung_area_ratio']*100:.1f}%")
    print(f"  Bounding box: {roi_info['bbox']}")
    print(f"  Contours found: {len(roi_info['contours'])}")

    # ROI-based preprocessing
    roi_preprocessed = roi_preprocessor.preprocess_with_roi(
        original_image,
        roi_info=roi_info,
        roi_clip_limit=3.0,
        background_clip_limit=1.5
    )

    # Test if ROI preprocessing helps
    pil_preprocessed = Image.fromarray(roi_preprocessed['enhanced_image']).convert('RGB')
    input_preprocessed = transform(pil_preprocessed).unsqueeze(0)

    with torch.no_grad():
        logits_preprocessed = model(input_preprocessed)
        probs_preprocessed = torch.softmax(logits_preprocessed, dim=1)
        normal_prob_preprocessed = probs_preprocessed[0, 1].item()
        pneumonia_prob_preprocessed = probs_preprocessed[0, 0].item()

    print(f"\nAfter ROI-based Preprocessing:")
    print(f"  NORMAL probability: {normal_prob_preprocessed*100:.2f}% (was {normal_prob*100:.2f}%)")
    print(f"  PNEUMONIA probability: {pneumonia_prob_preprocessed*100:.2f}% (was {pneumonia_prob*100:.2f}%)")

    improvement = pneumonia_prob_preprocessed - pneumonia_prob
    if improvement > 0:
        print(f"  ✓ IMPROVEMENT: +{improvement*100:.2f}% toward correct answer")
        did_improve = True
        fixed = pneumonia_prob_preprocessed > 0.5
    else:
        print(f"  ✗ WORSE: {improvement*100:.2f}% away from correct answer")
        did_improve = False
        fixed = False

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Row 1: Original, GradCAM, ROI
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_image, cmap='gray')
    ax1.set_title(f'Original Image\n{Path(case["path"]).name}', fontsize=10)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(gradcam_vis)
    ax2.set_title(f'GradCAM - Where Model Looks\nPredicted NORMAL {normal_prob*100:.1f}%', fontsize=10)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(roi_vis)
    ax3.set_title(f'Lung ROI Detection\n{roi_info["lung_area_ratio"]*100:.1f}% lung area', fontsize=10)
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(roi_preprocessed['enhanced_image'], cmap='gray')
    ax4.set_title(f'ROI-Enhanced Image\nPNEUMONIA {pneumonia_prob_preprocessed*100:.1f}%', fontsize=10)
    ax4.axis('off')

    # Row 2: Patch heatmaps
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(patch_heatmaps['brightness'])
    ax5.set_title(f'Patch Brightness\nAvg: {patch_analysis["brightness"].mean():.3f}', fontsize=10)
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(patch_heatmaps['contrast'])
    ax6.set_title(f'Patch Contrast\nAvg: {patch_analysis["contrast"].mean():.3f}', fontsize=10)
    ax6.axis('off')

    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(patch_heatmaps['sharpness'])
    ax7.set_title(f'Patch Sharpness\nAvg: {patch_analysis["sharpness"].mean():.3f}', fontsize=10)
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[1, 3])
    # GradCAM heatmap overlay
    ax8.imshow(gradcam_heatmap, cmap='jet')
    ax8.set_title('GradCAM Heatmap\n(Focus intensity)', fontsize=10)
    ax8.axis('off')

    # Row 3: Analysis summary
    ax9 = fig.add_subplot(gs[2, :])
    ax9.axis('off')

    summary_text = f"""
CASE ANALYSIS SUMMARY
{'='*100}

File: {Path(case['path']).name}
Ground Truth: PNEUMONIA (bacterial or viral)
Quality: {case['quality']} | Notes: {case['notes']}

{'='*100}
BASELINE MODEL PREDICTION:
  • Predicted: NORMAL with {normal_prob*100:.2f}% confidence (WRONG!)
  • True class (PNEUMONIA): {pneumonia_prob*100:.2f}%
  • This is a FALSE NEGATIVE - missed pneumonia case

AFTER ROI-BASED PREPROCESSING:
  • NORMAL: {normal_prob_preprocessed*100:.2f}% (was {normal_prob*100:.2f}%)
  • PNEUMONIA: {pneumonia_prob_preprocessed*100:.2f}% (was {pneumonia_prob*100:.2f}%)
  • Change: {'+' if improvement > 0 else ''}{improvement*100:.2f}% toward correct answer
  • Result: {'✓ FIXED - Now predicts correctly!' if fixed else '✓ IMPROVED but still wrong' if did_improve else '✗ DID NOT HELP'}

PATCH QUALITY METRICS:
  • Brightness: avg={patch_analysis['brightness'].mean():.3f}, std={patch_analysis['brightness'].std():.3f}
  • Contrast: avg={patch_analysis['contrast'].mean():.3f}, std={patch_analysis['contrast'].std():.3f}
  • Sharpness: avg={patch_analysis['sharpness'].mean():.3f}, std={patch_analysis['sharpness'].std():.3f}

ROI DETECTION:
  • Lung area: {roi_info['lung_area_ratio']*100:.1f}% of image
  • Contours detected: {len(roi_info['contours'])}

GRADCAM INSIGHTS:
  • Model is focusing on regions shown in heatmap when predicting NORMAL
  • Red/yellow areas = high attention, blue = low attention
  • Check if model is looking at the right regions (lungs vs background)
{'='*100}
    """

    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Save figure
    save_path = save_dir / f"{Path(case['path']).stem}_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {save_path}")
    plt.close()

    # Return analysis results
    return {
        'filename': Path(case['path']).name,
        'filepath': str(case['path']),
        'baseline_normal_prob': float(normal_prob),
        'baseline_pneumonia_prob': float(pneumonia_prob),
        'preprocessed_normal_prob': float(normal_prob_preprocessed),
        'preprocessed_pneumonia_prob': float(pneumonia_prob_preprocessed),
        'improvement': float(improvement),
        'did_improve': bool(did_improve),
        'fixed': bool(fixed),
        'patch_brightness_avg': float(patch_analysis['brightness'].mean()),
        'patch_brightness_std': float(patch_analysis['brightness'].std()),
        'patch_contrast_avg': float(patch_analysis['contrast'].mean()),
        'patch_contrast_std': float(patch_analysis['contrast'].std()),
        'patch_sharpness_avg': float(patch_analysis['sharpness'].mean()),
        'patch_sharpness_std': float(patch_analysis['sharpness'].std()),
        'lung_area_ratio': float(roi_info['lung_area_ratio']),
        'quality': case['quality'],
        'baseline_confidence': float(case['confidence']),
        'notes': case['notes'],
        'visualization_path': str(save_path)
    }


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("FALSE NEGATIVE ANALYSIS WITH GRADCAM, PATCHES, AND ROI")
    print("="*80)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = project_root / "validation" / "reports" / f"false_negative_analysis_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {save_dir}")

    # Load model
    model = load_model()
    if model is None:
        print("ERROR: Could not load model. Exiting.")
        return

    # Analyze each false negative case
    results = []
    for i, case in enumerate(FALSE_NEGATIVES, 1):
        print(f"\n\nCASE {i}/{len(FALSE_NEGATIVES)}")
        result = analyze_single_case(case, model, save_dir)
        if result:
            results.append(result)

    # Generate summary report
    print(f"\n\n{'='*80}")
    print("SUMMARY REPORT")
    print(f"{'='*80}")

    df = pd.DataFrame(results)

    print(f"\nTotal false negatives analyzed: {len(results)}")
    print(f"Cases improved by ROI preprocessing: {df['did_improve'].sum()}/{len(results)}")
    print(f"Cases FIXED by ROI preprocessing: {df['fixed'].sum()}/{len(results)}")
    print(f"Average improvement: {df['improvement'].mean()*100:.2f}%")
    print(f"Best improvement: {df['improvement'].max()*100:.2f}%")
    print(f"Worst case: {df['improvement'].min()*100:.2f}%")

    # Save JSON summary
    summary = {
        'analysis_date': timestamp,
        'total_cases': len(results),
        'improved_cases': int(df['did_improve'].sum()),
        'fixed_cases': int(df['fixed'].sum()),
        'average_improvement': float(df['improvement'].mean()),
        'cases': results,
        'key_findings': [
            f"{df['did_improve'].sum()} out of 6 cases showed improvement with ROI preprocessing",
            f"{df['fixed'].sum()} out of 6 cases were completely fixed (now predict correctly)",
            f"Average improvement: {df['improvement'].mean()*100:.2f}% toward correct answer",
            f"Best case improved by {df['improvement'].max()*100:.2f}%",
            "ROI-based preprocessing shows {'significant' if df['did_improve'].sum() >= 4 else 'limited'} potential"
        ]
    }

    json_path = save_dir / "analysis_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved JSON summary to {json_path}")

    # Save summary CSV
    csv_path = save_dir / "analysis_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved summary CSV to {csv_path}")

    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Before vs After probabilities
    ax1 = axes[0, 0]
    x = np.arange(len(results))
    width = 0.35
    ax1.bar(x - width/2, df['baseline_pneumonia_prob'], width, label='Before ROI', alpha=0.8)
    ax1.bar(x + width/2, df['preprocessed_pneumonia_prob'], width, label='After ROI', alpha=0.8)
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold')
    ax1.set_xlabel('Case Number')
    ax1.set_ylabel('PNEUMONIA Probability')
    ax1.set_title('Impact of ROI-Based Preprocessing on Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Improvement distribution
    ax2 = axes[0, 1]
    colors = ['green' if x > 0 else 'red' for x in df['improvement']]
    ax2.bar(range(len(results)), df['improvement']*100, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Case Number')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Improvement toward Correct Answer')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Quality metrics
    ax3 = axes[1, 0]
    metrics = ['brightness', 'contrast', 'sharpness']
    metric_values = [
        df['patch_brightness_avg'].mean(),
        df['patch_contrast_avg'].mean(),
        df['patch_sharpness_avg'].mean()
    ]
    ax3.bar(metrics, metric_values, alpha=0.7, color=['orange', 'blue', 'green'])
    ax3.set_ylabel('Average Score')
    ax3.set_title('Average Patch Quality Metrics')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Confidence vs Improvement
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df['baseline_confidence'], df['improvement']*100,
                         c=df['did_improve'].astype(int), cmap='RdYlGn',
                         s=100, alpha=0.7, edgecolors='black')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Baseline Confidence (WRONG)')
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('Baseline Confidence vs ROI Improvement')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    summary_vis_path = save_dir / "summary_visualization.png"
    plt.savefig(summary_vis_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved summary visualization to {summary_vis_path}")
    plt.close()

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {save_dir}")
    print(f"  • Individual case analyses: {len(results)} PNG files")
    print(f"  • Summary JSON: analysis_summary.json")
    print(f"  • Summary CSV: analysis_summary.csv")
    print(f"  • Summary visualization: summary_visualization.png")
    print(f"\nOutput directory: {save_dir}")


if __name__ == "__main__":
    main()
