#!/usr/bin/env python3
"""
Detailed Analysis of False Negative Cases

Analyzes the 21 false negative cases from the 1000-image test to:
1. Calculate detailed quality metrics for each
2. Compare with correctly classified pneumonia cases
3. Test different preprocessing strategies
4. Identify patterns and potential improvements
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
import json
from datetime import datetime
from tqdm import tqdm

from src.models.mobilenet import MobileNetFineTune
from src.data.datasets import get_medical_transforms
from src.preprocessing.intelligent_preprocessing import IntelligentPreprocessor


def load_model():
    """Load trained MobileNet model."""
    print("Loading MobileNet model...")
    model_path = project_root / "outputs" / "dgx_station_experiment" / "Best_MobilenetV1.pth"
    model = MobileNetFineTune(num_classes=2, freeze_layers=0)
    model.load_custom_weights(str(model_path))
    model.eval()
    return model


def calculate_detailed_metrics(image):
    """Calculate comprehensive quality metrics for an image."""
    metrics = {}

    # Basic metrics
    metrics['brightness'] = np.mean(image) / 255.0
    metrics['contrast'] = np.std(image) / 255.0

    # Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    metrics['sharpness'] = laplacian.var() / 10000.0

    # Histogram metrics
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    metrics['histogram_std'] = np.std(hist)
    metrics['histogram_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))

    # Edge density
    edges = cv2.Canny(image, 50, 150)
    metrics['edge_density'] = np.sum(edges > 0) / (image.shape[0] * image.shape[1])

    # Intensity distribution
    metrics['min_intensity'] = np.min(image)
    metrics['max_intensity'] = np.max(image)
    metrics['mean_intensity'] = np.mean(image)
    metrics['median_intensity'] = np.median(image)

    # Percentiles
    metrics['p10'] = np.percentile(image, 10)
    metrics['p25'] = np.percentile(image, 25)
    metrics['p75'] = np.percentile(image, 75)
    metrics['p90'] = np.percentile(image, 90)

    # Dynamic range
    metrics['dynamic_range'] = (metrics['max_intensity'] - metrics['min_intensity']) / 255.0

    return metrics


def test_preprocessing_strategies(image_path, model, transform, preprocessor):
    """Test multiple preprocessing strategies on a single image."""
    results = {}

    # Load original image
    original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # Strategy 1: No preprocessing
    pil_image = Image.fromarray(original).convert('RGB')
    input_tensor = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        results['none'] = {
            'pneumonia_prob': probs[0, 0].item(),
            'normal_prob': probs[0, 1].item()
        }

    # Strategy 2: Current conservative preprocessing
    quality = preprocessor.assess_image_quality(original)
    if quality.should_enhance:
        enhanced = preprocessor.enhance_image(original, quality)
    else:
        enhanced = original
    pil_image = Image.fromarray(enhanced).convert('RGB')
    input_tensor = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        results['conservative'] = {
            'pneumonia_prob': probs[0, 0].item(),
            'normal_prob': probs[0, 1].item(),
            'enhanced': quality.should_enhance
        }

    # Strategy 3: Light global CLAHE (always apply)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(original)
    pil_image = Image.fromarray(enhanced).convert('RGB')
    input_tensor = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        results['light_global'] = {
            'pneumonia_prob': probs[0, 0].item(),
            'normal_prob': probs[0, 1].item()
        }

    # Strategy 4: Medium global CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(original)
    pil_image = Image.fromarray(enhanced).convert('RGB')
    input_tensor = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        results['medium_global'] = {
            'pneumonia_prob': probs[0, 0].item(),
            'normal_prob': probs[0, 1].item()
        }

    # Strategy 5: ROI-based CLAHE (force apply)
    try:
        roi_info = preprocessor.roi_detector.detect_lung_roi(original)
        roi_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        bg_clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        enhanced_roi = roi_clahe.apply(original)
        enhanced_bg = bg_clahe.apply(original)
        mask = roi_info['mask']
        enhanced = np.where(mask > 0, enhanced_roi, enhanced_bg).astype(np.uint8)

        pil_image = Image.fromarray(enhanced).convert('RGB')
        input_tensor = transform(pil_image).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            results['roi_clahe'] = {
                'pneumonia_prob': probs[0, 0].item(),
                'normal_prob': probs[0, 1].item()
            }
    except Exception as e:
        results['roi_clahe'] = {
            'pneumonia_prob': 0.0,
            'normal_prob': 0.0,
            'error': str(e)
        }

    return results


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("DETAILED FALSE NEGATIVE ANALYSIS")
    print("="*80)

    # Load model
    model = load_model()
    transform = get_medical_transforms(is_training=False)
    preprocessor = IntelligentPreprocessor()

    # Load false negatives from latest test
    latest_test = project_root / "validation" / "reports" / "conservative_1000img_test_20251110_125248"
    fn_path = latest_test / "false_negatives.json"

    if not fn_path.exists():
        print(f"ERROR: Could not find false negatives file at {fn_path}")
        return

    with open(fn_path, 'r') as f:
        false_negatives = json.load(f)

    print(f"\nAnalyzing {len(false_negatives)} false negative cases...")

    # Analyze each false negative
    detailed_results = []

    for idx, fn_case in enumerate(tqdm(false_negatives, desc="Analyzing FN cases"), 1):
        image_path = Path(fn_case['path'])

        if not image_path.exists():
            print(f"\nWARNING: Image not found: {image_path}")
            continue

        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        # Calculate metrics
        metrics = calculate_detailed_metrics(image)

        # Test preprocessing strategies
        preprocessing_results = test_preprocessing_strategies(
            image_path, model, transform, preprocessor
        )

        # Combine results
        result = {
            'index': idx,
            'filename': fn_case['filename'],
            'path': str(image_path),
            'original_prediction': {
                'normal_prob': fn_case['normal_prob'],
                'pneumonia_prob': fn_case['pneumonia_prob']
            },
            'metrics': metrics,
            'preprocessing_tests': preprocessing_results
        }

        detailed_results.append(result)

    # Analyze patterns
    print("\n" + "="*80)
    print("PATTERN ANALYSIS")
    print("="*80)

    # Calculate average metrics
    avg_metrics = {}
    for key in detailed_results[0]['metrics'].keys():
        values = [r['metrics'][key] for r in detailed_results]
        avg_metrics[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    print("\nAverage Quality Metrics for False Negatives:")
    print(f"  Brightness: {avg_metrics['brightness']['mean']:.3f} ± {avg_metrics['brightness']['std']:.3f}")
    print(f"  Contrast: {avg_metrics['contrast']['mean']:.3f} ± {avg_metrics['contrast']['std']:.3f}")
    print(f"  Sharpness: {avg_metrics['sharpness']['mean']:.4f} ± {avg_metrics['sharpness']['std']:.4f}")
    print(f"  Edge Density: {avg_metrics['edge_density']['mean']:.3f} ± {avg_metrics['edge_density']['std']:.3f}")
    print(f"  Dynamic Range: {avg_metrics['dynamic_range']['mean']:.3f} ± {avg_metrics['dynamic_range']['std']:.3f}")

    # Find best preprocessing strategy for each case
    print("\n" + "="*80)
    print("PREPROCESSING STRATEGY EFFECTIVENESS")
    print("="*80)

    strategy_improvements = {
        'none': 0,
        'conservative': 0,
        'light_global': 0,
        'medium_global': 0,
        'roi_clahe': 0
    }

    best_strategies = []

    for result in detailed_results:
        best_strategy = None
        best_prob = result['original_prediction']['pneumonia_prob']

        for strategy, preds in result['preprocessing_tests'].items():
            if 'error' not in preds and preds['pneumonia_prob'] > best_prob:
                best_prob = preds['pneumonia_prob']
                best_strategy = strategy

        if best_strategy:
            strategy_improvements[best_strategy] += 1

            # Check if it would fix the misclassification
            if best_prob > 0.5:
                best_strategies.append({
                    'filename': result['filename'],
                    'strategy': best_strategy,
                    'original_prob': result['original_prediction']['pneumonia_prob'],
                    'improved_prob': best_prob,
                    'improvement': best_prob - result['original_prediction']['pneumonia_prob'],
                    'would_fix': True
                })
            else:
                best_strategies.append({
                    'filename': result['filename'],
                    'strategy': best_strategy,
                    'original_prob': result['original_prediction']['pneumonia_prob'],
                    'improved_prob': best_prob,
                    'improvement': best_prob - result['original_prediction']['pneumonia_prob'],
                    'would_fix': False
                })

    print("\nStrategy Improvement Counts:")
    for strategy, count in strategy_improvements.items():
        print(f"  {strategy}: {count} cases improved")

    # Count how many could be fixed
    fixable = sum(1 for s in best_strategies if s['would_fix'])
    print(f"\nPotentially Fixable: {fixable}/{len(false_negatives)} ({fixable/len(false_negatives)*100:.1f}%)")

    # Show top 10 most improvable cases
    print("\n" + "="*80)
    print("TOP 10 MOST IMPROVABLE CASES")
    print("="*80)

    best_strategies_sorted = sorted(best_strategies, key=lambda x: x['improvement'], reverse=True)[:10]

    for idx, case in enumerate(best_strategies_sorted, 1):
        print(f"\n{idx}. {case['filename']}")
        print(f"   Best Strategy: {case['strategy']}")
        print(f"   Original: {case['original_prob']*100:.2f}% pneumonia")
        print(f"   Improved: {case['improved_prob']*100:.2f}% pneumonia")
        print(f"   Gain: +{case['improvement']*100:.2f}%")
        print(f"   Would Fix: {'YES ✓' if case['would_fix'] else 'NO'}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = project_root / "validation" / "reports" / f"fn_detailed_analysis_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_to_native(obj.tolist())
        else:
            return obj

    # Save detailed results
    with open(save_dir / "detailed_results.json", 'w') as f:
        json.dump(convert_to_native(detailed_results), f, indent=2)

    # Save summary
    summary = {
        'total_false_negatives': len(false_negatives),
        'average_metrics': avg_metrics,
        'strategy_improvements': strategy_improvements,
        'potentially_fixable': fixable,
        'fixable_percentage': fixable / len(false_negatives) * 100,
        'top_improvable_cases': best_strategies_sorted[:10]
    }

    with open(save_dir / "summary.json", 'w') as f:
        json.dump(convert_to_native(summary), f, indent=2)

    print(f"\n" + "="*80)
    print("RESULTS SAVED")
    print("="*80)
    print(f"Directory: {save_dir}")
    print(f"  • detailed_results.json - Full analysis for all {len(false_negatives)} cases")
    print(f"  • summary.json - Summary statistics and patterns")

    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
