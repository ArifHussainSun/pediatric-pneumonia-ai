"""
Compare Original ROI vs Improved (A+C) ROI on 7 False Negatives

This script tests:
1. Baseline (no ROI preprocessing)
2. Original ROI (fixed clip_limit=3.0)
3. Improved ROI (A+C: adaptive clips + quality-aware skip)

Author: Analysis Team
Date: 2025-11-10
"""

import sys
from pathlib import Path
import json
import torch
import numpy as np
from PIL import Image
import cv2

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.mobilenet import MobileNetFineTune
from src.analysis.gradcam_roi_analysis import (
    LungROIDetector,
    ROIBasedPreprocessor,
    PatchAnalyzer
)


class ROIComparison:
    """Compare different ROI preprocessing approaches"""

    def __init__(self, model_path: Path):
        """Initialize with model"""
        self.device = torch.device('cpu')

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = MobileNetFineTune(num_classes=2).to(self.device)
        self.model.load_custom_weights(model_path)
        self.model.eval()
        print("Model loaded successfully!")

        # Initialize processors
        self.roi_detector = LungROIDetector()
        self.roi_processor = ROIBasedPreprocessor(self.roi_detector)
        self.patch_analyzer = PatchAnalyzer()

    def load_and_preprocess_image(self, image_path: Path) -> torch.Tensor:
        """Load image and apply standard preprocessing (resize, normalize)"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224), Image.LANCZOS)

        img_array = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std

        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0).float()
        return img_tensor.to(self.device)

    def apply_original_roi(self, image: np.ndarray) -> np.ndarray:
        """Apply original ROI preprocessing (fixed clip_limit=3.0)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Detect ROI
        roi_info = self.roi_detector.detect_lung_roi(gray)

        # Fixed clip limits (original approach)
        roi_clip_limit = 3.0
        background_clip_limit = 1.5

        # Apply CLAHE
        roi_clahe = cv2.createCLAHE(clipLimit=roi_clip_limit, tileGridSize=(8, 8))
        bg_clahe = cv2.createCLAHE(clipLimit=background_clip_limit, tileGridSize=(8, 8))

        enhanced_roi = roi_clahe.apply(gray)
        enhanced_bg = bg_clahe.apply(gray)

        mask = roi_info['mask']
        enhanced = np.where(mask > 0, enhanced_roi, enhanced_bg)

        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return enhanced_rgb

    def apply_improved_roi(self, image: np.ndarray) -> tuple:
        """Apply improved ROI preprocessing (A+C: adaptive + skip)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Get quality metrics
        patch_analysis = self.patch_analyzer.analyze_patches(gray)
        quality_metrics = {
            'brightness': float(np.mean(patch_analysis['brightness'])),
            'contrast': float(np.mean(patch_analysis['contrast'])),
            'sharpness': float(np.mean(patch_analysis['sharpness']))
        }

        # Apply adaptive preprocessing
        result = self.roi_processor.preprocess_with_adaptive_roi(
            gray,
            quality_metrics=quality_metrics
        )

        enhanced = result['enhanced_image']
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

        return enhanced_rgb, result

    def predict(self, img_tensor: torch.Tensor) -> dict:
        """Run prediction and return results"""
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)

            # Class 0 = PNEUMONIA, Class 1 = NORMAL (matches rerun_500_image_test.py)
            pneumonia_prob = probabilities[0][0].item()
            normal_prob = probabilities[0][1].item()

            predicted_class = 0 if pneumonia_prob > normal_prob else 1
            confidence = max(normal_prob, pneumonia_prob)

            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'normal_prob': normal_prob,
                'pneumonia_prob': pneumonia_prob,
                'prediction': 'PNEUMONIA' if predicted_class == 0 else 'NORMAL'
            }

    def compare_approaches(self, image_path: Path) -> dict:
        """Compare all three approaches on a single image"""
        print(f"\nAnalyzing: {image_path.name}")

        # Load original image
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)

        # Get quality metrics
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        patch_analysis = self.patch_analyzer.analyze_patches(gray)

        # Calculate overall metrics from patches
        quality_metrics = {
            'brightness': float(np.mean(patch_analysis['brightness'])),
            'contrast': float(np.mean(patch_analysis['contrast'])),
            'sharpness': float(np.mean(patch_analysis['sharpness']))
        }

        print(f"  Quality - Brightness: {quality_metrics['brightness']:.3f}, "
              f"Contrast: {quality_metrics['contrast']:.3f}")

        results = {
            'filename': image_path.name,
            'quality': quality_metrics,
            'approaches': {}
        }

        # 1. Baseline (no ROI)
        print("  Testing baseline (no ROI)...")
        baseline_tensor = self.load_and_preprocess_image(image_path)
        baseline_pred = self.predict(baseline_tensor)
        results['approaches']['baseline'] = baseline_pred
        print(f"    Baseline: {baseline_pred['prediction']} "
              f"({baseline_pred['pneumonia_prob']:.4f})")

        # 2. Original ROI (fixed clip_limit)
        print("  Testing original ROI (fixed clip=3.0)...")
        original_roi = self.apply_original_roi(img_np)
        original_roi_pil = Image.fromarray(original_roi.astype(np.uint8))

        # Save to temp and reload with preprocessing
        temp_path = image_path.parent / f"temp_original_{image_path.name}"
        original_roi_pil.save(temp_path)
        original_tensor = self.load_and_preprocess_image(temp_path)
        original_pred = self.predict(original_tensor)
        temp_path.unlink()  # Delete temp file

        results['approaches']['original_roi'] = original_pred
        print(f"    Original ROI: {original_pred['prediction']} "
              f"({original_pred['pneumonia_prob']:.4f})")

        # 3. Improved ROI (A+C: adaptive + skip)
        print("  Testing improved ROI (A+C)...")
        improved_roi, adaptive_info = self.apply_improved_roi(img_np)
        improved_roi_pil = Image.fromarray(improved_roi.astype(np.uint8))

        # Save to temp and reload with preprocessing
        temp_path = image_path.parent / f"temp_improved_{image_path.name}"
        improved_roi_pil.save(temp_path)
        improved_tensor = self.load_and_preprocess_image(temp_path)
        improved_pred = self.predict(improved_tensor)
        temp_path.unlink()  # Delete temp file

        results['approaches']['improved_roi'] = improved_pred
        results['adaptive_info'] = adaptive_info['enhancement_applied']

        print(f"    Improved ROI: {improved_pred['prediction']} "
              f"({improved_pred['pneumonia_prob']:.4f})")

        # Calculate improvements
        baseline_pneumonia = baseline_pred['pneumonia_prob']
        original_improvement = original_pred['pneumonia_prob'] - baseline_pneumonia
        improved_improvement = improved_pred['pneumonia_prob'] - baseline_pneumonia

        results['improvements'] = {
            'original_roi_delta': original_improvement,
            'improved_roi_delta': improved_improvement,
            'improved_vs_original': improved_improvement - original_improvement
        }

        print(f"  Improvements:")
        print(f"    Original ROI: {original_improvement:+.4f}")
        print(f"    Improved ROI: {improved_improvement:+.4f}")
        print(f"    Improved vs Original: {improved_improvement - original_improvement:+.4f}")

        # Did improved fix it?
        if baseline_pred['prediction'] == 'NORMAL' and improved_pred['prediction'] == 'PNEUMONIA':
            print(f"  ✓ Improved ROI FIXED this false negative!")
            results['improved_fixed'] = True
        elif baseline_pred['prediction'] == 'NORMAL' and original_pred['prediction'] == 'PNEUMONIA':
            if improved_pred['prediction'] == 'NORMAL':
                print(f"  ⚠ Original fixed it, but improved broke it again")
                results['improved_fixed'] = False
            else:
                print(f"  ✓ Both fixed it (original also worked)")
                results['improved_fixed'] = True
        else:
            print(f"  ✗ Still not fixed")
            results['improved_fixed'] = False

        return results


def main():
    """Run comparison on all 7 false negatives"""

    # Paths
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "outputs" / "dgx_station_experiment" / "Best_MobilenetV1.pth"

    # Load false negatives list
    fn_path = project_root / "validation" / "reports" / "rerun_test_20251110_005107" / "false_negatives.json"

    print("=" * 80)
    print("ROI APPROACH COMPARISON: Original vs Improved (A+C)")
    print("=" * 80)
    print("\nTesting 3 approaches:")
    print("  1. Baseline: No ROI preprocessing")
    print("  2. Original ROI: Fixed clip_limit=3.0 for ROI, 1.5 for background")
    print("  3. Improved ROI (A+C): Adaptive clips + quality-aware skip")
    print("\nImprovements implemented:")
    print("  A. Adaptive Enhancement: Clip limits based on brightness/contrast")
    print("  C. Quality-Aware Skip: Skip ROI for good quality images")
    print("=" * 80)

    # Load false negatives
    with open(fn_path, 'r') as f:
        false_negatives = json.load(f)

    print(f"\nLoaded {len(false_negatives)} false negatives")

    # Initialize comparator
    comparator = ROIComparison(model_path)

    # Run comparisons
    all_results = []

    for fn in false_negatives:
        image_path = Path(fn['path'])
        result = comparator.compare_approaches(image_path)
        all_results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    original_fixed = sum(1 for r in all_results
                         if r['approaches']['baseline']['prediction'] == 'NORMAL'
                         and r['approaches']['original_roi']['prediction'] == 'PNEUMONIA')

    improved_fixed = sum(1 for r in all_results if r.get('improved_fixed', False))

    original_improvements = [r['improvements']['original_roi_delta'] for r in all_results]
    improved_improvements = [r['improvements']['improved_roi_delta'] for r in all_results]

    avg_original = np.mean(original_improvements)
    avg_improved = np.mean(improved_improvements)

    print(f"\nFixed Count:")
    print(f"  Original ROI: {original_fixed}/7 ({original_fixed/7*100:.1f}%)")
    print(f"  Improved ROI: {improved_fixed}/7 ({improved_fixed/7*100:.1f}%)")

    print(f"\nAverage Pneumonia Probability Improvement:")
    print(f"  Original ROI: {avg_original:+.4f}")
    print(f"  Improved ROI: {avg_improved:+.4f}")
    print(f"  Difference: {avg_improved - avg_original:+.4f}")

    # Check if improved is better
    if improved_fixed >= original_fixed and avg_improved >= avg_original:
        print(f"\n✓ Improved ROI (A+C) is BETTER or EQUAL to original")
        recommendation = "Use Improved ROI (A+C)"
    elif improved_fixed > original_fixed:
        print(f"\n✓ Improved ROI (A+C) fixes more cases, but average improvement is lower")
        recommendation = "Use Improved ROI (A+C) - more fixes matter more"
    elif avg_improved > avg_original:
        print(f"\n⚠ Improved ROI (A+C) has better average, but fixes fewer cases")
        recommendation = "Consider hybrid approach or use Original ROI"
    else:
        print(f"\n✗ Original ROI is better")
        recommendation = "Use Original ROI"

    print(f"\nRecommendation: {recommendation}")

    # Detailed results per case
    print(f"\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    for i, result in enumerate(all_results, 1):
        print(f"\n{i}. {result['filename']}")
        print(f"   Quality: Brightness={result['quality']['brightness']:.3f}, "
              f"Contrast={result['quality']['contrast']:.3f}")

        baseline = result['approaches']['baseline']
        original = result['approaches']['original_roi']
        improved = result['approaches']['improved_roi']

        print(f"   Baseline:     {baseline['prediction']:9s} "
              f"(pneumonia={baseline['pneumonia_prob']:.4f})")
        print(f"   Original ROI: {original['prediction']:9s} "
              f"(pneumonia={original['pneumonia_prob']:.4f}) "
              f"[{result['improvements']['original_roi_delta']:+.4f}]")
        print(f"   Improved ROI: {improved['prediction']:9s} "
              f"(pneumonia={improved['pneumonia_prob']:.4f}) "
              f"[{result['improvements']['improved_roi_delta']:+.4f}]")

        # Show adaptive info
        if 'adaptive_info' in result:
            info = result['adaptive_info']
            if info['method'] == 'global_clahe':
                print(f"   Adaptive: Skipped ROI (good quality), used global CLAHE")
            else:
                print(f"   Adaptive: ROI clip={info['roi_clip_limit']:.1f}, "
                      f"BG clip={info['background_clip_limit']:.1f}")

        if result.get('improved_fixed'):
            print(f"   ✓ FIXED by Improved ROI")

    # Save results
    output_dir = project_root / "validation" / "reports" / "roi_comparison_20251110"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'original_fixed_count': original_fixed,
                'improved_fixed_count': improved_fixed,
                'original_avg_improvement': float(avg_original),
                'improved_avg_improvement': float(avg_improved),
                'recommendation': recommendation
            },
            'results': all_results
        }, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
