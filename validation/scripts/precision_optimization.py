#!/usr/bin/env python3
"""
Precision-Focused Threshold Optimization for Pneumonia Detection

This script optimizes the classification threshold to minimize false alarms (false positives)
while maintaining acceptable recall. Addresses the critical balance between sensitivity
and specificity in medical AI applications.

Key Features:
- Multi-threshold testing with precision focus
- False alarm rate analysis
- Clinical utility scoring
- ROC curve analysis for optimal threshold selection
"""

import pandas as pd
import numpy as np
import asyncio
import aiohttp
import random
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

async def predict_with_threshold(session, image_path, api_url, threshold=0.5):
    """Make prediction and apply custom threshold for pneumonia detection."""
    try:
        data = aiohttp.FormData()
        data.add_field('file',
                      open(image_path, 'rb'),
                      filename=image_path.name,
                      content_type='image/jpeg')

        async with session.post(
            f"{api_url}/predict",
            data=data,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                result = await response.json()

                # Get raw prediction and confidence
                raw_prediction = result['prediction']
                raw_confidence = result['confidence']

                # Calculate pneumonia confidence
                if raw_prediction == 'PNEUMONIA':
                    pneumonia_confidence = raw_confidence
                else:
                    pneumonia_confidence = 1 - raw_confidence

                # Apply custom threshold
                adjusted_prediction = 'PNEUMONIA' if pneumonia_confidence >= threshold else 'NORMAL'

                return {
                    'success': True,
                    'raw_prediction': raw_prediction,
                    'raw_confidence': raw_confidence,
                    'pneumonia_confidence': pneumonia_confidence,
                    'adjusted_prediction': adjusted_prediction,
                    'threshold_used': threshold
                }
            else:
                return {'success': False}

    except Exception as e:
        print(f"Error with {image_path.name}: {e}")
        return {'success': False}

def calculate_clinical_utility_score(tp, tn, fp, fn):
    """
    Calculate clinical utility score that heavily penalizes false alarms
    while maintaining sensitivity requirements.
    """
    # Weights for medical context: False alarms cause unnecessary anxiety/treatment
    # False negatives are critical but preprocessing should help reduce them
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Clinical utility: heavily weight false alarm reduction
    # while maintaining minimum sensitivity threshold
    if sensitivity < 0.85:  # Minimum 85% sensitivity for pneumonia detection
        return 0.0

    # Utility score: balance precision and sensitivity with false alarm penalty
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    utility = (sensitivity * 0.4) + (precision * 0.4) + ((1 - false_alarm_rate) * 0.2)

    return utility

async def precision_optimization_analysis(dataset_path, api_url, samples_per_class=200):
    """Comprehensive precision optimization with false alarm focus."""
    print("PRECISION-FOCUSED THRESHOLD OPTIMIZATION")
    print("=" * 50)
    print(f"Focus: Minimize false alarms while maintaining clinical sensitivity")
    print()

    dataset_path = Path(dataset_path)

    # Sample images
    normal_images = list((dataset_path / "NORMAL").glob("*.jpeg"))
    pneumonia_images = list((dataset_path / "PNEUMONIA").glob("*.jpeg"))

    normal_sample = random.sample(normal_images, min(samples_per_class, len(normal_images)))
    pneumonia_sample = random.sample(pneumonia_images, min(samples_per_class, len(pneumonia_images)))

    print(f"Testing {len(normal_sample)} normal + {len(pneumonia_sample)} pneumonia = {len(normal_sample) + len(pneumonia_sample)} images")

    # Test multiple thresholds focused on precision
    thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    all_results = []
    threshold_metrics = []

    async with aiohttp.ClientSession() as session:
        for threshold in thresholds_to_test:
            print(f"\nTesting threshold: {threshold}")

            results = []

            # Test normal images
            for img_path in normal_sample:
                result = await predict_with_threshold(session, img_path, api_url, threshold)
                if result['success']:
                    results.append({
                        'image_path': str(img_path),
                        'ground_truth': 'NORMAL',
                        'pneumonia_confidence': result['pneumonia_confidence'],
                        'adjusted_prediction': result['adjusted_prediction'],
                        'threshold': threshold,
                        'correct': result['adjusted_prediction'] == 'NORMAL'
                    })
                await asyncio.sleep(0.05)

            # Test pneumonia images
            for img_path in pneumonia_sample:
                result = await predict_with_threshold(session, img_path, api_url, threshold)
                if result['success']:
                    results.append({
                        'image_path': str(img_path),
                        'ground_truth': 'PNEUMONIA',
                        'pneumonia_confidence': result['pneumonia_confidence'],
                        'adjusted_prediction': result['adjusted_prediction'],
                        'threshold': threshold,
                        'correct': result['adjusted_prediction'] == 'PNEUMONIA'
                    })
                await asyncio.sleep(0.05)

            if results:
                # Calculate metrics for this threshold
                df = pd.DataFrame(results)

                tp = len(df[(df['ground_truth'] == 'PNEUMONIA') & (df['adjusted_prediction'] == 'PNEUMONIA')])
                tn = len(df[(df['ground_truth'] == 'NORMAL') & (df['adjusted_prediction'] == 'NORMAL')])
                fp = len(df[(df['ground_truth'] == 'NORMAL') & (df['adjusted_prediction'] == 'PNEUMONIA')])
                fn = len(df[(df['ground_truth'] == 'PNEUMONIA') & (df['adjusted_prediction'] == 'NORMAL')])

                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                accuracy = (tp + tn) / len(df)
                false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

                clinical_utility = calculate_clinical_utility_score(tp, tn, fp, fn)

                threshold_metrics.append({
                    'threshold': threshold,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'precision': precision,
                    'accuracy': accuracy,
                    'false_alarm_rate': false_alarm_rate,
                    'false_alarms': fp,
                    'missed_cases': fn,
                    'clinical_utility': clinical_utility,
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
                })

                print(f"  Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}")
                print(f"  Precision: {precision:.3f}, False Alarm Rate: {false_alarm_rate:.3f}")
                print(f"  False Alarms: {fp}, Missed Cases: {fn}")
                print(f"  Clinical Utility Score: {clinical_utility:.3f}")

                all_results.extend(results)

    # Analysis and recommendations
    print("\n" + "=" * 60)
    print("PRECISION OPTIMIZATION ANALYSIS")
    print("=" * 60)

    if threshold_metrics:
        metrics_df = pd.DataFrame(threshold_metrics)

        # Find optimal thresholds for different criteria
        optimal_utility = metrics_df.loc[metrics_df['clinical_utility'].idxmax()]
        optimal_precision = metrics_df.loc[metrics_df['precision'].idxmax()]
        min_false_alarms = metrics_df.loc[metrics_df['false_alarms'].idxmin()]

        print(f"\nOPTIMAL THRESHOLD RECOMMENDATIONS:")
        print(f"1. Best Clinical Utility: {optimal_utility['threshold']:.1f}")
        print(f"   - Sensitivity: {optimal_utility['sensitivity']:.1%}")
        print(f"   - Precision: {optimal_utility['precision']:.1%}")
        print(f"   - False Alarms: {optimal_utility['false_alarms']}")
        print(f"   - Utility Score: {optimal_utility['clinical_utility']:.3f}")

        print(f"\n2. Best Precision: {optimal_precision['threshold']:.1f}")
        print(f"   - Sensitivity: {optimal_precision['sensitivity']:.1%}")
        print(f"   - Precision: {optimal_precision['precision']:.1%}")
        print(f"   - False Alarms: {optimal_precision['false_alarms']}")

        print(f"\n3. Minimum False Alarms: {min_false_alarms['threshold']:.1f}")
        print(f"   - Sensitivity: {min_false_alarms['sensitivity']:.1%}")
        print(f"   - Precision: {min_false_alarms['precision']:.1%}")
        print(f"   - False Alarms: {min_false_alarms['false_alarms']}")

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save threshold comparison
        metrics_output_file = f"precision_optimization_{timestamp}.csv"
        metrics_df.to_csv(f"validation/reports/{metrics_output_file}", index=False)

        # Save all individual results
        all_results_df = pd.DataFrame(all_results)
        results_output_file = f"precision_results_{timestamp}.csv"
        all_results_df.to_csv(f"validation/reports/{results_output_file}", index=False)

        print(f"\nResults saved:")
        print(f"- Threshold metrics: validation/reports/{metrics_output_file}")
        print(f"- Individual results: validation/reports/{results_output_file}")

        return {
            'optimal_utility_threshold': optimal_utility['threshold'],
            'optimal_precision_threshold': optimal_precision['threshold'],
            'min_false_alarms_threshold': min_false_alarms['threshold'],
            'metrics': metrics_df.to_dict('records')
        }

    return None

async def main():
    dataset_path = "/Users/Adam/Downloads/College/Co op/Tech4Life-25Summer/chest_xray/train/"
    api_url = "http://localhost:8000"

    print("CLINICAL-GRADE PRECISION OPTIMIZATION")
    print("=" * 40)
    print("Objective: Minimize false alarms while maintaining sensitivity")
    print()

    result = await precision_optimization_analysis(
        dataset_path,
        api_url,
        samples_per_class=150  # Adequate sample for reliable statistics
    )

    if result:
        print(f"\nRECOMMENDED CLINICAL DEPLOYMENT THRESHOLD:")
        print(f"Use threshold: {result['optimal_utility_threshold']:.1f}")
        print(f"This balances sensitivity and false alarm rate for clinical use.")

if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    asyncio.run(main())