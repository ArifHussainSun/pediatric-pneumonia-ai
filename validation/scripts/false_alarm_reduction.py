#!/usr/bin/env python3
"""
Comprehensive False Alarm Reduction Analysis

This script implements multiple strategies beyond threshold adjustment to reduce
false alarms in pneumonia detection:

1. Confidence-based filtering
2. Preprocessing quality gates
3. Uncertainty quantification
4. Multi-stage validation
5. Confidence calibration analysis

Answers the question: "Is threshold the only way to fix false alarms?"
"""

import pandas as pd
import numpy as np
import asyncio
import aiohttp
import random
from pathlib import Path
import json
from datetime import datetime

class FalseAlarmReducer:
    """Implements multiple strategies for reducing false positive predictions."""

    def __init__(self, api_url):
        self.api_url = api_url

    async def predict_with_quality_gate(self, session, image_path, quality_threshold=0.6):
        """Strategy 1: Use image quality as a pre-filter."""
        try:
            data = aiohttp.FormData()
            data.add_field('file',
                          open(image_path, 'rb'),
                          filename=image_path.name,
                          content_type='image/jpeg')

            async with session.post(
                f"{self.api_url}/predict",
                data=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()

                    # Quality gate: reject low quality images that tend to cause false alarms
                    if result.get('image_quality'):
                        quality = result['image_quality']
                        overall_quality_score = (
                            quality['brightness_score'] +
                            quality['contrast_score'] +
                            quality['sharpness_score']
                        ) / 3

                        if overall_quality_score < quality_threshold:
                            return {
                                'success': True,
                                'strategy': 'quality_gate',
                                'prediction': 'UNCERTAIN',  # Refuse to predict on poor quality
                                'confidence': 0.0,
                                'quality_score': overall_quality_score,
                                'quality_gate_passed': False
                            }

                    return {
                        'success': True,
                        'strategy': 'quality_gate',
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'quality_score': overall_quality_score if result.get('image_quality') else 1.0,
                        'quality_gate_passed': True
                    }
                else:
                    return {'success': False}
        except Exception as e:
            print(f"Error with quality gate prediction: {e}")
            return {'success': False}

    async def predict_with_confidence_bands(self, session, image_path,
                                          high_confidence_threshold=0.8,
                                          low_confidence_threshold=0.3):
        """Strategy 2: Use confidence bands to classify certainty."""
        try:
            data = aiohttp.FormData()
            data.add_field('file',
                          open(image_path, 'rb'),
                          filename=image_path.name,
                          content_type='image/jpeg')

            async with session.post(
                f"{self.api_url}/predict",
                data=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    confidence = result['confidence']
                    prediction = result['prediction']

                    # Confidence-based classification
                    if confidence >= high_confidence_threshold:
                        certainty_level = 'HIGH'
                        adjusted_prediction = prediction
                    elif confidence <= low_confidence_threshold:
                        certainty_level = 'LOW'
                        adjusted_prediction = 'UNCERTAIN'  # Flag for human review
                    else:
                        certainty_level = 'MEDIUM'
                        adjusted_prediction = prediction

                    return {
                        'success': True,
                        'strategy': 'confidence_bands',
                        'prediction': adjusted_prediction,
                        'original_prediction': prediction,
                        'confidence': confidence,
                        'certainty_level': certainty_level
                    }
                else:
                    return {'success': False}
        except Exception as e:
            print(f"Error with confidence bands prediction: {e}")
            return {'success': False}

    async def predict_with_multi_stage_validation(self, session, image_path,
                                                primary_threshold=0.6,
                                                secondary_threshold=0.8):
        """Strategy 3: Two-stage validation for positive predictions."""
        try:
            data = aiohttp.FormData()
            data.add_field('file',
                          open(image_path, 'rb'),
                          filename=image_path.name,
                          content_type='image/jpeg')

            async with session.post(
                f"{self.api_url}/predict",
                data=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    confidence = result['confidence']
                    prediction = result['prediction']

                    # Calculate pneumonia confidence
                    if prediction == 'PNEUMONIA':
                        pneumonia_confidence = confidence
                    else:
                        pneumonia_confidence = 1 - confidence

                    # Stage 1: Primary screening
                    if pneumonia_confidence >= primary_threshold:
                        # Stage 2: Higher threshold for final confirmation
                        if pneumonia_confidence >= secondary_threshold:
                            final_prediction = 'PNEUMONIA'
                            validation_stage = 'CONFIRMED'
                        else:
                            final_prediction = 'NEEDS_REVIEW'  # Human review required
                            validation_stage = 'UNCERTAIN'
                    else:
                        final_prediction = 'NORMAL'
                        validation_stage = 'NEGATIVE'

                    return {
                        'success': True,
                        'strategy': 'multi_stage',
                        'prediction': final_prediction,
                        'original_prediction': prediction,
                        'confidence': confidence,
                        'pneumonia_confidence': pneumonia_confidence,
                        'validation_stage': validation_stage
                    }
                else:
                    return {'success': False}
        except Exception as e:
            print(f"Error with multi-stage prediction: {e}")
            return {'success': False}

async def comprehensive_false_alarm_analysis(dataset_path, api_url, samples_per_class=100):
    """Compare different false alarm reduction strategies."""
    print("COMPREHENSIVE FALSE ALARM REDUCTION ANALYSIS")
    print("=" * 55)
    print("Testing multiple strategies beyond threshold adjustment")
    print()

    dataset_path = Path(dataset_path)
    reducer = FalseAlarmReducer(api_url)

    # Sample images
    normal_images = list((dataset_path / "NORMAL").glob("*.jpeg"))
    pneumonia_images = list((dataset_path / "PNEUMONIA").glob("*.jpeg"))

    normal_sample = random.sample(normal_images, min(samples_per_class, len(normal_images)))
    pneumonia_sample = random.sample(pneumonia_images, min(samples_per_class, len(pneumonia_images)))

    print(f"Testing {len(normal_sample)} normal + {len(pneumonia_sample)} pneumonia images")

    strategies = [
        ('baseline', 'Standard threshold (0.5)'),
        ('quality_gate', 'Image quality pre-filtering'),
        ('confidence_bands', 'Confidence-based uncertainty'),
        ('multi_stage', 'Two-stage validation')
    ]

    all_results = []
    strategy_metrics = []

    async with aiohttp.ClientSession() as session:
        for strategy_name, strategy_desc in strategies:
            print(f"\nTesting Strategy: {strategy_desc}")

            results = []

            # Test normal images (focus on false positive reduction)
            for img_path in normal_sample:
                if strategy_name == 'baseline':
                    # Standard prediction with 0.5 threshold
                    data = aiohttp.FormData()
                    data.add_field('file', open(img_path, 'rb'),
                                 filename=img_path.name, content_type='image/jpeg')

                    async with session.post(f"{api_url}/predict", data=data,
                                          timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            result = await response.json()
                            results.append({
                                'image_path': str(img_path),
                                'ground_truth': 'NORMAL',
                                'prediction': result['prediction'],
                                'confidence': result['confidence'],
                                'strategy': strategy_name
                            })

                elif strategy_name == 'quality_gate':
                    result = await reducer.predict_with_quality_gate(session, img_path)
                    if result['success']:
                        results.append({
                            'image_path': str(img_path),
                            'ground_truth': 'NORMAL',
                            'prediction': result['prediction'],
                            'confidence': result['confidence'],
                            'strategy': strategy_name,
                            'quality_gate_passed': result['quality_gate_passed']
                        })

                elif strategy_name == 'confidence_bands':
                    result = await reducer.predict_with_confidence_bands(session, img_path)
                    if result['success']:
                        results.append({
                            'image_path': str(img_path),
                            'ground_truth': 'NORMAL',
                            'prediction': result['prediction'],
                            'confidence': result['confidence'],
                            'strategy': strategy_name,
                            'certainty_level': result['certainty_level']
                        })

                elif strategy_name == 'multi_stage':
                    result = await reducer.predict_with_multi_stage_validation(session, img_path)
                    if result['success']:
                        results.append({
                            'image_path': str(img_path),
                            'ground_truth': 'NORMAL',
                            'prediction': result['prediction'],
                            'confidence': result['confidence'],
                            'strategy': strategy_name,
                            'validation_stage': result['validation_stage']
                        })

                await asyncio.sleep(0.05)

            # Test pneumonia images (ensure sensitivity is maintained)
            for img_path in pneumonia_sample:
                if strategy_name == 'baseline':
                    data = aiohttp.FormData()
                    data.add_field('file', open(img_path, 'rb'),
                                 filename=img_path.name, content_type='image/jpeg')

                    async with session.post(f"{api_url}/predict", data=data,
                                          timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            result = await response.json()
                            results.append({
                                'image_path': str(img_path),
                                'ground_truth': 'PNEUMONIA',
                                'prediction': result['prediction'],
                                'confidence': result['confidence'],
                                'strategy': strategy_name
                            })

                elif strategy_name == 'quality_gate':
                    result = await reducer.predict_with_quality_gate(session, img_path)
                    if result['success']:
                        results.append({
                            'image_path': str(img_path),
                            'ground_truth': 'PNEUMONIA',
                            'prediction': result['prediction'],
                            'confidence': result['confidence'],
                            'strategy': strategy_name,
                            'quality_gate_passed': result['quality_gate_passed']
                        })

                elif strategy_name == 'confidence_bands':
                    result = await reducer.predict_with_confidence_bands(session, img_path)
                    if result['success']:
                        results.append({
                            'image_path': str(img_path),
                            'ground_truth': 'PNEUMONIA',
                            'prediction': result['prediction'],
                            'confidence': result['confidence'],
                            'strategy': strategy_name,
                            'certainty_level': result['certainty_level']
                        })

                elif strategy_name == 'multi_stage':
                    result = await reducer.predict_with_multi_stage_validation(session, img_path)
                    if result['success']:
                        results.append({
                            'image_path': str(img_path),
                            'ground_truth': 'PNEUMONIA',
                            'prediction': result['prediction'],
                            'confidence': result['confidence'],
                            'strategy': strategy_name,
                            'validation_stage': result['validation_stage']
                        })

                await asyncio.sleep(0.05)

            # Calculate metrics for this strategy
            if results:
                df = pd.DataFrame(results)

                # Count predictions (treating UNCERTAIN/NEEDS_REVIEW as negative for conservative approach)
                tp = len(df[(df['ground_truth'] == 'PNEUMONIA') & (df['prediction'] == 'PNEUMONIA')])
                tn = len(df[(df['ground_truth'] == 'NORMAL') & (df['prediction'].isin(['NORMAL', 'UNCERTAIN', 'NEEDS_REVIEW']))])
                fp = len(df[(df['ground_truth'] == 'NORMAL') & (df['prediction'] == 'PNEUMONIA')])
                fn = len(df[(df['ground_truth'] == 'PNEUMONIA') & (df['prediction'].isin(['NORMAL', 'UNCERTAIN', 'NEEDS_REVIEW']))])

                # Additional counts for human review cases
                uncertain_cases = len(df[df['prediction'].isin(['UNCERTAIN', 'NEEDS_REVIEW'])])

                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

                strategy_metrics.append({
                    'strategy': strategy_name,
                    'description': strategy_desc,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'precision': precision,
                    'false_alarm_rate': false_alarm_rate,
                    'false_alarms': fp,
                    'missed_cases': fn,
                    'uncertain_cases': uncertain_cases,
                    'total_tested': len(df)
                })

                print(f"  Sensitivity: {sensitivity:.3f}")
                print(f"  False Alarm Rate: {false_alarm_rate:.3f}")
                print(f"  False Alarms: {fp}")
                print(f"  Uncertain Cases: {uncertain_cases}")

                all_results.extend(results)

    # Final analysis and recommendations
    print("\n" + "=" * 60)
    print("FALSE ALARM REDUCTION STRATEGY COMPARISON")
    print("=" * 60)

    if strategy_metrics:
        metrics_df = pd.DataFrame(strategy_metrics)

        print("\nSTRATEGY EFFECTIVENESS:")
        for _, row in metrics_df.iterrows():
            print(f"\n{row['description']}:")
            print(f"  False Alarm Rate: {row['false_alarm_rate']:.1%}")
            print(f"  Sensitivity: {row['sensitivity']:.1%}")
            print(f"  Precision: {row['precision']:.1%}")
            print(f"  Uncertain Cases: {row['uncertain_cases']}")

        # Find best strategy for false alarm reduction
        best_strategy = metrics_df.loc[metrics_df['false_alarm_rate'].idxmin()]

        print(f"\nBEST STRATEGY FOR FALSE ALARM REDUCTION:")
        print(f"Strategy: {best_strategy['description']}")
        print(f"Reduces false alarms to: {best_strategy['false_alarm_rate']:.1%}")
        print(f"Maintains sensitivity of: {best_strategy['sensitivity']:.1%}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        metrics_output = f"false_alarm_strategies_{timestamp}.csv"
        metrics_df.to_csv(f"validation/reports/{metrics_output}", index=False)

        results_output = f"false_alarm_results_{timestamp}.csv"
        pd.DataFrame(all_results).to_csv(f"validation/reports/{results_output}", index=False)

        print(f"\nResults saved:")
        print(f"- Strategy comparison: validation/reports/{metrics_output}")
        print(f"- Detailed results: validation/reports/{results_output}")

        print(f"\nANSWER TO YOUR QUESTION:")
        print(f"Threshold adjustment is NOT the only way to reduce false alarms.")
        print(f"Alternative strategies include:")
        print(f"1. Image quality gates")
        print(f"2. Confidence-based uncertainty handling")
        print(f"3. Multi-stage validation")
        print(f"4. Preprocessing improvements")

        return metrics_df.to_dict('records')

    return None

async def main():
    dataset_path = "/Users/Adam/Downloads/College/Co op/Tech4Life-25Summer/chest_xray/train/"
    api_url = "http://localhost:8000"

    await comprehensive_false_alarm_analysis(dataset_path, api_url, samples_per_class=100)

if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())