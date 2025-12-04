#!/usr/bin/env python3
"""
Test threshold adjustment with intelligent preprocessing to improve recall and reduce false negatives.
Compares model performance with and without preprocessing pipeline.
"""

import pandas as pd
import numpy as np
import asyncio
import aiohttp
import random
from pathlib import Path
import json
from datetime import datetime

async def predict_with_threshold(session, image_path, api_url, threshold=0.1, use_preprocessing=True):
    """Make prediction and apply custom threshold for pneumonia detection."""
    try:
        data = aiohttp.FormData()
        data.add_field('file',
                      open(image_path, 'rb'),
                      filename=image_path.name,
                      content_type='image/jpeg')

        # Add preprocessing parameter
        if not use_preprocessing:
            data.add_field('disable_preprocessing', 'true')

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
                else:  # raw_prediction == 'NORMAL'
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

async def test_threshold_adjustment(dataset_path, api_url, threshold=0.1, samples_per_class=100, test_preprocessing=True):
    """Test threshold adjustment on a sample dataset with and without preprocessing."""
    if test_preprocessing:
        print(f"TESTING PREPROCESSING IMPACT WITH THRESHOLD ADJUSTMENT (threshold={threshold})")
        print("=" * 70)
    else:
        print(f"TESTING THRESHOLD ADJUSTMENT (threshold={threshold})")
        print("=" * 55)

    dataset_path = Path(dataset_path)

    # Sample images
    normal_images = list((dataset_path / "NORMAL").glob("*.jpeg"))
    pneumonia_images = list((dataset_path / "PNEUMONIA").glob("*.jpeg"))

    normal_sample = random.sample(normal_images, min(samples_per_class, len(normal_images)))
    pneumonia_sample = random.sample(pneumonia_images, min(samples_per_class, len(pneumonia_images)))

    print(f"Testing {len(normal_sample)} normal + {len(pneumonia_sample)} pneumonia = {len(normal_sample) + len(pneumonia_sample)} images")
    print()

    if test_preprocessing:
        # Test both with and without preprocessing for comparison
        results_with_preprocessing = []
        results_without_preprocessing = []

        async with aiohttp.ClientSession() as session:
            print("Testing WITH intelligent preprocessing...")

            # Test normal images with preprocessing
            for img_path in normal_sample:
                result = await predict_with_threshold(session, img_path, api_url, threshold, use_preprocessing=True)
                if result['success']:
                    results_with_preprocessing.append({
                        'image_path': str(img_path),
                        'ground_truth': 'NORMAL',
                        'raw_prediction': result['raw_prediction'],
                        'raw_confidence': result['raw_confidence'],
                        'pneumonia_confidence': result['pneumonia_confidence'],
                        'adjusted_prediction': result['adjusted_prediction'],
                        'threshold': threshold,
                        'raw_correct': result['raw_prediction'] == 'NORMAL',
                        'adjusted_correct': result['adjusted_prediction'] == 'NORMAL',
                        'preprocessing': True
                    })
                await asyncio.sleep(0.1)

            # Test pneumonia images with preprocessing
            for img_path in pneumonia_sample:
                result = await predict_with_threshold(session, img_path, api_url, threshold, use_preprocessing=True)
                if result['success']:
                    results_with_preprocessing.append({
                        'image_path': str(img_path),
                        'ground_truth': 'PNEUMONIA',
                        'raw_prediction': result['raw_prediction'],
                        'raw_confidence': result['raw_confidence'],
                        'pneumonia_confidence': result['pneumonia_confidence'],
                        'adjusted_prediction': result['adjusted_prediction'],
                        'threshold': threshold,
                        'raw_correct': result['raw_prediction'] == 'PNEUMONIA',
                        'adjusted_correct': result['adjusted_prediction'] == 'PNEUMONIA',
                        'preprocessing': True
                    })
                await asyncio.sleep(0.1)

            print(f"Processed with preprocessing: {len(normal_sample)} normal + {len(pneumonia_sample)} pneumonia")

            print("Testing WITHOUT preprocessing (baseline)...")

            # Test normal images without preprocessing
            for img_path in normal_sample:
                result = await predict_with_threshold(session, img_path, api_url, threshold, use_preprocessing=False)
                if result['success']:
                    results_without_preprocessing.append({
                        'image_path': str(img_path),
                        'ground_truth': 'NORMAL',
                        'raw_prediction': result['raw_prediction'],
                        'raw_confidence': result['raw_confidence'],
                        'pneumonia_confidence': result['pneumonia_confidence'],
                        'adjusted_prediction': result['adjusted_prediction'],
                        'threshold': threshold,
                        'raw_correct': result['raw_prediction'] == 'NORMAL',
                        'adjusted_correct': result['adjusted_prediction'] == 'NORMAL',
                        'preprocessing': False
                    })
                await asyncio.sleep(0.1)

            # Test pneumonia images without preprocessing
            for img_path in pneumonia_sample:
                result = await predict_with_threshold(session, img_path, api_url, threshold, use_preprocessing=False)
                if result['success']:
                    results_without_preprocessing.append({
                        'image_path': str(img_path),
                        'ground_truth': 'PNEUMONIA',
                        'raw_prediction': result['raw_prediction'],
                        'raw_confidence': result['raw_confidence'],
                        'pneumonia_confidence': result['pneumonia_confidence'],
                        'adjusted_prediction': result['adjusted_prediction'],
                        'threshold': threshold,
                        'raw_correct': result['raw_prediction'] == 'PNEUMONIA',
                        'adjusted_correct': result['adjusted_prediction'] == 'PNEUMONIA',
                        'preprocessing': False
                    })
                await asyncio.sleep(0.1)

            print(f"Processed without preprocessing: {len(normal_sample)} normal + {len(pneumonia_sample)} pneumonia")

        # Combine results for analysis
        results = results_with_preprocessing + results_without_preprocessing
    else:
        # Original single-mode testing
        results = []

        async with aiohttp.ClientSession() as session:
            # Test normal images
            for img_path in normal_sample:
                result = await predict_with_threshold(session, img_path, api_url, threshold)
                if result['success']:
                    results.append({
                        'image_path': str(img_path),
                        'ground_truth': 'NORMAL',
                        'raw_prediction': result['raw_prediction'],
                        'raw_confidence': result['raw_confidence'],
                        'pneumonia_confidence': result['pneumonia_confidence'],
                        'adjusted_prediction': result['adjusted_prediction'],
                        'threshold': threshold,
                        'raw_correct': result['raw_prediction'] == 'NORMAL',
                        'adjusted_correct': result['adjusted_prediction'] == 'NORMAL'
                    })
                await asyncio.sleep(0.1)

            print(f"Processed {len(normal_sample)} normal images...")

            # Test pneumonia images
            for img_path in pneumonia_sample:
                result = await predict_with_threshold(session, img_path, api_url, threshold)
                if result['success']:
                    results.append({
                        'image_path': str(img_path),
                        'ground_truth': 'PNEUMONIA',
                        'raw_prediction': result['raw_prediction'],
                        'raw_confidence': result['raw_confidence'],
                        'pneumonia_confidence': result['pneumonia_confidence'],
                        'adjusted_prediction': result['adjusted_prediction'],
                        'threshold': threshold,
                        'raw_correct': result['raw_prediction'] == 'PNEUMONIA',
                        'adjusted_correct': result['adjusted_prediction'] == 'PNEUMONIA'
                    })
                await asyncio.sleep(0.1)

            print(f"Processed {len(pneumonia_sample)} pneumonia images...")

    if not results:
        print("No results obtained!")
        return None

    # Analyze results
    df = pd.DataFrame(results)

    if test_preprocessing and 'preprocessing' in df.columns:
        # Preprocessing comparison analysis
        df_with_prep = df[df['preprocessing'] == True]
        df_without_prep = df[df['preprocessing'] == False]

        print()
        print("PREPROCESSING IMPACT ANALYSIS")
        print("=" * 40)

        # Analyze baseline (without preprocessing)
        baseline_adj_tp = len(df_without_prep[(df_without_prep['ground_truth'] == 'PNEUMONIA') & (df_without_prep['adjusted_prediction'] == 'PNEUMONIA')])
        baseline_adj_tn = len(df_without_prep[(df_without_prep['ground_truth'] == 'NORMAL') & (df_without_prep['adjusted_prediction'] == 'NORMAL')])
        baseline_adj_fp = len(df_without_prep[(df_without_prep['ground_truth'] == 'NORMAL') & (df_without_prep['adjusted_prediction'] == 'PNEUMONIA')])
        baseline_adj_fn = len(df_without_prep[(df_without_prep['ground_truth'] == 'PNEUMONIA') & (df_without_prep['adjusted_prediction'] == 'NORMAL')])

        baseline_recall = baseline_adj_tp / (baseline_adj_tp + baseline_adj_fn) if (baseline_adj_tp + baseline_adj_fn) > 0 else 0
        baseline_precision = baseline_adj_tp / (baseline_adj_tp + baseline_adj_fp) if (baseline_adj_tp + baseline_adj_fp) > 0 else 0
        baseline_accuracy = (baseline_adj_tp + baseline_adj_tn) / len(df_without_prep) if len(df_without_prep) > 0 else 0

        print("BASELINE (no preprocessing):")
        print(f"   TP: {baseline_adj_tp}, TN: {baseline_adj_tn}, FP: {baseline_adj_fp}, FN: {baseline_adj_fn}")
        print(f"   Accuracy: {baseline_accuracy*100:.1f}%, Recall: {baseline_recall*100:.1f}%, Precision: {baseline_precision*100:.1f}%")
        print(f"   Missed {baseline_adj_fn} pneumonia cases")

        # Analyze with preprocessing
        prep_adj_tp = len(df_with_prep[(df_with_prep['ground_truth'] == 'PNEUMONIA') & (df_with_prep['adjusted_prediction'] == 'PNEUMONIA')])
        prep_adj_tn = len(df_with_prep[(df_with_prep['ground_truth'] == 'NORMAL') & (df_with_prep['adjusted_prediction'] == 'NORMAL')])
        prep_adj_fp = len(df_with_prep[(df_with_prep['ground_truth'] == 'NORMAL') & (df_with_prep['adjusted_prediction'] == 'PNEUMONIA')])
        prep_adj_fn = len(df_with_prep[(df_with_prep['ground_truth'] == 'PNEUMONIA') & (df_with_prep['adjusted_prediction'] == 'NORMAL')])

        prep_recall = prep_adj_tp / (prep_adj_tp + prep_adj_fn) if (prep_adj_tp + prep_adj_fn) > 0 else 0
        prep_precision = prep_adj_tp / (prep_adj_tp + prep_adj_fp) if (prep_adj_tp + prep_adj_fp) > 0 else 0
        prep_accuracy = (prep_adj_tp + prep_adj_tn) / len(df_with_prep) if len(df_with_prep) > 0 else 0

        print()
        print("WITH INTELLIGENT PREPROCESSING:")
        print(f"   TP: {prep_adj_tp}, TN: {prep_adj_tn}, FP: {prep_adj_fp}, FN: {prep_adj_fn}")
        print(f"   Accuracy: {prep_accuracy*100:.1f}%, Recall: {prep_recall*100:.1f}%, Precision: {prep_precision*100:.1f}%")
        print(f"   Only missed {prep_adj_fn} pneumonia cases")

        print()
        print("PREPROCESSING IMPACT:")
        print(f"   Recall improvement: {(prep_recall - baseline_recall)*100:+.1f}% ({baseline_adj_fn - prep_adj_fn:+d} fewer missed cases)")
        print(f"   Precision change: {(prep_precision - baseline_precision)*100:+.1f}%")
        print(f"   Accuracy change: {(prep_accuracy - baseline_accuracy)*100:+.1f}%")
        print(f"   False alarms change: {prep_adj_fp - baseline_adj_fp:+d}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"preprocessing_comparison_{threshold}_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

        return {
            'threshold': threshold,
            'baseline_performance': {'recall': baseline_recall, 'precision': baseline_precision, 'accuracy': baseline_accuracy, 'fn': baseline_adj_fn, 'fp': baseline_adj_fp},
            'preprocessing_performance': {'recall': prep_recall, 'precision': prep_precision, 'accuracy': prep_accuracy, 'fn': prep_adj_fn, 'fp': prep_adj_fp},
            'improvement': {'recall_gain': prep_recall - baseline_recall, 'missed_reduction': baseline_adj_fn - prep_adj_fn}
        }

    else:
        # Standard threshold adjustment analysis
        raw_tp = len(df[(df['ground_truth'] == 'PNEUMONIA') & (df['raw_prediction'] == 'PNEUMONIA')])
        raw_tn = len(df[(df['ground_truth'] == 'NORMAL') & (df['raw_prediction'] == 'NORMAL')])
        raw_fp = len(df[(df['ground_truth'] == 'NORMAL') & (df['raw_prediction'] == 'PNEUMONIA')])
        raw_fn = len(df[(df['ground_truth'] == 'PNEUMONIA') & (df['raw_prediction'] == 'NORMAL')])

        adj_tp = len(df[(df['ground_truth'] == 'PNEUMONIA') & (df['adjusted_prediction'] == 'PNEUMONIA')])
        adj_tn = len(df[(df['ground_truth'] == 'NORMAL') & (df['adjusted_prediction'] == 'NORMAL')])
        adj_fp = len(df[(df['ground_truth'] == 'NORMAL') & (df['adjusted_prediction'] == 'PNEUMONIA')])
        adj_fn = len(df[(df['ground_truth'] == 'PNEUMONIA') & (df['adjusted_prediction'] == 'NORMAL')])

        print()
        print("COMPARISON RESULTS")
        print("=" * 30)

        print("ORIGINAL MODEL (default threshold):")
        print(f"   TP: {raw_tp}, TN: {raw_tn}, FP: {raw_fp}, FN: {raw_fn}")
        raw_recall = raw_tp / (raw_tp + raw_fn) if (raw_tp + raw_fn) > 0 else 0
        raw_precision = raw_tp / (raw_tp + raw_fp) if (raw_tp + raw_fp) > 0 else 0
        raw_accuracy = (raw_tp + raw_tn) / len(df)
        print(f"   Accuracy: {raw_accuracy*100:.1f}%, Recall: {raw_recall*100:.1f}%, Precision: {raw_precision*100:.1f}%")
        print(f"   Missed {raw_fn} pneumonia cases")

        print()
        print(f"THRESHOLD ADJUSTED (threshold={threshold}):")
        print(f"   TP: {adj_tp}, TN: {adj_tn}, FP: {adj_fp}, FN: {adj_fn}")
        adj_recall = adj_tp / (adj_tp + adj_fn) if (adj_tp + adj_fn) > 0 else 0
        adj_precision = adj_tp / (adj_tp + adj_fp) if (adj_tp + adj_fp) > 0 else 0
        adj_accuracy = (adj_tp + adj_tn) / len(df)
        print(f"   Accuracy: {adj_accuracy*100:.1f}%, Recall: {adj_recall*100:.1f}%, Precision: {adj_precision*100:.1f}%")
        print(f"   Only missed {adj_fn} pneumonia cases")

        print()
        print("IMPROVEMENT:")
        print(f"   Recall improvement: {(adj_recall - raw_recall)*100:+.1f}% ({raw_fn - adj_fn:+d} fewer missed cases)")
        print(f"   Precision change: {(adj_precision - raw_precision)*100:+.1f}%")
        print(f"   False alarms: {adj_fp - raw_fp:+d}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"threshold_test_{threshold}_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

        return {
            'threshold': threshold,
            'raw_performance': {'recall': raw_recall, 'precision': raw_precision, 'accuracy': raw_accuracy, 'fn': raw_fn, 'fp': raw_fp},
            'adjusted_performance': {'recall': adj_recall, 'precision': adj_precision, 'accuracy': adj_accuracy, 'fn': adj_fn, 'fp': adj_fp},
            'improvement': {'recall_gain': adj_recall - raw_recall, 'missed_reduction': raw_fn - adj_fn}
        }

async def main():
    dataset_path = "/Users/Adam/Downloads/College/Co op/Tech4Life-25Summer/chest_xray/train/"
    api_url = "http://localhost:8000"

    # Test preprocessing impact with optimal threshold
    print(f"\n{'='*70}")
    print("COMPREHENSIVE PREPROCESSING EVALUATION")
    print(f"{'='*70}")

    result = await test_threshold_adjustment(
        dataset_path,
        api_url,
        threshold=0.1,
        samples_per_class=50,  # Smaller sample for faster testing
        test_preprocessing=True
    )

    if result:
        print(f"\nPREPROCESSING IMPACT SUMMARY:")
        print(f"   Baseline (no preprocessing): {result['baseline_performance']['recall']*100:.1f}% recall, {result['baseline_performance']['fn']} missed")
        print(f"   With preprocessing: {result['preprocessing_performance']['recall']*100:.1f}% recall, {result['preprocessing_performance']['fn']} missed")
        print(f"   Improvement: {(result['preprocessing_performance']['recall'] - result['baseline_performance']['recall'])*100:+.1f}% recall gain")

    await asyncio.sleep(2)

    # Optional: Test different thresholds without preprocessing comparison
    print(f"\n{'='*60}")
    print("THRESHOLD COMPARISON (with preprocessing)")
    print(f"{'='*60}")

    thresholds_to_test = [0.05, 0.1, 0.15, 0.2]

    for threshold in thresholds_to_test:
        print(f"\nTesting threshold: {threshold}")
        result = await test_threshold_adjustment(
            dataset_path,
            api_url,
            threshold,
            samples_per_class=30,  # Smaller sample for comparison
            test_preprocessing=False
        )
        await asyncio.sleep(1)  # Brief pause between tests

if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    asyncio.run(main())