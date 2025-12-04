#!/usr/bin/env python3
"""
Quick Preprocessing Test - 50 images to verify the system works
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
import seaborn as sns

plt.style.use('seaborn-v0_8')

async def predict_with_preprocessing_control(session, image_path, api_url, use_preprocessing=True):
    """Make prediction with preprocessing control."""
    try:
        data = aiohttp.FormData()
        data.add_field('file',
                      open(image_path, 'rb'),
                      filename=image_path.name,
                      content_type='image/jpeg')

        if not use_preprocessing:
            data.add_field('disable_preprocessing', 'true')

        async with session.post(
            f"{api_url}/predict",
            data=data,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    'success': True,
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities'],
                    'preprocessing_used': use_preprocessing
                }
            else:
                return {'success': False}

    except Exception as e:
        print(f"Error with {image_path.name}: {e}")
        return {'success': False}

async def quick_test():
    """Quick test with 50 images."""
    dataset_path = Path("/Users/Adam/Downloads/College/Co op/Tech4Life-25Summer/chest_xray/train/")
    api_url = "http://localhost:8000"

    print("QUICK PREPROCESSING TEST - 50 IMAGES")
    print("=" * 40)

    # Sample 25 images of each type
    normal_images = list((dataset_path / "NORMAL").glob("*.jpeg"))
    pneumonia_images = list((dataset_path / "PNEUMONIA").glob("*.jpeg"))

    normal_sample = random.sample(normal_images, min(25, len(normal_images)))
    pneumonia_sample = random.sample(pneumonia_images, min(25, len(pneumonia_images)))

    all_images = normal_sample + pneumonia_sample
    print(f"Testing {len(all_images)} images...")

    results = []

    async with aiohttp.ClientSession() as session:
        # Test without preprocessing
        print("\nTesting WITHOUT preprocessing...")
        for img_path in all_images:
            ground_truth = 'NORMAL' if img_path.parent.name == 'NORMAL' else 'PNEUMONIA'
            result = await predict_with_preprocessing_control(session, img_path, api_url, use_preprocessing=False)
            if result['success']:
                results.append({
                    'image_path': str(img_path),
                    'ground_truth': ground_truth,
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'preprocessing_used': False
                })
            await asyncio.sleep(0.1)

        # Test with preprocessing
        print("Testing WITH preprocessing...")
        for img_path in all_images:
            ground_truth = 'NORMAL' if img_path.parent.name == 'NORMAL' else 'PNEUMONIA'
            result = await predict_with_preprocessing_control(session, img_path, api_url, use_preprocessing=True)
            if result['success']:
                results.append({
                    'image_path': str(img_path),
                    'ground_truth': ground_truth,
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'preprocessing_used': True
                })
            await asyncio.sleep(0.1)

    # Analyze results
    df = pd.DataFrame(results)

    baseline_df = df[df['preprocessing_used'] == False]
    preprocessing_df = df[df['preprocessing_used'] == True]

    def calculate_metrics(data_df):
        tp = len(data_df[(data_df['ground_truth'] == 'PNEUMONIA') & (data_df['prediction'] == 'PNEUMONIA')])
        tn = len(data_df[(data_df['ground_truth'] == 'NORMAL') & (data_df['prediction'] == 'NORMAL')])
        fp = len(data_df[(data_df['ground_truth'] == 'NORMAL') & (data_df['prediction'] == 'PNEUMONIA')])
        fn = len(data_df[(data_df['ground_truth'] == 'PNEUMONIA') & (data_df['prediction'] == 'NORMAL')])

        accuracy = (tp + tn) / len(data_df) if len(data_df) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'false_positives': fp,
            'false_negatives': fn,
            'avg_confidence': data_df['confidence'].mean()
        }

    baseline_metrics = calculate_metrics(baseline_df)
    preprocessing_metrics = calculate_metrics(preprocessing_df)

    print("\n" + "="*50)
    print("QUICK TEST RESULTS")
    print("="*50)

    print(f"\nBASELINE (No Preprocessing):")
    print(f"  Accuracy: {baseline_metrics['accuracy']:.3f}")
    print(f"  Sensitivity: {baseline_metrics['sensitivity']:.3f}")
    print(f"  Specificity: {baseline_metrics['specificity']:.3f}")
    print(f"  False Positives: {baseline_metrics['false_positives']}")
    print(f"  False Negatives: {baseline_metrics['false_negatives']}")
    print(f"  Avg Confidence: {baseline_metrics['avg_confidence']:.3f}")

    print(f"\nWITH PREPROCESSING:")
    print(f"  Accuracy: {preprocessing_metrics['accuracy']:.3f}")
    print(f"  Sensitivity: {preprocessing_metrics['sensitivity']:.3f}")
    print(f"  Specificity: {preprocessing_metrics['specificity']:.3f}")
    print(f"  False Positives: {preprocessing_metrics['false_positives']}")
    print(f"  False Negatives: {preprocessing_metrics['false_negatives']}")
    print(f"  Avg Confidence: {preprocessing_metrics['avg_confidence']:.3f}")

    print(f"\nIMPROVEMENTS:")
    print(f"  Accuracy: {(preprocessing_metrics['accuracy'] - baseline_metrics['accuracy'])*100:+.1f}%")
    print(f"  False Positive Reduction: {baseline_metrics['false_positives'] - preprocessing_metrics['false_positives']:+d}")
    print(f"  False Negative Reduction: {baseline_metrics['false_negatives'] - preprocessing_metrics['false_negatives']:+d}")
    print(f"  Confidence Improvement: {preprocessing_metrics['avg_confidence'] - baseline_metrics['avg_confidence']:+.3f}")

    # Quick visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy comparison
    methods = ['Baseline', 'Preprocessing']
    accuracies = [baseline_metrics['accuracy'], preprocessing_metrics['accuracy']]
    ax1.bar(methods, accuracies, color=['red', 'green'], alpha=0.7)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylim(0, 1)

    # Error comparison
    error_types = ['False\nPositives', 'False\nNegatives']
    baseline_errors = [baseline_metrics['false_positives'], baseline_metrics['false_negatives']]
    preprocessing_errors = [preprocessing_metrics['false_positives'], preprocessing_metrics['false_negatives']]

    x = np.arange(len(error_types))
    width = 0.35

    ax2.bar(x - width/2, baseline_errors, width, label='Baseline', color='red', alpha=0.7)
    ax2.bar(x + width/2, preprocessing_errors, width, label='Preprocessing', color='green', alpha=0.7)
    ax2.set_ylabel('Error Count')
    ax2.set_title('Error Reduction')
    ax2.set_xticks(x)
    ax2.set_xticklabels(error_types)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('validation/reports/quick_test_results.png', dpi=150, bbox_inches='tight')
    print(f"\nQuick test plot saved: validation/reports/quick_test_results.png")
    plt.show()

    return len(results) > 0

if __name__ == "__main__":
    random.seed(42)
    print("Starting quick preprocessing validation test...")
    result = asyncio.run(quick_test())
    if result:
        print("\nQuick test completed successfully!")
    else:
        print("\nQuick test failed!")