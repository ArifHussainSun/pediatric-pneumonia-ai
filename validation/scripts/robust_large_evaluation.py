#!/usr/bin/env python3
"""
Robust Large-Scale Model Evaluation
Handles API timeouts and connection issues gracefully
"""

import asyncio
import aiohttp
import random
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time

async def predict_with_retry(session, image_path, api_url, max_retries=3, delay=1):
    """Make prediction with retry logic."""
    for attempt in range(max_retries):
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
                    return {
                        'success': True,
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'processing_time': result.get('processing_time_ms', 0)
                    }
                else:
                    print(f"API error {response.status} for {image_path.name}")

        except Exception as e:
            print(f"Attempt {attempt+1} failed for {image_path.name}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff

    return {'success': False}

async def evaluate_batch(image_paths, labels, api_url, batch_size=10):
    """Evaluate a batch of images."""
    results = []

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            # Process batch
            tasks = [predict_with_retry(session, path, api_url) for path in batch_paths]
            batch_results = await asyncio.gather(*tasks)

            # Store results
            for path, label, result in zip(batch_paths, batch_labels, batch_results):
                if result['success']:
                    results.append({
                        'image_path': str(path),
                        'ground_truth': label,
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'processing_time_ms': result['processing_time'],
                        'correct': (result['prediction'] == label)
                    })
                else:
                    print(f"Failed to process {path.name}")

            print(f"Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)} images...")
            await asyncio.sleep(0.5)  # Small delay between batches

    return results

async def large_scale_evaluation(dataset_path, api_url, samples_per_class=500):
    """Run comprehensive evaluation."""
    print(f"🔬 Large-Scale Model Evaluation ({samples_per_class} per class)")
    print("=" * 60)

    dataset_path = Path(dataset_path)

    # Find image files
    normal_images = list((dataset_path / "NORMAL").glob("*.jpeg"))
    pneumonia_images = list((dataset_path / "PNEUMONIA").glob("*.jpeg"))

    print(f"Available images:")
    print(f"  Normal: {len(normal_images):,}")
    print(f"  Pneumonia: {len(pneumonia_images):,}")

    # Sample images
    if samples_per_class:
        normal_sample = random.sample(normal_images, min(samples_per_class, len(normal_images)))
        pneumonia_sample = random.sample(pneumonia_images, min(samples_per_class, len(pneumonia_images)))
    else:
        normal_sample = normal_images
        pneumonia_sample = pneumonia_images

    # Combine and shuffle
    all_images = normal_sample + pneumonia_sample
    all_labels = ['NORMAL'] * len(normal_sample) + ['PNEUMONIA'] * len(pneumonia_sample)

    # Shuffle to avoid bias
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    all_images, all_labels = zip(*combined)

    print(f"Testing {len(all_images):,} images total")
    print()

    # Run evaluation
    start_time = time.time()
    results = await evaluate_batch(all_images, all_labels, api_url, batch_size=5)
    end_time = time.time()

    if not results:
        print("❌ No results obtained!")
        return None

    # Save results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"large_eval_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    csv_file = output_dir / f"evaluation_results_{timestamp}.csv"
    df.to_csv(csv_file, index=False)

    # Calculate metrics
    total_samples = len(results)
    correct_predictions = len(df[df['correct'] == True])
    accuracy = correct_predictions / total_samples

    # Confusion matrix
    normal_df = df[df['ground_truth'] == 'NORMAL']
    pneumonia_df = df[df['ground_truth'] == 'PNEUMONIA']

    tn = len(normal_df[normal_df['prediction'] == 'NORMAL'])
    fp = len(normal_df[normal_df['prediction'] == 'PNEUMONIA'])
    fn = len(pneumonia_df[pneumonia_df['prediction'] == 'NORMAL'])
    tp = len(pneumonia_df[pneumonia_df['prediction'] == 'PNEUMONIA'])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate confidence intervals (bootstrap)
    n_bootstrap = 1000
    bootstrap_accuracies = []
    for _ in range(n_bootstrap):
        sample_df = df.sample(n=len(df), replace=True)
        sample_acc = len(sample_df[sample_df['correct'] == True]) / len(sample_df)
        bootstrap_accuracies.append(sample_acc)

    acc_ci_low, acc_ci_high = np.percentile(bootstrap_accuracies, [2.5, 97.5])

    # Results summary
    print("📊 LARGE-SCALE EVALUATION RESULTS")
    print("=" * 40)
    print(f"Total Images Processed: {total_samples:,}")
    print(f"Evaluation Time: {end_time - start_time:.1f} seconds")
    print()

    print("🎯 CONFUSION MATRIX")
    print(f"True Negatives:  {tn:,}")
    print(f"False Positives: {fp:,}")
    print(f"False Negatives: {fn:,}")
    print(f"True Positives:  {tp:,}")
    print()

    print("📈 PERFORMANCE METRICS")
    print(f"Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%) [{acc_ci_low*100:.2f}% - {acc_ci_high*100:.2f}%]")
    print(f"Precision:    {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:       {recall:.4f} ({recall*100:.2f}%)")
    print(f"Specificity:  {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"F1-Score:     {f1:.4f}")
    print()

    print("🏥 CLINICAL METRICS")
    print(f"Sensitivity: {recall*100:.2f}% (Pneumonia Detection Rate)")
    print(f"Specificity: {specificity*100:.2f}% (Normal Identification Rate)")
    print(f"PPV: {precision*100:.2f}% (Positive Predictive Value)")
    print(f"NPV: {tn/(tn+fn)*100:.2f}% (Negative Predictive Value)")
    print()

    # Confidence analysis
    normal_conf = df[df['ground_truth'] == 'NORMAL']['confidence']
    pneumonia_conf = df[df['ground_truth'] == 'PNEUMONIA']['confidence']

    print("📊 CONFIDENCE ANALYSIS")
    print(f"Normal - Mean: {normal_conf.mean():.3f}, Std: {normal_conf.std():.3f}")
    print(f"Pneumonia - Mean: {pneumonia_conf.mean():.3f}, Std: {pneumonia_conf.std():.3f}")
    print()

    # Save summary
    summary = {
        'total_samples': total_samples,
        'accuracy': accuracy,
        'accuracy_ci_95': [acc_ci_low, acc_ci_high],
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'evaluation_time_seconds': end_time - start_time
    }

    with open(output_dir / f"summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Results saved to {output_dir}/")
    print(f"📁 CSV: {csv_file}")

    return summary

if __name__ == "__main__":
    import sys

    dataset_path = "/Users/Adam/Downloads/College/Co op/Tech4Life-25Summer/chest_xray/train/"
    api_url = "http://localhost:8000"
    samples_per_class = 500  # Manageable size

    if len(sys.argv) > 1:
        samples_per_class = int(sys.argv[1])

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Run evaluation
    asyncio.run(large_scale_evaluation(dataset_path, api_url, samples_per_class))