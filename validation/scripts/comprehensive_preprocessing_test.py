#!/usr/bin/env python3
"""
Comprehensive Preprocessing Effectiveness Test

Tests 500 images to validate if intelligent preprocessing and CLAHE actually improve:
1. False positive reduction
2. False negative reduction
3. Overall accuracy and confidence scores

Generates detailed visualizations and statistical analysis.
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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

async def predict_with_preprocessing_control(session, image_path, api_url, use_preprocessing=True):
    """Make prediction with preprocessing control."""
    try:
        data = aiohttp.FormData()
        data.add_field('file',
                      open(image_path, 'rb'),
                      filename=image_path.name,
                      content_type='image/jpeg')

        # Control preprocessing
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
                    'processing_time': result['processing_time_ms'],
                    'image_quality': result.get('image_quality'),
                    'user_feedback': result.get('user_feedback'),
                    'preprocessing_used': use_preprocessing
                }
            else:
                return {'success': False}

    except Exception as e:
        print(f"Error with {image_path.name}: {e}")
        return {'success': False}

async def comprehensive_preprocessing_test(dataset_path, api_url, samples_per_class=250):
    """Run comprehensive test comparing preprocessing vs no preprocessing."""

    print("COMPREHENSIVE PREPROCESSING EFFECTIVENESS TEST")
    print("=" * 55)
    print(f"Testing {samples_per_class * 2} images total")
    print("Comparing: Baseline vs Intelligent Preprocessing + CLAHE")
    print()

    dataset_path = Path(dataset_path)

    # Sample images
    normal_images = list((dataset_path / "NORMAL").glob("*.jpeg"))
    pneumonia_images = list((dataset_path / "PNEUMONIA").glob("*.jpeg"))

    normal_sample = random.sample(normal_images, min(samples_per_class, len(normal_images)))
    pneumonia_sample = random.sample(pneumonia_images, min(samples_per_class, len(pneumonia_images)))

    print(f"Selected {len(normal_sample)} normal + {len(pneumonia_sample)} pneumonia images")

    all_results = []

    async with aiohttp.ClientSession() as session:
        print("\nPhase 1: Testing WITHOUT preprocessing (baseline)...")

        # Test without preprocessing
        for i, img_path in enumerate(normal_sample + pneumonia_sample):
            ground_truth = 'NORMAL' if img_path.parent.name == 'NORMAL' else 'PNEUMONIA'

            result = await predict_with_preprocessing_control(session, img_path, api_url, use_preprocessing=False)
            if result['success']:
                all_results.append({
                    'image_path': str(img_path),
                    'ground_truth': ground_truth,
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'normal_prob': result['probabilities']['NORMAL'],
                    'pneumonia_prob': result['probabilities']['PNEUMONIA'],
                    'processing_time': result['processing_time'],
                    'preprocessing_used': False,
                    'test_phase': 'baseline'
                })

            if i % 50 == 0:
                print(f"  Processed {i+1}/{len(normal_sample + pneumonia_sample)} images...")
            await asyncio.sleep(0.05)

        print(f"\nPhase 2: Testing WITH intelligent preprocessing + CLAHE...")

        # Test with preprocessing
        for i, img_path in enumerate(normal_sample + pneumonia_sample):
            ground_truth = 'NORMAL' if img_path.parent.name == 'NORMAL' else 'PNEUMONIA'

            result = await predict_with_preprocessing_control(session, img_path, api_url, use_preprocessing=True)
            if result['success']:
                all_results.append({
                    'image_path': str(img_path),
                    'ground_truth': ground_truth,
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'normal_prob': result['probabilities']['NORMAL'],
                    'pneumonia_prob': result['probabilities']['PNEUMONIA'],
                    'processing_time': result['processing_time'],
                    'preprocessing_used': True,
                    'test_phase': 'intelligent_preprocessing',
                    'image_quality': result.get('image_quality'),
                    'user_feedback': result.get('user_feedback')
                })

            if i % 50 == 0:
                print(f"  Processed {i+1}/{len(normal_sample + pneumonia_sample)} images...")
            await asyncio.sleep(0.05)

    return all_results

def analyze_results_and_generate_plots(results):
    """Analyze results and generate comprehensive visualizations."""

    df = pd.DataFrame(results)
    if df.empty:
        print("No results to analyze!")
        return None

    print(f"\nAnalyzing {len(df)} total predictions...")

    # Split baseline vs preprocessing results
    baseline_df = df[df['test_phase'] == 'baseline'].copy()
    preprocessing_df = df[df['test_phase'] == 'intelligent_preprocessing'].copy()

    # Calculate metrics for both approaches
    def calculate_metrics(data_df, name):
        tp = len(data_df[(data_df['ground_truth'] == 'PNEUMONIA') & (data_df['prediction'] == 'PNEUMONIA')])
        tn = len(data_df[(data_df['ground_truth'] == 'NORMAL') & (data_df['prediction'] == 'NORMAL')])
        fp = len(data_df[(data_df['ground_truth'] == 'NORMAL') & (data_df['prediction'] == 'PNEUMONIA')])
        fn = len(data_df[(data_df['ground_truth'] == 'PNEUMONIA') & (data_df['prediction'] == 'NORMAL')])

        accuracy = (tp + tn) / len(data_df) if len(data_df) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        # Calculate confidence scores
        avg_confidence = data_df['confidence'].mean()
        pneumonia_confidence = data_df[data_df['ground_truth'] == 'PNEUMONIA']['confidence'].mean()
        normal_confidence = data_df[data_df['ground_truth'] == 'NORMAL']['confidence'].mean()

        return {
            'name': name,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'true_negatives': tn,
            'avg_confidence': avg_confidence,
            'pneumonia_confidence': pneumonia_confidence,
            'normal_confidence': normal_confidence,
            'total_samples': len(data_df)
        }

    baseline_metrics = calculate_metrics(baseline_df, 'Baseline (No Preprocessing)')
    preprocessing_metrics = calculate_metrics(preprocessing_df, 'Intelligent Preprocessing + CLAHE')

    # Print detailed comparison
    print("\n" + "="*70)
    print("DETAILED PERFORMANCE COMPARISON")
    print("="*70)

    metrics_comparison = pd.DataFrame([baseline_metrics, preprocessing_metrics])

    print(f"\n{'Metric':<25} {'Baseline':<15} {'Preprocessing':<15} {'Improvement':<15}")
    print("-" * 70)

    for metric in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']:
        baseline_val = baseline_metrics[metric]
        preprocessing_val = preprocessing_metrics[metric]
        improvement = preprocessing_val - baseline_val
        improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0

        print(f"{metric.title():<25} {baseline_val:<15.3f} {preprocessing_val:<15.3f} {improvement:+.3f} ({improvement_pct:+.1f}%)")

    print(f"\n{'Error Type':<25} {'Baseline':<15} {'Preprocessing':<15} {'Reduction':<15}")
    print("-" * 70)
    print(f"{'False Positives':<25} {baseline_metrics['false_positives']:<15} {preprocessing_metrics['false_positives']:<15} {baseline_metrics['false_positives'] - preprocessing_metrics['false_positives']:+d}")
    print(f"{'False Negatives':<25} {baseline_metrics['false_negatives']:<15} {preprocessing_metrics['false_negatives']:<15} {baseline_metrics['false_negatives'] - preprocessing_metrics['false_negatives']:+d}")

    # Statistical significance test
    baseline_correct = baseline_df['prediction'] == baseline_df['ground_truth']
    preprocessing_correct = preprocessing_df['prediction'] == preprocessing_df['ground_truth']

    # McNemar's test for paired accuracy comparison
    from scipy.stats import chi2_contingency

    # Create contingency table
    both_correct = (baseline_correct & preprocessing_correct).sum()
    baseline_only = (baseline_correct & ~preprocessing_correct).sum()
    preprocessing_only = (~baseline_correct & preprocessing_correct).sum()
    both_wrong = (~baseline_correct & ~preprocessing_correct).sum()

    print(f"\nSTATISTICAL SIGNIFICANCE:")
    print(f"Both correct: {both_correct}")
    print(f"Only baseline correct: {baseline_only}")
    print(f"Only preprocessing correct: {preprocessing_only}")
    print(f"Both wrong: {both_wrong}")

    if baseline_only + preprocessing_only > 0:
        mcnemar_stat = (abs(baseline_only - preprocessing_only) - 1)**2 / (baseline_only + preprocessing_only)
        p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        print(f"McNemar's test p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("Improvement is statistically significant (p < 0.05)")
        else:
            print("Improvement is not statistically significant (p >= 0.05)")

    # Generate comprehensive visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create subplot figure
    fig = plt.figure(figsize=(20, 15))

    # 1. Performance Metrics Comparison
    ax1 = plt.subplot(2, 4, 1)
    metrics_names = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score']
    baseline_values = [baseline_metrics[m.lower().replace('-', '_')] for m in metrics_names]
    preprocessing_values = [preprocessing_metrics[m.lower().replace('-', '_')] for m in metrics_names]

    x = np.arange(len(metrics_names))
    width = 0.35

    ax1.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
    ax1.bar(x + width/2, preprocessing_values, width, label='Preprocessing', alpha=0.8)
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Error Reduction Analysis
    ax2 = plt.subplot(2, 4, 2)
    error_types = ['False Positives', 'False Negatives']
    baseline_errors = [baseline_metrics['false_positives'], baseline_metrics['false_negatives']]
    preprocessing_errors = [preprocessing_metrics['false_positives'], preprocessing_metrics['false_negatives']]

    x = np.arange(len(error_types))
    ax2.bar(x - width/2, baseline_errors, width, label='Baseline', alpha=0.8, color='red')
    ax2.bar(x + width/2, preprocessing_errors, width, label='Preprocessing', alpha=0.8, color='green')
    ax2.set_xlabel('Error Type')
    ax2.set_ylabel('Count')
    ax2.set_title('Error Reduction Analysis')
    ax2.set_xticks(x)
    ax2.set_xticklabels(error_types)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Confidence Score Distribution
    ax3 = plt.subplot(2, 4, 3)
    ax3.hist(baseline_df['confidence'], bins=30, alpha=0.7, label='Baseline', density=True)
    ax3.hist(preprocessing_df['confidence'], bins=30, alpha=0.7, label='Preprocessing', density=True)
    ax3.set_xlabel('Confidence Score')
    ax3.set_ylabel('Density')
    ax3.set_title('Confidence Score Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Confidence by Ground Truth
    ax4 = plt.subplot(2, 4, 4)
    confidence_data = []

    for truth in ['NORMAL', 'PNEUMONIA']:
        baseline_conf = baseline_df[baseline_df['ground_truth'] == truth]['confidence'].values
        preprocessing_conf = preprocessing_df[preprocessing_df['ground_truth'] == truth]['confidence'].values

        confidence_data.append(baseline_conf)
        confidence_data.append(preprocessing_conf)

    ax4.boxplot(confidence_data,
                labels=['Baseline\nNormal', 'Preprocessing\nNormal',
                       'Baseline\nPneumonia', 'Preprocessing\nPneumonia'])
    ax4.set_ylabel('Confidence Score')
    ax4.set_title('Confidence by Ground Truth')
    ax4.grid(True, alpha=0.3)

    # 5. Confusion Matrix - Baseline
    ax5 = plt.subplot(2, 4, 5)
    baseline_cm = np.array([[baseline_metrics['true_negatives'], baseline_metrics['false_positives']],
                           [baseline_metrics['false_negatives'], baseline_metrics['true_positives']]])

    sns.heatmap(baseline_cm, annot=True, fmt='d', ax=ax5,
                xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    ax5.set_title('Confusion Matrix - Baseline')
    ax5.set_ylabel('True Label')
    ax5.set_xlabel('Predicted Label')

    # 6. Confusion Matrix - Preprocessing
    ax6 = plt.subplot(2, 4, 6)
    preprocessing_cm = np.array([[preprocessing_metrics['true_negatives'], preprocessing_metrics['false_positives']],
                                [preprocessing_metrics['false_negatives'], preprocessing_metrics['true_positives']]])

    sns.heatmap(preprocessing_cm, annot=True, fmt='d', ax=ax6,
                xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    ax6.set_title('Confusion Matrix - Preprocessing')
    ax6.set_ylabel('True Label')
    ax6.set_xlabel('Predicted Label')

    # 7. Processing Time Comparison
    ax7 = plt.subplot(2, 4, 7)
    ax7.boxplot([baseline_df['processing_time'].values, preprocessing_df['processing_time'].values],
               labels=['Baseline', 'Preprocessing'])
    ax7.set_ylabel('Processing Time (ms)')
    ax7.set_title('Processing Time Comparison')
    ax7.grid(True, alpha=0.3)

    # 8. Improvement Summary
    ax8 = plt.subplot(2, 4, 8)
    improvements = {
        'Accuracy': (preprocessing_metrics['accuracy'] - baseline_metrics['accuracy']) * 100,
        'Sensitivity': (preprocessing_metrics['sensitivity'] - baseline_metrics['sensitivity']) * 100,
        'Specificity': (preprocessing_metrics['specificity'] - baseline_metrics['specificity']) * 100,
        'FP Reduction': baseline_metrics['false_positives'] - preprocessing_metrics['false_positives'],
        'FN Reduction': baseline_metrics['false_negatives'] - preprocessing_metrics['false_negatives']
    }

    colors = ['green' if v > 0 else 'red' for v in improvements.values()]
    bars = ax8.bar(range(len(improvements)), list(improvements.values()), color=colors, alpha=0.7)
    ax8.set_xlabel('Improvement Type')
    ax8.set_ylabel('Improvement Value')
    ax8.set_title('Preprocessing Improvements')
    ax8.set_xticks(range(len(improvements)))
    ax8.set_xticklabels(list(improvements.keys()), rotation=45)
    ax8.grid(True, alpha=0.3)
    ax8.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # Add value labels on bars
    for bar, value in zip(bars, improvements.values()):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1),
                f'{value:.1f}', ha='center', va='bottom' if height > 0 else 'top')

    plt.tight_layout()

    # Save the plot
    plot_filename = f"validation/reports/preprocessing_effectiveness_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nDetailed analysis plot saved: {plot_filename}")

    # Save detailed results
    results_filename = f"validation/reports/preprocessing_test_results_{timestamp}.csv"
    df.to_csv(results_filename, index=False)

    # Save metrics summary
    summary_filename = f"validation/reports/preprocessing_metrics_summary_{timestamp}.json"
    summary_data = {
        'test_timestamp': timestamp,
        'total_images_tested': len(df) // 2,  # Divided by 2 because we test each image twice
        'baseline_metrics': baseline_metrics,
        'preprocessing_metrics': preprocessing_metrics,
        'improvements': {
            'accuracy_improvement_pct': (preprocessing_metrics['accuracy'] - baseline_metrics['accuracy']) * 100,
            'sensitivity_improvement_pct': (preprocessing_metrics['sensitivity'] - baseline_metrics['sensitivity']) * 100,
            'specificity_improvement_pct': (preprocessing_metrics['specificity'] - baseline_metrics['specificity']) * 100,
            'false_positive_reduction': baseline_metrics['false_positives'] - preprocessing_metrics['false_positives'],
            'false_negative_reduction': baseline_metrics['false_negatives'] - preprocessing_metrics['false_negatives'],
            'confidence_improvement': preprocessing_metrics['avg_confidence'] - baseline_metrics['avg_confidence']
        }
    }

    with open(summary_filename, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"Results saved: {results_filename}")
    print(f"Summary saved: {summary_filename}")

    plt.show()

    return summary_data

async def main():
    dataset_path = "/Users/Adam/Downloads/College/Co op/Tech4Life-25Summer/chest_xray/train/"
    api_url = "http://localhost:8000"

    print("Starting comprehensive preprocessing effectiveness test...")
    print("This will test 500 images (250 normal + 250 pneumonia)")
    print("Each image tested twice: with and without preprocessing")
    print("=" * 60)

    # Run the comprehensive test
    results = await comprehensive_preprocessing_test(dataset_path, api_url, samples_per_class=250)

    if results:
        print(f"\nCollected {len(results)} total predictions")
        print("Generating analysis and visualizations...")

        # Analyze and generate plots
        summary = analyze_results_and_generate_plots(results)

        if summary:
            print("\n" + "="*60)
            print("FINAL SUMMARY: DOES PREPROCESSING HELP?")
            print("="*60)

            improvements = summary['improvements']

            print(f"Accuracy improvement: {improvements['accuracy_improvement_pct']:+.2f}%")
            print(f"Sensitivity improvement: {improvements['sensitivity_improvement_pct']:+.2f}%")
            print(f"Specificity improvement: {improvements['specificity_improvement_pct']:+.2f}%")
            print(f"False positives reduced by: {improvements['false_positive_reduction']}")
            print(f"False negatives reduced by: {improvements['false_negative_reduction']}")
            print(f"Average confidence improved by: {improvements['confidence_improvement']:+.3f}")

            # Final verdict
            total_improvement = (improvements['accuracy_improvement_pct'] +
                               improvements['sensitivity_improvement_pct'] +
                               improvements['specificity_improvement_pct']) / 3

            print(f"\nOVERALL VERDICT:")
            if total_improvement > 1.0:
                print("PREPROCESSING + CLAHE SIGNIFICANTLY IMPROVES PERFORMANCE")
            elif total_improvement > 0:
                print("PREPROCESSING + CLAHE PROVIDES MODEST IMPROVEMENTS")
            else:
                print("PREPROCESSING + CLAHE DOES NOT IMPROVE PERFORMANCE")

            print(f"Average metric improvement: {total_improvement:+.2f}%")

    else:
        print("Test failed - no results collected")

if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    asyncio.run(main())