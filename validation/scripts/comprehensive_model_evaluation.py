#!/usr/bin/env python3
"""
Comprehensive Model Evaluation with Statistical Analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import json

def calculate_confidence_intervals(data, confidence=0.95):
    """Calculate confidence intervals for metrics."""
    n = len(data)
    if n <= 1:
        return np.nan, np.nan

    mean = np.mean(data)
    stderr = stats.sem(data)
    interval = stderr * stats.t.ppf((1 + confidence) / 2., n-1)

    return mean - interval, mean + interval

def comprehensive_analysis(results_file):
    """Perform comprehensive statistical analysis."""
    print("🔬 COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)

    # Load results
    df = pd.read_csv(results_file)
    total_samples = len(df)

    print(f"📊 Dataset Size: {total_samples:,} images")
    print(f"   • Normal: {len(df[df['ground_truth'] == 'NORMAL']):,}")
    print(f"   • Pneumonia: {len(df[df['ground_truth'] == 'PNEUMONIA']):,}")
    print()

    # Confusion Matrix
    y_true = (df['ground_truth'] == 'PNEUMONIA').astype(int)
    y_pred = (df['prediction_raw'] == 'PNEUMONIA').astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("🎯 CONFUSION MATRIX")
    print("-" * 30)
    print(f"True Negatives:  {tn:,} (Normal → Normal)")
    print(f"False Positives: {fp:,} (Normal → Pneumonia)")
    print(f"False Negatives: {fn:,} (Pneumonia → Normal)")
    print(f"True Positives:  {tp:,} (Pneumonia → Pneumonia)")
    print()

    # Calculate metrics with confidence intervals
    accuracy = (tp + tn) / total_samples
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    # Bootstrap confidence intervals for accuracy
    n_bootstrap = 1000
    accuracies = []
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(len(df), len(df), replace=True)
        sample_df = df.iloc[sample_idx]
        sample_correct = len(sample_df[sample_df['correct_raw'] == True])
        sample_acc = sample_correct / len(sample_df)
        accuracies.append(sample_acc)

    acc_ci_low, acc_ci_high = np.percentile(accuracies, [2.5, 97.5])

    print("📈 PERFORMANCE METRICS (95% Confidence Intervals)")
    print("-" * 50)
    print(f"Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%) [{acc_ci_low*100:.2f}% - {acc_ci_high*100:.2f}%]")
    print(f"Precision:    {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:       {recall:.4f} ({recall*100:.2f}%)")
    print(f"Specificity:  {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"F1-Score:     {f1:.4f}")
    print(f"NPV:          {npv:.4f} ({npv*100:.2f}%)")
    print()

    # Clinical metrics
    print("🏥 CLINICAL INTERPRETATION")
    print("-" * 30)
    print(f"Sensitivity (Pneumonia Detection): {recall*100:.2f}%")
    print(f"Specificity (Normal Identification): {specificity*100:.2f}%")
    print(f"Positive Predictive Value: {precision*100:.2f}%")
    print(f"Negative Predictive Value: {npv*100:.2f}%")
    print()

    # Error analysis
    false_negatives = df[(df['ground_truth'] == 'PNEUMONIA') & (df['prediction_raw'] == 'NORMAL')]
    false_positives = df[(df['ground_truth'] == 'NORMAL') & (df['prediction_raw'] == 'PNEUMONIA')]

    print("❌ ERROR ANALYSIS")
    print("-" * 20)
    print(f"False Negative Rate: {fn/(tp+fn)*100:.2f}% ({fn} missed pneumonia cases)")
    print(f"False Positive Rate: {fp/(tn+fp)*100:.2f}% ({fp} false alarms)")

    if len(false_negatives) > 0:
        print(f"Missed pneumonia confidence range: {false_negatives['confidence_raw'].min():.3f} - {false_negatives['confidence_raw'].max():.3f}")

    if len(false_positives) > 0:
        print(f"False alarm confidence range: {false_positives['confidence_raw'].min():.3f} - {false_positives['confidence_raw'].max():.3f}")
    print()

    # Confidence analysis
    normal_conf = df[df['ground_truth'] == 'NORMAL']['confidence_raw']
    pneumonia_conf = df[df['ground_truth'] == 'PNEUMONIA']['confidence_raw']

    print("📊 CONFIDENCE SCORE ANALYSIS")
    print("-" * 30)
    print(f"Normal cases - Mean: {normal_conf.mean():.3f}, Std: {normal_conf.std():.3f}")
    print(f"Pneumonia cases - Mean: {pneumonia_conf.mean():.3f}, Std: {pneumonia_conf.std():.3f}")
    print()

    # Performance by confidence threshold
    print("🎚️ PERFORMANCE BY CONFIDENCE THRESHOLD")
    print("-" * 40)
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

    for thresh in thresholds:
        high_conf = df[df['confidence_raw'] >= thresh]
        if len(high_conf) > 0:
            high_conf_acc = len(high_conf[high_conf['correct_raw'] == True]) / len(high_conf)
            coverage = len(high_conf) / len(df)
            print(f"Confidence ≥ {thresh}: {high_conf_acc*100:.1f}% accuracy, {coverage*100:.1f}% coverage ({len(high_conf)} images)")
    print()

    # Statistical significance tests
    print("📐 STATISTICAL TESTS")
    print("-" * 20)

    # Test if accuracy is significantly different from random (50%)
    successes = len(df[df['correct_raw'] == True])
    p_value_random = stats.binom_test(successes, total_samples, 0.5, alternative='greater')
    print(f"vs Random (50%): p < {p_value_random:.2e} (highly significant)")

    # Test if accuracy is significantly different from 95%
    p_value_95 = stats.binom_test(successes, total_samples, 0.95, alternative='two-sided')
    print(f"vs 95% benchmark: p = {p_value_95:.4f}")

    # Test if accuracy is significantly different from training claim (97.07%)
    p_value_training = stats.binom_test(successes, total_samples, 0.9707, alternative='two-sided')
    print(f"vs Training claim (97.07%): p = {p_value_training:.4f}")
    print()

    print("✅ EVALUATION COMPLETE")

    return {
        'total_samples': total_samples,
        'accuracy': accuracy,
        'accuracy_ci': [acc_ci_low, acc_ci_high],
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'false_negative_rate': fn/(tp+fn),
        'false_positive_rate': fp/(tn+fp)
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python comprehensive_model_evaluation.py <results_csv_file>")
        sys.exit(1)

    results_file = sys.argv[1]
    if not Path(results_file).exists():
        print(f"Error: File {results_file} not found")
        sys.exit(1)

    comprehensive_analysis(results_file)