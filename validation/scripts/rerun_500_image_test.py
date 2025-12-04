#!/usr/bin/env python3
"""
Re-run 500 Image Validation Test with Current Model

This script re-runs the exact same test methodology as analyze_false_negatives.py
but on 500 images to find current false negatives (if any).

Uses direct model inference (not API) to match the analysis script methodology.
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
import random
from datetime import datetime
from tqdm import tqdm
import json

from src.models.mobilenet import MobileNetFineTune
from src.data.datasets import get_medical_transforms


def load_model():
    """Load trained MobileNet model (same as API server)."""
    print("Loading MobileNet model...")
    # Use the same model as API server (Best_MobilenetV1.pth)
    model_path = project_root / "outputs" / "dgx_station_experiment" / "Best_MobilenetV1.pth"

    model = MobileNetFineTune(num_classes=2, freeze_layers=0)

    if model_path.exists():
        model.load_custom_weights(str(model_path))
        print(f"✓ Loaded model from {model_path}")
    else:
        print(f"ERROR: Model not found at {model_path}")
        return None

    model.eval()
    return model


def predict_image(model, image_path, transform):
    """
    Predict single image using the model.

    Returns:
        dict with prediction, confidence, probabilities
    """
    try:
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        # Prepare for model
        pil_image = Image.fromarray(image).convert('RGB')
        input_tensor = transform(pil_image).unsqueeze(0)

        # Get prediction
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            normal_prob = probs[0, 1].item()
            pneumonia_prob = probs[0, 0].item()

            # Class 0 = PNEUMONIA, Class 1 = NORMAL
            predicted_class = 1 if normal_prob > 0.5 else 0
            confidence = max(normal_prob, pneumonia_prob)

            return {
                'predicted_class': predicted_class,
                'prediction': 'NORMAL' if predicted_class == 1 else 'PNEUMONIA',
                'confidence': confidence,
                'normal_prob': normal_prob,
                'pneumonia_prob': pneumonia_prob
            }

    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return None


def main():
    """Main validation pipeline."""
    print("="*80)
    print("500 IMAGE VALIDATION TEST - RE-RUN WITH CURRENT MODEL")
    print("="*80)
    print("\nThis test uses the SAME methodology as the false negative analysis")
    print("to find current false negatives with the current model.\n")

    # Load model
    model = load_model()
    if model is None:
        print("ERROR: Could not load model. Exiting.")
        return

    # Get transform
    transform = get_medical_transforms(is_training=False)

    # Load dataset
    dataset_path = Path("/Users/Adam/Downloads/College/Co op/Tech4Life-25Summer/chest_xray/train")

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return

    print(f"Loading images from: {dataset_path}")

    # Sample images (same as original test)
    normal_images = list((dataset_path / "NORMAL").glob("*.jpeg"))
    pneumonia_images = list((dataset_path / "PNEUMONIA").glob("*.jpeg"))

    print(f"Found {len(normal_images)} NORMAL images")
    print(f"Found {len(pneumonia_images)} PNEUMONIA images")

    # Sample 250 from each class (same as October 18 test)
    random.seed(42)  # Use same seed for reproducibility
    normal_sample = random.sample(normal_images, min(250, len(normal_images)))
    pneumonia_sample = random.sample(pneumonia_images, min(250, len(pneumonia_images)))

    print(f"\nSampled {len(normal_sample)} NORMAL images")
    print(f"Sampled {len(pneumonia_sample)} PNEUMONIA images")
    print(f"Total: {len(normal_sample) + len(pneumonia_sample)} images")

    # Run predictions
    results = []

    print("\nRunning predictions...")
    for img_path in tqdm(normal_sample + pneumonia_sample, desc="Processing images"):
        ground_truth = 'NORMAL' if img_path.parent.name == 'NORMAL' else 'PNEUMONIA'

        prediction = predict_image(model, img_path, transform)

        if prediction:
            results.append({
                'image_path': str(img_path),
                'filename': img_path.name,
                'ground_truth': ground_truth,
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'normal_prob': prediction['normal_prob'],
                'pneumonia_prob': prediction['pneumonia_prob']
            })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Calculate metrics
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    # Overall accuracy
    correct = (df['ground_truth'] == df['prediction']).sum()
    total = len(df)
    accuracy = correct / total

    print(f"\nOverall Accuracy: {accuracy*100:.2f}% ({correct}/{total})")

    # Confusion matrix
    true_positives = len(df[(df['ground_truth'] == 'PNEUMONIA') & (df['prediction'] == 'PNEUMONIA')])
    false_positives = len(df[(df['ground_truth'] == 'NORMAL') & (df['prediction'] == 'PNEUMONIA')])
    true_negatives = len(df[(df['ground_truth'] == 'NORMAL') & (df['prediction'] == 'NORMAL')])
    false_negatives = len(df[(df['ground_truth'] == 'PNEUMONIA') & (df['prediction'] == 'NORMAL')])

    print(f"\nConfusion Matrix:")
    print(f"  True Positives (correctly identified pneumonia): {true_positives}")
    print(f"  True Negatives (correctly identified normal): {true_negatives}")
    print(f"  False Positives (normal predicted as pneumonia): {false_positives}")
    print(f"  False Negatives (pneumonia predicted as normal): {false_negatives} ⚠️")

    # Metrics
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    print(f"\nDetailed Metrics:")
    print(f"  Sensitivity (Recall): {sensitivity*100:.2f}% - correctly identifies {sensitivity*100:.1f}% of pneumonia cases")
    print(f"  Specificity: {specificity*100:.2f}% - correctly identifies {specificity*100:.1f}% of normal cases")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  F1-Score: {f1*100:.2f}%")

    # Compare with October 18 results
    print(f"\n" + "="*80)
    print("COMPARISON WITH OCTOBER 18 TEST")
    print("="*80)
    print("\nOctober 18, 2025 Results:")
    print("  Accuracy: 98.6%")
    print("  False Negatives: 6")
    print("  False Positives: 1")

    print(f"\nCurrent Test Results:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  False Negatives: {false_negatives}")
    print(f"  False Positives: {false_positives}")

    accuracy_change = (accuracy - 0.986) * 100
    fn_change = false_negatives - 6
    fp_change = false_positives - 1

    print(f"\nChanges:")
    print(f"  Accuracy: {'+' if accuracy_change >= 0 else ''}{accuracy_change:.2f}%")
    print(f"  False Negatives: {'+' if fn_change >= 0 else ''}{fn_change}")
    print(f"  False Positives: {'+' if fp_change >= 0 else ''}{fp_change}")

    # List false negatives
    if false_negatives > 0:
        print(f"\n" + "="*80)
        print(f"FALSE NEGATIVE CASES ({false_negatives} total)")
        print("="*80)

        fn_df = df[(df['ground_truth'] == 'PNEUMONIA') & (df['prediction'] == 'NORMAL')]

        for i, row in fn_df.iterrows():
            print(f"\n{i+1}. {row['filename']}")
            print(f"   Predicted: NORMAL with {row['confidence']*100:.1f}% confidence")
            print(f"   Truth: PNEUMONIA")
            print(f"   NORMAL prob: {row['normal_prob']*100:.2f}%")
            print(f"   PNEUMONIA prob: {row['pneumonia_prob']*100:.2f}%")
            print(f"   Path: {row['image_path']}")

        # Save false negatives list
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = project_root / "validation" / "reports" / f"rerun_test_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        fn_list_path = save_dir / "false_negatives.json"
        fn_list = []
        for _, row in fn_df.iterrows():
            fn_list.append({
                'path': row['image_path'],
                'filename': row['filename'],
                'confidence': float(row['confidence']),
                'normal_prob': float(row['normal_prob']),
                'pneumonia_prob': float(row['pneumonia_prob'])
            })

        with open(fn_list_path, 'w') as f:
            json.dump(fn_list, f, indent=2)

        print(f"\n✓ Saved false negatives list to: {fn_list_path}")
    else:
        print(f"\n" + "="*80)
        print("🎉 NO FALSE NEGATIVES FOUND!")
        print("="*80)
        print("The model correctly identified all pneumonia cases.")

    # List false positives
    if false_positives > 0:
        print(f"\n" + "="*80)
        print(f"FALSE POSITIVE CASES ({false_positives} total)")
        print("="*80)

        fp_df = df[(df['ground_truth'] == 'NORMAL') & (df['prediction'] == 'PNEUMONIA')]

        for i, row in fp_df.iterrows():
            print(f"\n{i+1}. {row['filename']}")
            print(f"   Predicted: PNEUMONIA with {row['confidence']*100:.1f}% confidence")
            print(f"   Truth: NORMAL")
            print(f"   Path: {row['image_path']}")

    # Save full results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = project_root / "validation" / "reports" / f"rerun_test_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "full_results.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        'test_date': timestamp,
        'total_images': total,
        'accuracy': float(accuracy),
        'true_positives': int(true_positives),
        'true_negatives': int(true_negatives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1_score': float(f1),
        'comparison_oct_18': {
            'accuracy_change': float(accuracy_change),
            'fn_change': int(fn_change),
            'fp_change': int(fp_change)
        }
    }

    json_path = save_dir / "summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n" + "="*80)
    print("SAVED RESULTS")
    print("="*80)
    print(f"Directory: {save_dir}")
    print(f"  • full_results.csv - All {total} predictions")
    print(f"  • summary.json - Test summary and metrics")
    if false_negatives > 0:
        print(f"  • false_negatives.json - {false_negatives} false negative cases")

    print(f"\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

    if false_negatives > 0:
        print(f"\nNext step: Run analyze_false_negatives.py on the {false_negatives} cases")
        print(f"to understand why they failed with GradCAM/ROI/Patch analysis.")
    else:
        print("\nNo false negatives found! Model performance has improved.")


if __name__ == "__main__":
    main()
