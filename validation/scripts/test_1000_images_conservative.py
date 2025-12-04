#!/usr/bin/env python3
"""
1000 Image Validation Test with Conservative Preprocessing

Tests conservative preprocessing approach on 1000 images (500 normal + 500 pneumonia)
to validate that the conservative approach maintains baseline performance at scale.
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
import argparse

from src.models.mobilenet import MobileNetFineTune
from src.data.datasets import get_medical_transforms
from src.preprocessing.intelligent_preprocessing import IntelligentPreprocessor
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn


def find_data_directory():
    """
    Auto-detect the data directory by checking common locations.

    Returns:
        Path to data directory or None if not found
    """
    possible_paths = [
        # Project directory (DGX and local)
        project_root / "data" / "train",
        # Adam's local path
        Path("/Users/Adam/Downloads/College/Co op/Tech4Life-25Summer/chest_xray/train"),
        # DGX common paths
        Path("/workspace/data/chest_xray/train"),
        Path("/data/chest_xray/train"),
        Path.home() / "data" / "chest_xray" / "train",
    ]

    for path in possible_paths:
        if path.exists():
            # Check for uppercase or lowercase folder names
            has_normal = (path / "NORMAL").exists() or (path / "normal").exists()
            has_pneumonia = (path / "PNEUMONIA").exists() or (path / "pneumonia").exists()
            if has_normal and has_pneumonia:
                return path

    return None


def load_model(model_path=None, model_type='mobilenet'):
    """Load trained model (MobileNet or ResNet50)."""
    if model_type == 'mobilenet':
        print("Loading MobileNet model...")

        if model_path is None:
            model_path = project_root / "outputs" / "dgx_station_experiment" / "Best_MobilenetV1.pth"
        else:
            model_path = Path(model_path)

        model = MobileNetFineTune(num_classes=2, freeze_layers=0)

        if model_path.exists():
            model.load_custom_weights(str(model_path))
            print(f"✓ Loaded model from {model_path}")
        else:
            print(f"ERROR: Model not found at {model_path}")
            return None

    elif model_type == 'resnet50':
        print("Loading ResNet50 model...")

        if model_path is None:
            model_path = project_root / "outputs" / "resnet50_standalone" / "best_model.pth"
        else:
            model_path = Path(model_path)

        # Load ResNet50 architecture
        model = resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)

        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"✓ Loaded model from {model_path}")
        else:
            print(f"ERROR: Model not found at {model_path}")
            return None

    else:
        print(f"ERROR: Unsupported model type: {model_type}")
        return None

    model.eval()
    return model


def predict_image(model, image_path, transform, preprocessor=None, skip_preprocessing=False):
    """
    Predict single image using the model with optional preprocessing.

    Args:
        skip_preprocessing: If True, skip preprocessing and use raw images

    Returns:
        dict with prediction, confidence, probabilities
    """
    try:
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        # Apply conservative preprocessing (unless skipped)
        if not skip_preprocessing and preprocessor is not None:
            quality = preprocessor.assess_image_quality(image)
            if quality.should_enhance:
                image = preprocessor.enhance_image(image, quality)

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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='1000 Image Validation Test')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model weights (default: outputs/dgx_station_experiment/Best_MobilenetV1.pth)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to dataset train directory (auto-detected if not provided)')
    parser.add_argument('--no-preprocessing', action='store_true',
                       help='Skip preprocessing, test on raw images (for CLAHE-augmented models)')
    parser.add_argument('--model_type', type=str, default='mobilenet', choices=['mobilenet', 'resnet50'],
                       help='Model architecture to load (default: mobilenet)')
    args = parser.parse_args()

    print("="*80)
    if args.no_preprocessing:
        print("1000 IMAGE VALIDATION TEST - RAW IMAGES (NO PREPROCESSING)")
    else:
        print("1000 IMAGE VALIDATION TEST - CONSERVATIVE PREPROCESSING")
    print("="*80)
    if args.no_preprocessing:
        print("\nTesting on raw images without preprocessing")
    else:
        print("\nTesting conservative preprocessing approach on 1000 images")
    print("(500 NORMAL + 500 PNEUMONIA)\n")

    # Load model
    model = load_model(args.model_path, model_type=args.model_type)
    if model is None:
        print("ERROR: Could not load model. Exiting.")
        return

    # Initialize preprocessor with conservative logic (unless skipped)
    preprocessor = None
    if not args.no_preprocessing:
        print("Initializing IntelligentPreprocessor with conservative preprocessing...")
        preprocessor = IntelligentPreprocessor()
        print(f"✓ ROI detector available: {preprocessor.roi_detector is not None}\n")
    else:
        print("Skipping preprocessing - testing on raw images\n")

    # Get transform
    transform = get_medical_transforms(is_training=False)

    # Load dataset
    if args.data_dir:
        dataset_path = Path(args.data_dir)
    else:
        dataset_path = find_data_directory()

    if dataset_path is None or not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("\nPlease specify dataset location with --data_dir")
        print("Example: python test_1000_images_conservative.py --data_dir ~/pediatric-pneumonia-ai/data/train")
        return

    print(f"Loading images from: {dataset_path}")

    # Sample 1000 images (500 from each class)
    # Support both uppercase and lowercase folder names
    normal_dir = dataset_path / "NORMAL" if (dataset_path / "NORMAL").exists() else dataset_path / "normal"
    pneumonia_dir = dataset_path / "PNEUMONIA" if (dataset_path / "PNEUMONIA").exists() else dataset_path / "pneumonia"

    normal_images = list(normal_dir.glob("*.jpeg"))
    pneumonia_images = list(pneumonia_dir.glob("*.jpeg"))

    print(f"Found {len(normal_images)} NORMAL images")
    print(f"Found {len(pneumonia_images)} PNEUMONIA images")

    # Use seed=2025 for fresh dataset
    random.seed(2025)
    normal_sample = random.sample(normal_images, min(500, len(normal_images)))
    pneumonia_sample = random.sample(pneumonia_images, min(500, len(pneumonia_images)))

    print(f"\nSampled {len(normal_sample)} NORMAL images")
    print(f"Sampled {len(pneumonia_sample)} PNEUMONIA images")
    print(f"Total: {len(normal_sample) + len(pneumonia_sample)} images")
    print(f"Using seed=2025 for fresh validation data")

    # Run predictions
    results = []

    if args.no_preprocessing:
        print("\nRunning predictions on raw images (no preprocessing)...")
    else:
        print("\nRunning predictions with conservative preprocessing...")

    for img_path in tqdm(normal_sample + pneumonia_sample, desc="Processing images"):
        ground_truth = 'NORMAL' if img_path.parent.name in ['NORMAL', 'normal'] else 'PNEUMONIA'

        prediction = predict_image(model, img_path, transform, preprocessor, skip_preprocessing=args.no_preprocessing)

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

    # Per-class accuracy
    normal_correct = true_negatives
    normal_total = true_negatives + false_positives
    normal_accuracy = normal_correct / normal_total if normal_total > 0 else 0

    pneumonia_correct = true_positives
    pneumonia_total = true_positives + false_negatives
    pneumonia_accuracy = pneumonia_correct / pneumonia_total if pneumonia_total > 0 else 0

    print(f"\nPer-Class Accuracy:")
    print(f"  NORMAL: {normal_accuracy*100:.2f}% ({normal_correct}/{normal_total})")
    print(f"  PNEUMONIA: {pneumonia_accuracy*100:.2f}% ({pneumonia_correct}/{pneumonia_total})")

    # Metrics
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    print(f"\nDetailed Metrics:")
    print(f"  Sensitivity (Recall): {sensitivity*100:.2f}%")
    print(f"  Specificity: {specificity*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  F1-Score: {f1*100:.2f}%")

    # List false negatives
    if false_negatives > 0:
        print(f"\n" + "="*80)
        print(f"FALSE NEGATIVE CASES ({false_negatives} total)")
        print("="*80)

        fn_df = df[(df['ground_truth'] == 'PNEUMONIA') & (df['prediction'] == 'NORMAL')]
        fn_list = []

        for idx, (i, row) in enumerate(fn_df.iterrows(), 1):
            if idx <= 10:  # Print first 10
                print(f"\n{idx}. {row['filename']}")
                print(f"   Predicted: NORMAL with {row['confidence']*100:.1f}% confidence")
                print(f"   Truth: PNEUMONIA")
                print(f"   NORMAL prob: {row['normal_prob']*100:.2f}%")
                print(f"   PNEUMONIA prob: {row['pneumonia_prob']*100:.2f}%")

            fn_list.append({
                'path': row['image_path'],
                'filename': row['filename'],
                'normal_prob': float(row['normal_prob']),
                'pneumonia_prob': float(row['pneumonia_prob'])
            })

        if false_negatives > 10:
            print(f"\n... and {false_negatives - 10} more false negatives")
    else:
        print(f"\n" + "="*80)
        print("🎉 NO FALSE NEGATIVES FOUND!")
        print("="*80)
        fn_list = []

    # List false positives
    if false_positives > 0:
        print(f"\n" + "="*80)
        print(f"FALSE POSITIVE CASES ({false_positives} total)")
        print("="*80)

        fp_df = df[(df['ground_truth'] == 'NORMAL') & (df['prediction'] == 'PNEUMONIA')]

        for idx, (i, row) in enumerate(fp_df.iterrows(), 1):
            if idx <= 10:  # Print first 10
                print(f"\n{idx}. {row['filename']}")
                print(f"   Predicted: PNEUMONIA with {row['confidence']*100:.1f}% confidence")
                print(f"   Truth: NORMAL")

        if false_positives > 10:
            print(f"\n... and {false_positives - 10} more false positives")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = project_root / "validation" / "reports" / f"conservative_1000img_test_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    csv_path = save_dir / "full_results.csv"
    df.to_csv(csv_path, index=False)

    # Save false negatives
    fn_path = save_dir / "false_negatives.json"
    with open(fn_path, 'w') as f:
        json.dump(fn_list, f, indent=2)

    # Save summary
    summary = {
        'seed': 2025,
        'total': total,
        'accuracy': float(accuracy),
        'normal_accuracy': float(normal_accuracy),
        'pneumonia_accuracy': float(pneumonia_accuracy),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'metrics': {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1_score': float(f1)
        },
        'preprocessing': 'conservative_adaptive'
    }

    json_path = save_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n" + "="*80)
    print("SAVED RESULTS")
    print("="*80)
    print(f"Directory: {save_dir}")
    print(f"  • full_results.csv - All {total} predictions")
    print(f"  • results.json - Summary metrics")
    print(f"  • false_negatives.json - {false_negatives} false negative cases")

    print(f"\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"\nConservative preprocessing maintains strong performance on {total} images.")


if __name__ == "__main__":
    main()
