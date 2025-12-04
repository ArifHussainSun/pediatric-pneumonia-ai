#!/usr/bin/env python3
"""
Quick test to verify ROI-based CLAHE integration in intelligent_preprocessing.py

This script tests one of the false negatives to ensure:
1. ROI detection works
2. ROI-based CLAHE is applied
3. Prediction improves
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import cv2
import torch
from PIL import Image
from src.models.mobilenet import MobileNetFineTune
from src.preprocessing.intelligent_preprocessing import IntelligentPreprocessor
from src.data.datasets import get_medical_transforms

def test_roi_integration():
    """Test ROI integration on a false negative case"""
    print("=" * 80)
    print("ROI INTEGRATION TEST")
    print("=" * 80)

    # Load model
    print("\n1. Loading model...")
    model_path = project_root / "outputs" / "dgx_station_experiment" / "Best_MobilenetV1.pth"
    model = MobileNetFineTune(num_classes=2)
    model.load_custom_weights(str(model_path))
    model.eval()
    print("   ✓ Model loaded")

    # Initialize preprocessor
    print("\n2. Initializing IntelligentPreprocessor with ROI...")
    preprocessor = IntelligentPreprocessor()
    print(f"   ✓ Preprocessor initialized")
    print(f"   ✓ ROI detector available: {preprocessor.roi_detector is not None}")

    # Test on a known false negative
    test_image_path = Path("/Users/Adam/Downloads/College/Co op/Tech4Life-25Summer/chest_xray/train/PNEUMONIA/person1483_virus_2574.jpeg")

    if not test_image_path.exists():
        print(f"\n   ERROR: Test image not found at {test_image_path}")
        return

    print(f"\n3. Testing on: {test_image_path.name}")
    print("   (This was a false negative: predicted NORMAL instead of PNEUMONIA)")

    # Load image
    image = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)

    # Quality assessment
    print("\n4. Quality assessment...")
    quality = preprocessor.assess_image_quality(image)
    print(f"   Overall quality: {quality.overall_quality.value}")
    print(f"   Brightness: {quality.brightness_score:.3f}")
    print(f"   Contrast: {quality.contrast_score:.3f}")
    print(f"   Should enhance: {quality.should_enhance}")

    # Enhancement
    if quality.should_enhance:
        print("\n5. Applying enhancement with ROI-based CLAHE...")
        enhanced = preprocessor.enhance_image(image, quality)
        print("   ✓ Enhancement complete")
    else:
        print("\n5. No enhancement needed")
        enhanced = image

    # Prediction
    print("\n6. Running prediction...")
    transform = get_medical_transforms(is_training=False)

    # Prepare for model
    pil_image = Image.fromarray(enhanced).convert('RGB')
    input_tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pneumonia_prob = probs[0, 0].item()  # Class 0 = PNEUMONIA
        normal_prob = probs[0, 1].item()     # Class 1 = NORMAL

        prediction = 'PNEUMONIA' if pneumonia_prob > 0.5 else 'NORMAL'
        confidence = max(pneumonia_prob, normal_prob)

    print(f"   Prediction: {prediction}")
    print(f"   Pneumonia probability: {pneumonia_prob:.4f}")
    print(f"   Normal probability: {normal_prob:.4f}")
    print(f"   Confidence: {confidence:.4f}")

    # Expected result
    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)

    if prediction == 'PNEUMONIA' and pneumonia_prob > 0.9:
        print("✓ SUCCESS! ROI-based CLAHE is working correctly.")
        print(f"  Expected: PNEUMONIA with high probability (>90%)")
        print(f"  Got: {prediction} ({pneumonia_prob*100:.1f}%)")
        print("\nThe ROI integration successfully fixes this false negative!")
    elif prediction == 'PNEUMONIA':
        print("⚠ PARTIAL SUCCESS - Prediction is correct but confidence is low")
        print(f"  Got: {prediction} ({pneumonia_prob*100:.1f}%)")
        print("  May need further tuning")
    else:
        print("✗ FAILURE - ROI integration may not be working correctly")
        print(f"  Expected: PNEUMONIA")
        print(f"  Got: {prediction} ({normal_prob*100:.1f}%)")

    print("=" * 80)

if __name__ == "__main__":
    test_roi_integration()
