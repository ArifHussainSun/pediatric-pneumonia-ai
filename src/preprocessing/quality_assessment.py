"""
Image Quality Assessment for Medical X-ray Images

This module provides functions to assess the quality of chest X-ray images
before model inference to detect poor exposure, low contrast, and other
quality issues that may affect prediction confidence.
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def assess_image_quality(image: Union[np.ndarray, Image.Image]) -> Dict[str, Union[str, float, bool]]:
    """
    Assess the quality of a chest X-ray image.

    Args:
        image: Input image as numpy array or PIL Image

    Returns:
        Dictionary containing quality metrics and assessment
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        if image.mode != 'L':  # Convert to grayscale if needed
            image = image.convert('L')
        img_array = np.array(image)
    else:
        img_array = image

    # Ensure we have a 2D grayscale image
    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=2)

    # Calculate quality metrics
    brightness = float(np.mean(img_array))
    contrast = float(np.std(img_array))

    # Calculate additional metrics
    dynamic_range = float(np.max(img_array) - np.min(img_array))
    histogram_entropy = calculate_histogram_entropy(img_array)

    # Assess quality levels
    brightness_quality = assess_brightness_quality(brightness)
    contrast_quality = assess_contrast_quality(contrast)
    overall_quality = determine_overall_quality(brightness_quality, contrast_quality, dynamic_range)

    return {
        'brightness': brightness,
        'contrast': contrast,
        'dynamic_range': dynamic_range,
        'histogram_entropy': histogram_entropy,
        'brightness_quality': brightness_quality,
        'contrast_quality': contrast_quality,
        'overall_quality': overall_quality,
        'is_acceptable': overall_quality in ['good', 'acceptable']
    }


def assess_brightness_quality(brightness: float) -> str:
    """
    Assess brightness quality based on medical imaging standards.

    Args:
        brightness: Mean pixel intensity (0-255)

    Returns:
        Quality level: 'too_dark', 'dark', 'good', 'bright', 'too_bright'
    """
    if brightness < 30:
        return 'too_dark'
    elif brightness < 60:
        return 'dark'
    elif brightness > 200:
        return 'too_bright'
    elif brightness > 160:
        return 'bright'
    else:
        return 'good'


def assess_contrast_quality(contrast: float) -> str:
    """
    Assess contrast quality based on standard deviation.

    Args:
        contrast: Standard deviation of pixel intensities

    Returns:
        Quality level: 'very_low', 'low', 'good', 'high'
    """
    if contrast < 15:
        return 'very_low'
    elif contrast < 30:
        return 'low'
    elif contrast > 80:
        return 'high'
    else:
        return 'good'


def determine_overall_quality(brightness_quality: str, contrast_quality: str, dynamic_range: float) -> str:
    """
    Determine overall image quality based on multiple factors.

    Args:
        brightness_quality: Brightness assessment
        contrast_quality: Contrast assessment
        dynamic_range: Pixel intensity range

    Returns:
        Overall quality: 'poor', 'acceptable', 'good', 'excellent'
    """
    # Poor quality conditions
    if brightness_quality in ['too_dark', 'too_bright']:
        return 'poor'
    if contrast_quality == 'very_low':
        return 'poor'
    if dynamic_range < 50:  # Very low dynamic range
        return 'poor'

    # Good quality conditions
    if (brightness_quality == 'good' and
        contrast_quality == 'good' and
        dynamic_range > 150):
        return 'excellent'

    # Acceptable quality conditions
    if (brightness_quality in ['good', 'dark', 'bright'] and
        contrast_quality in ['good', 'low']):
        return 'good'

    return 'acceptable'


def calculate_histogram_entropy(image: np.ndarray) -> float:
    """
    Calculate histogram entropy as a measure of image information content.

    Args:
        image: Input grayscale image

    Returns:
        Histogram entropy value
    """
    # Calculate histogram
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))

    # Normalize histogram to get probabilities
    hist = hist + 1e-7  # Add small value to avoid log(0)
    hist = hist / np.sum(hist)

    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist))

    return float(entropy)


def get_quality_warnings(quality_assessment: Dict) -> List[str]:
    """
    Generate user-friendly warnings based on quality assessment.

    Args:
        quality_assessment: Output from assess_image_quality()

    Returns:
        List of warning messages for the user
    """
    warnings = []

    brightness_quality = quality_assessment['brightness_quality']
    contrast_quality = quality_assessment['contrast_quality']
    overall_quality = quality_assessment['overall_quality']

    # Brightness warnings
    if brightness_quality == 'too_dark':
        warnings.append("Image is too dark. Please increase lighting or adjust camera exposure.")
    elif brightness_quality == 'dark':
        warnings.append("Image appears underexposed. Consider improving lighting for better accuracy.")
    elif brightness_quality == 'too_bright':
        warnings.append("Image is overexposed. Please reduce lighting or adjust camera exposure.")
    elif brightness_quality == 'bright':
        warnings.append("Image is quite bright. Consider reducing exposure for optimal results.")

    # Contrast warnings
    if contrast_quality == 'very_low':
        warnings.append("Image has very low contrast. This may significantly affect prediction accuracy.")
    elif contrast_quality == 'low':
        warnings.append("Image has low contrast. Consider adjusting camera settings or lighting.")

    # Overall quality warnings
    if overall_quality == 'poor':
        warnings.append("Image quality is poor and may lead to unreliable results. Please retake the image.")
    elif overall_quality == 'acceptable':
        warnings.append("Image quality is acceptable but could be improved for better confidence.")

    # Dynamic range warning
    if quality_assessment['dynamic_range'] < 50:
        warnings.append("Image has limited tonal range. This may affect detail visibility.")

    return warnings


def is_image_suitable_for_analysis(quality_assessment: Dict) -> bool:
    """
    Determine if image quality is suitable for medical analysis.

    Args:
        quality_assessment: Output from assess_image_quality()

    Returns:
        True if image is suitable for analysis, False otherwise
    """
    return quality_assessment['is_acceptable']


def get_quality_score(quality_assessment: Dict) -> float:
    """
    Calculate a numeric quality score (0-100) based on assessment.

    Args:
        quality_assessment: Output from assess_image_quality()

    Returns:
        Quality score from 0 (poor) to 100 (excellent)
    """
    overall_quality = quality_assessment['overall_quality']

    quality_scores = {
        'poor': 25,
        'acceptable': 50,
        'good': 75,
        'excellent': 95
    }

    base_score = quality_scores.get(overall_quality, 50)

    # Adjust based on specific metrics
    brightness_quality = quality_assessment['brightness_quality']
    contrast_quality = quality_assessment['contrast_quality']

    # Brightness adjustments
    if brightness_quality == 'good':
        base_score += 5
    elif brightness_quality in ['too_dark', 'too_bright']:
        base_score -= 15

    # Contrast adjustments
    if contrast_quality == 'good':
        base_score += 5
    elif contrast_quality == 'very_low':
        base_score -= 10

    # Ensure score stays within bounds
    return max(0, min(100, base_score))


if __name__ == "__main__":
    # Test with a sample image
    test_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    assessment = assess_image_quality(test_image)
    warnings = get_quality_warnings(assessment)

    print("Quality Assessment:")
    for key, value in assessment.items():
        print(f"  {key}: {value}")

    print("\nWarnings:")
    for warning in warnings:
        print(f"  - {warning}")

    print(f"\nQuality Score: {get_quality_score(assessment)}/100")