"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) Enhancement

This module provides medical-grade image enhancement using CLAHE,
which is the gold standard for chest X-ray preprocessing in clinical settings.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def apply_clahe_enhancement(
    image: Union[np.ndarray, Image.Image],
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE enhancement to improve image contrast.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) is the
    medical imaging standard for enhancing chest X-rays. It improves
    local contrast while preventing over-amplification of noise.

    Args:
        image: Input image as numpy array or PIL Image
        clip_limit: Contrast limiting threshold (higher = more contrast)
        tile_grid_size: Size of neighborhood for local enhancement

    Returns:
        Enhanced image as numpy array
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        if image.mode != 'L':  # Convert to grayscale if needed
            image = image.convert('L')
        img_array = np.array(image, dtype=np.uint8)
    else:
        img_array = image.astype(np.uint8)

    # Ensure we have a 2D grayscale image
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE enhancement
    enhanced = clahe.apply(img_array)

    logger.debug(f"Applied CLAHE enhancement with clip_limit={clip_limit}, tile_grid_size={tile_grid_size}")

    return enhanced


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """
    Apply gamma correction for brightness adjustment.

    Args:
        image: Input grayscale image
        gamma: Gamma value (>1 brightens, <1 darkens)

    Returns:
        Gamma-corrected image
    """
    # Build lookup table for gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)

    # Apply gamma correction
    corrected = cv2.LUT(image, table)

    logger.debug(f"Applied gamma correction with gamma={gamma}")

    return corrected


def normalize_intensity(image: np.ndarray, target_mean: float = 127.0, target_std: float = 50.0) -> np.ndarray:
    """
    Normalize image intensity to target statistics.

    Args:
        image: Input grayscale image
        target_mean: Target mean intensity
        target_std: Target standard deviation

    Returns:
        Normalized image
    """
    # Calculate current statistics
    current_mean = np.mean(image)
    current_std = np.std(image)

    # Avoid division by zero
    if current_std < 1e-6:
        logger.warning("Image has very low standard deviation, skipping normalization")
        return image

    # Normalize to zero mean, unit variance
    normalized = (image - current_mean) / current_std

    # Scale to target statistics
    result = normalized * target_std + target_mean

    # Clip to valid range
    result = np.clip(result, 0, 255).astype(np.uint8)

    logger.debug(f"Normalized intensity: {current_mean:.1f}±{current_std:.1f} → {target_mean}±{target_std}")

    return result


def preprocess_medical_image(
    image: Union[np.ndarray, Image.Image],
    apply_clahe: bool = True,
    apply_gamma: bool = False,
    apply_normalization: bool = True,
    clahe_clip_limit: float = 2.0,
    gamma_value: float = 1.2,
    target_size: Optional[Tuple[int, int]] = (224, 224)
) -> np.ndarray:
    """
    Complete preprocessing pipeline for medical chest X-ray images.

    This function applies the full preprocessing pipeline including
    CLAHE enhancement, gamma correction, intensity normalization,
    and resizing for model input.

    Args:
        image: Input image
        apply_clahe: Whether to apply CLAHE enhancement
        apply_gamma: Whether to apply gamma correction
        apply_normalization: Whether to apply intensity normalization
        clahe_clip_limit: CLAHE contrast limiting threshold
        gamma_value: Gamma correction value
        target_size: Target size for model input (None to skip resizing)

    Returns:
        Preprocessed image ready for model inference
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        if image.mode != 'L':
            image = image.convert('L')
        img_array = np.array(image, dtype=np.uint8)
    else:
        img_array = image.astype(np.uint8)

    # Ensure grayscale
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    processed = img_array.copy()

    # Apply CLAHE enhancement (medical standard)
    if apply_clahe:
        processed = apply_clahe_enhancement(processed, clip_limit=clahe_clip_limit)

    # Apply gamma correction if needed
    if apply_gamma:
        processed = apply_gamma_correction(processed, gamma=gamma_value)

    # Apply intensity normalization
    if apply_normalization:
        processed = normalize_intensity(processed)

    # Resize to target size for model input
    if target_size is not None:
        processed = cv2.resize(processed, target_size, interpolation=cv2.INTER_AREA)

    logger.info(f"Preprocessed medical image: CLAHE={apply_clahe}, Gamma={apply_gamma}, "
                f"Normalization={apply_normalization}, Size={target_size}")

    return processed


def calculate_optimal_gamma(image: np.ndarray) -> float:
    """
    Calculate optimal gamma value based on image characteristics.

    Args:
        image: Input grayscale image

    Returns:
        Optimal gamma value for correction
    """
    mean_intensity = np.mean(image)

    # Calculate gamma based on mean intensity
    # Dark images (mean < 85) need gamma < 1 (brightening)
    # Bright images (mean > 170) need gamma > 1 (darkening)
    if mean_intensity < 85:
        gamma = 0.7 + (mean_intensity / 85) * 0.3  # 0.7 to 1.0
    elif mean_intensity > 170:
        gamma = 1.0 + ((mean_intensity - 170) / 85) * 0.5  # 1.0 to 1.5
    else:
        gamma = 1.0  # No correction needed

    return gamma


def enhance_for_low_quality_image(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """
    Special enhancement pipeline for low-quality images.

    Applies more aggressive enhancement for images with poor
    exposure, low contrast, or other quality issues.

    Args:
        image: Input low-quality image

    Returns:
        Enhanced image with aggressive processing
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        if image.mode != 'L':
            image = image.convert('L')
        img_array = np.array(image, dtype=np.uint8)
    else:
        img_array = image.astype(np.uint8)

    # Ensure grayscale
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    processed = img_array.copy()

    # Calculate optimal gamma
    optimal_gamma = calculate_optimal_gamma(processed)

    # Apply gamma correction first for exposure issues
    if optimal_gamma != 1.0:
        processed = apply_gamma_correction(processed, gamma=optimal_gamma)

    # Apply more aggressive CLAHE for low contrast
    processed = apply_clahe_enhancement(processed, clip_limit=3.0, tile_grid_size=(6, 6))

    # Apply intensity normalization
    processed = normalize_intensity(processed)

    logger.info(f"Applied aggressive enhancement: gamma={optimal_gamma:.2f}, "
                f"CLAHE clip_limit=3.0, tile_grid=(6,6)")

    return processed


if __name__ == "__main__":
    # Test the enhancement functions
    test_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)

    print("Testing CLAHE enhancement...")
    enhanced = apply_clahe_enhancement(test_image)
    print(f"Original: mean={np.mean(test_image):.1f}, std={np.std(test_image):.1f}")
    print(f"Enhanced: mean={np.mean(enhanced):.1f}, std={np.std(enhanced):.1f}")

    print("\nTesting complete preprocessing pipeline...")
    processed = preprocess_medical_image(test_image)
    print(f"Processed: mean={np.mean(processed):.1f}, std={np.std(processed):.1f}")

    print("\nTesting aggressive enhancement...")
    aggressive = enhance_for_low_quality_image(test_image)
    print(f"Aggressive: mean={np.mean(aggressive):.1f}, std={np.std(aggressive):.1f}")

    print("\nCLAHE enhancement module ready!")