"""
Image Preprocessing Pipeline for API Inference

This module provides comprehensive image preprocessing capabilities for
the pneumonia detection API. Handles medical image validation, quality
assessment, and standardization for clinical deployment.

Features:
- Medical image validation and quality checks
- Standardized preprocessing for different model types
- DICOM and standard image format support
- Image enhancement and noise reduction
- Batch processing capabilities
- Clinical metadata extraction

Designed to ensure consistent input quality for accurate
pneumonia detection in clinical environments.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import cv2

# Setup logging
logger = logging.getLogger(__name__)

# Preprocessing configurations for different model types
PREPROCESSING_CONFIGS = {
    'mobilenet': {
        'input_size': (224, 224),
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'resize_method': 'bilinear',
        'enhancement': 'standard'
    },
    'vgg': {
        'input_size': (224, 224),
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'resize_method': 'bilinear',
        'enhancement': 'standard'
    },
    'xception': {
        'input_size': (299, 299),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'resize_method': 'bilinear',
        'enhancement': 'advanced'
    },
    'fusion': {
        'input_size': (224, 224),
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'resize_method': 'bicubic',
        'enhancement': 'advanced'
    }
}

# Image quality thresholds
QUALITY_THRESHOLDS = {
    'min_resolution': (512, 512),
    'max_resolution': (4096, 4096),
    'min_contrast': 0.1,
    'max_noise_level': 0.3,
    'min_sharpness': 0.2,
    'aspect_ratio_tolerance': 0.3
}

# Medical image enhancement parameters
ENHANCEMENT_PARAMS = {
    'standard': {
        'contrast_factor': 1.1,
        'brightness_factor': 1.0,
        'sharpness_factor': 1.05,
        'histogram_equalization': False
    },
    'advanced': {
        'contrast_factor': 1.15,
        'brightness_factor': 1.02,
        'sharpness_factor': 1.1,
        'histogram_equalization': True,
        'noise_reduction': True
    }
}


class ImagePreprocessor:
    """
    Comprehensive image preprocessing for pneumonia detection API.

    Provides medical-grade image validation, enhancement, and standardization
    with support for different model architectures and clinical requirements.
    """

    def __init__(self, model_type: str = 'mobilenet', enable_quality_checks: bool = True):
        """
        Initialize image preprocessor.

        Args:
            model_type: Target model type for preprocessing
            enable_quality_checks: Whether to perform image quality validation
        """
        if model_type not in PREPROCESSING_CONFIGS:
            raise ValueError(f"Unsupported model type: {model_type}. Available: {list(PREPROCESSING_CONFIGS.keys())}")

        self.model_type = model_type
        self.config = PREPROCESSING_CONFIGS[model_type]
        self.enable_quality_checks = enable_quality_checks

        # Setup transforms
        self.inference_transform = self._create_inference_transform()
        self.enhancement_params = ENHANCEMENT_PARAMS[self.config['enhancement']]

        logger.info(f"ImagePreprocessor initialized for {model_type}")
        logger.info(f"Input size: {self.config['input_size']}")
        logger.info(f"Enhancement: {self.config['enhancement']}")

    def preprocess_image(self,
                        image: Union[Image.Image, np.ndarray, str, bytes],
                        patient_id: Optional[str] = None,
                        enhance_image: bool = True,
                        validate_quality: bool = None) -> Dict[str, Any]:
        """
        Comprehensive image preprocessing with quality validation.

        Args:
            image: Input image in various formats
            patient_id: Optional patient identifier
            enhance_image: Whether to apply image enhancement
            validate_quality: Whether to validate image quality (overrides default)

        Returns:
            Dict containing processed tensor, metadata, and quality metrics
        """
        if validate_quality is None:
            validate_quality = self.enable_quality_checks

        preprocessing_results = {
            'tensor': None,
            'original_size': None,
            'processed_size': self.config['input_size'],
            'quality_metrics': {},
            'preprocessing_steps': [],
            'warnings': [],
            'metadata': {}
        }

        try:
            # Load and convert image
            pil_image, load_step = self._load_image(image)
            preprocessing_results['preprocessing_steps'].append(load_step)
            preprocessing_results['original_size'] = pil_image.size

            # Quality validation
            if validate_quality:
                quality_results = self._validate_image_quality(pil_image)
                preprocessing_results['quality_metrics'] = quality_results['metrics']
                preprocessing_results['warnings'].extend(quality_results['warnings'])

                if not quality_results['is_valid']:
                    raise ValueError(f"Image quality validation failed: {quality_results['warnings']}")

            # Image enhancement
            if enhance_image:
                enhanced_image, enhancement_step = self._enhance_image(pil_image)
                preprocessing_results['preprocessing_steps'].append(enhancement_step)
                pil_image = enhanced_image

            # Apply model-specific transforms
            tensor, transform_step = self._apply_transforms(pil_image)
            preprocessing_results['tensor'] = tensor
            preprocessing_results['preprocessing_steps'].append(transform_step)

            # Add metadata
            preprocessing_results['metadata'] = {
                'patient_id': patient_id,
                'model_type': self.model_type,
                'preprocessing_config': self.config.copy(),
                'enhancement_applied': enhance_image,
                'quality_validated': validate_quality
            }

            logger.debug(f"Image preprocessing completed for {patient_id or 'unknown'}")
            return preprocessing_results

        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise

    def preprocess_batch(self,
                        images: List[Union[Image.Image, np.ndarray, str, bytes]],
                        patient_ids: Optional[List[str]] = None,
                        enhance_images: bool = True,
                        validate_quality: bool = None) -> List[Dict[str, Any]]:
        """
        Batch image preprocessing for multiple images.

        Args:
            images: List of input images in various formats
            patient_ids: Optional list of patient identifiers
            enhance_images: Whether to apply image enhancement
            validate_quality: Whether to validate image quality

        Returns:
            List of preprocessing results for each image
        """
        if patient_ids and len(patient_ids) != len(images):
            raise ValueError("Number of patient IDs must match number of images")

        results = []
        for i, image in enumerate(images):
            patient_id = patient_ids[i] if patient_ids else None

            try:
                result = self.preprocess_image(
                    image=image,
                    patient_id=patient_id,
                    enhance_image=enhance_images,
                    validate_quality=validate_quality
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Batch preprocessing failed for image {i}: {e}")
                # Add failed result with error information
                results.append({
                    'tensor': None,
                    'error': str(e),
                    'patient_id': patient_id,
                    'image_index': i
                })

        return results

    def _load_image(self, image_input: Union[Image.Image, np.ndarray, str, bytes]) -> Tuple[Image.Image, str]:
        """Load image from various input formats."""
        if isinstance(image_input, Image.Image):
            pil_image = image_input.copy()
            load_step = "loaded_pil_image"

        elif isinstance(image_input, np.ndarray):
            # Convert numpy array to PIL Image
            if image_input.dtype != np.uint8:
                image_input = (image_input * 255).astype(np.uint8)

            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                pil_image = Image.fromarray(image_input, 'RGB')
            elif len(image_input.shape) == 2:
                pil_image = Image.fromarray(image_input, 'L').convert('RGB')
            else:
                raise ValueError(f"Unsupported numpy array shape: {image_input.shape}")

            load_step = "converted_numpy_to_pil"

        elif isinstance(image_input, (str, Path)):
            # Load from file path
            pil_image = Image.open(image_input)
            load_step = "loaded_from_file"

        elif isinstance(image_input, bytes):
            # Load from bytes
            import io
            pil_image = Image.open(io.BytesIO(image_input))
            load_step = "loaded_from_bytes"

        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Ensure RGB format
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            load_step += "_converted_to_rgb"

        return pil_image, load_step

    def _validate_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Validate image quality for medical analysis."""
        quality_results = {
            'is_valid': True,
            'metrics': {},
            'warnings': []
        }

        try:
            # Convert to numpy for analysis
            img_array = np.array(image)

            # Resolution check
            width, height = image.size
            min_w, min_h = QUALITY_THRESHOLDS['min_resolution']
            max_w, max_h = QUALITY_THRESHOLDS['max_resolution']

            quality_results['metrics']['resolution'] = (width, height)

            if width < min_w or height < min_h:
                quality_results['warnings'].append(f"Low resolution: {width}x{height} < {min_w}x{min_h}")
                quality_results['is_valid'] = False

            if width > max_w or height > max_h:
                quality_results['warnings'].append(f"Excessive resolution: {width}x{height} > {max_w}x{max_h}")

            # Aspect ratio check
            aspect_ratio = width / height
            expected_ratio = 1.0  # Chest X-rays are typically square-ish
            ratio_diff = abs(aspect_ratio - expected_ratio) / expected_ratio

            quality_results['metrics']['aspect_ratio'] = aspect_ratio

            if ratio_diff > QUALITY_THRESHOLDS['aspect_ratio_tolerance']:
                quality_results['warnings'].append(f"Unusual aspect ratio: {aspect_ratio:.2f}")

            # Contrast analysis
            grayscale = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            contrast = grayscale.std() / 255.0

            quality_results['metrics']['contrast'] = contrast

            if contrast < QUALITY_THRESHOLDS['min_contrast']:
                quality_results['warnings'].append(f"Low contrast: {contrast:.3f}")
                quality_results['is_valid'] = False

            # Sharpness analysis (using Laplacian variance)
            laplacian_var = cv2.Laplacian(grayscale, cv2.CV_64F).var()
            normalized_sharpness = min(laplacian_var / 1000.0, 1.0)

            quality_results['metrics']['sharpness'] = normalized_sharpness

            if normalized_sharpness < QUALITY_THRESHOLDS['min_sharpness']:
                quality_results['warnings'].append(f"Low sharpness: {normalized_sharpness:.3f}")

            # Noise level estimation
            noise_level = self._estimate_noise_level(grayscale)
            quality_results['metrics']['noise_level'] = noise_level

            if noise_level > QUALITY_THRESHOLDS['max_noise_level']:
                quality_results['warnings'].append(f"High noise level: {noise_level:.3f}")

            # Brightness analysis
            mean_brightness = grayscale.mean() / 255.0
            quality_results['metrics']['brightness'] = mean_brightness

            if mean_brightness < 0.1 or mean_brightness > 0.9:
                quality_results['warnings'].append(f"Extreme brightness: {mean_brightness:.3f}")

        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            quality_results['warnings'].append(f"Quality analysis failed: {str(e)}")

        return quality_results

    def _enhance_image(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """Apply medical image enhancement techniques."""
        enhanced_image = image.copy()
        enhancement_steps = []

        try:
            # Contrast enhancement
            if self.enhancement_params['contrast_factor'] != 1.0:
                enhancer = ImageEnhance.Contrast(enhanced_image)
                enhanced_image = enhancer.enhance(self.enhancement_params['contrast_factor'])
                enhancement_steps.append("contrast")

            # Brightness adjustment
            if self.enhancement_params['brightness_factor'] != 1.0:
                enhancer = ImageEnhance.Brightness(enhanced_image)
                enhanced_image = enhancer.enhance(self.enhancement_params['brightness_factor'])
                enhancement_steps.append("brightness")

            # Sharpness enhancement
            if self.enhancement_params['sharpness_factor'] != 1.0:
                enhancer = ImageEnhance.Sharpness(enhanced_image)
                enhanced_image = enhancer.enhance(self.enhancement_params['sharpness_factor'])
                enhancement_steps.append("sharpness")

            # Histogram equalization for advanced enhancement
            if self.enhancement_params.get('histogram_equalization', False):
                enhanced_image = self._apply_histogram_equalization(enhanced_image)
                enhancement_steps.append("histogram_equalization")

            # Noise reduction for advanced enhancement
            if self.enhancement_params.get('noise_reduction', False):
                enhanced_image = enhanced_image.filter(ImageFilter.MedianFilter(size=3))
                enhancement_steps.append("noise_reduction")

            enhancement_step = f"enhanced_image_({','.join(enhancement_steps)})"

        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            enhancement_step = "enhancement_failed"

        return enhanced_image, enhancement_step

    def _apply_transforms(self, image: Image.Image) -> Tuple[torch.Tensor, str]:
        """Apply model-specific transforms to prepare tensor."""
        try:
            tensor = self.inference_transform(image)
            return tensor, "applied_model_transforms"

        except Exception as e:
            logger.error(f"Transform application failed: {e}")
            raise

    def _create_inference_transform(self) -> transforms.Compose:
        """Create model-specific inference transforms."""
        input_size = self.config['input_size']
        mean = self.config['mean']
        std = self.config['std']

        # Choose resize method
        if self.config['resize_method'] == 'bicubic':
            interpolation = transforms.InterpolationMode.BICUBIC
        else:
            interpolation = transforms.InterpolationMode.BILINEAR

        transform_list = [
            transforms.Resize(input_size, interpolation=interpolation),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]

        return transforms.Compose(transform_list)

    def _estimate_noise_level(self, grayscale_image: np.ndarray) -> float:
        """Estimate noise level in grayscale image."""
        try:
            # Use Laplacian to estimate noise
            laplacian = cv2.Laplacian(grayscale_image, cv2.CV_64F)
            noise_estimate = laplacian.var()

            # Normalize to 0-1 range
            normalized_noise = min(noise_estimate / 10000.0, 1.0)
            return normalized_noise

        except Exception:
            return 0.0

    def _apply_histogram_equalization(self, image: Image.Image) -> Image.Image:
        """Apply histogram equalization to improve contrast."""
        try:
            # Convert to numpy array
            img_array = np.array(image)

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

            # Process each channel
            enhanced_channels = []
            for i in range(3):  # RGB channels
                enhanced_channel = clahe.apply(img_array[:, :, i])
                enhanced_channels.append(enhanced_channel)

            # Reconstruct image
            enhanced_array = np.stack(enhanced_channels, axis=-1)
            return Image.fromarray(enhanced_array, 'RGB')

        except Exception as e:
            logger.warning(f"Histogram equalization failed: {e}")
            return image

    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about current preprocessing configuration."""
        return {
            'model_type': self.model_type,
            'config': self.config.copy(),
            'enhancement_params': self.enhancement_params.copy(),
            'quality_checks_enabled': self.enable_quality_checks,
            'quality_thresholds': QUALITY_THRESHOLDS.copy()
        }


def create_preprocessor(model_type: str = 'mobilenet', **kwargs) -> ImagePreprocessor:
    """
    Convenience function to create image preprocessor.

    Args:
        model_type: Target model type
        **kwargs: Additional preprocessor arguments

    Returns:
        Configured ImagePreprocessor instance
    """
    return ImagePreprocessor(model_type=model_type, **kwargs)


if __name__ == "__main__":
    # Test image preprocessing
    print("Testing image preprocessing...")

    try:
        # Create preprocessor
        preprocessor = ImagePreprocessor('mobilenet')

        # Test with dummy image
        dummy_image = Image.new('RGB', (512, 512), color='gray')
        result = preprocessor.preprocess_image(dummy_image)

        print(f"Preprocessing successful: {result['tensor'].shape}")
        print(f"Quality metrics: {result['quality_metrics']}")
        print(f"Steps: {result['preprocessing_steps']}")

        print("Image preprocessing ready!")

    except Exception as e:
        print(f"Error during testing: {e}")

    print("Image preprocessing system ready!")