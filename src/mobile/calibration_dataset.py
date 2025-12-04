"""
Calibration Dataset Generator for Mobile Model Quantization

This module generates representative datasets for TensorFlow Lite quantization
from medical imaging data. Designed specifically for pneumonia detection models
deployed on Android devices.

Features:
- Representative data sampling from training/validation sets
- Medical image preprocessing for mobile inference
- Calibration dataset optimization for different deployment targets
- Quality validation for calibration effectiveness
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
import random

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image

# Setup logging
logger = logging.getLogger(__name__)

# Medical imaging preprocessing parameters
MEDICAL_IMAGE_STATS = {
    'mean': [0.485, 0.456, 0.406],  # ImageNet pretrained means
    'std': [0.229, 0.224, 0.225]   # ImageNet pretrained stds
}

# Calibration dataset configurations
CALIBRATION_CONFIGS = {
    'android_tablet': {
        'dataset_size': 100,
        'batch_size': 8,
        'image_quality': 'high',
        'diversity_factor': 0.8
    },
    'android_phone': {
        'dataset_size': 50,
        'batch_size': 4,
        'image_quality': 'medium',
        'diversity_factor': 0.9
    },
    'embedded_device': {
        'dataset_size': 25,
        'batch_size': 2,
        'image_quality': 'low',
        'diversity_factor': 1.0
    }
}


class CalibrationDatasetGenerator:
    """
    Generates representative datasets for mobile model quantization.

    Optimized for medical imaging data with focus on preserving
    diagnostic accuracy during quantization process.
    """

    def __init__(self, data_dir: Union[str, Path], target_platform: str = 'android_tablet'):
        """
        Initialize calibration dataset generator.

        Args:
            data_dir: Directory containing training/validation images
            target_platform: Target deployment platform configuration
        """
        self.data_dir = Path(data_dir)
        self.target_platform = target_platform

        if target_platform not in CALIBRATION_CONFIGS:
            raise ValueError(f"Unsupported platform: {target_platform}")

        self.config = CALIBRATION_CONFIGS[target_platform]

        # Initialize preprocessing transforms
        self.transform = self._create_mobile_transforms()

        logger.info(f"CalibrationDatasetGenerator initialized for {target_platform}")
        logger.info(f"Dataset size: {self.config['dataset_size']} samples")

    def _create_mobile_transforms(self) -> transforms.Compose:
        """Create preprocessing transforms optimized for mobile inference."""

        # Base transforms for medical imaging
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEDICAL_IMAGE_STATS['mean'],
                std=MEDICAL_IMAGE_STATS['std']
            )
        ]

        # Add quality-specific transforms
        if self.config['image_quality'] == 'low':
            # Simulate lower quality for embedded devices
            transform_list.insert(-2, transforms.RandomAdjustSharpness(sharpness_factor=0.8))

        return transforms.Compose(transform_list)

    def generate_calibration_dataset(self,
                                   output_path: Optional[Union[str, Path]] = None,
                                   class_balance: bool = True) -> List[np.ndarray]:
        """
        Generate representative calibration dataset.

        Args:
            output_path: Optional path to save calibration dataset
            class_balance: Whether to balance normal/pneumonia samples

        Returns:
            List of preprocessed calibration images as numpy arrays
        """
        logger.info("Generating calibration dataset...")

        # Find all image files
        image_files = self._find_medical_images()

        if len(image_files) < self.config['dataset_size']:
            logger.warning(f"Found only {len(image_files)} images, requested {self.config['dataset_size']}")

        # Sample representative images
        selected_images = self._sample_representative_images(
            image_files,
            target_count=self.config['dataset_size'],
            balance_classes=class_balance
        )

        # Preprocess images
        calibration_data = []
        for img_path in selected_images:
            try:
                image = self._load_and_preprocess_image(img_path)
                calibration_data.append(image)

            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                continue

        # Validate dataset quality
        self._validate_calibration_dataset(calibration_data)

        # Save dataset if requested
        if output_path:
            self._save_calibration_dataset(calibration_data, output_path)

        logger.info(f"Generated calibration dataset with {len(calibration_data)} samples")
        return calibration_data

    def _find_medical_images(self) -> List[Path]:
        """Find all medical image files in data directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        image_files = []

        for ext in image_extensions:
            # Search in common medical imaging directory structures
            pattern_paths = [
                self.data_dir.glob(f"**/*{ext}"),
                self.data_dir.glob(f"**/*{ext.upper()}")
            ]

            for pattern in pattern_paths:
                image_files.extend(list(pattern))

        # Remove duplicates
        unique_files = list(set(image_files))

        logger.info(f"Found {len(unique_files)} medical images")
        return unique_files

    def _sample_representative_images(self,
                                    image_files: List[Path],
                                    target_count: int,
                                    balance_classes: bool = True) -> List[Path]:
        """Sample representative images for calibration."""

        if balance_classes:
            # Try to balance normal vs pneumonia cases
            normal_images = [f for f in image_files if 'normal' in str(f).lower()]
            pneumonia_images = [f for f in image_files if 'pneumonia' in str(f).lower()]

            if normal_images and pneumonia_images:
                # Sample equally from both classes
                samples_per_class = target_count // 2

                selected_normal = random.sample(
                    normal_images,
                    min(samples_per_class, len(normal_images))
                )
                selected_pneumonia = random.sample(
                    pneumonia_images,
                    min(samples_per_class, len(pneumonia_images))
                )

                selected_images = selected_normal + selected_pneumonia

                # Fill remaining slots if needed
                remaining = target_count - len(selected_images)
                if remaining > 0:
                    other_images = [f for f in image_files if f not in selected_images]
                    additional = random.sample(other_images, min(remaining, len(other_images)))
                    selected_images.extend(additional)

                logger.info(f"Balanced sampling: {len(selected_normal)} normal, {len(selected_pneumonia)} pneumonia")
                return selected_images

        # Fallback to random sampling
        return random.sample(image_files, min(target_count, len(image_files)))

    def _load_and_preprocess_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess a single medical image."""

        # Load image
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Apply mobile-optimized transforms
            tensor_image = self.transform(img)

            # Convert to numpy array for TensorFlow Lite
            numpy_image = tensor_image.numpy()

            return numpy_image

    def _validate_calibration_dataset(self, calibration_data: List[np.ndarray]):
        """Validate quality of calibration dataset."""

        if not calibration_data:
            raise ValueError("Calibration dataset is empty")

        # Check data shape consistency
        first_shape = calibration_data[0].shape
        for i, data in enumerate(calibration_data):
            if data.shape != first_shape:
                logger.warning(f"Shape mismatch at index {i}: {data.shape} vs {first_shape}")

        # Check data distribution
        all_data = np.stack(calibration_data)
        mean_val = np.mean(all_data)
        std_val = np.std(all_data)

        logger.info(f"Calibration dataset statistics:")
        logger.info(f"  Shape: {first_shape}")
        logger.info(f"  Mean: {mean_val:.4f}")
        logger.info(f"  Std: {std_val:.4f}")
        logger.info(f"  Min: {np.min(all_data):.4f}")
        logger.info(f"  Max: {np.max(all_data):.4f}")

        # Validate distribution is reasonable for medical images
        if abs(mean_val) > 3.0 or std_val < 0.1 or std_val > 5.0:
            logger.warning("Calibration dataset has unusual distribution - check preprocessing")

    def _save_calibration_dataset(self, calibration_data: List[np.ndarray], output_path: Union[str, Path]):
        """Save calibration dataset to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Stack data and save as numpy array
        dataset_array = np.stack(calibration_data)
        np.save(output_path, dataset_array)

        # Save metadata
        metadata = {
            'target_platform': self.target_platform,
            'dataset_size': len(calibration_data),
            'shape': dataset_array.shape,
            'config': self.config
        }

        metadata_path = output_path.with_suffix('.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Calibration dataset saved to {output_path}")

    def generate_for_all_platforms(self, output_dir: Union[str, Path]) -> Dict[str, List[np.ndarray]]:
        """Generate calibration datasets for all mobile platforms."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for platform in CALIBRATION_CONFIGS.keys():
            logger.info(f"Generating calibration dataset for {platform}...")

            # Create platform-specific generator
            generator = CalibrationDatasetGenerator(self.data_dir, platform)

            # Generate dataset
            calibration_data = generator.generate_calibration_dataset(
                output_path=output_dir / f"calibration_{platform}.npy"
            )

            results[platform] = calibration_data

        logger.info("Calibration datasets generated for all platforms")
        return results


def create_calibration_dataset(data_dir: Union[str, Path],
                             target_platform: str = 'android_tablet',
                             output_path: Optional[Union[str, Path]] = None) -> List[np.ndarray]:
    """
    Convenience function to create calibration dataset.

    Args:
        data_dir: Directory containing medical images
        target_platform: Target deployment platform
        output_path: Optional path to save dataset

    Returns:
        List of calibration images as numpy arrays
    """
    generator = CalibrationDatasetGenerator(data_dir, target_platform)
    return generator.generate_calibration_dataset(output_path)


if __name__ == "__main__":
    # Test calibration dataset generation
    print("Testing calibration dataset generation...")

    # Example usage
    try:
        # Use sample data directory (adjust path as needed)
        data_dir = "data/chest_xray"

        if Path(data_dir).exists():
            # Generate for tablet deployment
            calibration_data = create_calibration_dataset(
                data_dir,
                target_platform='android_tablet',
                output_path='outputs/calibration_android_tablet.npy'
            )

            print(f"Generated {len(calibration_data)} calibration samples")
            print(f"Sample shape: {calibration_data[0].shape}")
        else:
            print(f"Data directory {data_dir} not found - using dummy data")

            # Generate dummy calibration data for testing
            dummy_data = [np.random.randn(3, 224, 224).astype(np.float32) for _ in range(10)]
            print(f"Generated {len(dummy_data)} dummy calibration samples")

    except Exception as e:
        print(f"Error during testing: {e}")

    print("Calibration dataset utilities ready!")