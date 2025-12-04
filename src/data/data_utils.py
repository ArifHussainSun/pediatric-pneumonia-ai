"""
Data utilities for pediatric pneumonia detection.

This module contains comprehensive data processing utilities including:
- Image augmentation for medical X-rays
- Statistical feature extraction
- ROI (Region of Interest) analysis
- Data splitting with class balance
- Batch processing for feature extraction

Key features:
- Efficient OpenCV-based augmentation pipeline
- Comprehensive statistical feature sets
- Spatial analysis through ROI grids
- Medical data-specific preprocessing
"""

import cv2
import numpy as np
import random
import os
import pandas as pd
from pathlib import Path
import shutil
from scipy import stats
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union


def augment_image_efficient(image: np.ndarray) -> np.ndarray:
    """
    Apply efficient random augmentation techniques to chest X-ray images.

    Optimized for medical imaging with denoising and realistic transformations
    that preserve medical accuracy.

    Args:
        image (np.ndarray): Input grayscale X-ray image

    Returns:
        np.ndarray: Processed and augmented image
    """
    # Ensure image is in the right format
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image.squeeze()
    h, w = image.shape
    image = image.astype(np.uint8)

    # Step 1: Remove noise from X-ray (medical images often have noise)
    denoised_image = cv2.fastNlMeansDenoising(
        image, None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # Step 2: Random horizontal flip (50% chance)
    # Safe for chest X-rays as lungs are symmetric
    if random.getrandbits(1):
        cv2.flip(denoised_image, 1, denoised_image)

    # Step 3: Apply random shear transformation
    # Simulates variations in patient positioning
    vertical_shear = np.tan(np.radians(random.uniform(0, 10)))
    horizontal_shear = np.tan(np.radians(random.uniform(0, 10)))

    shear_matrix = np.array([
        [1, horizontal_shear, 0],
        [vertical_shear, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    denoised_image = cv2.warpPerspective(
        denoised_image, shear_matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Step 4: Random rotation (-3 to +3 degrees)
    # Small rotations for realistic patient positioning variations
    rotation_angle = random.uniform(-3, 3)
    rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), rotation_angle, 1.0)
    denoised_image = cv2.warpAffine(
        denoised_image, rotation_matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return denoised_image


def extract_features(image: np.ndarray) -> Dict[str, float]:
    """
    Extract comprehensive statistical features from chest X-ray images.

    Computes statistical properties useful for traditional ML algorithms
    and ensemble methods.

    Args:
        image (np.ndarray): Grayscale image as 2D array

    Returns:
        dict: Statistical features including basic stats, distribution measures,
              percentiles, energy, and entropy
    """
    pixels = image.flatten().astype(float)
    features = {}

    # Basic statistical measures
    features['maximum'] = np.max(pixels)
    features['minimum'] = np.min(pixels)
    features['mean'] = np.mean(pixels)
    features['median'] = np.median(pixels)
    features['std_dev'] = np.std(pixels)

    # Mode (most common pixel value)
    features['mode'] = stats.mode(pixels, keepdims=True)[0][0]

    # Distribution shape measures
    if np.std(pixels) < 1e-10:
        features['skewness'] = 0.0
        features['kurtosis'] = 0.0
    else:
        skew_val = stats.skew(pixels)
        features['skewness'] = 0.0 if np.isnan(skew_val) else skew_val

        kurt_val = stats.kurtosis(pixels)
        features['kurtosis'] = 0.0 if np.isnan(kurt_val) else kurt_val

    # Percentile features
    features['quantile_2.5'] = np.percentile(pixels, 2.5)
    features['quantile_5'] = np.percentile(pixels, 5)
    features['quantile_10'] = np.percentile(pixels, 10)
    features['quantile_90'] = np.percentile(pixels, 90)
    features['quantile_95'] = np.percentile(pixels, 95)
    features['quantile_97.5'] = np.percentile(pixels, 97.5)

    # Energy measure
    features['absolute_energy'] = np.sum(pixels ** 2)

    # Entropy (measure of texture/randomness)
    hist, _ = np.histogram(pixels, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    features['entropy'] = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0

    return features


def extract_roi_features(image: np.ndarray, grid_size: Tuple[int, int] = (4, 4)) -> Dict[str, float]:
    """
    Extract statistical features from each ROI in a grid pattern.

    Provides spatial information by analyzing different regions of the X-ray
    separately, which can help identify localized pneumonia patterns.

    Args:
        image (np.ndarray): Grayscale image as 2D array
        grid_size (tuple): Grid dimensions (rows, cols) for ROI division

    Returns:
        dict: Statistical features for each ROI region
    """
    height, width = image.shape
    rows, cols = grid_size

    roi_height = height // rows
    roi_width = width // cols

    # Extract all ROIs first
    roi_pixels_list = []
    for row in range(rows):
        for col in range(cols):
            start_row = row * roi_height
            end_row = start_row + roi_height
            start_col = col * roi_width
            end_col = start_col + roi_width

            roi = image[start_row:end_row, start_col:end_col]
            roi_pixels = roi.flatten().astype(np.float32)
            roi_pixels_list.append(roi_pixels)

    # Extract features from each ROI
    all_features = {}
    percentiles = [2.5, 5, 10, 50, 90, 95, 97.5]

    for roi_idx, pixels in enumerate(roi_pixels_list):
        roi_num = roi_idx + 1

        # Basic statistics
        features = {
            f'maximum_roi_{roi_num}': np.max(pixels),
            f'minimum_roi_{roi_num}': np.min(pixels),
            f'mean_roi_{roi_num}': np.mean(pixels),
            f'std_dev_roi_{roi_num}': np.std(pixels)
        }

        # Percentiles
        perc_values = np.percentile(pixels, percentiles)
        features.update({
            f'quantile_2.5_roi_{roi_num}': perc_values[0],
            f'quantile_5_roi_{roi_num}': perc_values[1],
            f'quantile_10_roi_{roi_num}': perc_values[2],
            f'median_roi_{roi_num}': perc_values[3],
            f'quantile_90_roi_{roi_num}': perc_values[4],
            f'quantile_95_roi_{roi_num}': perc_values[5],
            f'quantile_97.5_roi_{roi_num}': perc_values[6],
        })

        # Energy
        features[f'absolute_energy_roi_{roi_num}'] = np.sum(pixels * pixels)

        # Handle uniform regions efficiently
        std_val = features[f'std_dev_roi_{roi_num}']
        if std_val < 1e-10:
            features[f'mode_roi_{roi_num}'] = features[f'mean_roi_{roi_num}']
            features[f'skewness_roi_{roi_num}'] = 0.0
            features[f'kurtosis_roi_{roi_num}'] = 0.0
            features[f'entropy_roi_{roi_num}'] = 0.0
        else:
            # Mode
            try:
                mode_result = stats.mode(pixels, keepdims=True)
                features[f'mode_roi_{roi_num}'] = mode_result[0][0]
            except:
                features[f'mode_roi_{roi_num}'] = features[f'mean_roi_{roi_num}']

            # Distribution shape
            skew_val = stats.skew(pixels)
            features[f'skewness_roi_{roi_num}'] = 0.0 if np.isnan(skew_val) else skew_val

            kurt_val = stats.kurtosis(pixels)
            features[f'kurtosis_roi_{roi_num}'] = 0.0 if np.isnan(kurt_val) else kurt_val

            # Entropy for reasonably sized ROIs
            if len(pixels) > 100:
                hist, _ = np.histogram(pixels, bins=128, range=(0, 255), density=True)
                hist = hist[hist > 1e-10]
                if len(hist) > 0:
                    features[f'entropy_roi_{roi_num}'] = -np.sum(hist * np.log2(hist))
                else:
                    features[f'entropy_roi_{roi_num}'] = 0.0
            else:
                features[f'entropy_roi_{roi_num}'] = 0.0

        all_features.update(features)

    return all_features


class DataSplitter:
    """
    Utility for splitting medical image datasets into training and testing sets.

    Maintains class balance and ensures proper data separation for ML training.
    Designed specifically for binary medical classification tasks.
    """

    def __init__(self, base_dir: Union[str, Path] = ".", train_ratio: float = 0.8):
        """
        Initialize the data splitter.

        Args:
            base_dir: Directory containing train and test folders
            train_ratio: Fraction of data for training (0.8 = 80% train, 20% test)
        """
        self.base_dir = Path(base_dir)
        self.train_ratio = train_ratio
        self.test_ratio = 1 - train_ratio

        # Directory paths
        self.original_train_dir = self.base_dir / "train"
        self.original_test_dir = self.base_dir / "test"
        self.new_train_dir = self.base_dir / "new_train"
        self.new_test_dir = self.base_dir / "new_test"

        # Medical image classes
        self.classes = ["NORMAL", "PNEUMONIA"]

    def aggregate_files(self) -> Dict[str, List[Tuple[Path, str]]]:
        """Collect all image files from both train and test directories."""
        aggregated = {"NORMAL": [], "PNEUMONIA": []}

        for class_name in self.classes:
            # Training directory
            train_class_dir = self.original_train_dir / class_name
            if train_class_dir.exists():
                files = (list(train_class_dir.glob("*.jpeg")) +
                        list(train_class_dir.glob("*.jpg")) +
                        list(train_class_dir.glob("*.png")))
                aggregated[class_name].extend([(f, "train") for f in files])

            # Test directory
            test_class_dir = self.original_test_dir / class_name
            if test_class_dir.exists():
                files = (list(test_class_dir.glob("*.jpeg")) +
                        list(test_class_dir.glob("*.jpg")) +
                        list(test_class_dir.glob("*.png")))
                aggregated[class_name].extend([(f, "test") for f in files])

        return aggregated

    def calculate_split_counts(self, pneumonia_count: int, normal_count: int) -> Dict[str, Dict[str, int]]:
        """
        Calculate optimal split counts maintaining class balance.

        Args:
            pneumonia_count: Total pneumonia images
            normal_count: Total normal images

        Returns:
            Dictionary with split counts for each class
        """
        target_pneumonia_test = int(pneumonia_count * self.test_ratio)
        target_pneumonia_train = pneumonia_count - target_pneumonia_test

        # Match normal test count to pneumonia for balanced testing
        target_normal_test = target_pneumonia_test
        target_normal_train = normal_count - target_normal_test

        return {
            "pneumonia": {"train": target_pneumonia_train, "test": target_pneumonia_test},
            "normal": {"train": target_normal_train, "test": target_normal_test}
        }

    def create_directories(self):
        """Create new directory structure for split data."""
        for dir_path in [self.new_train_dir, self.new_test_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)

        for split in ["new_train", "new_test"]:
            for class_name in self.classes:
                dir_path = self.base_dir / split / class_name
                dir_path.mkdir(parents=True, exist_ok=True)

    def split_data(self) -> Dict[str, Dict[str, int]]:
        """
        Split data according to specified ratio.

        Returns:
            Dictionary with actual split counts
        """
        print("Aggregating files...")
        aggregated = self.aggregate_files()

        pneumonia_files = aggregated["PNEUMONIA"]
        normal_files = aggregated["NORMAL"]

        print(f"Total PNEUMONIA files: {len(pneumonia_files)}")
        print(f"Total NORMAL files: {len(normal_files)}")

        split_counts = self.calculate_split_counts(len(pneumonia_files), len(normal_files))

        print("\nTarget split counts:")
        print(f"PNEUMONIA - Train: {split_counts['pneumonia']['train']}, Test: {split_counts['pneumonia']['test']}")
        print(f"NORMAL - Train: {split_counts['normal']['train']}, Test: {split_counts['normal']['test']}")

        self.create_directories()

        # Shuffle for random distribution
        random.shuffle(pneumonia_files)
        random.shuffle(normal_files)

        self._copy_split_files(pneumonia_files, normal_files, split_counts)

        print("\nData split completed successfully!")
        return split_counts

    def _copy_split_files(self, pneumonia_files: List[Tuple[Path, str]],
                         normal_files: List[Tuple[Path, str]],
                         split_counts: Dict[str, Dict[str, int]]):
        """Copy files to designated directories."""
        print("\nCopying files...")

        # Split files
        pneumonia_train = pneumonia_files[:split_counts["pneumonia"]["train"]]
        pneumonia_test = pneumonia_files[split_counts["pneumonia"]["train"]:]

        normal_test = normal_files[:split_counts["normal"]["test"]]
        normal_train = normal_files[split_counts["normal"]["test"]:]

        # Copy to directories
        for file_path, _ in pneumonia_train:
            dest = self.new_train_dir / "PNEUMONIA" / file_path.name
            shutil.copy2(file_path, dest)

        for file_path, _ in pneumonia_test:
            dest = self.new_test_dir / "PNEUMONIA" / file_path.name
            shutil.copy2(file_path, dest)

        for file_path, _ in normal_train:
            dest = self.new_train_dir / "NORMAL" / file_path.name
            shutil.copy2(file_path, dest)

        for file_path, _ in normal_test:
            dest = self.new_test_dir / "NORMAL" / file_path.name
            shutil.copy2(file_path, dest)


def extract_all_features(input_dir: str, output_file: str,
                        roi_scheme: Tuple[int, int] = (16, 16)) -> pd.DataFrame:
    """
    Extract features from all images in dataset and save to CSV.

    Processes entire directories of X-ray images and creates feature files
    suitable for traditional machine learning algorithms.

    Args:
        input_dir: Directory containing NORMAL and PNEUMONIA subdirectories
        output_file: Path for output CSV file
        roi_scheme: Grid size for ROI analysis

    Returns:
        pd.DataFrame: Feature dataframe with extracted features
    """
    source_dir = os.path.join(os.getcwd(), input_dir)
    normal_dir = os.path.join(source_dir, 'NORMAL')
    pneumonia_dir = os.path.join(source_dir, 'PNEUMONIA')

    data = []
    count = 0

    print(f"Extracting features using {roi_scheme[0]}x{roi_scheme[1]} ROI scheme...")

    # Process NORMAL images
    print("Processing NORMAL images...")
    if os.path.exists(normal_dir):
        normal_files = [f for f in os.listdir(normal_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for filename in tqdm(normal_files, desc="NORMAL"):
            path = os.path.join(normal_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            features = extract_roi_features(img, grid_size=roi_scheme)
            features['image_id'] = filename
            features['label'] = 1  # NORMAL = 1
            data.append(features)
            count += 1

    # Process PNEUMONIA images
    print("Processing PNEUMONIA images...")
    if os.path.exists(pneumonia_dir):
        pneumonia_files = [f for f in os.listdir(pneumonia_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for filename in tqdm(pneumonia_files, desc="PNEUMONIA"):
            path = os.path.join(pneumonia_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            features = extract_roi_features(img, grid_size=roi_scheme)
            features['image_id'] = filename
            features['label'] = 0  # PNEUMONIA = 0
            data.append(features)
            count += 1

    # Create and save DataFrame
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

    print(f"\nFeature extraction completed!")
    print(f"Total images processed: {count}")
    print(f"Features per image: {len(df.columns) - 2}")
    print(f"Output saved to: {output_file}")

    return df


def load_image_safe(image_path: Union[str, Path], grayscale: bool = True) -> Optional[np.ndarray]:
    """
    Safely load an image with error handling.

    Args:
        image_path: Path to image file
        grayscale: Whether to load as grayscale

    Returns:
        np.ndarray or None: Loaded image or None if failed
    """
    try:
        if grayscale:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        return None


def get_dataset_stats(data_dir: Union[str, Path]) -> Dict[str, int]:
    """
    Get statistics about a medical image dataset.

    Args:
        data_dir: Root directory containing class subdirectories

    Returns:
        dict: Dataset statistics including class counts
    """
    data_dir = Path(data_dir)
    stats = {}

    for class_name in ["NORMAL", "PNEUMONIA"]:
        class_dir = data_dir / class_name
        if class_dir.exists():
            files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
            stats[class_name] = len(files)
        else:
            stats[class_name] = 0

    stats['total'] = sum(stats.values())
    return stats


if __name__ == "__main__":
    # Test data utilities
    print("Testing data utilities...")

    # Create sample image for testing
    sample_img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)

    # Test feature extraction
    features = extract_features(sample_img)
    print(f"Extracted {len(features)} global features")

    # Test ROI features
    roi_features = extract_roi_features(sample_img, grid_size=(4, 4))
    print(f"Extracted {len(roi_features)} ROI features")

    # Test augmentation
    augmented = augment_image_efficient(sample_img)
    print(f"Augmentation successful: {augmented.shape}")

    print("Data utilities ready for use!")