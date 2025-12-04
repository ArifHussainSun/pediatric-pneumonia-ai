#!/usr/bin/env python3
"""
Intelligent Preprocessing Pipeline for Medical Image Quality Control
- Quality validation and user feedback
- Adaptive enhancement based on image characteristics
- Autoencoder-based noise reduction
- ROI-based contrast enhancement for lung regions
- False positive/negative reduction
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ImageQuality(Enum):
    """Image quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"

@dataclass
class QualityAssessment:
    """Comprehensive image quality assessment"""
    overall_quality: ImageQuality
    brightness_score: float
    contrast_score: float
    sharpness_score: float
    edge_density_score: float
    noise_level: float
    positioning_score: float
    artifacts_detected: bool
    recommendations: list
    should_enhance: bool
    should_reject: bool
    user_message: Optional[str] = None

class MedicalImageAutoencoder(nn.Module):
    """Autoencoder for medical image denoising and enhancement"""

    def __init__(self, input_channels=1):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, input_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class LungROIDetector:
    """
    Detect lung regions of interest in chest X-rays using traditional CV methods.

    Uses edge detection and morphological operations to identify lung areas
    for targeted preprocessing.
    """

    def __init__(self):
        pass

    def detect_lung_roi(self, image: np.ndarray) -> Dict:
        """
        Detect lung regions in chest X-ray.

        Args:
            image: Grayscale chest X-ray image

        Returns:
            dict with 'mask', 'bbox', 'contours', 'lung_area_ratio'
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        h, w = image.shape

        # Normalize image
        if image.max() > 1.0:
            normalized = image.astype(np.float32) / 255.0
        else:
            normalized = image.astype(np.float32)

        # Apply gentle blur to reduce noise
        blurred = cv2.GaussianBlur((normalized * 255).astype(np.uint8), (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 30, 100)

        # Morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Dilate to connect regions
        dilated = cv2.dilate(closed, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create lung mask
        lung_mask = np.zeros((h, w), dtype=np.uint8)

        # Filter contours by size and position
        min_area = (h * w) * 0.05  # At least 5% of image

        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Check if contour is in central region using centroid
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # Lung regions should be in center 80% of image
                    if (w * 0.1 < cx < w * 0.9) and (h * 0.1 < cy < h * 0.9):
                        valid_contours.append(contour)

        # Draw valid contours on mask
        if valid_contours:
            cv2.drawContours(lung_mask, valid_contours, -1, 255, -1)

            # Get bounding box
            all_points = np.vstack(valid_contours)
            x, y, cw, ch = cv2.boundingRect(all_points)
            bbox = (x, y, cw, ch)
        else:
            # Fallback to center region
            bbox = (int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6))
            lung_mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 255

        return {
            'mask': lung_mask,
            'bbox': bbox,
            'contours': valid_contours,
            'lung_area_ratio': np.sum(lung_mask > 0) / (h * w)
        }

class IntelligentPreprocessor:
    """Main preprocessing pipeline with quality control and enhancement"""

    def __init__(self, autoencoder_path: Optional[str] = None):
        self.autoencoder = None
        # Use the working ROI detector from analysis module
        try:
            from src.analysis.gradcam_roi_analysis import LungROIDetector as WorkingROIDetector
            self.roi_detector = WorkingROIDetector()
        except ImportError:
            # Fallback to local implementation
            self.roi_detector = LungROIDetector()
        if autoencoder_path:
            self.load_autoencoder(autoencoder_path)

    def load_autoencoder(self, model_path: str):
        """Load pre-trained autoencoder"""
        try:
            self.autoencoder = MedicalImageAutoencoder()
            self.autoencoder.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.autoencoder.eval()
            logger.info("Autoencoder loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load autoencoder: {e}")

    def assess_image_quality(self, image: Union[np.ndarray, Image.Image]) -> QualityAssessment:
        """Comprehensive image quality assessment with user feedback"""

        if isinstance(image, Image.Image):
            image = np.array(image.convert('L'))

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Normalize to 0-255
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Calculate quality metrics
        brightness = self._assess_brightness(image)
        contrast = self._assess_contrast(image)
        sharpness = self._assess_sharpness(image)
        edge_density = self._assess_edge_density(image)
        noise = self._assess_noise(image)
        positioning = self._assess_positioning(image)
        artifacts = self._detect_artifacts(image)

        # Determine overall quality and recommendations
        quality, recommendations, should_enhance, should_reject, user_message = self._determine_quality_action(
            brightness, contrast, sharpness, edge_density, noise, positioning, artifacts
        )

        return QualityAssessment(
            overall_quality=quality,
            brightness_score=brightness,
            contrast_score=contrast,
            sharpness_score=sharpness,
            edge_density_score=edge_density,
            noise_level=noise,
            positioning_score=positioning,
            artifacts_detected=artifacts,
            recommendations=recommendations,
            should_enhance=should_enhance,
            should_reject=should_reject,
            user_message=user_message
        )

    def _assess_brightness(self, image: np.ndarray) -> float:
        """Assess image brightness (0-1, 0.5 is optimal)"""
        mean_brightness = np.mean(image) / 255.0
        # Optimal brightness for chest X-rays is around 0.3-0.7
        if 0.3 <= mean_brightness <= 0.7:
            return 1.0
        elif 0.1 <= mean_brightness <= 0.9:
            return 0.7
        else:
            return 0.3

    def _assess_contrast(self, image: np.ndarray) -> float:
        """Assess image contrast using standard deviation"""
        std = np.std(image) / 255.0
        # Good contrast has std > 0.15
        if std > 0.2:
            return 1.0
        elif std > 0.1:
            return 0.7
        else:
            return 0.3

    def _assess_sharpness(self, image: np.ndarray) -> float:
        """Assess image sharpness using Laplacian variance"""
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        # Normalize sharpness score
        if laplacian_var > 500:
            return 1.0
        elif laplacian_var > 100:
            return 0.7
        else:
            return 0.3

    def _assess_edge_density(self, image: np.ndarray) -> float:
        """Assess edge density using Canny edge detection"""
        edges = cv2.Canny(image, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = image.shape[0] * image.shape[1]
        edge_density = edge_pixels / total_pixels

        # Normalize edge density score
        if edge_density > 0.02:
            return 1.0
        elif edge_density > 0.01:
            return 0.7
        else:
            return 0.3

    def _assess_noise(self, image: np.ndarray) -> float:
        """Assess noise level (0 = no noise, 1 = very noisy)"""
        # Use bilateral filter to estimate noise
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        noise_estimate = np.mean(np.abs(image.astype(float) - filtered.astype(float))) / 255.0
        return min(noise_estimate * 5, 1.0)  # Scale to 0-1

    def _assess_positioning(self, image: np.ndarray) -> float:
        """Assess if chest is properly positioned (simplified)"""
        h, w = image.shape
        center_region = image[h//4:3*h//4, w//4:3*w//4]
        edge_region = np.concatenate([
            image[:h//8, :].flatten(),
            image[-h//8:, :].flatten(),
            image[:, :w//8].flatten(),
            image[:, -w//8:].flatten()
        ])

        # Good positioning has content in center, less in edges
        center_intensity = np.mean(center_region)
        edge_intensity = np.mean(edge_region)

        if center_intensity > edge_intensity * 1.2:
            return 1.0
        elif center_intensity > edge_intensity:
            return 0.7
        else:
            return 0.5

    def _detect_artifacts(self, image: np.ndarray) -> bool:
        """Detect common imaging artifacts"""
        # Check for excessive saturation (clipped regions)
        clipped_pixels = np.sum((image == 0) | (image == 255))
        total_pixels = image.size
        clipped_ratio = clipped_pixels / total_pixels

        return clipped_ratio > 0.05  # More than 5% clipped pixels

    def _determine_quality_action(self, brightness: float, contrast: float, sharpness: float,
                                edge_density: float, noise: float, positioning: float, artifacts: bool) -> Tuple:
        """Determine overall quality and required actions"""

        # Calculate weighted overall score
        weights = {
            'brightness': 0.15,
            'contrast': 0.20,
            'sharpness': 0.20,
            'edge_density': 0.15,
            'noise': -0.15,  # Negative because high noise is bad
            'positioning': 0.15
        }

        overall_score = (
            weights['brightness'] * brightness +
            weights['contrast'] * contrast +
            weights['sharpness'] * sharpness +
            weights['edge_density'] * edge_density +
            weights['noise'] * (1 - noise) +  # Invert noise
            weights['positioning'] * positioning
        )

        if artifacts:
            overall_score *= 0.8  # Penalize artifacts

        recommendations = []
        user_message = None

        # Determine quality level and actions
        if overall_score >= 0.8:
            quality = ImageQuality.EXCELLENT
            should_enhance = False
            should_reject = False
        elif overall_score >= 0.65:
            quality = ImageQuality.GOOD
            should_enhance = False
            should_reject = False
        elif overall_score >= 0.45:
            quality = ImageQuality.ACCEPTABLE
            should_enhance = True
            should_reject = False
            recommendations.append("Image will be enhanced for better analysis")
        elif overall_score >= 0.25:
            quality = ImageQuality.POOR
            should_enhance = True
            should_reject = False
            recommendations.append("Poor image quality detected - enhancement applied")
            user_message = "Image quality is poor. Consider retaking with better lighting and positioning for optimal results."
        else:
            quality = ImageQuality.UNACCEPTABLE
            should_enhance = False
            should_reject = True
            user_message = "Image quality too poor for reliable analysis. Please retake with:\n" + \
                          "• Better lighting\n• Proper positioning\n• Stable camera\n• Clear image without artifacts"

        # Add specific recommendations
        if brightness < 0.5:
            recommendations.append("Increase lighting")
        if contrast < 0.5:
            recommendations.append("Improve contrast")
        if sharpness < 0.5:
            recommendations.append("Ensure image is in focus")
        if edge_density < 0.5:
            recommendations.append("Improve image detail and clarity")
        if noise > 0.6:
            recommendations.append("Reduce camera noise")
        if positioning < 0.6:
            recommendations.append("Center the chest in the image")
        if artifacts:
            recommendations.append("Avoid overexposure and artifacts")

        return quality, recommendations, should_enhance, should_reject, user_message

    def enhance_image(self, image: Union[np.ndarray, Image.Image],
                     quality_assessment: QualityAssessment) -> np.ndarray:
        """
        Apply VERY conservative adaptive enhancement based on quality assessment.

        Key principle: Baseline (no enhancement) achieves 98.40% accuracy.
        ROI-based CLAHE with high clip limits (3.0) caused accuracy to drop to 89.60%.

        Strategy:
        - Only apply minimal enhancement to extremely poor quality images
        - Use much lower clip limits to avoid over-enhancement artifacts
        - Skip enhancement for most images to preserve baseline performance
        """

        if isinstance(image, Image.Image):
            image = np.array(image.convert('L'))

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # CONSERVATIVE PREPROCESSING DECISION TREE
        contrast = quality_assessment.contrast_score
        brightness = quality_assessment.brightness_score
        sharpness = quality_assessment.sharpness_score
        edge_density = quality_assessment.edge_density_score

        # Case 1: VERY low contrast (<0.10) - These benefit from minimal ROI-based CLAHE
        if contrast < 0.10:
            logger.info(f"Very low contrast ({contrast:.3f}) detected - applying minimal ROI-based CLAHE")
            return self._apply_adaptive_clahe(image, mode='minimal_roi')

        # Case 2: Low contrast + dark (<0.15 contrast AND <0.40 brightness) - Light global CLAHE
        elif contrast < 0.15 and brightness < 0.40:
            logger.info(f"Low contrast ({contrast:.3f}) + dark ({brightness:.3f}) - applying light global CLAHE")
            return self._apply_adaptive_clahe(image, mode='light_global')

        # Case 3: Extremely dark (<0.25 brightness) - Brightness correction only
        elif brightness < 0.25:
            logger.info(f"Extremely dark ({brightness:.3f}) - applying brightness correction")
            return self._correct_brightness(image, brightness)

        # Case 4 (NEW): Very low sharpness AND edge density - Light global CLAHE
        # Based on FN analysis: 13/21 (61.9%) false negatives had low sharpness/edge density
        # These blurry pneumonia cases have acceptable contrast but need enhancement
        # Testing results (1000-image validation):
        #   Baseline (no Case 4): 21 FN, 3 FP, 97.60% accuracy
        #   0.35 OR + medium (2.5): 35 FN, 25 FP, 94.00% (fixed 8, broke 22) ❌
        #   0.15 AND + medium (2.5): 21 FN, 3 FP, 97.60% (no change)
        #   0.30 AND + light (2.0): 20 FN, 4 FP, 97.60% (fixed 6, broke 5) ← BEST
        #   0.30 AND + medium (2.5): 24 FN, 6 FP, 97.00% (net -3) ❌
        #   0.30 AND + contrast filter: 25 FN, 3 FP, 97.20% (net -4) ❌
        # CONCLUSION: 0.30 AND + light CLAHE gives marginal +1 improvement
        elif sharpness <= 0.30 and edge_density <= 0.30:
            logger.info(f"Very low sharpness ({sharpness:.3f}) AND edge density ({edge_density:.3f}) - applying light global CLAHE")
            return self._apply_adaptive_clahe(image, mode='light_global')

        # Case 5: DEFAULT - NO enhancement (preserves 98.40% baseline accuracy)
        else:
            logger.info(f"Acceptable quality (C={contrast:.3f}, B={brightness:.3f}, S={sharpness:.3f}, E={edge_density:.3f}) - no enhancement applied")
            return image

    def _apply_autoencoder_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply autoencoder for denoising"""
        if not self.autoencoder:
            return image

        try:
            # Prepare image for autoencoder
            original_shape = image.shape
            image_tensor = torch.FloatTensor(image / 255.0).unsqueeze(0).unsqueeze(0)

            # Resize to standard size if needed (e.g., 256x256)
            if image_tensor.shape[-1] != 256 or image_tensor.shape[-2] != 256:
                image_tensor = F.interpolate(image_tensor, size=(256, 256), mode='bilinear')

            with torch.no_grad():
                denoised = self.autoencoder(image_tensor)

            # Convert back and resize to original
            denoised = denoised.squeeze().numpy() * 255
            if original_shape != (256, 256):
                denoised = cv2.resize(denoised, (original_shape[1], original_shape[0]))

            return denoised.astype(np.uint8)

        except Exception as e:
            logger.warning(f"Autoencoder denoising failed: {e}")
            return image

    def _apply_adaptive_clahe(self, image: np.ndarray, mode: str = 'light_global', clip_limit: float = 2.0) -> np.ndarray:
        """
        Apply CLAHE with different modes based on image quality.

        Based on validation analysis:
        - Original ROI approach (clip=3.0/1.5) dropped accuracy from 98.40% to 89.60%
        - Cause: Over-enhancement creating pneumonia-like artifacts in normal lungs
        - Solution: Much more conservative clip limits and selective application

        Modes:
            'minimal_roi': Light ROI-based CLAHE for extremely low contrast images
                          (roi_clip=1.5, bg_clip=1.0)
            'light_global': Light global CLAHE for dark low-contrast images
                           (clip=2.0)
            'medium_global': Medium global CLAHE for blurry/low-detail images
                            (clip=2.5)
            'none': No enhancement (fallback)

        Args:
            image: Grayscale chest X-ray image
            mode: Enhancement mode to apply
            clip_limit: Legacy parameter for backward compatibility

        Returns:
            Enhanced image
        """
        if mode == 'minimal_roi':
            try:
                # Detect lung ROI
                roi_info = self.roi_detector.detect_lung_roi(image)

                # CONSERVATIVE clip limits (increased from 1.2/0.8, still much lower than original 3.0/1.5)
                # These values avoid creating artifacts while still helping very low contrast cases
                roi_clip_limit = 1.5  # Light enhancement for lung regions
                background_clip_limit = 1.0  # Light enhancement for background

                # Create CLAHE processors
                roi_clahe = cv2.createCLAHE(clipLimit=roi_clip_limit, tileGridSize=(8, 8))
                bg_clahe = cv2.createCLAHE(clipLimit=background_clip_limit, tileGridSize=(8, 8))

                # Apply to respective regions
                enhanced_roi = roi_clahe.apply(image)
                enhanced_bg = bg_clahe.apply(image)

                # Combine using lung mask
                mask = roi_info['mask']
                enhanced = np.where(mask > 0, enhanced_roi, enhanced_bg)

                logger.info(f"Applied light ROI-based CLAHE (roi={roi_clip_limit}, bg={background_clip_limit})")
                return enhanced.astype(np.uint8)

            except Exception as e:
                logger.warning(f"Minimal ROI-based CLAHE failed, falling back to light global: {e}")
                mode = 'light_global'  # Fall through to light global

        if mode == 'light_global':
            # Light global CLAHE - conservative enhancement (increased from 1.5 to 2.0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            logger.info("Applied light global CLAHE (clip=2.0)")
            return clahe.apply(image)

        if mode == 'medium_global':
            # Medium global CLAHE - for blurry pneumonia cases with acceptable contrast
            # Based on FN analysis: 14/21 false negatives benefited from clip=3.0
            # Using 2.5 as a balanced approach (less aggressive than analysis, more than light)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            logger.info("Applied medium global CLAHE (clip=2.5)")
            return clahe.apply(image)

        # Fallback: no enhancement
        logger.info("No CLAHE enhancement applied")
        return image

    def _correct_brightness(self, image: np.ndarray, brightness_score: float) -> np.ndarray:
        """Correct brightness based on assessment"""
        if brightness_score < 0.3:  # Too dark
            gamma = 0.7  # Brighten
        elif brightness_score > 0.8:  # Too bright
            gamma = 1.3  # Darken
        else:
            return image

        # Apply gamma correction
        gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, gamma_table)

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction"""
        return cv2.bilateralFilter(image, 9, 75, 75)

    def _apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp masking for sharpening"""
        gaussian = cv2.GaussianBlur(image, (0, 0), 1.0)
        return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

    def process_image(self, image: Union[np.ndarray, Image.Image]) -> Dict:
        """Complete preprocessing pipeline"""

        # Step 1: Quality assessment
        quality = self.assess_image_quality(image)

        # Step 2: Decision based on quality
        if quality.should_reject:
            return {
                'success': False,
                'quality_assessment': quality,
                'enhanced_image': None,
                'user_message': quality.user_message,
                'should_retry': True
            }

        # Step 3: Enhancement if needed
        if quality.should_enhance:
            enhanced_image = self.enhance_image(image, quality)

            # Re-assess enhanced image
            enhanced_quality = self.assess_image_quality(enhanced_image)

            return {
                'success': True,
                'original_quality': quality,
                'enhanced_quality': enhanced_quality,
                'enhanced_image': enhanced_image,
                'user_message': quality.user_message,
                'enhancement_applied': True,
                'should_retry': False
            }
        else:
            # No enhancement needed
            if isinstance(image, Image.Image):
                image_array = np.array(image.convert('L'))
            else:
                image_array = image

            return {
                'success': True,
                'original_quality': quality,
                'enhanced_quality': quality,
                'enhanced_image': image_array,
                'user_message': None,
                'enhancement_applied': False,
                'should_retry': False
            }