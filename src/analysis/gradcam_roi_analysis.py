#!/usr/bin/env python3
"""
GradCAM, ROI, and Patch-Based Analysis for Medical Image Interpretation

This module provides comprehensive analysis tools for understanding model decisions
and preprocessing effectiveness:
- GradCAM visualization for model attention
- Patch-based quality analysis
- Lung ROI detection and segmentation
- ROI-based targeted preprocessing
- Comprehensive visualization tools
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (GradCAM) for visualizing model attention.

    Shows which regions of the image the model focuses on when making predictions.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Initialize GradCAM.

        Args:
            model: PyTorch model to analyze
            target_layer: Layer to visualize (default: last conv layer)
        """
        self.model = model
        self.model.eval()

        # Auto-detect target layer if not specified
        if target_layer is None:
            target_layer = self._find_target_layer()

        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _find_target_layer(self) -> nn.Module:
        """Find the last convolutional layer in the model."""
        # For MobileNetFineTune, use conv_head
        if hasattr(self.model, 'conv_head'):
            logger.info("Using conv_head as target layer for GradCAM")
            return self.model.conv_head
        # For MobileNetV2, use the last layer in features
        elif hasattr(self.model, 'mobilenet') and hasattr(self.model.mobilenet, 'features'):
            return self.model.mobilenet.features[-1]
        else:
            raise ValueError("Could not auto-detect target layer. Please specify manually.")

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate GradCAM heatmap.

        Args:
            input_tensor: Input image tensor [1, 3, H, W]
            target_class: Target class index (None = predicted class)

        Returns:
            heatmap: GradCAM heatmap as numpy array [H, W]
        """
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)

        # Use predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass on target class
        output[0, target_class].backward()

        # Compute GradCAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU to focus on positive contributions
        cam = F.relu(cam)

        # Normalize to 0-1
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()

    def visualize(self,
                  image: Union[np.ndarray, Image.Image],
                  input_tensor: torch.Tensor,
                  target_class: Optional[int] = None,
                  alpha: float = 0.4) -> np.ndarray:
        """
        Generate visualization with GradCAM overlay.

        Args:
            image: Original image
            input_tensor: Preprocessed input tensor
            target_class: Target class for visualization
            alpha: Overlay transparency (0-1)

        Returns:
            visualization: RGB image with heatmap overlay
        """
        # Generate CAM
        cam = self.generate_cam(input_tensor, target_class)

        # Prepare image
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))

        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Normalize image to 0-255
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Overlay heatmap
        visualization = (1 - alpha) * image + alpha * heatmap
        visualization = np.uint8(visualization)

        return visualization


class PatchAnalyzer:
    """
    Patch-based quality analysis for medical images.

    Divides images into grid patches and analyzes quality metrics per patch.
    """

    def __init__(self, grid_size: Tuple[int, int] = (8, 8)):
        """
        Initialize patch analyzer.

        Args:
            grid_size: Grid dimensions (rows, cols)
        """
        self.grid_size = grid_size

    def analyze_patches(self, image: np.ndarray) -> Dict:
        """
        Analyze image quality metrics for each patch.

        Args:
            image: Grayscale image array

        Returns:
            analysis: Dictionary with patch-wise metrics
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        h, w = image.shape
        grid_h, grid_w = self.grid_size

        patch_h = h // grid_h
        patch_w = w // grid_w

        # Initialize metric grids
        brightness_grid = np.zeros(self.grid_size)
        contrast_grid = np.zeros(self.grid_size)
        sharpness_grid = np.zeros(self.grid_size)

        # Analyze each patch
        for i in range(grid_h):
            for j in range(grid_w):
                # Extract patch
                y_start = i * patch_h
                y_end = (i + 1) * patch_h
                x_start = j * patch_w
                x_end = (j + 1) * patch_w
                patch = image[y_start:y_end, x_start:x_end]

                # Calculate metrics
                brightness_grid[i, j] = np.mean(patch) / 255.0
                contrast_grid[i, j] = np.std(patch) / 255.0

                # Sharpness using Laplacian variance
                laplacian = cv2.Laplacian(patch, cv2.CV_64F)
                sharpness_grid[i, j] = laplacian.var()

        # Normalize sharpness to 0-1 range
        if sharpness_grid.max() > 0:
            sharpness_grid = sharpness_grid / sharpness_grid.max()

        return {
            'brightness': brightness_grid,
            'contrast': contrast_grid,
            'sharpness': sharpness_grid,
            'patch_size': (patch_h, patch_w),
            'grid_size': self.grid_size
        }

    def visualize_patches(self,
                         image: np.ndarray,
                         analysis: Optional[Dict] = None) -> np.ndarray:
        """
        Visualize patch-based analysis with grid overlay.

        Args:
            image: Original image
            analysis: Pre-computed analysis (will compute if None)

        Returns:
            visualization: Image with grid and quality overlay
        """
        if analysis is None:
            analysis = self.analyze_patches(image)

        # Convert to RGB if needed
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            vis = image.copy()

        if vis.max() <= 1.0:
            vis = (vis * 255).astype(np.uint8)

        h, w = image.shape[:2] if len(image.shape) == 2 else (image.shape[0], image.shape[1])
        grid_h, grid_w = self.grid_size
        patch_h, patch_w = analysis['patch_size']

        # Draw grid
        for i in range(1, grid_h):
            y = i * patch_h
            cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)

        for j in range(1, grid_w):
            x = j * patch_w
            cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1)

        return vis

    def create_heatmaps(self, analysis: Dict) -> Dict[str, np.ndarray]:
        """
        Create heatmap visualizations for each metric.

        Args:
            analysis: Patch analysis results

        Returns:
            heatmaps: Dictionary of colored heatmaps
        """
        heatmaps = {}

        for metric_name in ['brightness', 'contrast', 'sharpness']:
            metric_grid = analysis[metric_name]

            # Normalize to 0-255
            normalized = ((metric_grid - metric_grid.min()) /
                         (metric_grid.max() - metric_grid.min() + 1e-8) * 255).astype(np.uint8)

            # Apply colormap
            heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            heatmaps[metric_name] = heatmap

        return heatmaps


class LungROIDetector:
    """
    Detect lung regions of interest in chest X-rays using traditional CV methods.
    """

    def __init__(self):
        """Initialize lung ROI detector."""
        pass

    def detect_lung_roi(self, image: np.ndarray) -> Dict:
        """
        Detect lung regions using edge detection and morphological operations.

        Args:
            image: Grayscale chest X-ray image

        Returns:
            roi_info: Dictionary with ROI mask and bounding boxes
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Normalize to 0-255
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Apply CLAHE for better edge detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create lung mask (focus on center region)
        h, w = image.shape
        lung_mask = np.zeros((h, w), dtype=np.uint8)

        # Filter contours by size and position
        min_area = (h * w) * 0.05  # At least 5% of image

        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Check if contour is in central region
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # Lung regions should be in center 80% of image
                    if (w * 0.1 < cx < w * 0.9) and (h * 0.1 < cy < h * 0.9):
                        valid_contours.append(contour)

        # Draw valid contours on mask
        cv2.drawContours(lung_mask, valid_contours, -1, 255, -1)

        # Get bounding box of entire lung region
        if len(valid_contours) > 0:
            all_points = np.vstack(valid_contours)
            x, y, w_box, h_box = cv2.boundingRect(all_points)
            bbox = (x, y, w_box, h_box)
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

    def visualize_roi(self, image: np.ndarray, roi_info: Dict) -> np.ndarray:
        """
        Visualize detected ROI on image.

        Args:
            image: Original image
            roi_info: ROI detection results

        Returns:
            visualization: Image with ROI overlay
        """
        # Convert to RGB
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            vis = image.copy()

        if vis.max() <= 1.0:
            vis = (vis * 255).astype(np.uint8)

        # Create colored mask
        mask_colored = np.zeros_like(vis)
        mask_colored[roi_info['mask'] > 0] = [0, 255, 0]  # Green for lung region

        # Overlay with transparency
        vis = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)

        # Draw bounding box
        x, y, w, h = roi_info['bbox']
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw contours
        cv2.drawContours(vis, roi_info['contours'], -1, (0, 255, 255), 2)

        return vis


class ROIBasedPreprocessor:
    """
    Apply targeted preprocessing to lung ROI regions.
    """

    def __init__(self, roi_detector: Optional[LungROIDetector] = None):
        """
        Initialize ROI-based preprocessor.

        Args:
            roi_detector: LungROIDetector instance (creates new if None)
        """
        self.roi_detector = roi_detector or LungROIDetector()

    def preprocess_with_adaptive_roi(self,
                                     image: np.ndarray,
                                     quality_metrics: Optional[Dict] = None,
                                     roi_info: Optional[Dict] = None) -> Dict:
        """
        Apply adaptive ROI preprocessing based on image quality assessment.

        This improved version:
        - Skips ROI for good/excellent quality images (prevents degradation)
        - Adjusts enhancement strength based on brightness and contrast
        - Uses quality metrics to make intelligent decisions

        Args:
            image: Input grayscale image
            quality_metrics: Quality assessment metrics (brightness, contrast, sharpness)
            roi_info: Pre-computed ROI info (will detect if None)

        Returns:
            result: Dictionary with enhanced image and metadata
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Calculate quality metrics if not provided
        if quality_metrics is None:
            quality_metrics = self._assess_basic_quality(image)

        brightness = quality_metrics.get('brightness', 0.5)
        contrast = quality_metrics.get('contrast', 0.5)

        # Decision: Skip ROI for good quality images
        # Based on analysis: cases that got worse had moderate quality (brightness > 0.55, contrast > 0.09)
        if brightness > 0.55 and contrast > 0.09:
            # Good quality - use light global CLAHE instead of ROI
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)

            return {
                'enhanced_image': enhanced.astype(np.uint8),
                'roi_info': None,
                'enhancement_applied': {
                    'method': 'global_clahe',
                    'reason': 'good_quality_detected',
                    'roi_clip_limit': None,
                    'background_clip_limit': None,
                    'global_clip_limit': 2.0
                },
                'skipped_roi': True
            }

        # Detect ROI if not provided
        if roi_info is None:
            roi_info = self.roi_detector.detect_lung_roi(image)

        # Adaptive clip limits based on quality
        # Very dark images (brightness < 0.4) need strong enhancement
        if brightness < 0.4:
            roi_clip_limit = 3.5
        # Dark images (brightness < 0.5) need moderate-strong enhancement
        elif brightness < 0.5:
            roi_clip_limit = 3.0
        # Moderate brightness needs moderate enhancement
        else:
            roi_clip_limit = 2.5

        # Adjust for contrast
        if contrast < 0.1:
            # Very low contrast - boost enhancement
            roi_clip_limit += 0.5
        elif contrast < 0.12:
            # Low contrast - slight boost
            roi_clip_limit += 0.3

        # Cap maximum
        roi_clip_limit = min(roi_clip_limit, 4.0)

        # Background always half of ROI
        background_clip_limit = roi_clip_limit * 0.5

        # Create CLAHE processors
        roi_clahe = cv2.createCLAHE(clipLimit=roi_clip_limit, tileGridSize=(8, 8))
        bg_clahe = cv2.createCLAHE(clipLimit=background_clip_limit, tileGridSize=(8, 8))

        # Apply different enhancement to ROI vs background
        enhanced_roi = roi_clahe.apply(image)
        enhanced_bg = bg_clahe.apply(image)

        # Combine using mask
        mask = roi_info['mask']
        enhanced = np.where(mask > 0, enhanced_roi, enhanced_bg)

        return {
            'enhanced_image': enhanced.astype(np.uint8),
            'roi_info': roi_info,
            'enhancement_applied': {
                'method': 'adaptive_roi_clahe',
                'roi_clip_limit': float(roi_clip_limit),
                'background_clip_limit': float(background_clip_limit),
                'brightness_score': float(brightness),
                'contrast_score': float(contrast)
            },
            'skipped_roi': False
        }

    def _assess_basic_quality(self, image: np.ndarray) -> Dict:
        """Quick quality assessment for adaptive preprocessing."""
        # Brightness
        brightness = np.mean(image) / 255.0

        # Contrast (standard deviation)
        contrast = np.std(image) / 255.0

        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        sharpness = min(laplacian_var / 500.0, 1.0)  # Normalize

        return {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness
        }

    def preprocess_with_roi(self,
                           image: np.ndarray,
                           roi_info: Optional[Dict] = None,
                           roi_clip_limit: float = 3.0,
                           background_clip_limit: float = 1.5) -> Dict:
        """
        Apply different preprocessing to ROI vs background.

        Args:
            image: Input grayscale image
            roi_info: Pre-computed ROI info (will detect if None)
            roi_clip_limit: CLAHE clip limit for lung regions
            background_clip_limit: CLAHE clip limit for background

        Returns:
            result: Dictionary with enhanced image and metadata
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Detect ROI if not provided
        if roi_info is None:
            roi_info = self.roi_detector.detect_lung_roi(image)

        # Create CLAHE processors
        roi_clahe = cv2.createCLAHE(clipLimit=roi_clip_limit, tileGridSize=(8, 8))
        bg_clahe = cv2.createCLAHE(clipLimit=background_clip_limit, tileGridSize=(8, 8))

        # Apply different enhancement to ROI vs background
        enhanced_roi = roi_clahe.apply(image)
        enhanced_bg = bg_clahe.apply(image)

        # Combine using mask
        mask = roi_info['mask']
        enhanced = np.where(mask > 0, enhanced_roi, enhanced_bg)

        return {
            'enhanced_image': enhanced.astype(np.uint8),
            'roi_info': roi_info,
            'enhancement_applied': {
                'roi_clip_limit': roi_clip_limit,
                'background_clip_limit': background_clip_limit
            }
        }


def create_comprehensive_visualization(
    original_image: np.ndarray,
    preprocessed_image: np.ndarray,
    model: nn.Module,
    input_tensor: torch.Tensor,
    prediction: Dict,
    save_path: Optional[Path] = None
) -> np.ndarray:
    """
    Create comprehensive visualization with GradCAM, patches, and ROI.

    Args:
        original_image: Original X-ray image
        preprocessed_image: Preprocessed image
        model: PyTorch model
        input_tensor: Preprocessed input tensor
        prediction: Model prediction results
        save_path: Optional path to save visualization

    Returns:
        combined_vis: Combined visualization image
    """
    # Initialize analyzers
    gradcam = GradCAM(model)
    patch_analyzer = PatchAnalyzer(grid_size=(8, 8))
    roi_detector = LungROIDetector()

    # Generate visualizations
    gradcam_vis = gradcam.visualize(preprocessed_image, input_tensor)

    patch_analysis = patch_analyzer.analyze_patches(preprocessed_image)
    patch_heatmaps = patch_analyzer.create_heatmaps(patch_analysis)

    roi_info = roi_detector.detect_lung_roi(preprocessed_image)
    roi_vis = roi_detector.visualize_roi(preprocessed_image, roi_info)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Preprocessed image
    axes[0, 1].imshow(preprocessed_image, cmap='gray')
    axes[0, 1].set_title('Preprocessed Image')
    axes[0, 1].axis('off')

    # GradCAM
    axes[0, 2].imshow(gradcam_vis)
    pred_class = 'PNEUMONIA' if prediction['class'] == 0 else 'NORMAL'
    axes[0, 2].set_title(f'GradCAM - {pred_class} ({prediction["confidence"]:.1f}%)')
    axes[0, 2].axis('off')

    # Patch brightness heatmap
    axes[1, 0].imshow(patch_heatmaps['brightness'])
    axes[1, 0].set_title('Patch Brightness')
    axes[1, 0].axis('off')

    # Patch contrast heatmap
    axes[1, 1].imshow(patch_heatmaps['contrast'])
    axes[1, 1].set_title('Patch Contrast')
    axes[1, 1].axis('off')

    # ROI visualization
    axes[1, 2].imshow(roi_vis)
    axes[1, 2].set_title(f'Lung ROI ({roi_info["lung_area_ratio"]*100:.1f}% area)')
    axes[1, 2].axis('off')

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comprehensive visualization to {save_path}")

    # Convert to numpy array
    fig.canvas.draw()
    combined_vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    combined_vis = combined_vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return combined_vis
