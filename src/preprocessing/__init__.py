"""
Image Preprocessing Module for Pediatric Pneumonia Detection

This module provides image quality assessment and enhancement utilities
for medical chest X-ray images to improve model confidence and accuracy.

Available Functions:
- assess_image_quality: Evaluate image brightness, contrast, and quality
- apply_clahe_enhancement: Medical-grade contrast enhancement
- preprocess_medical_image: Complete preprocessing pipeline
- IntelligentPreprocessor: Advanced preprocessing with autoencoder support
"""

from .quality_assessment import assess_image_quality, get_quality_warnings
from .clahe_enhancement import apply_clahe_enhancement, preprocess_medical_image
from .intelligent_preprocessing import IntelligentPreprocessor, QualityAssessment, ImageQuality

__all__ = [
    'assess_image_quality',
    'get_quality_warnings',
    'apply_clahe_enhancement',
    'preprocess_medical_image',
    'IntelligentPreprocessor',
    'QualityAssessment',
    'ImageQuality'
]