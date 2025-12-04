#!/usr/bin/env python3
"""
Android Model Export Script

Export trained MobileNet models to TensorFlow Lite format for Android deployment.
Handles model conversion, quantization, and packaging for mobile devices.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.models import create_model
from src.mobile.tflite_export import TensorFlowLiteExporter
from src.mobile.calibration_dataset import create_calibration_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_for_android(model_path, model_type='mobilenet', output_dir='android_exports'):
    """Export model for Android deployment."""

    logger.info(f"Starting Android export for {model_type} model...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = create_model(model_type, num_classes=2)

    # Load trained weights
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Create TensorFlow Lite exporter
    exporter = TensorFlowLiteExporter(model, model_type)

    # Export for Android tablets
    tablet_path = output_path / f"{model_type}_android_tablet.tflite"
    success = exporter.export_to_tflite(
        tablet_path,
        target_platform='android_tablet'
    )

    if success:
        logger.info(f"Android tablet model exported: {tablet_path}")
    else:
        logger.error("Android tablet export failed")

    # Export for Android phones
    phone_path = output_path / f"{model_type}_android_phone.tflite"
    success = exporter.export_to_tflite(
        phone_path,
        target_platform='android_phone'
    )

    if success:
        logger.info(f"Android phone model exported: {phone_path}")
    else:
        logger.error("Android phone export failed")

    logger.info("Android export completed!")


def main():
    parser = argparse.ArgumentParser(description='Export models for Android deployment')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--model_type', default='mobilenet', help='Model type')
    parser.add_argument('--output_dir', default='android_exports', help='Output directory')

    args = parser.parse_args()

    export_for_android(args.model_path, args.model_type, args.output_dir)


if __name__ == "__main__":
    main()