#!/usr/bin/env python3
"""
Windows Desktop Export Script

Export trained MobileNet models to ONNX format for Windows tablet deployment.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_for_windows(model_path, model_type='mobilenet', output_dir='windows_exports'):
    """Export model for Windows desktop deployment."""

    logger.info(f"Starting Windows export for {model_type} model...")

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

    # Export to ONNX (full precision for Windows)
    onnx_path = output_path / f"{model_type}_windows.onnx"

    logger.info("Exporting to ONNX format for Windows...")

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )

    logger.info(f"Windows ONNX model exported: {onnx_path}")
    logger.info("Windows export completed!")


def main():
    parser = argparse.ArgumentParser(description='Export models for Windows desktop deployment')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--model_type', default='mobilenet', help='Model type')
    parser.add_argument('--output_dir', default='windows_exports', help='Output directory')

    args = parser.parse_args()

    export_for_windows(args.model_path, args.model_type, args.output_dir)


if __name__ == "__main__":
    main()