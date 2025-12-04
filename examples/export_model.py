#!/usr/bin/env python3
"""
Example script for exporting trained pneumonia detection models.

This script demonstrates how to:
- Export models to different formats (ONNX, TorchScript)
- Configure exports for different deployment scenarios
- Validate exported models
- Generate deployment packages

Usage:
    python examples/export_model.py --model_path outputs/xception_model.pth --model_type xception
    python examples/export_model.py --model_path outputs/mobilenet_model.pth --model_type mobilenet --deployment mobile_devices
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.export import ModelExporter, export_model, validate_exported_model
from src.export.config_manager import ExportConfigManager
from src.models import create_model

import torch
import json


def main():
    parser = argparse.ArgumentParser(description='Export a trained pneumonia detection model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--model_type', type=str, required=True,
                       help='Model architecture type (xception, vgg, mobilenet, etc.)')
    parser.add_argument('--output_dir', type=str, default='exports',
                       help='Output directory for exported models')
    parser.add_argument('--formats', nargs='+', default=['onnx', 'torchscript'],
                       choices=['onnx', 'torchscript', 'coreml', 'tflite'],
                       help='Export formats')
    parser.add_argument('--deployment', type=str, default='cloud_inference',
                       help='Deployment target (cloud_inference, mobile_devices, etc.)')
    parser.add_argument('--platform', type=str,
                       help='Target platform (ios, android, windows, etc.)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate exported models')
    parser.add_argument('--optimize', action='store_true', default=True,
                       help='Apply optimizations during export')

    args = parser.parse_args()

    print(f"Exporting {args.model_type} model from {args.model_path}")
    print(f"Target deployment: {args.deployment}")
    print(f"Export formats: {args.formats}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration manager
    config_manager = ExportConfigManager()

    # Validate export request
    for format_type in args.formats:
        validation = config_manager.validate_export_request(
            args.model_type, format_type, args.deployment
        )

        if not validation['valid']:
            print(f"ERROR: Invalid export configuration for {format_type}")
            for error in validation['errors']:
                print(f"  - {error}")
            continue

        if validation['warnings']:
            print(f"WARNINGS for {format_type} export:")
            for warning in validation['warnings']:
                print(f"  - {warning}")

        if validation['recommendations']:
            print(f"RECOMMENDATIONS for {format_type} export:")
            for rec in validation['recommendations']:
                print(f"  - {rec}")

    # Load model
    print(f"Loading {args.model_type} model...")
    model = create_model(args.model_type, num_classes=2)

    # Load trained weights
    checkpoint = torch.load(args.model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("   Loaded model weights from checkpoint")
    else:
        model.load_state_dict(checkpoint)
        print("   Loaded model weights directly")

    # Create exporter
    exporter = ModelExporter(model, model_type=args.model_type)

    # Export to each format
    export_results = {}

    for format_type in args.formats:
        print(f"\nExporting to {format_type.upper()}...")

        # Get export settings
        export_settings = config_manager.get_export_settings(
            args.model_type, format_type, args.deployment, args.platform
        )

        # Create format-specific output path
        format_dir = output_dir / format_type
        format_dir.mkdir(exist_ok=True)

        output_path = format_dir / f"{args.model_type}_pneumonia.{format_type}"

        try:
            if format_type == 'onnx':
                success = exporter.export_to_onnx(
                    output_path,
                    opset_version=export_settings.get('opset_version', 11),
                    optimize=args.optimize,
                    verify=args.validate
                )
            elif format_type == 'torchscript':
                success = exporter.export_to_torchscript(
                    output_path,
                    method=export_settings.get('method', 'trace'),
                    optimize=args.optimize,
                    mobile_optimize=export_settings.get('mobile_optimize', False)
                )
            else:
                print(f"   Format {format_type} not yet implemented in this example")
                success = False

            export_results[format_type] = {
                'success': success,
                'output_path': str(output_path) if success else None,
                'settings': export_settings
            }

            if success:
                print(f"   Successfully exported to: {output_path}")

                # Get model info
                model_info = exporter.get_model_info()
                export_results[format_type]['model_info'] = model_info

                # Validate if requested
                if args.validate:
                    print(f"   Validating {format_type} export...")
                    validation_result = validate_exported_model(
                        model, output_path, format_type
                    )
                    export_results[format_type]['validation'] = validation_result

                    if validation_result.get('valid', False):
                        print(f"   Validation passed!")
                    else:
                        print(f"   Validation failed: {validation_result.get('error', 'Unknown error')}")

            else:
                print(f"   Failed to export to {format_type}")

        except Exception as e:
            print(f"   Error exporting to {format_type}: {e}")
            export_results[format_type] = {
                'success': False,
                'error': str(e),
                'settings': export_settings
            }

    # Save export summary
    summary = {
        'model_type': args.model_type,
        'model_path': str(args.model_path),
        'deployment_target': args.deployment,
        'platform': args.platform,
        'export_results': export_results,
        'configuration': {
            'recommended_formats': config_manager.get_recommended_formats(
                args.model_type, args.deployment
            ),
            'recommended_models': config_manager.get_recommended_models(args.deployment)
        }
    }

    summary_path = output_dir / f"{args.model_type}_export_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nExport completed!")
    print(f"Summary saved to: {summary_path}")
    print(f"Exported models saved to: {output_dir}")

    # Print summary
    successful_exports = [fmt for fmt, result in export_results.items() if result['success']]
    failed_exports = [fmt for fmt, result in export_results.items() if not result['success']]

    if successful_exports:
        print(f"Successful exports: {', '.join(successful_exports)}")
    if failed_exports:
        print(f"Failed exports: {', '.join(failed_exports)}")


if __name__ == "__main__":
    main()