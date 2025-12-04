#!/usr/bin/env python3
"""
Example script for exporting models for specific deployment scenarios.

This script demonstrates complete deployment workflows:
- Cloud inference deployment (ONNX + TorchScript)
- Mobile device deployment (CoreML + TensorFlow Lite)
- Edge computing deployment (optimized ONNX)
- Web deployment (ONNX.js compatible)

Usage:
    python examples/export_for_deployment.py --model_path outputs/mobilenet_model.pth --deployment mobile_devices
    python examples/export_for_deployment.py --model_path outputs/xception_model.pth --deployment cloud_inference
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.export import export_all_models
from src.export.config_manager import ExportConfigManager
from src.models import create_model

import torch
import json
import time


DEPLOYMENT_WORKFLOWS = {
    'cloud_inference': {
        'description': 'High-performance cloud deployment for AWS/Azure/GCP',
        'recommended_models': ['xception', 'fusion', 'vgg'],
        'formats': ['onnx', 'torchscript'],
        'optimizations': ['fp16_precision', 'dynamic_batching']
    },
    'mobile_devices': {
        'description': 'iPad/tablet deployment for clinical point-of-care',
        'recommended_models': ['mobilenet'],
        'formats': ['coreml', 'tflite', 'onnx'],
        'optimizations': ['quantization', 'mobile_specific']
    },
    'edge_computing': {
        'description': 'Edge devices with limited compute resources',
        'recommended_models': ['mobilenet', 'vgg'],
        'formats': ['onnx', 'tflite'],
        'optimizations': ['quantization', 'memory_optimization']
    },
    'web_deployment': {
        'description': 'Browser-based inference with ONNX.js',
        'recommended_models': ['mobilenet', 'vgg'],
        'formats': ['onnx'],
        'optimizations': ['quantization', 'size_optimization']
    },
    'embedded_systems': {
        'description': 'Microcontrollers and very limited compute',
        'recommended_models': ['mobilenet'],
        'formats': ['tflite'],
        'optimizations': ['int8_quantization', 'extreme_compression']
    }
}


def main():
    parser = argparse.ArgumentParser(description='Export model for specific deployment scenario')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--deployment', type=str, required=True,
                       choices=list(DEPLOYMENT_WORKFLOWS.keys()),
                       help='Deployment scenario')
    parser.add_argument('--model_type', type=str,
                       help='Model type (if not specified, uses recommended models)')
    parser.add_argument('--output_dir', type=str, default='deployment_exports',
                       help='Output directory for deployment package')
    parser.add_argument('--validate', action='store_true',
                       help='Validate all exported models')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarks')

    args = parser.parse_args()

    deployment_info = DEPLOYMENT_WORKFLOWS[args.deployment]
    print(f"Deployment Scenario: {args.deployment}")
    print(f"Description: {deployment_info['description']}")
    print(f"Recommended models: {deployment_info['recommended_models']}")
    print(f"Export formats: {deployment_info['formats']}")

    # Create output directory
    output_dir = Path(args.output_dir) / args.deployment
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config_manager = ExportConfigManager()

    # Determine model type
    if args.model_type:
        model_types = [args.model_type]
        print(f"Using specified model: {args.model_type}")
    else:
        model_types = deployment_info['recommended_models']
        print(f"Using recommended models: {model_types}")

    # Check if model type is recommended for deployment
    if args.model_type and args.model_type not in deployment_info['recommended_models']:
        print(f"WARNING: {args.model_type} is not recommended for {args.deployment} deployment")
        print(f"Recommended models: {deployment_info['recommended_models']}")

    deployment_results = {}

    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Processing {model_type} model for {args.deployment} deployment")
        print(f"{'='*60}")

        # Create model-specific output directory
        model_output_dir = output_dir / model_type
        model_output_dir.mkdir(exist_ok=True)

        try:
            # Load model
            print(f"Loading {model_type} model...")
            model = create_model(model_type, num_classes=2)

            # Load weights
            checkpoint = torch.load(args.model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            print(f"Model loaded successfully")

            # Export for deployment scenario
            from src.export import ModelExporter

            exporter = ModelExporter(model, model_type=model_type)
            model_results = {}

            # Export to each format for this deployment
            for format_type in deployment_info['formats']:
                print(f"\nExporting {model_type} to {format_type} for {args.deployment}...")

                # Get deployment-specific settings
                export_settings = config_manager.get_export_settings(
                    model_type, format_type, args.deployment
                )

                output_path = model_output_dir / f"{model_type}_pneumonia_{args.deployment}.{format_type}"

                start_time = time.time()

                try:
                    if format_type == 'onnx':
                        success = exporter.export_to_onnx(
                            output_path,
                            opset_version=export_settings.get('opset_version', 11),
                            optimize=True,
                            verify=args.validate
                        )
                    elif format_type == 'torchscript':
                        success = exporter.export_to_torchscript(
                            output_path,
                            method=export_settings.get('method', 'trace'),
                            optimize=True,
                            mobile_optimize=export_settings.get('mobile_optimize', False)
                        )
                    else:
                        print(f"   Format {format_type} export will be implemented in mobile optimization phase")
                        success = False

                    export_time = time.time() - start_time

                    if success:
                        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                        print(f"   Export successful: {output_path}")
                        print(f"   File size: {file_size:.2f} MB")
                        print(f"   Export time: {export_time:.2f} seconds")

                        model_results[format_type] = {
                            'success': True,
                            'output_path': str(output_path),
                            'file_size_mb': file_size,
                            'export_time_seconds': export_time,
                            'settings': export_settings
                        }

                        # Run benchmark if requested
                        if args.benchmark:
                            print(f"   Running performance benchmark...")
                            # Basic benchmark - will be enhanced in later phases
                            benchmark_result = run_basic_benchmark(exporter, format_type, output_path)
                            model_results[format_type]['benchmark'] = benchmark_result

                    else:
                        print(f"   Export failed")
                        model_results[format_type] = {
                            'success': False,
                            'error': 'Export failed'
                        }

                except Exception as e:
                    print(f"   Export error: {e}")
                    model_results[format_type] = {
                        'success': False,
                        'error': str(e)
                    }

            deployment_results[model_type] = model_results

        except Exception as e:
            print(f"Failed to process {model_type}: {e}")
            deployment_results[model_type] = {'error': str(e)}

    # Generate deployment package summary
    package_summary = {
        'deployment_scenario': args.deployment,
        'description': deployment_info['description'],
        'model_path': str(args.model_path),
        'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': deployment_results,
        'deployment_config': {
            'recommended_models': deployment_info['recommended_models'],
            'formats': deployment_info['formats'],
            'optimizations': deployment_info['optimizations']
        }
    }

    # Save deployment summary
    summary_path = output_dir / f"{args.deployment}_deployment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(package_summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Deployment Export Complete!")
    print(f"{'='*60}")
    print(f"Scenario: {args.deployment}")
    print(f"Output directory: {output_dir}")
    print(f"Summary: {summary_path}")

    # Print success summary
    total_exports = 0
    successful_exports = 0

    for model_type, results in deployment_results.items():
        if 'error' not in results:
            for format_type, result in results.items():
                total_exports += 1
                if result.get('success', False):
                    successful_exports += 1

    print(f"Successful exports: {successful_exports}/{total_exports}")

    if successful_exports == total_exports:
        print("All exports completed successfully!")
    elif successful_exports > 0:
        print("Some exports completed successfully. Check summary for details.")
    else:
        print("No exports were successful. Check logs for errors.")


def run_basic_benchmark(exporter, format_type, model_path):
    """Run basic performance benchmark on exported model."""
    try:
        if format_type == 'onnx':
            import onnxruntime as ort
            import numpy as np

            session = ort.InferenceSession(str(model_path))
            input_name = session.get_inputs()[0].name

            # Warm up
            dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            for _ in range(5):
                session.run(None, {input_name: dummy_input})

            # Benchmark
            times = []
            for _ in range(50):
                start_time = time.time()
                session.run(None, {input_name: dummy_input})
                times.append(time.time() - start_time)

            return {
                'avg_inference_time_ms': np.mean(times) * 1000,
                'std_inference_time_ms': np.std(times) * 1000,
                'min_inference_time_ms': np.min(times) * 1000,
                'max_inference_time_ms': np.max(times) * 1000
            }

        elif format_type == 'torchscript':
            import torch

            model = torch.jit.load(str(model_path))
            model.eval()

            # Warm up
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                for _ in range(5):
                    model(dummy_input)

            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(50):
                    start_time = time.time()
                    model(dummy_input)
                    times.append(time.time() - start_time)

            return {
                'avg_inference_time_ms': np.mean(times) * 1000,
                'std_inference_time_ms': np.std(times) * 1000,
                'min_inference_time_ms': np.min(times) * 1000,
                'max_inference_time_ms': np.max(times) * 1000
            }

        else:
            return {'note': f'Benchmark not implemented for {format_type}'}

    except Exception as e:
        return {'error': f'Benchmark failed: {e}'}


if __name__ == "__main__":
    main()