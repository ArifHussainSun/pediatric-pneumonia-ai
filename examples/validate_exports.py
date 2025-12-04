#!/usr/bin/env python3
"""
Example script for validating exported models.

This script demonstrates comprehensive validation of exported models:
- Numerical accuracy validation
- Performance benchmarking
- Cross-format comparison
- Medical image processing validation
- Deployment readiness assessment

Usage:
    python examples/validate_exports.py --exports_dir exports/
    python examples/validate_exports.py --model_path exports/onnx/xception_pneumonia.onnx --format onnx
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.export.validator import ExportValidator
from src.models import create_model
from src.data import create_data_loaders

import torch
import json
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Validate exported pneumonia detection models')
    parser.add_argument('--exports_dir', type=str,
                       help='Directory containing exported models')
    parser.add_argument('--model_path', type=str,
                       help='Specific model file to validate')
    parser.add_argument('--format', type=str,
                       choices=['onnx', 'torchscript', 'coreml', 'tflite'],
                       help='Format of specific model file')
    parser.add_argument('--original_model', type=str,
                       help='Path to original PyTorch model for comparison')
    parser.add_argument('--model_type', type=str, default='xception',
                       help='Model architecture type')
    parser.add_argument('--data_dir', type=str,
                       help='Path to test data for validation')
    parser.add_argument('--output_dir', type=str, default='validation_results',
                       help='Output directory for validation reports')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples for validation')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarks')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original model if provided
    original_model = None
    if args.original_model:
        print(f"Loading original model: {args.original_model}")
        original_model = create_model(args.model_type, num_classes=2)
        checkpoint = torch.load(args.original_model, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            original_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            original_model.load_state_dict(checkpoint)
        original_model.eval()
        print("Original model loaded successfully")

    # Load test data if provided
    test_data = None
    if args.data_dir:
        print(f"Loading test data from: {args.data_dir}")
        try:
            _, test_loader = create_data_loaders(
                train_dir=Path(args.data_dir),
                test_dir=Path(args.data_dir),
                batch_size=32,
                distributed=False
            )
            # Convert to list of samples for validation
            test_data = []
            for batch_idx, (images, labels) in enumerate(test_loader):
                for i in range(images.size(0)):
                    test_data.append((images[i].numpy(), labels[i].item()))
                    if len(test_data) >= args.num_samples:
                        break
                if len(test_data) >= args.num_samples:
                    break
            print(f"Loaded {len(test_data)} test samples")
        except Exception as e:
            print(f"Failed to load test data: {e}")
            test_data = None

    # Create validator
    validator = ExportValidator()

    validation_results = {}

    if args.model_path and args.format:
        # Validate single model
        print(f"\nValidating single model: {args.model_path}")
        model_path = Path(args.model_path)

        if not model_path.exists():
            print(f"ERROR: Model file not found: {model_path}")
            return

        result = validate_single_model(
            validator, original_model, model_path, args.format,
            test_data, args.benchmark
        )
        validation_results[str(model_path)] = result

    elif args.exports_dir:
        # Validate all models in directory
        print(f"\nValidating all models in: {args.exports_dir}")
        exports_dir = Path(args.exports_dir)

        if not exports_dir.exists():
            print(f"ERROR: Exports directory not found: {exports_dir}")
            return

        # Find all exported model files
        model_files = []

        # ONNX files
        model_files.extend([(f, 'onnx') for f in exports_dir.rglob('*.onnx')])

        # TorchScript files
        model_files.extend([(f, 'torchscript') for f in exports_dir.rglob('*.pt')])
        model_files.extend([(f, 'torchscript') for f in exports_dir.rglob('*.pth') if 'torchscript' in f.name])

        # CoreML files
        model_files.extend([(f, 'coreml') for f in exports_dir.rglob('*.mlmodel')])

        # TensorFlow Lite files
        model_files.extend([(f, 'tflite') for f in exports_dir.rglob('*.tflite')])

        print(f"Found {len(model_files)} exported models")

        for model_path, format_type in model_files:
            print(f"\nValidating {format_type}: {model_path.name}")

            result = validate_single_model(
                validator, original_model, model_path, format_type,
                test_data, args.benchmark
            )
            validation_results[str(model_path)] = result

    else:
        print("ERROR: Must specify either --model_path with --format, or --exports_dir")
        return

    # Generate validation report
    report = generate_validation_report(validation_results, args)

    # Save detailed results
    results_path = output_dir / "validation_results.json"
    with open(results_path, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)

    # Save human-readable report
    report_path = output_dir / "validation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nValidation completed!")
    print(f"Results saved to: {results_path}")
    print(f"Report saved to: {report_path}")
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(report)


def validate_single_model(validator, original_model, model_path, format_type, test_data, benchmark):
    """Validate a single exported model."""
    result = {
        'model_path': str(model_path),
        'format': format_type,
        'file_size_mb': model_path.stat().st_size / (1024 ** 2),
        'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    try:
        # Basic validation
        print(f"   Running validation...")
        validation_result = validator.validate_export(
            original_model=original_model,
            exported_model_path=model_path,
            format_type=format_type,
            test_data=test_data,
            num_samples=min(50, len(test_data)) if test_data else 50
        )

        result.update(validation_result)

        # Performance benchmark
        if benchmark:
            print(f"   Running performance benchmark...")
            benchmark_result = run_performance_benchmark(model_path, format_type)
            result['benchmark'] = benchmark_result

        # Medical validation
        if test_data:
            print(f"   Running medical validation...")
            medical_result = run_medical_validation(model_path, format_type, test_data[:20])
            result['medical_validation'] = medical_result

        print(f"   Validation completed - {'PASSED' if result.get('valid', False) else 'FAILED'}")

    except Exception as e:
        print(f"   Validation error: {e}")
        result.update({
            'valid': False,
            'error': str(e)
        })

    return result


def run_performance_benchmark(model_path, format_type, num_runs=50):
    """Run performance benchmark on exported model."""
    try:
        times = []

        if format_type == 'onnx':
            import onnxruntime as ort

            session = ort.InferenceSession(str(model_path))
            input_name = session.get_inputs()[0].name
            dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

            # Warm up
            for _ in range(5):
                session.run(None, {input_name: dummy_input})

            # Benchmark
            for _ in range(num_runs):
                start_time = time.time()
                session.run(None, {input_name: dummy_input})
                times.append(time.time() - start_time)

        elif format_type == 'torchscript':
            model = torch.jit.load(str(model_path))
            model.eval()
            dummy_input = torch.randn(1, 3, 224, 224)

            # Warm up
            with torch.no_grad():
                for _ in range(5):
                    model(dummy_input)

            # Benchmark
            with torch.no_grad():
                for _ in range(num_runs):
                    start_time = time.time()
                    model(dummy_input)
                    times.append(time.time() - start_time)

        else:
            return {'note': f'Benchmark not implemented for {format_type}'}

        return {
            'avg_inference_time_ms': float(np.mean(times) * 1000),
            'std_inference_time_ms': float(np.std(times) * 1000),
            'min_inference_time_ms': float(np.min(times) * 1000),
            'max_inference_time_ms': float(np.max(times) * 1000),
            'p95_inference_time_ms': float(np.percentile(times, 95) * 1000),
            'throughput_fps': float(1.0 / np.mean(times))
        }

    except Exception as e:
        return {'error': f'Benchmark failed: {e}'}


def run_medical_validation(model_path, format_type, test_samples):
    """Run medical-specific validation checks."""
    try:
        correct_predictions = 0
        total_predictions = 0
        confidence_scores = []

        if format_type == 'onnx':
            import onnxruntime as ort

            session = ort.InferenceSession(str(model_path))
            input_name = session.get_inputs()[0].name

            for image, label in test_samples:
                input_data = np.expand_dims(image, axis=0).astype(np.float32)
                output = session.run(None, {input_name: input_data})[0]

                # Apply softmax to get probabilities
                probs = np.exp(output) / np.sum(np.exp(output))
                predicted_class = np.argmax(probs)
                confidence = float(np.max(probs))

                confidence_scores.append(confidence)
                if predicted_class == label:
                    correct_predictions += 1
                total_predictions += 1

        elif format_type == 'torchscript':
            model = torch.jit.load(str(model_path))
            model.eval()

            with torch.no_grad():
                for image, label in test_samples:
                    input_tensor = torch.from_numpy(image).unsqueeze(0)
                    output = model(input_tensor)

                    # Apply softmax
                    probs = torch.softmax(output, dim=1)
                    predicted_class = torch.argmax(probs, dim=1).item()
                    confidence = float(torch.max(probs))

                    confidence_scores.append(confidence)
                    if predicted_class == label:
                        correct_predictions += 1
                    total_predictions += 1

        else:
            return {'note': f'Medical validation not implemented for {format_type}'}

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0

        return {
            'accuracy': float(accuracy),
            'total_samples': total_predictions,
            'correct_predictions': correct_predictions,
            'average_confidence': float(avg_confidence),
            'min_confidence': float(np.min(confidence_scores)) if confidence_scores else 0,
            'max_confidence': float(np.max(confidence_scores)) if confidence_scores else 0,
            'clinical_threshold_met': accuracy >= 0.95  # 95% accuracy threshold for medical use
        }

    except Exception as e:
        return {'error': f'Medical validation failed: {e}'}


def generate_validation_report(validation_results, args):
    """Generate human-readable validation report."""
    report_lines = []

    report_lines.append("PEDIATRIC PNEUMONIA AI - MODEL VALIDATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total models validated: {len(validation_results)}")
    report_lines.append("")

    # Summary statistics
    total_models = len(validation_results)
    valid_models = sum(1 for result in validation_results.values() if result.get('valid', False))

    report_lines.append("SUMMARY")
    report_lines.append("-" * 20)
    report_lines.append(f"Valid models: {valid_models}/{total_models}")
    report_lines.append(f"Success rate: {valid_models/total_models*100:.1f}%" if total_models > 0 else "Success rate: 0%")
    report_lines.append("")

    # Format breakdown
    format_counts = {}
    for result in validation_results.values():
        fmt = result.get('format', 'unknown')
        format_counts[fmt] = format_counts.get(fmt, 0) + 1

    report_lines.append("FORMATS")
    report_lines.append("-" * 20)
    for fmt, count in format_counts.items():
        valid_count = sum(1 for result in validation_results.values()
                         if result.get('format') == fmt and result.get('valid', False))
        report_lines.append(f"{fmt.upper()}: {valid_count}/{count} valid")
    report_lines.append("")

    # Individual model results
    report_lines.append("INDIVIDUAL RESULTS")
    report_lines.append("-" * 30)

    for model_path, result in validation_results.items():
        model_name = Path(model_path).name
        status = "PASS" if result.get('valid', False) else "FAIL"
        file_size = result.get('file_size_mb', 0)

        report_lines.append(f"{model_name}")
        report_lines.append(f"  Status: {status}")
        report_lines.append(f"  Format: {result.get('format', 'unknown')}")
        report_lines.append(f"  Size: {file_size:.2f} MB")

        if 'numerical_accuracy' in result:
            acc = result['numerical_accuracy']
            report_lines.append(f"  Numerical accuracy: {acc:.6f}")

        if 'benchmark' in result and 'avg_inference_time_ms' in result['benchmark']:
            bench = result['benchmark']
            report_lines.append(f"  Avg inference: {bench['avg_inference_time_ms']:.2f} ms")
            report_lines.append(f"  Throughput: {bench.get('throughput_fps', 0):.1f} FPS")

        if 'medical_validation' in result:
            med = result['medical_validation']
            if 'accuracy' in med:
                report_lines.append(f"  Medical accuracy: {med['accuracy']:.3f}")
                report_lines.append(f"  Clinical ready: {'YES' if med.get('clinical_threshold_met', False) else 'NO'}")

        if 'error' in result:
            report_lines.append(f"  Error: {result['error']}")

        report_lines.append("")

    return "\n".join(report_lines)


if __name__ == "__main__":
    main()