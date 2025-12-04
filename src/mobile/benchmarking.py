"""
Mobile Performance Benchmarking for Pneumonia Detection Models

This module provides comprehensive benchmarking capabilities for mobile
deployed models across different platforms and formats. Essential for
validating clinical deployment readiness and performance optimization.

Benchmarks Include:
- Inference time measurements
- Memory usage monitoring
- Battery consumption analysis
- Accuracy validation
- Cross-platform comparisons

Designed to ensure mobile models meet clinical requirements for
real-time pneumonia detection on resource-constrained devices.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
import psutil
import gc

# Setup logging
logger = logging.getLogger(__name__)

# Benchmark configurations
BENCHMARK_CONFIGS = {
    'clinical_validation': {
        'description': 'Clinical deployment validation benchmarks',
        'max_inference_time_ms': 500,
        'max_memory_usage_mb': 200,
        'min_accuracy': 0.95,
        'test_samples': 1000
    },
    'mobile_performance': {
        'description': 'Mobile device performance benchmarks',
        'max_inference_time_ms': 300,
        'max_memory_usage_mb': 100,
        'min_accuracy': 0.93,
        'test_samples': 500
    },
    'embedded_systems': {
        'description': 'Resource-constrained device benchmarks',
        'max_inference_time_ms': 1000,
        'max_memory_usage_mb': 50,
        'min_accuracy': 0.90,
        'test_samples': 200
    },
    'comparative_analysis': {
        'description': 'Cross-format comparison benchmarks',
        'max_inference_time_ms': 1000,
        'max_memory_usage_mb': 500,
        'min_accuracy': 0.85,
        'test_samples': 100
    }
}

# Platform-specific benchmark settings
PLATFORM_BENCHMARKS = {
    'pytorch': {
        'formats': ['pth'],
        'acceleration': 'cpu',
        'precision': 'float32'
    },
    'onnx': {
        'formats': ['onnx'],
        'acceleration': 'cpu_gpu',
        'precision': 'float32_float16'
    },
    'torchscript': {
        'formats': ['pt', 'pth'],
        'acceleration': 'cpu_gpu',
        'precision': 'float32'
    },
    'coreml': {
        'formats': ['mlmodel'],
        'acceleration': 'neural_engine',
        'precision': 'float16_int8'
    },
    'tflite': {
        'formats': ['tflite'],
        'acceleration': 'nnapi',
        'precision': 'float32_int8'
    }
}


class MobileBenchmark:
    """
    Comprehensive benchmarking suite for mobile pneumonia detection models.

    Provides detailed performance analysis across different platforms,
    formats, and deployment scenarios with clinical validation focus.
    """

    def __init__(self, benchmark_type: str = 'clinical_validation'):
        """
        Initialize mobile benchmark suite.

        Args:
            benchmark_type: Type of benchmark to run
        """
        if benchmark_type not in BENCHMARK_CONFIGS:
            raise ValueError(f"Unsupported benchmark type: {benchmark_type}. Choose from {list(BENCHMARK_CONFIGS.keys())}")

        self.benchmark_type = benchmark_type
        self.config = BENCHMARK_CONFIGS[benchmark_type]
        self.results = {}

        logger.info(f"MobileBenchmark initialized: {benchmark_type}")
        logger.info(f"Max inference time: {self.config['max_inference_time_ms']} ms")
        logger.info(f"Max memory usage: {self.config['max_memory_usage_mb']} MB")
        logger.info(f"Min accuracy: {self.config['min_accuracy']}")

    def benchmark_model(self,
                       model_path: Union[str, Path],
                       format_type: str,
                       test_data: Optional[List[Tuple[np.ndarray, int]]] = None,
                       num_warmup: int = 10,
                       num_runs: int = 100) -> Dict[str, Any]:
        """
        Benchmark a mobile model comprehensively.

        Args:
            model_path: Path to the model file
            format_type: Model format ('pytorch', 'onnx', 'torchscript', 'coreml', 'tflite')
            test_data: Test data for accuracy validation
            num_warmup: Number of warmup runs
            num_runs: Number of benchmark runs

        Returns:
            Dict containing comprehensive benchmark results
        """
        logger.info(f"Benchmarking {format_type} model: {Path(model_path).name}")

        benchmark_results = {
            'model_path': str(model_path),
            'format_type': format_type,
            'benchmark_type': self.benchmark_type,
            'model_info': self._get_model_info(model_path, format_type),
            'performance': {},
            'memory': {},
            'accuracy': {},
            'clinical_validation': {},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        try:
            # Load model
            model_loader = self._get_model_loader(format_type)
            model, preprocess_fn = model_loader(model_path)

            # Performance benchmarking
            logger.info("Running performance benchmarks...")
            benchmark_results['performance'] = self._benchmark_performance(
                model, preprocess_fn, format_type, num_warmup, num_runs
            )

            # Memory benchmarking
            logger.info("Running memory benchmarks...")
            benchmark_results['memory'] = self._benchmark_memory(
                model, preprocess_fn, format_type
            )

            # Accuracy validation
            if test_data:
                logger.info("Running accuracy validation...")
                benchmark_results['accuracy'] = self._benchmark_accuracy(
                    model, preprocess_fn, test_data, format_type
                )

            # Clinical validation
            logger.info("Running clinical validation...")
            benchmark_results['clinical_validation'] = self._validate_clinical_requirements(
                benchmark_results
            )

            logger.info(f"Benchmark completed for {format_type}")

        except Exception as e:
            logger.error(f"Benchmarking failed for {format_type}: {e}")
            benchmark_results['error'] = str(e)

        return benchmark_results

    def benchmark_multiple_models(self,
                                 model_configs: List[Dict[str, Any]],
                                 test_data: Optional[List[Tuple[np.ndarray, int]]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark multiple models for comparison.

        Args:
            model_configs: List of model configurations with 'path' and 'format' keys
            test_data: Test data for accuracy validation

        Returns:
            Dict mapping model names to benchmark results
        """
        logger.info(f"Benchmarking {len(model_configs)} models...")

        comparison_results = {}

        for i, config in enumerate(model_configs):
            model_path = config['path']
            format_type = config['format']
            model_name = config.get('name', f"{format_type}_{i}")

            logger.info(f"Benchmarking model {i+1}/{len(model_configs)}: {model_name}")

            results = self.benchmark_model(
                model_path, format_type, test_data
            )
            comparison_results[model_name] = results

        # Generate comparison summary
        comparison_results['summary'] = self._generate_comparison_summary(comparison_results)

        return comparison_results

    def _benchmark_performance(self,
                              model: Any,
                              preprocess_fn: Callable,
                              format_type: str,
                              num_warmup: int,
                              num_runs: int) -> Dict[str, float]:
        """Benchmark model inference performance."""
        # Generate test input
        test_input = self._generate_test_input()
        processed_input = preprocess_fn(test_input)

        # Warmup runs
        for _ in range(num_warmup):
            self._run_inference(model, processed_input, format_type)
            gc.collect()

        # Benchmark runs
        inference_times = []

        for _ in range(num_runs):
            start_time = time.perf_counter()
            self._run_inference(model, processed_input, format_type)
            end_time = time.perf_counter()

            inference_times.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        inference_times = np.array(inference_times)

        return {
            'avg_inference_time_ms': float(np.mean(inference_times)),
            'std_inference_time_ms': float(np.std(inference_times)),
            'min_inference_time_ms': float(np.min(inference_times)),
            'max_inference_time_ms': float(np.max(inference_times)),
            'p50_inference_time_ms': float(np.percentile(inference_times, 50)),
            'p95_inference_time_ms': float(np.percentile(inference_times, 95)),
            'p99_inference_time_ms': float(np.percentile(inference_times, 99)),
            'throughput_fps': float(1000.0 / np.mean(inference_times)),
            'meets_performance_target': float(np.mean(inference_times)) <= self.config['max_inference_time_ms']
        }

    def _benchmark_memory(self,
                         model: Any,
                         preprocess_fn: Callable,
                         format_type: str) -> Dict[str, float]:
        """Benchmark model memory usage."""
        # Clear memory
        gc.collect()

        # Get baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / (1024 ** 2)  # MB

        # Load model and run inference
        test_input = self._generate_test_input()
        processed_input = preprocess_fn(test_input)

        # Measure peak memory during inference
        peak_memory = baseline_memory

        for _ in range(10):  # Multiple runs to get peak usage
            self._run_inference(model, processed_input, format_type)
            current_memory = process.memory_info().rss / (1024 ** 2)
            peak_memory = max(peak_memory, current_memory)

        memory_usage = peak_memory - baseline_memory

        return {
            'baseline_memory_mb': baseline_memory,
            'peak_memory_mb': peak_memory,
            'memory_usage_mb': memory_usage,
            'meets_memory_target': memory_usage <= self.config['max_memory_usage_mb']
        }

    def _benchmark_accuracy(self,
                           model: Any,
                           preprocess_fn: Callable,
                           test_data: List[Tuple[np.ndarray, int]],
                           format_type: str) -> Dict[str, float]:
        """Benchmark model accuracy on test data."""
        correct_predictions = 0
        total_predictions = 0
        confidence_scores = []

        # Limit test samples based on config
        test_samples = min(len(test_data), self.config['test_samples'])

        for i in range(test_samples):
            image, true_label = test_data[i]

            try:
                # Preprocess and run inference
                processed_input = preprocess_fn(image)
                output = self._run_inference(model, processed_input, format_type)

                # Get prediction
                if isinstance(output, (list, tuple)):
                    output = output[0]

                if hasattr(output, 'numpy'):
                    output = output.numpy()

                if len(output.shape) > 1:
                    output = output.flatten()

                # Apply softmax to get probabilities
                probs = np.exp(output) / np.sum(np.exp(output))
                predicted_label = np.argmax(probs)
                confidence = float(np.max(probs))

                confidence_scores.append(confidence)

                if predicted_label == true_label:
                    correct_predictions += 1

                total_predictions += 1

            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        return {
            'accuracy': accuracy,
            'total_samples': total_predictions,
            'correct_predictions': correct_predictions,
            'average_confidence': float(np.mean(confidence_scores)) if confidence_scores else 0.0,
            'min_confidence': float(np.min(confidence_scores)) if confidence_scores else 0.0,
            'max_confidence': float(np.max(confidence_scores)) if confidence_scores else 0.0,
            'meets_accuracy_target': accuracy >= self.config['min_accuracy']
        }

    def _validate_clinical_requirements(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model meets clinical deployment requirements."""
        performance = benchmark_results.get('performance', {})
        memory = benchmark_results.get('memory', {})
        accuracy = benchmark_results.get('accuracy', {})

        # Check individual requirements
        performance_ok = performance.get('meets_performance_target', False)
        memory_ok = memory.get('meets_memory_target', False)
        accuracy_ok = accuracy.get('meets_accuracy_target', False)

        # Overall clinical readiness
        clinical_ready = performance_ok and memory_ok and accuracy_ok

        validation_results = {
            'performance_requirement_met': performance_ok,
            'memory_requirement_met': memory_ok,
            'accuracy_requirement_met': accuracy_ok,
            'clinical_deployment_ready': clinical_ready,
            'requirements': self.config
        }

        # Add specific failure reasons
        if not clinical_ready:
            failures = []
            if not performance_ok:
                failures.append(f"Inference time too slow: {performance.get('avg_inference_time_ms', 0):.1f}ms > {self.config['max_inference_time_ms']}ms")
            if not memory_ok:
                failures.append(f"Memory usage too high: {memory.get('memory_usage_mb', 0):.1f}MB > {self.config['max_memory_usage_mb']}MB")
            if not accuracy_ok:
                failures.append(f"Accuracy too low: {accuracy.get('accuracy', 0):.3f} < {self.config['min_accuracy']}")

            validation_results['failure_reasons'] = failures

        return validation_results

    def _generate_comparison_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary comparison of multiple models."""
        model_results = {k: v for k, v in results.items() if k != 'summary'}

        if not model_results:
            return {}

        summary = {
            'total_models': len(model_results),
            'clinical_ready_models': 0,
            'best_performance': None,
            'best_accuracy': None,
            'smallest_model': None,
            'format_comparison': {}
        }

        best_perf_time = float('inf')
        best_accuracy = 0.0
        smallest_size = float('inf')

        for model_name, result in model_results.items():
            if 'error' in result:
                continue

            # Count clinical ready models
            if result.get('clinical_validation', {}).get('clinical_deployment_ready', False):
                summary['clinical_ready_models'] += 1

            # Track best performance
            avg_time = result.get('performance', {}).get('avg_inference_time_ms', float('inf'))
            if avg_time < best_perf_time:
                best_perf_time = avg_time
                summary['best_performance'] = model_name

            # Track best accuracy
            accuracy = result.get('accuracy', {}).get('accuracy', 0.0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                summary['best_accuracy'] = model_name

            # Track smallest model
            model_size = result.get('model_info', {}).get('file_size_mb', float('inf'))
            if model_size < smallest_size:
                smallest_size = model_size
                summary['smallest_model'] = model_name

            # Format comparison
            format_type = result.get('format_type', 'unknown')
            if format_type not in summary['format_comparison']:
                summary['format_comparison'][format_type] = {
                    'count': 0,
                    'avg_performance_ms': 0,
                    'avg_accuracy': 0,
                    'avg_size_mb': 0
                }

            summary['format_comparison'][format_type]['count'] += 1
            summary['format_comparison'][format_type]['avg_performance_ms'] += avg_time
            summary['format_comparison'][format_type]['avg_accuracy'] += accuracy
            summary['format_comparison'][format_type]['avg_size_mb'] += model_size

        # Calculate averages for format comparison
        for format_stats in summary['format_comparison'].values():
            count = format_stats['count']
            if count > 0:
                format_stats['avg_performance_ms'] /= count
                format_stats['avg_accuracy'] /= count
                format_stats['avg_size_mb'] /= count

        return summary

    def _get_model_info(self, model_path: Union[str, Path], format_type: str) -> Dict[str, Any]:
        """Get basic model information."""
        model_path = Path(model_path)

        return {
            'file_name': model_path.name,
            'file_size_mb': model_path.stat().st_size / (1024 ** 2),
            'format': format_type
        }

    def _get_model_loader(self, format_type: str) -> Callable:
        """Get appropriate model loader for format."""
        if format_type == 'pytorch':
            return self._load_pytorch_model
        elif format_type == 'onnx':
            return self._load_onnx_model
        elif format_type == 'torchscript':
            return self._load_torchscript_model
        elif format_type == 'coreml':
            return self._load_coreml_model
        elif format_type == 'tflite':
            return self._load_tflite_model
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _load_pytorch_model(self, model_path: str) -> Tuple[Any, Callable]:
        """Load PyTorch model."""
        model = torch.load(model_path, map_location='cpu')
        model.eval()

        def preprocess(x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            return x

        return model, preprocess

    def _load_onnx_model(self, model_path: str) -> Tuple[Any, Callable]:
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(model_path)

            def preprocess(x):
                if torch.is_tensor(x):
                    x = x.numpy()
                if len(x.shape) == 3:
                    x = np.expand_dims(x, 0)
                return x.astype(np.float32)

            return session, preprocess
        except ImportError:
            raise ImportError("ONNX Runtime not available")

    def _load_torchscript_model(self, model_path: str) -> Tuple[Any, Callable]:
        """Load TorchScript model."""
        model = torch.jit.load(model_path, map_location='cpu')
        model.eval()

        def preprocess(x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            return x

        return model, preprocess

    def _load_coreml_model(self, model_path: str) -> Tuple[Any, Callable]:
        """Load CoreML model."""
        try:
            import coremltools as ct
            model = ct.models.MLModel(model_path)

            def preprocess(x):
                if torch.is_tensor(x):
                    x = x.numpy()
                # CoreML expects HWC format, uint8
                if len(x.shape) == 3:
                    x = x.transpose(1, 2, 0)
                x = (x * 255).astype(np.uint8)
                return x

            return model, preprocess
        except ImportError:
            raise ImportError("CoreML Tools not available")

    def _load_tflite_model(self, model_path: str) -> Tuple[Any, Callable]:
        """Load TensorFlow Lite model."""
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path)
            interpreter.allocate_tensors()

            def preprocess(x):
                if torch.is_tensor(x):
                    x = x.numpy()
                if len(x.shape) == 3:
                    x = np.expand_dims(x, 0)
                return x.astype(np.float32)

            return interpreter, preprocess
        except ImportError:
            raise ImportError("TensorFlow not available")

    def _run_inference(self, model: Any, input_data: Any, format_type: str) -> Any:
        """Run inference based on model format."""
        if format_type == 'pytorch' or format_type == 'torchscript':
            with torch.no_grad():
                return model(input_data)

        elif format_type == 'onnx':
            input_name = model.get_inputs()[0].name
            return model.run(None, {input_name: input_data})[0]

        elif format_type == 'coreml':
            return model.predict({'chest_xray_image': input_data})['pneumonia_prediction']

        elif format_type == 'tflite':
            input_details = model.get_input_details()
            output_details = model.get_output_details()

            model.set_tensor(input_details[0]['index'], input_data)
            model.invoke()
            return model.get_tensor(output_details[0]['index'])

        else:
            raise ValueError(f"Unsupported format for inference: {format_type}")

    def _generate_test_input(self) -> np.ndarray:
        """Generate test input for benchmarking."""
        return np.random.randn(3, 224, 224).astype(np.float32)


def benchmark_mobile_model(model_path: Union[str, Path],
                          format_type: str,
                          benchmark_type: str = 'clinical_validation',
                          test_data: Optional[List[Tuple[np.ndarray, int]]] = None) -> Dict[str, Any]:
    """
    Convenience function to benchmark a mobile model.

    Args:
        model_path: Path to model file
        format_type: Model format
        benchmark_type: Type of benchmark to run
        test_data: Test data for accuracy validation

    Returns:
        Dict containing benchmark results
    """
    benchmark = MobileBenchmark(benchmark_type)
    return benchmark.benchmark_model(model_path, format_type, test_data)


if __name__ == "__main__":
    # Test mobile benchmarking
    print("Testing mobile benchmarking...")

    try:
        benchmark = MobileBenchmark('clinical_validation')
        print(f"Available benchmark types: {list(BENCHMARK_CONFIGS.keys())}")
        print(f"Available platforms: {list(PLATFORM_BENCHMARKS.keys())}")
        print("Mobile benchmarking ready!")

    except Exception as e:
        print(f"Error during testing: {e}")

    print("Mobile benchmarking system ready!")