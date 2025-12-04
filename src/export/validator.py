"""
Export Validation and Testing for Pediatric Pneumonia Detection Models

This module provides comprehensive validation and testing functionality for
exported models. Ensures that exported models maintain accuracy, performance,
and compatibility across different formats and platforms.

Validation includes:
- Numerical accuracy validation
- Performance benchmarking
- Format-specific testing
- Cross-platform compatibility
- Medical image preprocessing validation
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import tempfile

from .config_manager import ExportConfigManager
from ..data import get_medical_transforms

# Setup logging
logger = logging.getLogger(__name__)


class ExportValidator:
    """
    Comprehensive validator for exported pneumonia detection models.

    Validates exported models against original PyTorch models to ensure:
    - Numerical accuracy preservation
    - Performance requirements
    - Format-specific functionality
    - Medical image processing compatibility
    """

    def __init__(self, config_manager: Optional[ExportConfigManager] = None):
        """
        Initialize export validator.

        Args:
            config_manager: Optional configuration manager for validation settings
        """
        self.config_manager = config_manager or ExportConfigManager()
        self.validation_config = self.config_manager.get_validation_settings()

        # Get validation thresholds
        self.accuracy_threshold = self.validation_config.get('accuracy_threshold', 0.95)
        self.max_inference_time = self.validation_config.get('max_inference_time_ms', 500)
        self.max_model_size = self.validation_config.get('max_model_size_mb', 50)
        self.numerical_precision = self.validation_config.get('numerical_precision', 1e-5)

        logger.info("ExportValidator initialized with validation thresholds")

    def validate_export(self,
                       original_model: nn.Module,
                       exported_model_path: Union[str, Path],
                       format_type: str,
                       test_data: Optional[torch.Tensor] = None,
                       num_samples: int = 100) -> Dict[str, Any]:
        """
        Comprehensive validation of an exported model.

        Args:
            original_model: Original PyTorch model
            exported_model_path: Path to exported model
            format_type: Export format ('onnx', 'torchscript', 'coreml', 'tflite')
            test_data: Optional test data tensor
            num_samples: Number of test samples for validation

        Returns:
            Dict containing validation results
        """
        results = {
            'format': format_type,
            'model_path': str(exported_model_path),
            'validation_passed': False,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }

        try:
            # Basic file validation
            if not Path(exported_model_path).exists():
                results['errors'].append(f"Exported model file not found: {exported_model_path}")
                return results

            # Model size validation
            model_size_mb = Path(exported_model_path).stat().st_size / (1024 ** 2)
            results['metrics']['model_size_mb'] = model_size_mb

            if model_size_mb > self.max_model_size:
                results['warnings'].append(
                    f"Model size ({model_size_mb:.2f}MB) exceeds threshold ({self.max_model_size}MB)"
                )

            # Load exported model
            exported_model = self._load_exported_model(exported_model_path, format_type)
            if exported_model is None:
                results['errors'].append(f"Failed to load {format_type} model")
                return results

            # Generate test data if not provided
            if test_data is None:
                test_data = self._generate_test_data(num_samples)

            # Numerical accuracy validation
            accuracy_results = self._validate_numerical_accuracy(
                original_model, exported_model, test_data, format_type
            )
            results['metrics'].update(accuracy_results)

            # Performance validation
            performance_results = self._validate_performance(
                exported_model, test_data, format_type
            )
            results['metrics'].update(performance_results)

            # Medical image processing validation
            medical_results = self._validate_medical_processing(
                original_model, exported_model, format_type
            )
            results['metrics'].update(medical_results)

            # Overall validation status
            results['validation_passed'] = self._determine_validation_status(results)

            logger.info(f"Validation completed for {format_type} model: {results['validation_passed']}")

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            results['errors'].append(f"Validation error: {str(e)}")

        return results

    def _load_exported_model(self, model_path: Union[str, Path], format_type: str):
        """Load exported model based on format type."""
        try:
            if format_type == 'onnx':
                return self._load_onnx_model(model_path)
            elif format_type == 'torchscript':
                return self._load_torchscript_model(model_path)
            elif format_type == 'coreml':
                return self._load_coreml_model(model_path)
            elif format_type == 'tflite':
                return self._load_tflite_model(model_path)
            else:
                logger.error(f"Unsupported format type: {format_type}")
                return None

        except Exception as e:
            logger.error(f"Failed to load {format_type} model: {e}")
            return None

    def _load_onnx_model(self, model_path: Union[str, Path]):
        """Load ONNX model for validation."""
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(model_path))
            return {
                'session': session,
                'input_name': session.get_inputs()[0].name,
                'output_name': session.get_outputs()[0].name
            }

        except ImportError:
            logger.error("ONNX Runtime not available for validation")
            return None

    def _load_torchscript_model(self, model_path: Union[str, Path]):
        """Load TorchScript model for validation."""
        model = torch.jit.load(str(model_path))
        model.eval()
        return model

    def _load_coreml_model(self, model_path: Union[str, Path]):
        """Load CoreML model for validation."""
        try:
            import coremltools as ct

            model = ct.models.MLModel(str(model_path))
            return model

        except ImportError:
            logger.error("CoreML tools not available for validation")
            return None

    def _load_tflite_model(self, model_path: Union[str, Path]):
        """Load TensorFlow Lite model for validation."""
        try:
            import tensorflow as tf

            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()

            return {
                'interpreter': interpreter,
                'input_details': interpreter.get_input_details(),
                'output_details': interpreter.get_output_details()
            }

        except ImportError:
            logger.error("TensorFlow not available for validation")
            return None

    def _generate_test_data(self, num_samples: int) -> torch.Tensor:
        """Generate test data for validation."""
        # Generate realistic medical image data
        test_data = torch.randn(num_samples, 3, 224, 224)

        # Apply medical image transforms
        transform = get_medical_transforms(is_training=False)

        # Convert to PIL images and back to simulate real preprocessing
        processed_data = []
        for i in range(num_samples):
            # Convert tensor to PIL image
            img_array = (test_data[i] * 0.5 + 0.5).clamp(0, 1)  # Denormalize
            img_array = (img_array * 255).byte().permute(1, 2, 0).numpy()
            pil_image = Image.fromarray(img_array)

            # Apply transforms
            transformed = transform(pil_image)
            processed_data.append(transformed)

        return torch.stack(processed_data)

    def _validate_numerical_accuracy(self,
                                   original_model: nn.Module,
                                   exported_model: Any,
                                   test_data: torch.Tensor,
                                   format_type: str) -> Dict[str, float]:
        """Validate numerical accuracy between original and exported models."""
        results = {}

        try:
            original_model.eval()

            # Get original model outputs
            with torch.no_grad():
                original_outputs = original_model(test_data)

            # Get exported model outputs
            exported_outputs = self._run_exported_inference(
                exported_model, test_data, format_type
            )

            if exported_outputs is None:
                results['numerical_accuracy'] = 0.0
                results['max_difference'] = float('inf')
                return results

            # Calculate differences
            if isinstance(exported_outputs, torch.Tensor):
                exported_outputs_np = exported_outputs.cpu().numpy()
            else:
                exported_outputs_np = exported_outputs

            original_outputs_np = original_outputs.cpu().numpy()

            # Calculate metrics
            max_diff = np.abs(original_outputs_np - exported_outputs_np).max()
            mean_diff = np.abs(original_outputs_np - exported_outputs_np).mean()
            relative_error = np.abs(original_outputs_np - exported_outputs_np) / (np.abs(original_outputs_np) + 1e-8)
            max_relative_error = relative_error.max()

            results['max_difference'] = float(max_diff)
            results['mean_difference'] = float(mean_diff)
            results['max_relative_error'] = float(max_relative_error)

            # Determine numerical accuracy
            if max_diff < self.numerical_precision:
                results['numerical_accuracy'] = 1.0
            else:
                results['numerical_accuracy'] = 1.0 - min(max_relative_error, 1.0)

            logger.info(f"Numerical accuracy: {results['numerical_accuracy']:.4f}, "
                       f"max diff: {max_diff:.2e}")

        except Exception as e:
            logger.error(f"Numerical accuracy validation failed: {e}")
            results['numerical_accuracy'] = 0.0
            results['validation_error'] = str(e)

        return results

    def _run_exported_inference(self,
                              exported_model: Any,
                              test_data: torch.Tensor,
                              format_type: str) -> Optional[np.ndarray]:
        """Run inference on exported model."""
        try:
            if format_type == 'onnx':
                session = exported_model['session']
                input_name = exported_model['input_name']
                outputs = session.run(None, {input_name: test_data.numpy()})
                return outputs[0]

            elif format_type == 'torchscript':
                with torch.no_grad():
                    outputs = exported_model(test_data)
                return outputs

            elif format_type == 'tflite':
                interpreter = exported_model['interpreter']
                input_details = exported_model['input_details']
                output_details = exported_model['output_details']

                # Run inference for each sample
                all_outputs = []
                for i in range(test_data.shape[0]):
                    input_data = test_data[i:i+1].numpy().astype(np.float32)
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    all_outputs.append(output_data)

                return np.concatenate(all_outputs, axis=0)

            else:
                logger.error(f"Inference not implemented for format: {format_type}")
                return None

        except Exception as e:
            logger.error(f"Exported model inference failed: {e}")
            return None

    def _validate_performance(self,
                            exported_model: Any,
                            test_data: torch.Tensor,
                            format_type: str) -> Dict[str, float]:
        """Validate performance requirements."""
        results = {}

        try:
            # Warmup
            for _ in range(5):
                self._run_exported_inference(exported_model, test_data[:1], format_type)

            # Benchmark
            times = []
            for i in range(min(50, test_data.shape[0])):
                start_time = time.time()
                self._run_exported_inference(exported_model, test_data[i:i+1], format_type)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms

            results['mean_inference_time_ms'] = float(np.mean(times))
            results['std_inference_time_ms'] = float(np.std(times))
            results['max_inference_time_ms'] = float(np.max(times))
            results['min_inference_time_ms'] = float(np.min(times))

            # Check against threshold
            if results['mean_inference_time_ms'] > self.max_inference_time:
                results['performance_warning'] = (
                    f"Mean inference time ({results['mean_inference_time_ms']:.2f}ms) "
                    f"exceeds threshold ({self.max_inference_time}ms)"
                )

            logger.info(f"Performance: {results['mean_inference_time_ms']:.2f}ms average")

        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            results['performance_error'] = str(e)

        return results

    def _validate_medical_processing(self,
                                   original_model: nn.Module,
                                   exported_model: Any,
                                   format_type: str) -> Dict[str, Any]:
        """Validate medical image processing pipeline."""
        results = {}

        try:
            # Test with medical image samples
            test_cases = self.validation_config.get('test_cases', [])

            for test_case in test_cases:
                case_name = test_case.get('name', 'unknown')
                expected_class = test_case.get('expected_class')
                confidence_threshold = test_case.get('confidence_threshold', 0.8)

                # Generate synthetic test case (in real scenario, use actual medical images)
                if 'normal' in case_name.lower():
                    # Generate normal-like patterns
                    test_image = torch.randn(1, 3, 224, 224) * 0.3 + 0.1
                else:
                    # Generate pneumonia-like patterns
                    test_image = torch.randn(1, 3, 224, 224) * 0.5 + 0.3

                # Run inference
                original_output = original_model(test_image)
                exported_output = self._run_exported_inference(exported_model, test_image, format_type)

                if exported_output is not None:
                    # Convert to probabilities
                    original_probs = torch.softmax(original_output, dim=1)
                    exported_probs = torch.softmax(torch.tensor(exported_output), dim=1)

                    # Check predictions match
                    original_pred = original_probs.argmax(dim=1).item()
                    exported_pred = exported_probs.argmax(dim=1).item()

                    results[f'{case_name}_prediction_match'] = original_pred == exported_pred
                    results[f'{case_name}_confidence_diff'] = float(
                        torch.abs(original_probs - exported_probs).max()
                    )

        except Exception as e:
            logger.error(f"Medical processing validation failed: {e}")
            results['medical_processing_error'] = str(e)

        return results

    def _determine_validation_status(self, results: Dict[str, Any]) -> bool:
        """Determine overall validation status."""
        if results['errors']:
            return False

        metrics = results.get('metrics', {})

        # Check numerical accuracy
        numerical_accuracy = metrics.get('numerical_accuracy', 0.0)
        if numerical_accuracy < 0.99:  # Very strict for medical applications
            return False

        # Check performance (if measured)
        mean_inference_time = metrics.get('mean_inference_time_ms')
        if mean_inference_time and mean_inference_time > self.max_inference_time:
            return False

        # Check medical processing cases
        for key, value in metrics.items():
            if 'prediction_match' in key and not value:
                return False

        return True


def validate_exported_models(export_directory: Union[str, Path],
                            original_model: nn.Module,
                            model_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Validate all exported models in a directory.

    Args:
        export_directory: Directory containing exported models
        original_model: Original PyTorch model for comparison
        model_type: Type of model for configuration

    Returns:
        Dict mapping model paths to validation results
    """
    validator = ExportValidator()
    results = {}

    export_dir = Path(export_directory)
    if not export_dir.exists():
        logger.error(f"Export directory not found: {export_directory}")
        return results

    # Find exported models
    model_files = {
        'onnx': list(export_dir.glob('**/*.onnx')),
        'torchscript': list(export_dir.glob('**/*.pt')),
        'coreml': list(export_dir.glob('**/*.mlmodel')),
        'tflite': list(export_dir.glob('**/*.tflite'))
    }

    for format_type, files in model_files.items():
        for model_path in files:
            logger.info(f"Validating {format_type} model: {model_path}")

            validation_result = validator.validate_export(
                original_model, model_path, format_type
            )

            results[str(model_path)] = validation_result

    return results


def generate_validation_report(validation_results: Dict[str, Dict[str, Any]],
                             output_path: Union[str, Path]) -> bool:
    """
    Generate a comprehensive validation report.

    Args:
        validation_results: Results from model validation
        output_path: Path to save the report

    Returns:
        bool: True if report generated successfully
    """
    try:
        with open(output_path, 'w') as f:
            f.write("# Model Export Validation Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            total_models = len(validation_results)
            passed_models = sum(1 for r in validation_results.values() if r['validation_passed'])

            f.write(f"## Summary\n\n")
            f.write(f"- Total models validated: {total_models}\n")
            f.write(f"- Models passed: {passed_models}\n")
            f.write(f"- Models failed: {total_models - passed_models}\n")
            f.write(f"- Success rate: {passed_models/total_models*100:.1f}%\n\n")

            f.write("## Detailed Results\n\n")

            for model_path, result in validation_results.items():
                f.write(f"### {Path(model_path).name}\n\n")
                f.write(f"- **Format**: {result['format']}\n")
                f.write(f"- **Status**: {'✅ PASSED' if result['validation_passed'] else '❌ FAILED'}\n")

                if result['errors']:
                    f.write(f"- **Errors**: {', '.join(result['errors'])}\n")

                if result['warnings']:
                    f.write(f"- **Warnings**: {', '.join(result['warnings'])}\n")

                metrics = result.get('metrics', {})
                if metrics:
                    f.write("- **Metrics**:\n")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            f.write(f"  - {key}: {value:.4f}\n")
                        else:
                            f.write(f"  - {key}: {value}\n")

                f.write("\n")

        logger.info(f"Validation report generated: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to generate validation report: {e}")
        return False


if __name__ == "__main__":
    # Test export validator
    print("Testing ExportValidator...")

    try:
        from ..models import create_model

        # Create dummy model and validator
        model = create_model('xception', num_classes=2)
        validator = ExportValidator()

        print("ExportValidator ready!")

    except Exception as e:
        print(f"Error during testing: {e}")

    print("Export validation system ready!")