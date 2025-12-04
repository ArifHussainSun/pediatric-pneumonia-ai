"""
CoreML Export Utilities for iOS Deployment

This module provides specialized CoreML export functionality for deploying
pneumonia detection models on iOS devices (iPads, iPhones). CoreML enables
high-performance inference with hardware acceleration on Apple devices.

Features:
- Model conversion to CoreML format
- iOS-specific optimizations
- Neural Engine acceleration
- Metadata and documentation generation
- Performance validation

Designed for clinical deployment on iPads where offline inference
is critical for patient care in areas with limited connectivity.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

# CoreML deployment configurations
COREML_CONFIGS = {
    'ios_13': {
        'minimum_deployment_target': 'iOS13',
        'compute_precision': 'Float32',
        'neural_engine': False,
        'description': 'Compatible with iOS 13+ devices'
    },
    'ios_14': {
        'minimum_deployment_target': 'iOS14',
        'compute_precision': 'Float16',
        'neural_engine': True,
        'description': 'Optimized for iOS 14+ with Neural Engine'
    },
    'ios_15': {
        'minimum_deployment_target': 'iOS15',
        'compute_precision': 'Float16',
        'neural_engine': True,
        'description': 'Latest optimizations for iOS 15+'
    },
    'ipad_pro': {
        'minimum_deployment_target': 'iOS14',
        'compute_precision': 'Float16',
        'neural_engine': True,
        'description': 'Optimized for iPad Pro with M1/M2 chips'
    }
}

# Model-specific CoreML optimizations
COREML_MODEL_OPTIMIZATIONS = {
    'mobilenet': {
        'recommended_precision': 'Float16',
        'neural_engine_compatible': True,
        'expected_performance': 'excellent',
        'memory_usage': 'low'
    },
    'xception': {
        'recommended_precision': 'Float32',
        'neural_engine_compatible': False,
        'expected_performance': 'good',
        'memory_usage': 'medium'
    },
    'vgg': {
        'recommended_precision': 'Float16',
        'neural_engine_compatible': True,
        'expected_performance': 'good',
        'memory_usage': 'medium'
    },
    'fusion': {
        'recommended_precision': 'Float32',
        'neural_engine_compatible': False,
        'expected_performance': 'fair',
        'memory_usage': 'high'
    }
}


class CoreMLExporter:
    """
    Specialized CoreML exporter for pneumonia detection models.

    Provides iOS-specific optimizations and configurations for
    clinical deployment on Apple devices.
    """

    def __init__(self, model: nn.Module, model_type: str):
        """
        Initialize CoreML exporter.

        Args:
            model: PyTorch model to export
            model_type: Type of model ('mobilenet', 'xception', etc.)
        """
        self.model = model
        self.model_type = model_type

        if model_type not in COREML_MODEL_OPTIMIZATIONS:
            logger.warning(f"No specific optimizations for {model_type}, using default")
            self.optimizations = COREML_MODEL_OPTIMIZATIONS['mobilenet']
        else:
            self.optimizations = COREML_MODEL_OPTIMIZATIONS[model_type]

        # Set model to evaluation mode
        self.model.eval()

        logger.info(f"CoreMLExporter initialized for {model_type}")
        logger.info(f"Recommended precision: {self.optimizations['recommended_precision']}")
        logger.info(f"Neural Engine compatible: {self.optimizations['neural_engine_compatible']}")

    def export_to_coreml(self,
                        output_path: Union[str, Path],
                        ios_deployment: str = 'ios_14',
                        model_name: str = None,
                        model_description: str = None,
                        input_name: str = 'chest_xray_image',
                        output_name: str = 'pneumonia_prediction') -> bool:
        """
        Export model to CoreML format for iOS deployment.

        Args:
            output_path: Path to save CoreML model
            ios_deployment: iOS deployment target configuration
            model_name: Name for the CoreML model
            model_description: Description for the CoreML model
            input_name: Name for the input tensor
            output_name: Name for the output tensor

        Returns:
            bool: True if export successful
        """
        if ios_deployment not in COREML_CONFIGS:
            raise ValueError(f"Unsupported iOS deployment: {ios_deployment}. Choose from {list(COREML_CONFIGS.keys())}")

        config = COREML_CONFIGS[ios_deployment]

        try:
            import coremltools as ct

            logger.info(f"Exporting to CoreML for {ios_deployment}...")

            # Prepare model for tracing
            example_input = torch.randn(1, 3, 224, 224)
            traced_model = torch.jit.trace(self.model, example_input)

            # Set model metadata
            if model_name is None:
                model_name = f"{self.model_type.title()} Pneumonia Detection"

            if model_description is None:
                model_description = f"Pediatric pneumonia detection model ({self.model_type}) optimized for {ios_deployment}"

            # Configure input/output
            input_config = ct.ImageType(
                name=input_name,
                shape=example_input.shape,
                bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225],  # ImageNet normalization
                scale=1.0/(0.229*255.0),  # ImageNet normalization
                color_layout=ct.colorlayout.RGB
            )

            # Set compute precision
            if config['compute_precision'] == 'Float16':
                compute_precision = ct.precision.FLOAT16
            else:
                compute_precision = ct.precision.FLOAT32

            # Convert to CoreML
            coreml_model = ct.convert(
                traced_model,
                inputs=[input_config],
                outputs=[ct.TensorType(name=output_name)],
                minimum_deployment_target=getattr(ct.target, config['minimum_deployment_target']),
                compute_precision=compute_precision
            )

            # Set model metadata
            coreml_model.short_description = model_name
            coreml_model.description = model_description
            coreml_model.author = "Tech4Life Pediatric Pneumonia AI Team"
            coreml_model.license = "Medical Use Only"
            coreml_model.version = "1.0.0"

            # Add input/output descriptions
            coreml_model.input_description[input_name] = "Chest X-ray image (224x224 RGB)"
            coreml_model.output_description[output_name] = "Pneumonia prediction probabilities [Normal, Pneumonia]"

            # Add medical-specific metadata
            coreml_model.user_defined_metadata['medical_use'] = 'pediatric_pneumonia_detection'
            coreml_model.user_defined_metadata['input_requirements'] = 'chest_xray_224x224_rgb'
            coreml_model.user_defined_metadata['output_classes'] = 'normal,pneumonia'
            coreml_model.user_defined_metadata['accuracy_threshold'] = '0.95'
            coreml_model.user_defined_metadata['model_type'] = self.model_type
            coreml_model.user_defined_metadata['deployment_target'] = ios_deployment

            # Save CoreML model
            coreml_model.save(str(output_path))

            # Get model information
            model_size = Path(output_path).stat().st_size / (1024 ** 2)

            logger.info(f"CoreML export successful: {output_path}")
            logger.info(f"Model size: {model_size:.2f} MB")
            logger.info(f"Deployment target: {config['minimum_deployment_target']}")
            logger.info(f"Compute precision: {config['compute_precision']}")

            return True

        except ImportError:
            logger.error("CoreML Tools not available. Install with: pip install coremltools")
            return False
        except Exception as e:
            logger.error(f"CoreML export failed: {e}")
            return False

    def export_optimized_coreml(self,
                               output_dir: Union[str, Path],
                               quantize: bool = True,
                               validate_fn: Optional[callable] = None) -> Dict[str, Any]:
        """
        Export CoreML model with comprehensive optimizations.

        Args:
            output_dir: Directory to save optimized CoreML models
            quantize: Whether to apply quantization
            validate_fn: Function to validate model accuracy

        Returns:
            Dict containing export results for different configurations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        export_results = {}

        # Export configurations to test
        configurations = ['ios_13', 'ios_14', 'ipad_pro']

        for config_name in configurations:
            config = COREML_CONFIGS[config_name]
            logger.info(f"Exporting CoreML for {config_name}...")

            try:
                # Standard export
                model_name = f"{self.model_type}_pneumonia_{config_name}"
                output_path = output_dir / f"{model_name}.mlmodel"

                success = self.export_to_coreml(
                    output_path,
                    ios_deployment=config_name,
                    model_name=f"{self.model_type.title()} Pneumonia Detection ({config_name})"
                )

                if success:
                    model_size = output_path.stat().st_size / (1024 ** 2)

                    result = {
                        'success': True,
                        'output_path': str(output_path),
                        'model_size_mb': model_size,
                        'config': config,
                        'optimizations': self.optimizations
                    }

                    # Validate accuracy if function provided
                    if validate_fn:
                        try:
                            accuracy = self._validate_coreml_accuracy(output_path, validate_fn)
                            result['accuracy'] = accuracy
                        except Exception as e:
                            logger.warning(f"Validation failed for {config_name}: {e}")

                    # Export quantized version if requested
                    if quantize and config['compute_precision'] == 'Float16':
                        quantized_path = output_dir / f"{model_name}_quantized.mlmodel"
                        quantized_success = self._export_quantized_coreml(output_path, quantized_path)

                        if quantized_success:
                            quantized_size = quantized_path.stat().st_size / (1024 ** 2)
                            result['quantized'] = {
                                'success': True,
                                'output_path': str(quantized_path),
                                'model_size_mb': quantized_size,
                                'size_reduction': 1 - (quantized_size / model_size)
                            }

                    export_results[config_name] = result

                else:
                    export_results[config_name] = {
                        'success': False,
                        'error': 'Export failed'
                    }

            except Exception as e:
                logger.error(f"Export failed for {config_name}: {e}")
                export_results[config_name] = {
                    'success': False,
                    'error': str(e)
                }

        return export_results

    def _export_quantized_coreml(self, original_path: Path, output_path: Path) -> bool:
        """Export quantized version of CoreML model."""
        try:
            import coremltools as ct

            # Load original model
            model = ct.models.MLModel(str(original_path))

            # Apply quantization
            quantized_model = ct.models.neural_network.quantization_utils.quantize_weights(
                model, nbits=8
            )

            # Save quantized model
            quantized_model.save(str(output_path))

            logger.info(f"Quantized CoreML model saved: {output_path}")
            return True

        except Exception as e:
            logger.error(f"CoreML quantization failed: {e}")
            return False

    def _validate_coreml_accuracy(self, model_path: Path, validate_fn: callable) -> float:
        """Validate CoreML model accuracy."""
        try:
            import coremltools as ct

            # Load CoreML model
            coreml_model = ct.models.MLModel(str(model_path))

            # Create wrapper function for validation
            def coreml_predict(inputs):
                # Convert PyTorch tensor to CoreML input format
                if torch.is_tensor(inputs):
                    inputs = inputs.numpy()

                # Ensure correct shape and type
                if len(inputs.shape) == 4:
                    batch_predictions = []
                    for i in range(inputs.shape[0]):
                        img = inputs[i].transpose(1, 2, 0)  # CHW to HWC
                        img = (img * 255).astype(np.uint8)  # Convert to uint8
                        pred = coreml_model.predict({'chest_xray_image': img})
                        batch_predictions.append(pred['pneumonia_prediction'])
                    return np.array(batch_predictions)
                else:
                    img = inputs.transpose(1, 2, 0)
                    img = (img * 255).astype(np.uint8)
                    pred = coreml_model.predict({'chest_xray_image': img})
                    return pred['pneumonia_prediction']

            # Use validation function with CoreML wrapper
            accuracy = validate_fn(coreml_predict)
            return accuracy

        except Exception as e:
            logger.error(f"CoreML validation failed: {e}")
            return 0.0

    def get_coreml_info(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """Get detailed information about a CoreML model."""
        try:
            import coremltools as ct

            model = ct.models.MLModel(str(model_path))

            # Get model specifications
            spec = model.get_spec()

            info = {
                'file_path': str(model_path),
                'file_size_mb': Path(model_path).stat().st_size / (1024 ** 2),
                'description': spec.description,
                'metadata': dict(spec.metadata.userDefined),
                'inputs': [],
                'outputs': [],
                'compute_precision': str(spec.description.metadata.versionInfo)
            }

            # Get input information
            for input_desc in spec.description.input:
                input_info = {
                    'name': input_desc.name,
                    'type': str(input_desc.type).split('.')[1] if '.' in str(input_desc.type) else str(input_desc.type)
                }

                if input_desc.type.HasField('imageType'):
                    img_type = input_desc.type.imageType
                    input_info.update({
                        'width': img_type.width,
                        'height': img_type.height,
                        'color_space': str(img_type.colorSpace).split('.')[-1]
                    })

                info['inputs'].append(input_info)

            # Get output information
            for output_desc in spec.description.output:
                output_info = {
                    'name': output_desc.name,
                    'type': str(output_desc.type).split('.')[1] if '.' in str(output_desc.type) else str(output_desc.type)
                }
                info['outputs'].append(output_info)

            return info

        except ImportError:
            return {'error': 'CoreML Tools not available'}
        except Exception as e:
            return {'error': f'Failed to inspect CoreML model: {e}'}


def export_coreml_for_ios(model_path: Union[str, Path],
                         model_type: str,
                         output_dir: Union[str, Path],
                         ios_targets: List[str] = None) -> Dict[str, bool]:
    """
    Export CoreML models optimized for different iOS deployment scenarios.

    Args:
        model_path: Path to trained PyTorch model
        model_type: Type of model ('mobilenet', 'xception', etc.)
        output_dir: Directory to save CoreML models
        ios_targets: List of iOS deployment targets

    Returns:
        Dict mapping iOS targets to export success status
    """
    if ios_targets is None:
        ios_targets = ['ios_13', 'ios_14', 'ipad_pro']

    from ..models import create_model

    results = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load model
        model = create_model(model_type, num_classes=2)

        # Load trained weights
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        # Create CoreML exporter
        exporter = CoreMLExporter(model, model_type)

        # Export for each iOS target
        for ios_target in ios_targets:
            target_dir = output_dir / ios_target
            target_dir.mkdir(exist_ok=True)

            model_name = f"{model_type}_pneumonia_{ios_target}"
            output_path = target_dir / f"{model_name}.mlmodel"

            success = exporter.export_to_coreml(
                output_path, ios_target, model_name
            )
            results[ios_target] = success

        logger.info(f"CoreML export completed for {model_type} model")

    except Exception as e:
        logger.error(f"CoreML export failed: {e}")
        for target in ios_targets:
            results[target] = False

    return results


if __name__ == "__main__":
    # Test CoreML export functionality
    print("Testing CoreML export utilities...")

    try:
        from ..models import create_model

        # Create dummy model
        model = create_model('mobilenet', num_classes=2)
        exporter = CoreMLExporter(model, 'mobilenet')

        print(f"Available iOS targets: {list(COREML_CONFIGS.keys())}")
        print("CoreML export utilities ready!")

    except Exception as e:
        print(f"Error during testing: {e}")

    print("CoreML utilities ready for iOS deployment!")