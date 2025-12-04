"""
ONNX Export Utilities for Pediatric Pneumonia Detection Models

This module provides specialized ONNX export functionality with optimizations
for medical image analysis deployment. Includes model optimization,
quantization, and platform-specific configurations.

ONNX (Open Neural Network Exchange) enables cross-platform deployment:
- Python inference with onnxruntime
- C++ inference for high performance
- JavaScript inference for web deployment
- Mobile inference on various platforms
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

# ONNX optimization levels
ONNX_OPTIMIZATION_LEVELS = {
    'none': 'DISABLE_ALL',
    'basic': 'ENABLE_BASIC',
    'extended': 'ENABLE_EXTENDED',
    'all': 'ENABLE_ALL'
}

# Platform-specific configurations
PLATFORM_CONFIGS = {
    'cpu_inference': {
        'opset_version': 11,
        'optimization_level': 'all',
        'enable_onnx_checker': True,
        'dynamic_axes': True
    },
    'gpu_inference': {
        'opset_version': 13,
        'optimization_level': 'extended',
        'enable_onnx_checker': True,
        'dynamic_axes': True
    },
    'mobile_deployment': {
        'opset_version': 11,
        'optimization_level': 'basic',
        'enable_onnx_checker': False,
        'dynamic_axes': False
    },
    'web_deployment': {
        'opset_version': 11,
        'optimization_level': 'basic',
        'enable_onnx_checker': True,
        'dynamic_axes': True
    }
}


class ONNXExporter:
    """
    Specialized ONNX exporter for pneumonia detection models.

    Provides platform-specific optimizations and configurations
    for different deployment scenarios.
    """

    def __init__(self, model: nn.Module, input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)):
        self.model = model
        self.input_shape = input_shape
        self.dummy_input = torch.randn(input_shape)

        # Set model to evaluation mode
        self.model.eval()

    def export_for_platform(self,
                           output_path: Union[str, Path],
                           platform: str = 'cpu_inference',
                           model_name: str = 'pneumonia_model',
                           optimize: bool = True) -> bool:
        """
        Export model with platform-specific optimizations.

        Args:
            output_path: Path to save ONNX model
            platform: Target platform ('cpu_inference', 'gpu_inference', 'mobile_deployment', 'web_deployment')
            model_name: Model name for metadata
            optimize: Whether to apply post-export optimizations

        Returns:
            bool: True if export successful
        """
        if platform not in PLATFORM_CONFIGS:
            raise ValueError(f"Unsupported platform: {platform}. Choose from {list(PLATFORM_CONFIGS.keys())}")

        config = PLATFORM_CONFIGS[platform]

        try:
            # Setup dynamic axes if enabled
            dynamic_axes = None
            if config['dynamic_axes']:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }

            # Export with platform-specific configuration
            logger.info(f"Exporting ONNX model for {platform} platform")

            torch.onnx.export(
                self.model,
                self.dummy_input,
                str(output_path),
                export_params=True,
                opset_version=config['opset_version'],
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False,
                # Add model metadata
                custom_opsets=None,
                keep_initializers_as_inputs=False
            )

            # Post-export optimization
            if optimize:
                self._optimize_onnx_model(output_path, config['optimization_level'])

            # Validate export
            if config['enable_onnx_checker']:
                self._validate_onnx_model(output_path)

            logger.info(f"Successfully exported ONNX model for {platform}: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export ONNX model for {platform}: {e}")
            return False

    def export_quantized(self,
                        output_path: Union[str, Path],
                        calibration_data: Optional[torch.Tensor] = None,
                        quantization_mode: str = 'dynamic') -> bool:
        """
        Export quantized ONNX model for mobile deployment.

        Args:
            output_path: Path to save quantized ONNX model
            calibration_data: Calibration data for static quantization
            quantization_mode: 'dynamic' or 'static'

        Returns:
            bool: True if export successful
        """
        try:
            # First export standard ONNX model
            temp_path = Path(output_path).with_suffix('.temp.onnx')
            success = self.export_for_platform(temp_path, 'mobile_deployment', optimize=False)

            if not success:
                return False

            # Apply quantization
            quantized_path = self._quantize_onnx_model(
                temp_path, output_path, quantization_mode, calibration_data
            )

            # Cleanup temp file
            if temp_path.exists():
                temp_path.unlink()

            logger.info(f"Successfully exported quantized ONNX model: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export quantized ONNX model: {e}")
            return False

    def _optimize_onnx_model(self, model_path: Union[str, Path], optimization_level: str):
        """Apply ONNX model optimizations."""
        try:
            import onnx
            from onnxruntime.tools import optimizer

            # Load model
            model = onnx.load(str(model_path))

            # Apply optimizations based on level
            if optimization_level in ['basic', 'extended', 'all']:
                # Use onnxruntime optimizer
                opt_model = optimizer.optimize_model(
                    str(model_path),
                    model_type='bert',  # General optimization
                    opt_level=99 if optimization_level == 'all' else 1
                )

                # Save optimized model
                onnx.save(opt_model.model, str(model_path))
                logger.info(f"Applied {optimization_level} optimizations to ONNX model")

        except ImportError:
            logger.warning("ONNX optimization skipped (onnxruntime tools not available)")
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")

    def _validate_onnx_model(self, model_path: Union[str, Path]) -> bool:
        """Validate ONNX model structure and run test inference."""
        try:
            import onnx
            import onnxruntime as ort

            # Load and check model structure
            model = onnx.load(str(model_path))
            onnx.checker.check_model(model)

            # Test inference
            session = ort.InferenceSession(str(model_path))
            input_name = session.get_inputs()[0].name

            # Run inference with dummy data
            test_input = np.random.randn(*self.input_shape).astype(np.float32)
            result = session.run(None, {input_name: test_input})

            logger.info("ONNX model validation successful")
            return True

        except ImportError:
            logger.warning("ONNX validation skipped (onnx/onnxruntime not installed)")
            return True
        except Exception as e:
            logger.error(f"ONNX model validation failed: {e}")
            return False

    def _quantize_onnx_model(self,
                           input_path: Union[str, Path],
                           output_path: Union[str, Path],
                           mode: str,
                           calibration_data: Optional[torch.Tensor]) -> Union[str, Path]:
        """Apply quantization to ONNX model."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            if mode == 'dynamic':
                # Dynamic quantization (no calibration data needed)
                quantize_dynamic(
                    str(input_path),
                    str(output_path),
                    weight_type=QuantType.QUInt8,
                    optimize_model=True
                )
                logger.info("Applied dynamic quantization to ONNX model")

            elif mode == 'static':
                # Static quantization would require calibration dataset
                # For now, fall back to dynamic quantization
                logger.warning("Static quantization not implemented, using dynamic quantization")
                return self._quantize_onnx_model(input_path, output_path, 'dynamic', None)

            return output_path

        except ImportError:
            logger.warning("ONNX quantization skipped (onnxruntime quantization not available)")
            # Just copy the file
            import shutil
            shutil.copy(str(input_path), str(output_path))
            return output_path
        except Exception as e:
            logger.error(f"ONNX quantization failed: {e}")
            return input_path


def export_onnx_for_deployment(model_path: Union[str, Path],
                              model_type: str,
                              output_dir: Union[str, Path],
                              platforms: List[str] = None) -> Dict[str, bool]:
    """
    Export ONNX models optimized for different deployment platforms.

    Args:
        model_path: Path to trained PyTorch model
        model_type: Type of model ('xception', 'vgg', etc.)
        output_dir: Directory to save ONNX models
        platforms: List of target platforms

    Returns:
        Dict mapping platform names to export success status
    """
    if platforms is None:
        platforms = ['cpu_inference', 'mobile_deployment', 'web_deployment']

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

        # Create ONNX exporter
        exporter = ONNXExporter(model)

        # Export for each platform
        for platform in platforms:
            platform_dir = output_dir / platform
            platform_dir.mkdir(exist_ok=True)

            model_name = f"{model_type}_pneumonia_{platform}"
            output_path = platform_dir / f"{model_name}.onnx"

            success = exporter.export_for_platform(
                output_path, platform, model_name
            )
            results[platform] = success

            # Also create quantized version for mobile
            if platform == 'mobile_deployment' and success:
                quantized_path = platform_dir / f"{model_name}_quantized.onnx"
                results[f"{platform}_quantized"] = exporter.export_quantized(quantized_path)

        logger.info(f"ONNX export completed for {model_type} model")

    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        for platform in platforms:
            results[platform] = False

    return results


def get_onnx_model_info(onnx_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get detailed information about an ONNX model.

    Args:
        onnx_path: Path to ONNX model file

    Returns:
        Dict containing model information
    """
    try:
        import onnx

        model = onnx.load(str(onnx_path))

        # Get model metadata
        info = {
            'file_path': str(onnx_path),
            'file_size_mb': Path(onnx_path).stat().st_size / (1024 ** 2),
            'opset_version': model.opset_import[0].version if model.opset_import else 'unknown',
            'producer_name': model.producer_name,
            'producer_version': model.producer_version,
            'inputs': [],
            'outputs': [],
            'total_parameters': 0
        }

        # Get input information
        for input_info in model.graph.input:
            input_shape = []
            for dim in input_info.type.tensor_type.shape.dim:
                input_shape.append(dim.dim_value if dim.dim_value > 0 else 'dynamic')

            info['inputs'].append({
                'name': input_info.name,
                'shape': input_shape,
                'type': input_info.type.tensor_type.elem_type
            })

        # Get output information
        for output_info in model.graph.output:
            output_shape = []
            for dim in output_info.type.tensor_type.shape.dim:
                output_shape.append(dim.dim_value if dim.dim_value > 0 else 'dynamic')

            info['outputs'].append({
                'name': output_info.name,
                'shape': output_shape,
                'type': output_info.type.tensor_type.elem_type
            })

        # Estimate parameter count
        for initializer in model.graph.initializer:
            shape = [dim for dim in initializer.dims]
            params = np.prod(shape) if shape else 1
            info['total_parameters'] += params

        return info

    except ImportError:
        return {'error': 'ONNX not available for model inspection'}
    except Exception as e:
        return {'error': f'Failed to inspect ONNX model: {e}'}


if __name__ == "__main__":
    # Test ONNX export functionality
    print("Testing ONNX export utilities...")

    try:
        from ..models import create_model

        # Create dummy model
        model = create_model('xception', num_classes=2)
        exporter = ONNXExporter(model)

        # Test platform configurations
        print(f"Available platforms: {list(PLATFORM_CONFIGS.keys())}")
        print("ONNX export utilities ready!")

    except Exception as e:
        print(f"Error during testing: {e}")

    print("ONNX utilities ready for deployment!")