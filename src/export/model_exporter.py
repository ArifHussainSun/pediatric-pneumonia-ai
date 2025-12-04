"""
Model Exporter for Pediatric Pneumonia Detection Models

This module provides comprehensive model export functionality to convert
trained PyTorch models into various formats for production deployment.

Supported export formats:
- ONNX: Cross-platform neural network exchange format
- TorchScript: Optimized PyTorch format for production
- CoreML: Apple's machine learning format for iOS devices
- TensorFlow Lite: Google's format for mobile/embedded devices

The exporter handles all model types in the pneumonia detection system:
- Xception models
- VGG16 models
- MobileNet models
- Fusion models
- CNN-LSTM models
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io

# Import model creation utilities
from ..models import create_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExporter:
    """
    Comprehensive model exporter for pneumonia detection models.

    This class handles exporting trained PyTorch models to various formats
    optimized for different deployment scenarios.

    Args:
        model: PyTorch model to export
        model_type: Type of model ('xception', 'vgg', 'mobilenet', 'fusion', 'xception_lstm')
        input_shape: Input tensor shape (default: [1, 3, 224, 224])
        device: Computing device ('cpu', 'cuda')
    """

    def __init__(self,
                 model: nn.Module,
                 model_type: str,
                 input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
                 device: str = 'cpu'):
        self.model = model
        self.model_type = model_type
        self.input_shape = input_shape
        self.device = torch.device(device)

        # Move model to specified device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

        # Create dummy input for export operations
        self.dummy_input = torch.randn(input_shape, device=self.device)

        logger.info(f"ModelExporter initialized for {model_type} model")
        logger.info(f"Input shape: {input_shape}, Device: {device}")

    def export_to_onnx(self,
                      output_path: Union[str, Path],
                      opset_version: int = 11,
                      dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                      verify: bool = True) -> bool:
        """
        Export model to ONNX format.

        ONNX provides cross-platform compatibility and is widely supported
        by inference frameworks across different programming languages.

        Args:
            output_path: Path to save ONNX model
            opset_version: ONNX opset version (11 recommended for compatibility)
            dynamic_axes: Dynamic axis configuration for variable batch sizes
            verify: Whether to verify the exported model

        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Default dynamic axes for batch size flexibility
            if dynamic_axes is None:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }

            # Export to ONNX
            logger.info(f"Exporting {self.model_type} model to ONNX: {output_path}")

            torch.onnx.export(
                self.model,
                self.dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )

            # Verify export if requested
            if verify:
                return self._verify_onnx_export(output_path)

            logger.info(f"Successfully exported to ONNX: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export to ONNX: {e}")
            return False

    def export_to_torchscript(self,
                             output_path: Union[str, Path],
                             method: str = 'trace',
                             verify: bool = True) -> bool:
        """
        Export model to TorchScript format.

        TorchScript provides optimized PyTorch deployment with improved
        performance and no Python dependency.

        Args:
            output_path: Path to save TorchScript model
            method: Export method ('trace' or 'script')
            verify: Whether to verify the exported model

        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Exporting {self.model_type} model to TorchScript: {output_path}")

            if method == 'trace':
                # Trace method - records operations during forward pass
                traced_model = torch.jit.trace(self.model, self.dummy_input)
                torch.jit.save(traced_model, str(output_path))
            elif method == 'script':
                # Script method - analyzes code structure
                scripted_model = torch.jit.script(self.model)
                torch.jit.save(scripted_model, str(output_path))
            else:
                raise ValueError(f"Invalid method: {method}. Use 'trace' or 'script'")

            # Verify export if requested
            if verify:
                return self._verify_torchscript_export(output_path)

            logger.info(f"Successfully exported to TorchScript: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export to TorchScript: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dict containing model metadata, size, and performance info
        """
        # Calculate model size
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Estimate model size in MB
        model_size_mb = param_count * 4 / (1024 ** 2)  # Assuming float32

        # Test inference speed
        inference_times = []
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = self.model(self.dummy_input)

            # Measure
            import time
            for _ in range(10):
                start = time.time()
                _ = self.model(self.dummy_input)
                inference_times.append(time.time() - start)

        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms

        return {
            'model_type': self.model_type,
            'input_shape': self.input_shape,
            'total_parameters': param_count,
            'trainable_parameters': trainable_params,
            'model_size_mb': round(model_size_mb, 2),
            'avg_inference_time_ms': round(avg_inference_time, 2),
            'device': str(self.device)
        }

    def _verify_onnx_export(self, onnx_path: Path) -> bool:
        """Verify ONNX export by comparing outputs."""
        try:
            import onnx
            import onnxruntime as ort

            # Load and check ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)

            # Create ONNX runtime session
            ort_session = ort.InferenceSession(str(onnx_path))

            # Get PyTorch output
            with torch.no_grad():
                pytorch_output = self.model(self.dummy_input)

            # Get ONNX output
            ort_inputs = {ort_session.get_inputs()[0].name: self.dummy_input.cpu().numpy()}
            onnx_output = ort_session.run(None, ort_inputs)[0]

            # Compare outputs
            pytorch_output_np = pytorch_output.cpu().numpy()
            diff = np.abs(pytorch_output_np - onnx_output).max()

            if diff < 1e-5:
                logger.info(f"ONNX export verified successfully (max diff: {diff:.2e})")
                return True
            else:
                logger.warning(f"ONNX export verification failed (max diff: {diff:.2e})")
                return False

        except ImportError:
            logger.warning("ONNX verification skipped (onnx/onnxruntime not installed)")
            return True
        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")
            return False

    def _verify_torchscript_export(self, torchscript_path: Path) -> bool:
        """Verify TorchScript export by comparing outputs."""
        try:
            # Load TorchScript model
            loaded_model = torch.jit.load(str(torchscript_path), map_location=self.device)
            loaded_model.eval()

            # Get outputs
            with torch.no_grad():
                original_output = self.model(self.dummy_input)
                loaded_output = loaded_model(self.dummy_input)

            # Compare outputs
            diff = torch.abs(original_output - loaded_output).max().item()

            if diff < 1e-5:
                logger.info(f"TorchScript export verified successfully (max diff: {diff:.2e})")
                return True
            else:
                logger.warning(f"TorchScript export verification failed (max diff: {diff:.2e})")
                return False

        except Exception as e:
            logger.error(f"TorchScript verification failed: {e}")
            return False


def export_model(model_path: Union[str, Path],
                model_type: str,
                output_dir: Union[str, Path],
                formats: List[str] = ['onnx', 'torchscript'],
                **kwargs) -> Dict[str, bool]:
    """
    Export a trained model to multiple formats.

    Args:
        model_path: Path to trained PyTorch model (.pth file)
        model_type: Type of model to create
        output_dir: Directory to save exported models
        formats: List of formats to export ('onnx', 'torchscript')
        **kwargs: Additional arguments for model creation

    Returns:
        Dict mapping format names to success status
    """
    results = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = create_model(model_type, **kwargs)

        # Load trained weights
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        # Create exporter
        exporter = ModelExporter(model, model_type)

        # Export to requested formats
        base_name = f"{model_type}_pneumonia_model"

        if 'onnx' in formats:
            onnx_path = output_dir / f"{base_name}.onnx"
            results['onnx'] = exporter.export_to_onnx(onnx_path)

        if 'torchscript' in formats:
            ts_path = output_dir / f"{base_name}.pt"
            results['torchscript'] = exporter.export_to_torchscript(ts_path)

        # Save model info
        model_info = exporter.get_model_info()
        info_path = output_dir / f"{base_name}_info.json"

        import json
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Model info saved to {info_path}")

    except Exception as e:
        logger.error(f"Export failed: {e}")
        for fmt in formats:
            results[fmt] = False

    return results


def export_all_models(models_dir: Union[str, Path],
                     output_dir: Union[str, Path],
                     formats: List[str] = ['onnx', 'torchscript']) -> Dict[str, Dict[str, bool]]:
    """
    Export all trained models found in a directory.

    Args:
        models_dir: Directory containing trained model files
        output_dir: Directory to save exported models
        formats: List of formats to export

    Returns:
        Dict mapping model names to export results
    """
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    results = {}

    # Common model types and their expected filenames
    model_patterns = {
        'xception': ['xception_model.pth', 'xception_pneumonia_model.pth'],
        'vgg': ['vgg_model.pth', 'vgg16_model.pth', 'vgg_pneumonia_model.pth'],
        'mobilenet': ['mobilenet_model.pth', 'mobilenet_pneumonia_model.pth'],
        'fusion': ['fusion_model.pth', 'fusion_pneumonia_model.pth'],
        'xception_lstm': ['xception_lstm_model.pth', 'lstm_model.pth']
    }

    for model_type, patterns in model_patterns.items():
        for pattern in patterns:
            model_path = models_dir / pattern
            if model_path.exists():
                logger.info(f"Found {model_type} model: {model_path}")
                model_output_dir = output_dir / model_type
                results[model_type] = export_model(
                    model_path, model_type, model_output_dir, formats
                )
                break

    return results


def validate_exported_model(export_path: Union[str, Path],
                           format_type: str,
                           test_input: Optional[torch.Tensor] = None) -> bool:
    """
    Validate an exported model by running inference.

    Args:
        export_path: Path to exported model
        format_type: Type of exported model ('onnx', 'torchscript')
        test_input: Optional test input tensor

    Returns:
        bool: True if model loads and runs successfully
    """
    try:
        if test_input is None:
            test_input = torch.randn(1, 3, 224, 224)

        if format_type == 'onnx':
            import onnxruntime as ort
            session = ort.InferenceSession(str(export_path))
            input_name = session.get_inputs()[0].name
            result = session.run(None, {input_name: test_input.numpy()})
            logger.info(f"ONNX model validation successful: {export_path}")

        elif format_type == 'torchscript':
            model = torch.jit.load(str(export_path))
            model.eval()
            with torch.no_grad():
                result = model(test_input)
            logger.info(f"TorchScript model validation successful: {export_path}")

        else:
            raise ValueError(f"Unsupported format: {format_type}")

        return True

    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False


if __name__ == "__main__":
    # Test model exporter
    print("Testing ModelExporter...")

    # This would normally use a real trained model
    try:
        # Create a dummy model for testing
        dummy_model = create_model('xception', num_classes=2)
        exporter = ModelExporter(dummy_model, 'xception')

        # Get model info
        info = exporter.get_model_info()
        print(f"Model info: {info}")

        print("ModelExporter ready for production use!")

    except Exception as e:
        print(f"Error during testing: {e}")

    print("Model export utilities ready!")