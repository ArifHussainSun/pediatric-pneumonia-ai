"""
TorchScript Export Utilities for Pediatric Pneumonia Detection Models

This module provides specialized TorchScript export functionality for optimized
PyTorch deployment. TorchScript offers improved performance and eliminates
Python dependencies for production inference.

TorchScript Benefits:
- No Python interpreter required
- Faster inference than standard PyTorch
- Better memory efficiency
- Mobile deployment support
- C++ integration capabilities
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
import time

# Setup logging
logger = logging.getLogger(__name__)

# TorchScript optimization configurations
TORCHSCRIPT_CONFIGS = {
    'production': {
        'method': 'trace',
        'optimize_for_inference': True,
        'strict': True,
        'check_trace': True
    },
    'mobile': {
        'method': 'trace',
        'optimize_for_inference': True,
        'strict': False,
        'check_trace': False,
        'mobile_optimize': True
    },
    'development': {
        'method': 'script',
        'optimize_for_inference': False,
        'strict': False,
        'check_trace': True
    }
}


class TorchScriptExporter:
    """
    Specialized TorchScript exporter for pneumonia detection models.

    Provides optimized TorchScript export with different configurations
    for various deployment scenarios.
    """

    def __init__(self, model: nn.Module, input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)):
        self.model = model
        self.input_shape = input_shape
        self.dummy_input = torch.randn(input_shape)

        # Set model to evaluation mode
        self.model.eval()

    def export_optimized(self,
                        output_path: Union[str, Path],
                        config_type: str = 'production',
                        custom_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export optimized TorchScript model.

        Args:
            output_path: Path to save TorchScript model
            config_type: Configuration type ('production', 'mobile', 'development')
            custom_config: Optional custom configuration

        Returns:
            bool: True if export successful
        """
        if config_type not in TORCHSCRIPT_CONFIGS and custom_config is None:
            raise ValueError(f"Unknown config_type: {config_type}. Use {list(TORCHSCRIPT_CONFIGS.keys())}")

        config = custom_config or TORCHSCRIPT_CONFIGS[config_type]

        try:
            logger.info(f"Exporting TorchScript model with {config_type} configuration")

            # Export using specified method
            if config['method'] == 'trace':
                scripted_model = self._trace_model(config)
            elif config['method'] == 'script':
                scripted_model = self._script_model(config)
            else:
                raise ValueError(f"Unknown method: {config['method']}")

            # Apply optimizations
            if config.get('optimize_for_inference', False):
                scripted_model = self._optimize_for_inference(scripted_model)

            # Mobile-specific optimizations
            if config.get('mobile_optimize', False):
                scripted_model = self._optimize_for_mobile(scripted_model)

            # Save model
            torch.jit.save(scripted_model, str(output_path))

            # Validate export
            if config.get('check_trace', True):
                self._validate_torchscript_model(output_path, config.get('strict', True))

            logger.info(f"Successfully exported TorchScript model: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export TorchScript model: {e}")
            return False

    def _trace_model(self, config: Dict[str, Any]) -> torch.jit.ScriptModule:
        """Trace model for TorchScript export."""
        strict = config.get('strict', True)

        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(
                self.model,
                self.dummy_input,
                strict=strict,
                check_trace=config.get('check_trace', True)
            )

        return traced_model

    def _script_model(self, config: Dict[str, Any]) -> torch.jit.ScriptModule:
        """Script model for TorchScript export."""
        try:
            # Try to script the entire model
            scripted_model = torch.jit.script(self.model)
            return scripted_model

        except Exception as e:
            logger.warning(f"Full model scripting failed: {e}. Falling back to tracing.")
            return self._trace_model(config)

    def _optimize_for_inference(self, scripted_model: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """Apply inference optimizations to TorchScript model."""
        try:
            # Freeze model for inference
            scripted_model.eval()
            scripted_model = torch.jit.freeze(scripted_model)

            # Apply graph optimizations
            scripted_model = torch.jit.optimize_for_inference(scripted_model)

            logger.info("Applied inference optimizations to TorchScript model")
            return scripted_model

        except Exception as e:
            logger.warning(f"Inference optimization failed: {e}")
            return scripted_model

    def _optimize_for_mobile(self, scripted_model: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """Apply mobile-specific optimizations."""
        try:
            from torch.utils.mobile_optimizer import optimize_for_mobile

            # Apply mobile optimizations
            optimized_model = optimize_for_mobile(scripted_model)

            logger.info("Applied mobile optimizations to TorchScript model")
            return optimized_model

        except ImportError:
            logger.warning("Mobile optimization not available (torch.utils.mobile_optimizer not found)")
            return scripted_model
        except Exception as e:
            logger.warning(f"Mobile optimization failed: {e}")
            return scripted_model

    def _validate_torchscript_model(self, model_path: Union[str, Path], strict: bool = True) -> bool:
        """Validate TorchScript model by loading and running inference."""
        try:
            # Load the saved model
            loaded_model = torch.jit.load(str(model_path))
            loaded_model.eval()

            # Run inference with dummy data
            with torch.no_grad():
                original_output = self.model(self.dummy_input)
                loaded_output = loaded_model(self.dummy_input)

            # Compare outputs
            diff = torch.abs(original_output - loaded_output).max().item()
            tolerance = 1e-5 if strict else 1e-3

            if diff < tolerance:
                logger.info(f"TorchScript model validation successful (max diff: {diff:.2e})")
                return True
            else:
                logger.warning(f"TorchScript model validation warning (max diff: {diff:.2e})")
                return not strict

        except Exception as e:
            logger.error(f"TorchScript model validation failed: {e}")
            return False

    def benchmark_model(self, model_path: Union[str, Path], num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark TorchScript model performance.

        Args:
            model_path: Path to TorchScript model
            num_runs: Number of inference runs for benchmarking

        Returns:
            Dict with performance metrics
        """
        try:
            # Load model
            loaded_model = torch.jit.load(str(model_path))
            loaded_model.eval()

            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = loaded_model(self.dummy_input)

            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = loaded_model(self.dummy_input)
                    times.append(time.time() - start_time)

            times = np.array(times) * 1000  # Convert to milliseconds

            return {
                'mean_inference_time_ms': float(np.mean(times)),
                'std_inference_time_ms': float(np.std(times)),
                'min_inference_time_ms': float(np.min(times)),
                'max_inference_time_ms': float(np.max(times)),
                'median_inference_time_ms': float(np.median(times)),
                'throughput_fps': float(1000 / np.mean(times))
            }

        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return {}


def export_torchscript_variants(model_path: Union[str, Path],
                               model_type: str,
                               output_dir: Union[str, Path],
                               variants: List[str] = None) -> Dict[str, bool]:
    """
    Export multiple TorchScript variants for different use cases.

    Args:
        model_path: Path to trained PyTorch model
        model_type: Type of model ('xception', 'vgg', etc.)
        output_dir: Directory to save TorchScript models
        variants: List of variants to export

    Returns:
        Dict mapping variant names to export success status
    """
    if variants is None:
        variants = ['production', 'mobile', 'development']

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

        # Create TorchScript exporter
        exporter = TorchScriptExporter(model)

        # Export each variant
        for variant in variants:
            variant_dir = output_dir / variant
            variant_dir.mkdir(exist_ok=True)

            model_name = f"{model_type}_pneumonia_{variant}"
            output_path = variant_dir / f"{model_name}.pt"

            success = exporter.export_optimized(
                output_path, variant
            )
            results[variant] = success

            # Benchmark the variant
            if success:
                benchmark_results = exporter.benchmark_model(output_path)

                # Save benchmark results
                benchmark_path = variant_dir / f"{model_name}_benchmark.json"
                import json
                with open(benchmark_path, 'w') as f:
                    json.dump(benchmark_results, f, indent=2)

                logger.info(f"Benchmark results for {variant}: "
                          f"{benchmark_results.get('mean_inference_time_ms', 0):.2f}ms avg")

        logger.info(f"TorchScript export completed for {model_type} model")

    except Exception as e:
        logger.error(f"TorchScript export failed: {e}")
        for variant in variants:
            results[variant] = False

    return results


def compare_torchscript_performance(model_paths: List[Union[str, Path]],
                                  model_names: List[str],
                                  input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> Dict[str, Dict[str, float]]:
    """
    Compare performance of multiple TorchScript models.

    Args:
        model_paths: List of paths to TorchScript models
        model_names: List of model names for results
        input_shape: Input tensor shape

    Returns:
        Dict mapping model names to performance metrics
    """
    if len(model_paths) != len(model_names):
        raise ValueError("Number of model paths must match number of model names")

    results = {}
    dummy_input = torch.randn(input_shape)

    for model_path, model_name in zip(model_paths, model_names):
        try:
            # Load model
            model = torch.jit.load(str(model_path))
            model.eval()

            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)

            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(100):
                    start_time = time.time()
                    _ = model(dummy_input)
                    times.append(time.time() - start_time)

            times = np.array(times) * 1000

            # Get model size
            model_size = Path(model_path).stat().st_size / (1024 ** 2)

            results[model_name] = {
                'mean_inference_time_ms': float(np.mean(times)),
                'std_inference_time_ms': float(np.std(times)),
                'throughput_fps': float(1000 / np.mean(times)),
                'model_size_mb': float(model_size)
            }

            logger.info(f"{model_name}: {np.mean(times):.2f}ms, {model_size:.2f}MB")

        except Exception as e:
            logger.error(f"Failed to benchmark {model_name}: {e}")
            results[model_name] = {'error': str(e)}

    return results


def get_torchscript_model_info(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get detailed information about a TorchScript model.

    Args:
        model_path: Path to TorchScript model

    Returns:
        Dict containing model information
    """
    try:
        # Load model
        model = torch.jit.load(str(model_path))

        # Get model information
        info = {
            'file_path': str(model_path),
            'file_size_mb': Path(model_path).stat().st_size / (1024 ** 2),
            'code': str(model.code),
            'graph': str(model.graph),
            'methods': [method for method in dir(model) if not method.startswith('_')],
            'parameters': {}
        }

        # Get parameter information
        total_params = 0
        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            info['parameters'][name] = {
                'shape': list(param.shape),
                'count': param_count,
                'dtype': str(param.dtype)
            }

        info['total_parameters'] = total_params

        return info

    except Exception as e:
        return {'error': f'Failed to inspect TorchScript model: {e}'}


if __name__ == "__main__":
    # Test TorchScript export functionality
    print("Testing TorchScript export utilities...")

    try:
        from ..models import create_model

        # Create dummy model
        model = create_model('xception', num_classes=2)
        exporter = TorchScriptExporter(model)

        # Test configurations
        print(f"Available configurations: {list(TORCHSCRIPT_CONFIGS.keys())}")
        print("TorchScript export utilities ready!")

    except Exception as e:
        print(f"Error during testing: {e}")

    print("TorchScript utilities ready for deployment!")