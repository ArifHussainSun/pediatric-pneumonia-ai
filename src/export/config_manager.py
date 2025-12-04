"""
Export Configuration Manager for Pediatric Pneumonia Detection Models

This module handles loading, validation, and management of export configurations
for different deployment scenarios. Provides a centralized way to manage
export settings across different models, platforms, and use cases.

The configuration system supports:
- Model-specific export settings
- Platform-specific optimizations
- Quality profiles for different accuracy/efficiency trade-offs
- Deployment target configurations
- Validation and monitoring settings
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import warnings
warnings.filterwarnings('ignore')

import yaml

# Setup logging
logger = logging.getLogger(__name__)


class ExportConfigManager:
    """
    Configuration manager for model export settings.

    Handles loading, validation, and access to export configurations
    for different models, platforms, and deployment scenarios.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        if config_path is None:
            # Use default config file
            config_path = Path(__file__).parent.parent.parent / "configs" / "export_config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

        logger.info(f"ExportConfigManager initialized with config: {self.config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            logger.info("Configuration loaded successfully")
            return config

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        return {
            'global': {
                'input_shape': [1, 3, 224, 224],
                'output_dir': './exports',
                'verify_exports': True
            },
            'models': {
                'xception': {
                    'recommended_formats': ['onnx', 'torchscript'],
                    'export_settings': {
                        'onnx': {'opset_version': 11},
                        'torchscript': {'method': 'trace'}
                    }
                }
            }
        }

    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model type.

        Args:
            model_type: Type of model ('xception', 'vgg', 'mobilenet', etc.)

        Returns:
            Dict containing model-specific configuration
        """
        models_config = self.config.get('models', {})

        if model_type not in models_config:
            logger.warning(f"No configuration found for model type: {model_type}")
            return self._get_default_model_config()

        return models_config[model_type]

    def get_deployment_config(self, deployment_target: str) -> Dict[str, Any]:
        """
        Get configuration for a specific deployment target.

        Args:
            deployment_target: Target deployment ('cloud_inference', 'mobile_devices', etc.)

        Returns:
            Dict containing deployment-specific configuration
        """
        deployment_config = self.config.get('deployment_targets', {})

        if deployment_target not in deployment_config:
            logger.warning(f"No configuration found for deployment target: {deployment_target}")
            return {}

        return deployment_config[deployment_target]

    def get_platform_config(self, platform: str) -> Dict[str, Any]:
        """
        Get configuration for a specific platform.

        Args:
            platform: Target platform ('ios', 'android', 'windows', etc.)

        Returns:
            Dict containing platform-specific configuration
        """
        platform_config = self.config.get('platforms', {})

        if platform not in platform_config:
            logger.warning(f"No configuration found for platform: {platform}")
            return {}

        return platform_config[platform]

    def get_quality_profile(self, profile_name: str) -> Dict[str, Any]:
        """
        Get quality profile configuration.

        Args:
            profile_name: Quality profile ('maximum_accuracy', 'balanced', 'maximum_efficiency')

        Returns:
            Dict containing quality profile configuration
        """
        quality_profiles = self.config.get('quality_profiles', {})

        if profile_name not in quality_profiles:
            logger.warning(f"No configuration found for quality profile: {profile_name}")
            return {}

        return quality_profiles[profile_name]

    def get_export_settings(self,
                           model_type: str,
                           format_type: str,
                           deployment_target: Optional[str] = None,
                           platform: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive export settings for a specific scenario.

        Args:
            model_type: Type of model
            format_type: Export format ('onnx', 'torchscript', 'coreml', 'tflite')
            deployment_target: Optional deployment target
            platform: Optional target platform

        Returns:
            Dict containing merged export settings
        """
        # Start with global settings
        settings = self.config.get('global', {}).copy()

        # Add model-specific settings
        model_config = self.get_model_config(model_type)
        model_export_settings = model_config.get('export_settings', {}).get(format_type, {})
        settings.update(model_export_settings)

        # Add deployment target settings
        if deployment_target:
            deployment_config = self.get_deployment_config(deployment_target)
            deployment_optimizations = deployment_config.get('optimizations', {})
            settings.update(deployment_optimizations)

        # Add platform-specific settings
        if platform:
            platform_config = self.get_platform_config(platform)
            platform_optimizations = platform_config.get('optimizations', {})
            settings.update(platform_optimizations)

        return settings

    def get_recommended_formats(self, model_type: str, deployment_target: Optional[str] = None) -> List[str]:
        """
        Get recommended export formats for a model and deployment scenario.

        Args:
            model_type: Type of model
            deployment_target: Optional deployment target

        Returns:
            List of recommended format names
        """
        # Get model's recommended formats
        model_config = self.get_model_config(model_type)
        model_formats = model_config.get('recommended_formats', ['onnx'])

        # Filter by deployment target if specified
        if deployment_target:
            deployment_config = self.get_deployment_config(deployment_target)
            deployment_formats = deployment_config.get('formats', [])

            if deployment_formats:
                # Intersection of model and deployment formats
                recommended = [fmt for fmt in model_formats if fmt in deployment_formats]
                return recommended if recommended else model_formats

        return model_formats

    def get_recommended_models(self, deployment_target: str) -> List[str]:
        """
        Get recommended models for a deployment target.

        Args:
            deployment_target: Target deployment scenario

        Returns:
            List of recommended model types
        """
        deployment_config = self.get_deployment_config(deployment_target)
        return deployment_config.get('recommended_models', ['xception'])

    def validate_export_request(self,
                               model_type: str,
                               format_type: str,
                               deployment_target: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate an export request against configuration.

        Args:
            model_type: Type of model
            format_type: Export format
            deployment_target: Optional deployment target

        Returns:
            Dict containing validation results and recommendations
        """
        result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }

        # Check if model type is supported
        if model_type not in self.config.get('models', {}):
            result['warnings'].append(f"Model type '{model_type}' not in configuration")

        # Check if format is recommended for model
        recommended_formats = self.get_recommended_formats(model_type, deployment_target)
        if format_type not in recommended_formats:
            result['warnings'].append(
                f"Format '{format_type}' not recommended for model '{model_type}'. "
                f"Recommended: {recommended_formats}"
            )

        # Check deployment target compatibility
        if deployment_target:
            deployment_config = self.get_deployment_config(deployment_target)
            if not deployment_config:
                result['errors'].append(f"Unknown deployment target: '{deployment_target}'")
                result['valid'] = False
            else:
                # Check if model is recommended for deployment target
                recommended_models = deployment_config.get('recommended_models', [])
                if model_type not in recommended_models:
                    result['recommendations'].append(
                        f"Consider using {recommended_models} for {deployment_target} deployment"
                    )

        return result

    def get_validation_settings(self) -> Dict[str, Any]:
        """Get validation settings for exported models."""
        return self.config.get('validation', {})

    def get_monitoring_settings(self) -> Dict[str, Any]:
        """Get monitoring settings for export process."""
        return self.config.get('monitoring', {})

    def _get_default_model_config(self) -> Dict[str, Any]:
        """Get default model configuration."""
        return {
            'description': 'Default model configuration',
            'priority': 'balance',
            'recommended_formats': ['onnx', 'torchscript'],
            'export_settings': {
                'onnx': {
                    'opset_version': 11,
                    'optimization_level': 'basic'
                },
                'torchscript': {
                    'method': 'trace',
                    'optimize_for_inference': True
                }
            }
        }

    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates
        """
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_update(self.config, updates)
        logger.info("Configuration updated")

    def save_config(self, output_path: Optional[Union[str, Path]] = None):
        """
        Save current configuration to file.

        Args:
            output_path: Path to save configuration. If None, overwrites original file.
        """
        if output_path is None:
            output_path = self.config_path

        try:
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)

            logger.info(f"Configuration saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def list_available_options(self) -> Dict[str, List[str]]:
        """
        List all available configuration options.

        Returns:
            Dict containing lists of available options
        """
        return {
            'models': list(self.config.get('models', {}).keys()),
            'deployment_targets': list(self.config.get('deployment_targets', {}).keys()),
            'platforms': list(self.config.get('platforms', {}).keys()),
            'quality_profiles': list(self.config.get('quality_profiles', {}).keys())
        }

    def print_summary(self):
        """Print a summary of the loaded configuration."""
        options = self.list_available_options()

        print("Export Configuration Summary")
        print("=" * 40)
        print(f"Configuration file: {self.config_path}")
        print(f"Models available: {len(options['models'])}")
        print(f"  - {', '.join(options['models'])}")
        print(f"Deployment targets: {len(options['deployment_targets'])}")
        print(f"  - {', '.join(options['deployment_targets'])}")
        print(f"Platforms: {len(options['platforms'])}")
        print(f"  - {', '.join(options['platforms'])}")
        print(f"Quality profiles: {len(options['quality_profiles'])}")
        print(f"  - {', '.join(options['quality_profiles'])}")


# Convenience functions for direct access
def load_export_config(config_path: Optional[Union[str, Path]] = None) -> ExportConfigManager:
    """Load export configuration manager."""
    return ExportConfigManager(config_path)


def get_model_export_settings(model_type: str,
                             format_type: str,
                             config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Get export settings for a specific model and format."""
    config_manager = ExportConfigManager(config_path)
    return config_manager.get_export_settings(model_type, format_type)


def validate_export_config(model_type: str,
                          format_type: str,
                          deployment_target: Optional[str] = None,
                          config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Validate an export configuration."""
    config_manager = ExportConfigManager(config_path)
    return config_manager.validate_export_request(model_type, format_type, deployment_target)


if __name__ == "__main__":
    # Test configuration manager
    print("Testing ExportConfigManager...")

    try:
        # Create config manager
        config_manager = ExportConfigManager()
        config_manager.print_summary()

        # Test getting model config
        xception_config = config_manager.get_model_config('xception')
        print(f"\nXception config: {xception_config}")

        # Test validation
        validation = config_manager.validate_export_request('mobilenet', 'coreml', 'mobile_devices')
        print(f"\nValidation result: {validation}")

        print("\nExportConfigManager ready!")

    except Exception as e:
        print(f"Error during testing: {e}")

    print("Export configuration system ready!")