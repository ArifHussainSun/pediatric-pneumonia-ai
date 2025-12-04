"""
API Configuration Management

Handles loading and validation of API deployment configurations
for production environments. Supports environment-specific settings
and runtime configuration updates.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    timeout_seconds: int = 30
    max_request_size_mb: int = 10
    log_level: str = "INFO"


@dataclass
class ModelConfig:
    file: str
    device: str = "cpu"
    max_memory_mb: int = 512
    warmup_samples: int = 5


@dataclass
class BatchConfig:
    max_concurrent: int = 5
    max_batch_size: int = 20
    timeout_seconds: int = 300
    queue_size_limit: int = 100


@dataclass
class SecurityConfig:
    enable_auth: bool = False
    api_key_header: str = "X-API-Key"
    allowed_ips: list = None
    enable_https: bool = False


class APIConfig:
    """
    Configuration manager for API deployment settings.

    Loads settings from YAML files with environment variable overrides
    and provides typed access to configuration values.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize API configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "api_config.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()

        # Initialize typed configs
        self.server = ServerConfig(**self._config.get('server', {}))
        self.batch = BatchConfig(**self._config.get('batch_processing', {}))
        self.security = SecurityConfig(**self._config.get('security', {}))

        # Model configs
        self.models = {}
        for name, config in self._config.get('models', {}).items():
            self.models[name] = ModelConfig(**config)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with environment overrides."""
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            config = {}

        # Apply environment variable overrides
        config = self._apply_env_overrides(config)

        return config

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        # Server overrides
        if 'API_HOST' in os.environ:
            config.setdefault('server', {})['host'] = os.environ['API_HOST']
        if 'API_PORT' in os.environ:
            config.setdefault('server', {})['port'] = int(os.environ['API_PORT'])
        if 'API_WORKERS' in os.environ:
            config.setdefault('server', {})['workers'] = int(os.environ['API_WORKERS'])

        # Security overrides
        if 'API_ENABLE_AUTH' in os.environ:
            config.setdefault('security', {})['enable_auth'] = os.environ['API_ENABLE_AUTH'].lower() == 'true'

        return config

    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration."""
        return self._config.get('cors', {
            'allow_origins': ['*'],
            'allow_credentials': True,
            'allow_methods': ['GET', 'POST', 'OPTIONS'],
            'allow_headers': ['*']
        })

    def get_rate_limiting_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration."""
        return self._config.get('rate_limiting', {
            'requests_per_minute': 120,
            'burst_limit': 10,
            'enable_rate_limiting': True
        })

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return self._config.get('monitoring', {
            'enable_metrics': True,
            'metrics_endpoint': '/metrics',
            'health_check_interval': 30,
            'log_requests': True,
            'track_performance': True
        })

    def get_clinical_config(self) -> Dict[str, Any]:
        """Get clinical settings."""
        return self._config.get('clinical', {
            'default_confidence_threshold': 0.5,
            'enable_quality_checks': True,
            'require_patient_id': False,
            'audit_logging': True
        })

    def update_config(self, updates: Dict[str, Any]):
        """Update configuration values at runtime."""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_update(self._config, updates)

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Validate server config
        if self.server.port < 1 or self.server.port > 65535:
            issues.append(f"Invalid port: {self.server.port}")

        if self.server.workers < 1:
            issues.append(f"Invalid worker count: {self.server.workers}")

        # Validate models
        if not self.models:
            issues.append("No models configured")

        return issues


def load_api_config(config_path: Optional[str] = None) -> APIConfig:
    """Load API configuration from file."""
    return APIConfig(config_path)


if __name__ == "__main__":
    # Test configuration loading
    config = APIConfig()
    issues = config.validate_config()

    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration valid")
        print(f"Server: {config.server.host}:{config.server.port}")
        print(f"Models: {list(config.models.keys())}")