"""
Cloud-Agnostic Configuration Manager

This module handles configuration management across different cloud providers
and deployment environments, maintaining compatibility and ease of migration.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Cloud-agnostic storage configuration."""
    type: str = "local"  # local, gcs, s3, azure
    bucket_name: Optional[str] = None
    credentials_path: Optional[str] = None
    prefix: str = "potassium-pipeline"
    data_path: str = "/app/data"
    models_path: str = "/app/models"
    reports_path: str = "/app/reports"
    logs_path: str = "/app/logs"


@dataclass
class ComputeConfig:
    """Cloud-agnostic compute configuration."""
    gpu_enabled: bool = True
    gpu_memory_fraction: float = 0.8
    cpu_cores: Optional[int] = None
    memory_limit: Optional[str] = None


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    timeout: int = 300
    max_file_size: str = "100MB"
    cors_origins: list = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]


@dataclass
class SecurityConfig:
    """Security configuration."""
    api_key_required: bool = False
    api_key: Optional[str] = None
    rate_limiting_enabled: bool = False
    requests_per_minute: int = 60


class CloudManager:
    """Manages cloud-agnostic configuration and deployment settings."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize cloud manager with configuration."""
        self.config_path = config_path or Path(__file__).parent.parent.parent / "config" / "cloud_config.yml"
        self._config = self._load_config()
        self._detect_cloud_provider()
        self._apply_overrides()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            else:
                logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
                return {}
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            return {}
    
    def _detect_cloud_provider(self) -> Optional[str]:
        """Detect current cloud provider from environment."""
        # Google Cloud Platform
        if os.getenv('GOOGLE_CLOUD_PROJECT') or os.path.exists('/var/secrets/google'):
            self.cloud_provider = 'gcp'
            logger.info("Detected Google Cloud Platform")
            return 'gcp'
        
        # Amazon Web Services
        elif os.getenv('AWS_REGION') or os.path.exists('/opt/aws'):
            self.cloud_provider = 'aws'
            logger.info("Detected Amazon Web Services")
            return 'aws'
        
        # Microsoft Azure
        elif os.getenv('AZURE_SUBSCRIPTION_ID') or os.path.exists('/var/lib/waagent'):
            self.cloud_provider = 'azure'
            logger.info("Detected Microsoft Azure")
            return 'azure'
        
        # Local/Other
        else:
            self.cloud_provider = 'local'
            logger.info("No cloud provider detected, using local configuration")
            return 'local'
    
    def _apply_overrides(self):
        """Apply environment-specific and cloud-specific overrides."""
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Apply cloud-specific overrides
        if hasattr(self, 'cloud_provider') and self.cloud_provider in self._config.get('cloud_providers', {}):
            cloud_config = self._config['cloud_providers'][self.cloud_provider]
            self._deep_update(self._config, cloud_config)
            logger.info(f"Applied {self.cloud_provider} specific configuration")
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            # API Configuration
            'API_HOST': ('api', 'host'),
            'API_PORT': ('api', 'port'),
            'API_WORKERS': ('api', 'workers'),
            'API_TIMEOUT': ('api', 'timeout'),
            
            # Storage Configuration
            'STORAGE_TYPE': ('storage', 'type'),
            'STORAGE_BUCKET': ('storage', 'bucket_name'),
            'STORAGE_CREDENTIALS': ('storage', 'credentials_path'),
            'DATA_PATH': ('storage', 'data_path'),
            'MODELS_PATH': ('storage', 'models_path'),
            'REPORTS_PATH': ('storage', 'reports_path'),
            'LOGS_PATH': ('storage', 'logs_path'),
            
            # Compute Configuration
            'GPU_ENABLED': ('compute', 'gpu_enabled'),
            'CPU_CORES': ('compute', 'cpu_cores'),
            'MEMORY_LIMIT': ('compute', 'memory_limit'),
            
            # Security
            'API_KEY': ('security', 'api_key'),
            'API_KEY_REQUIRED': ('security', 'api_key_required'),
            
            # Cloud Provider
            'CLOUD_PROVIDER': ('app', 'cloud_provider'),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(self._config, config_path, self._convert_env_value(value))
                logger.debug(f"Applied environment override: {env_var} -> {config_path}")
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type."""
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # String
        return value
    
    def _set_nested_value(self, config: Dict, path: tuple, value: Any):
        """Set a nested value in configuration dictionary."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """Recursively update base dictionary with update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    def get_storage_config(self) -> StorageConfig:
        """Get storage configuration."""
        storage_config = self._config.get('storage', {})
        return StorageConfig(**storage_config)
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration."""
        api_config = self._config.get('api', {})
        return APIConfig(**api_config)
    
    def get_compute_config(self) -> ComputeConfig:
        """Get compute configuration."""
        compute_config = self._config.get('compute', {})
        return ComputeConfig(**compute_config)
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        security_config = self._config.get('security', {})
        return SecurityConfig(**security_config)
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration section or full configuration."""
        if section:
            return self._config.get(section, {})
        return self._config
    
    def is_cloud_deployment(self) -> bool:
        """Check if running in cloud environment."""
        return hasattr(self, 'cloud_provider') and self.cloud_provider != 'local'
    
    def get_cloud_provider(self) -> str:
        """Get detected cloud provider."""
        return getattr(self, 'cloud_provider', 'local')
    
    def get_storage_client(self):
        """Get cloud-appropriate storage client."""
        storage_config = self.get_storage_config()
        
        if storage_config.type == 'gcs':
            from google.cloud import storage
            return storage.Client()
        
        elif storage_config.type == 's3':
            import boto3
            return boto3.client('s3')
        
        elif storage_config.type == 'azure':
            from azure.storage.blob import BlobServiceClient
            return BlobServiceClient.from_connection_string(
                os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            )
        
        else:
            # Local storage - return None (use filesystem directly)
            return None
    
    def setup_cloud_logging(self):
        """Setup cloud-appropriate logging."""
        if self.get_cloud_provider() == 'gcp':
            try:
                import google.cloud.logging
                client = google.cloud.logging.Client()
                client.setup_logging()
                logger.info("Enabled Google Cloud Logging")
            except ImportError:
                logger.warning("Google Cloud Logging not available")
        
        elif self.get_cloud_provider() == 'aws':
            # AWS CloudWatch logging setup would go here
            pass
        
        elif self.get_cloud_provider() == 'azure':
            # Azure Monitor logging setup would go here
            pass
    
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration for inspection."""
        return {
            'cloud_provider': self.get_cloud_provider(),
            'storage': asdict(self.get_storage_config()),
            'api': asdict(self.get_api_config()),
            'compute': asdict(self.get_compute_config()),
            'security': asdict(self.get_security_config()),
            'full_config': self._config
        }


# Global cloud manager instance
_cloud_manager = None

def get_cloud_manager(config_path: Optional[Union[str, Path]] = None) -> CloudManager:
    """Get singleton cloud manager instance."""
    global _cloud_manager
    if _cloud_manager is None:
        _cloud_manager = CloudManager(config_path)
    return _cloud_manager