"""
Configuration Manager

This module provides utilities to save, load, and manage pipeline configurations
for training and fine-tuning experiments.
"""
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging
from dataclasses import asdict

from pydantic import BaseModel
from .pipeline_config import Config, ModelParamsConfig, TunerConfig, AutoGluonConfig, ObjectiveConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages saving, loading, and applying pipeline configurations."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
    def save_config(self, config: Config, name: str, description: str = "") -> Path:
        """
        Save a configuration to disk.
        
        Args:
            config: The pipeline configuration to save
            name: Name for the configuration file
            description: Optional description of the configuration
            
        Returns:
            Path to the saved configuration file
        """
        # Use existing run_timestamp from config if available, otherwise create new one
        timestamp = getattr(config, 'run_timestamp', None) or datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.yaml"
        config_path = self.config_dir / filename
        
        # Convert config to dictionary, excluding non-serializable fields
        config_dict = self._config_to_dict(config)
        
        # Add metadata
        config_dict['_metadata'] = {
            'name': name,
            'description': description,
            'created_at': timestamp,
            'config_version': '1.0'
        }
        
        # Save as YAML for readability
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
        logger.info(f"Configuration saved to: {config_path}")
        return config_path
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a configuration from disk.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
                
        logger.info(f"Configuration loaded from: {config_path}")
        return config_dict
    
    def apply_config(self, base_config: Config, stored_config_path: Path, 
                    override_params: Optional[Dict[str, Any]] = None) -> Config:
        """
        Apply a stored configuration to a base configuration.
        
        Args:
            base_config: The base configuration to modify
            stored_config_path: Path to the stored configuration
            override_params: Optional parameters to override
            
        Returns:
            Modified configuration
        """
        stored_config = self.load_config(stored_config_path)
        
        # Remove metadata if present
        if '_metadata' in stored_config:
            metadata = stored_config.pop('_metadata')
            logger.info(f"Applying config: {metadata.get('name', 'unnamed')} - {metadata.get('description', '')}")
        
        # Apply stored configuration
        updated_config = self._merge_configs(base_config, stored_config)
        
        # Apply any override parameters
        if override_params:
            updated_config = self._apply_overrides(updated_config, override_params)
            
        return updated_config
    
    def list_configs(self) -> List[Dict[str, Any]]:
        """
        List all available configurations with their metadata.
        
        Returns:
            List of configuration metadata
        """
        configs = []
        
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                config_dict = self.load_config(config_file)
                metadata = config_dict.get('_metadata', {})
                metadata['file_path'] = str(config_file)
                configs.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to read config {config_file}: {e}")
                
        # Sort by creation time (newest first)
        configs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return configs
    
    def create_training_config(self, base_config: Config, 
                             models: List[str],
                             feature_strategies: List[str],
                             use_gpu: bool = False,
                             use_sample_weights: bool = True) -> Config:
        """
        Create a training configuration with specified parameters.
        
        Args:
            base_config: Base configuration to modify
            models: List of models to train
            feature_strategies: List of feature strategies to use
            use_gpu: Whether to use GPU acceleration
            use_sample_weights: Whether to use sample weights
            
        Returns:
            Training configuration
        """
        # Clone the config
        import copy
        training_config = copy.deepcopy(base_config)
        
        # Apply training-specific settings
        training_config.models_to_train = models
        training_config.feature_strategies = feature_strategies
        training_config.use_gpu = use_gpu
        training_config.use_sample_weights = use_sample_weights
        
        # Update timestamp
        training_config.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Created training config: {len(models)} models, {len(feature_strategies)} strategies, GPU={use_gpu}")
        return training_config
    
    def create_tuning_config(self, base_config: Config,
                           models_to_tune: List[str],
                           n_trials: int = 200,
                           timeout: int = 3600,
                           objective_function: str = 'distribution_based',
                           use_gpu: bool = False) -> Config:
        """
        Create a hyperparameter tuning configuration.
        
        Args:
            base_config: Base configuration to modify
            models_to_tune: List of models to tune
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            objective_function: Objective function name
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            Tuning configuration
        """
        # Clone the config
        import copy
        tuning_config = copy.deepcopy(base_config)
        
        # Apply tuning-specific settings
        tuning_config.tuner.models_to_tune = models_to_tune
        tuning_config.tuner.n_trials = n_trials
        tuning_config.tuner.timeout = timeout
        tuning_config.tuner.objective_function_name = objective_function
        tuning_config.use_gpu = use_gpu
        
        # Update timestamp
        tuning_config.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Created tuning config: {len(models_to_tune)} models, {n_trials} trials, {timeout}s timeout")
        return tuning_config
    
    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """Convert Config object to dictionary, handling Pydantic models."""
        # Use Pydantic's dict() method for proper serialization
        config_dict = config.dict()
        
        # Remove non-serializable path objects and convert to strings
        self._convert_paths_to_strings(config_dict)
        
        return config_dict
    
    def _convert_paths_to_strings(self, obj):
        """Recursively convert Path objects to strings for serialization."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, Path):
                    obj[key] = str(value)
                elif isinstance(value, (dict, list)):
                    self._convert_paths_to_strings(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, Path):
                    obj[i] = str(item)
                elif isinstance(item, (dict, list)):
                    self._convert_paths_to_strings(item)
    
    def _merge_configs(self, base_config: Config, stored_config: Dict[str, Any]) -> Config:
        """Merge stored configuration into base configuration with proper nested handling."""
        import copy
        
        merged_config = copy.deepcopy(base_config)
        
        # Define known nested config fields
        nested_configs = {
            'model_params': 'ModelParamsConfig',
            'autogluon': 'AutoGluonConfig', 
            'tuner': 'TunerConfig',
            'dimension_reduction': 'DimensionReductionConfig'
        }
        
        for key, value in stored_config.items():
            if not hasattr(merged_config, key) or key.startswith('_'):
                continue
                
            try:
                current_attr = getattr(merged_config, key)
                
                # Handle nested config objects
                if key in nested_configs and isinstance(value, dict) and isinstance(current_attr, BaseModel):
                    logger.debug(f"Merging nested config: {key}")
                    merged_nested = self._merge_nested_config(current_attr, value)
                    setattr(merged_config, key, merged_nested)
                else:
                    # Handle simple fields
                    setattr(merged_config, key, value)
                    logger.debug(f"Set config field: {key}={value}")
                    
            except Exception as e:
                logger.warning(f"Failed to set {key}={value}: {e}")
        
        return merged_config
    
    def _merge_nested_config(self, base_nested: BaseModel, stored_nested: Dict[str, Any]) -> BaseModel:
        """Merge nested configuration objects."""
        import copy
        
        # Create a copy of the base nested config
        merged_nested = copy.deepcopy(base_nested)
        
        # Update fields in the nested config
        for nested_key, nested_value in stored_nested.items():
            if hasattr(merged_nested, nested_key):
                try:
                    current_nested_attr = getattr(merged_nested, nested_key)
                    
                    # Check for further nesting (like tuner.objectives)
                    if isinstance(nested_value, dict) and isinstance(current_nested_attr, BaseModel):
                        # Recursively merge further nested objects
                        further_merged = self._merge_nested_config(current_nested_attr, nested_value)
                        setattr(merged_nested, nested_key, further_merged)
                        logger.debug(f"Merged further nested config: {nested_key}")
                    else:
                        setattr(merged_nested, nested_key, nested_value)
                        logger.debug(f"Set nested field: {nested_key}={nested_value}")
                        
                except Exception as e:
                    logger.warning(f"Failed to set nested {nested_key}={nested_value}: {e}")
        
        return merged_nested
    
    def _apply_overrides(self, config: Config, overrides: Dict[str, Any]) -> Config:
        """Apply override parameters to configuration."""
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.info(f"Override applied: {key}={value}")
            else:
                logger.warning(f"Override key not found in config: {key}")
        
        return config


# Predefined configuration templates
class ConfigTemplates:
    """Predefined configuration templates for common scenarios."""
    
    @staticmethod
    def get_quick_training_template() -> Dict[str, Any]:
        """Template for quick training with basic models."""
        return {
            'models_to_train': ['extratrees', 'xgboost', 'lightgbm'],
            'feature_strategies': ['simple_only'],
            'use_sample_weights': True,
            'sample_weight_method': 'distribution_based'
        }
    
    @staticmethod
    def get_comprehensive_training_template() -> Dict[str, Any]:
        """Template for comprehensive training with all models."""
        return {
            'models_to_train': [
                'ridge', 'lasso', 'random_forest', 'xgboost', 
                'lightgbm', 'catboost', 'extratrees', 'neural_network_light'
            ],
            'feature_strategies': ['simple_only', 'full_context'],
            'use_sample_weights': True,
            'sample_weight_method': 'distribution_based'
        }
    
    @staticmethod
    def get_neural_network_template() -> Dict[str, Any]:
        """Template for neural network training."""
        return {
            'models_to_train': ['neural_network', 'neural_network_light'],
            'feature_strategies': ['simple_only'],
            'use_sample_weights': True,
            'sample_weight_method': 'distribution_based'
        }
    
    @staticmethod
    def get_tuning_template() -> Dict[str, Any]:
        """Template for hyperparameter tuning."""
        return {
            'tuner': {
                'models_to_tune': ['xgboost', 'lightgbm', 'neural_network_light'],
                'n_trials': 100,
                'timeout': 3600,
                'objective_function_name': 'distribution_based'
            }
        }


# Global config manager instance
config_manager = ConfigManager()