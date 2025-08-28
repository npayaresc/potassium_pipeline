#!/usr/bin/env python3
"""
Configuration CLI

Command-line interface for managing pipeline configurations.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional
import yaml

from src.config.config_manager import ConfigManager, ConfigTemplates
from src.config.pipeline_config import Config
from main import create_default_config


def list_configs(config_manager: ConfigManager):
    """List all available configurations."""
    configs = config_manager.list_configs()
    
    if not configs:
        print("No saved configurations found.")
        return
    
    print("\nAvailable Configurations:")
    print("=" * 50)
    
    for i, config in enumerate(configs, 1):
        name = config.get('name', 'unnamed')
        description = config.get('description', 'No description')
        created_at = config.get('created_at', 'Unknown')
        file_path = config.get('file_path', '')
        
        print(f"{i}. {name}")
        print(f"   Description: {description}")
        print(f"   Created: {created_at}")
        print(f"   File: {Path(file_path).name}")
        print()


def save_current_config(config_manager: ConfigManager, name: str, description: str):
    """Save the current default configuration."""
    try:
        # Create default config (same as main.py does)
        config = create_default_config()
        
        # Save configuration
        saved_path = config_manager.save_config(config, name, description)
        print(f"✅ Configuration saved successfully to: {saved_path}")
        
    except Exception as e:
        print(f"❌ Failed to save configuration: {e}")
        sys.exit(1)


def create_training_config(config_manager: ConfigManager, 
                         config_name: Optional[str],
                         models: List[str],
                         strategies: List[str],
                         gpu: bool,
                         weights: bool,
                         output_file: str):
    """Create a training configuration."""
    try:
        # Load base config
        if config_name:
            # Load from saved config
            config_files = config_manager.list_configs()
            matching_config = None
            
            for config_meta in config_files:
                if config_meta.get('name') == config_name:
                    matching_config = Path(config_meta['file_path'])
                    break
            
            if not matching_config:
                print(f"❌ Configuration '{config_name}' not found.")
                print("Available configurations:")
                list_configs(config_manager)
                sys.exit(1)
            
            base_config_dict = config_manager.load_config(matching_config)
            base_config = create_default_config()
            base_config = config_manager._merge_configs(base_config, base_config_dict)
        else:
            # Use default config
            base_config = create_default_config()
        
        # Create training config
        training_config = config_manager.create_training_config(
            base_config=base_config,
            models=models,
            feature_strategies=strategies,
            use_gpu=gpu,
            use_sample_weights=weights
        )
        
        # Save to output file
        output_path = Path(output_file)
        config_dict = config_manager._config_to_dict(training_config)
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        print(f"✅ Training configuration created: {output_path}")
        print(f"   Models: {', '.join(models)}")
        print(f"   Strategies: {', '.join(strategies)}")
        print(f"   GPU: {gpu}")
        print(f"   Sample weights: {weights}")
        
    except Exception as e:
        print(f"❌ Failed to create training configuration: {e}")
        sys.exit(1)


def create_tuning_config(config_manager: ConfigManager,
                        config_name: Optional[str],
                        models: List[str],
                        trials: int,
                        timeout: int,
                        objective: str,
                        gpu: bool,
                        output_file: str):
    """Create a tuning configuration."""
    try:
        # Load base config (same logic as training)
        if config_name:
            config_files = config_manager.list_configs()
            matching_config = None
            
            for config_meta in config_files:
                if config_meta.get('name') == config_name:
                    matching_config = Path(config_meta['file_path'])
                    break
            
            if not matching_config:
                print(f"❌ Configuration '{config_name}' not found.")
                sys.exit(1)
            
            base_config_dict = config_manager.load_config(matching_config)
            base_config = create_default_config()
            base_config = config_manager._merge_configs(base_config, base_config_dict)
        else:
            base_config = create_default_config()
        
        # Create tuning config
        tuning_config = config_manager.create_tuning_config(
            base_config=base_config,
            models_to_tune=models,
            n_trials=trials,
            timeout=timeout,
            objective_function=objective,
            use_gpu=gpu
        )
        
        # Save to output file
        output_path = Path(output_file)
        config_dict = config_manager._config_to_dict(tuning_config)
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        print(f"✅ Tuning configuration created: {output_path}")
        print(f"   Models: {', '.join(models)}")
        print(f"   Trials: {trials}")
        print(f"   Timeout: {timeout}s")
        print(f"   Objective: {objective}")
        print(f"   GPU: {gpu}")
        
    except Exception as e:
        print(f"❌ Failed to create tuning configuration: {e}")
        sys.exit(1)


def show_templates():
    """Show available configuration templates."""
    print("\nAvailable Configuration Templates:")
    print("=" * 40)
    
    templates = {
        'quick-training': ConfigTemplates.get_quick_training_template(),
        'comprehensive-training': ConfigTemplates.get_comprehensive_training_template(),
        'neural-network': ConfigTemplates.get_neural_network_template(),
        'tuning': ConfigTemplates.get_tuning_template()
    }
    
    for name, template in templates.items():
        print(f"\n{name}:")
        print(yaml.dump(template, default_flow_style=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Pipeline Configuration Manager")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List configs
    list_parser = subparsers.add_parser('list', help='List saved configurations')
    
    # Save current config
    save_parser = subparsers.add_parser('save', help='Save current configuration')
    save_parser.add_argument('name', help='Configuration name')
    save_parser.add_argument('--description', '-d', default='', help='Configuration description')
    
    # Create training config
    train_parser = subparsers.add_parser('create-training', help='Create training configuration')
    train_parser.add_argument('--base-config', '-c', help='Base configuration name to use')
    train_parser.add_argument('--models', '-m', nargs='+', 
                             default=['extratrees', 'xgboost', 'lightgbm'],
                             help='Models to train')
    train_parser.add_argument('--strategies', '-s', nargs='+',
                             default=['simple_only'],
                             help='Feature strategies to use')
    train_parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
    train_parser.add_argument('--no-weights', action='store_true', help='Disable sample weights')
    train_parser.add_argument('--output', '-o', default='training_config.yaml',
                             help='Output configuration file')
    
    # Create tuning config
    tune_parser = subparsers.add_parser('create-tuning', help='Create tuning configuration')
    tune_parser.add_argument('--base-config', '-c', help='Base configuration name to use')
    tune_parser.add_argument('--models', '-m', nargs='+',
                            default=['xgboost', 'lightgbm'],
                            help='Models to tune')
    tune_parser.add_argument('--trials', '-t', type=int, default=100,
                            help='Number of optimization trials')
    tune_parser.add_argument('--timeout', type=int, default=3600,
                            help='Timeout in seconds')
    tune_parser.add_argument('--objective', default='distribution_based',
                            choices=['r2', 'robust', 'concentration_weighted', 'mape_focused',
                                   'robust_v2', 'weighted_r2', 'balanced_mae', 'quantile_weighted',
                                   'distribution_based', 'hybrid_weighted'],
                            help='Objective function')
    tune_parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
    tune_parser.add_argument('--output', '-o', default='tuning_config.yaml',
                            help='Output configuration file')
    
    # Show templates
    templates_parser = subparsers.add_parser('templates', help='Show configuration templates')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    if args.command == 'list':
        list_configs(config_manager)
        
    elif args.command == 'save':
        save_current_config(config_manager, args.name, args.description)
        
    elif args.command == 'create-training':
        create_training_config(
            config_manager=config_manager,
            config_name=args.base_config,
            models=args.models,
            strategies=args.strategies,
            gpu=args.gpu,
            weights=not args.no_weights,
            output_file=args.output
        )
        
    elif args.command == 'create-tuning':
        create_tuning_config(
            config_manager=config_manager,
            config_name=args.base_config,
            models=args.models,
            trials=args.trials,
            timeout=args.timeout,
            objective=args.objective,
            gpu=args.gpu,
            output_file=args.output
        )
        
    elif args.command == 'templates':
        show_templates()


if __name__ == '__main__':
    main()