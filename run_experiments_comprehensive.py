#!/usr/bin/env python3
"""
Comprehensive ML Experiment Runner for Magnesium Pipeline
Runs traditional ML experiments with all applicable parameters from AutoGluon experiments
Includes: feature engineering, sample weighting, dimensionality reduction, GPU support
"""

import os
import sys
import time
import json
import yaml
import subprocess
import tempfile
import itertools
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./experiment_logs/comprehensive_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveExperimentRunner:
    def __init__(self):
        self.setup_directories()
        self.load_config()
        self.setup_tracking()
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            './experiment_results',
            './experiment_logs',
            './experiment_configs'
        ]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            
    def load_config(self):
        """Load configuration from staging config"""
        staging_config_path = "/home/payanico/magnesium_pipeline/config/staging_config.yml"
        
        try:
            with open(staging_config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Extract GCP configuration
            gcp_config = config.get('gcp', {})
            self.project_id = gcp_config.get('project_id', 'mapana-ai-models')
            self.region = gcp_config.get('region', 'us-central1')
            self.bucket_name = config.get('bucket_name', 'staging-magnesium-data')
            self.cloud_storage_prefix = config.get('prefix', 'magnesium-pipeline-staging')
            
        except Exception as e:
            logger.warning(f"Could not load staging config: {e}. Using defaults.")
            self.project_id = 'mapana-ai-models'
            self.region = 'us-central1'
            self.bucket_name = 'staging-magnesium-data'
            self.cloud_storage_prefix = 'magnesium-pipeline-staging'
            
        # Fixed configuration
        self.repo_name = 'magnesium-repo'
        self.image_name = 'magnesium-pipeline'
        self.service_name = 'ml-experiments'
        
        logger.info(f"Configuration loaded:")
        logger.info(f"  PROJECT_ID: {self.project_id}")
        logger.info(f"  REGION: {self.region}")
        logger.info(f"  BUCKET_NAME: {self.bucket_name}")
        
    def setup_tracking(self):
        """Setup experiment tracking CSV"""
        self.tracker_file = './experiment_results/comprehensive_tracker.csv'
        if not Path(self.tracker_file).exists():
            with open(self.tracker_file, 'w') as f:
                f.write("experiment_id,timestamp,training_mode,strategy,models,config_name,gpu,status,job_id,notes\n")
                
    def generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pid = os.getpid()
        return f"ml_{timestamp}_{pid}_{int(time.time() * 1000) % 100000}"
        
    def define_experiment_parameters(self) -> Dict[str, Any]:
        """Define all experiment parameters applicable to traditional ML models"""
        return {
            'strategies': ['simple_only', 'full_context', 'Mg_only'],
            
            'training_modes': {
                'train': {'timeout': 7200},  # 2 hours
                'tune': {'timeout': 10800},  # 3 hours  
                'optimize-xgboost': {'timeout': 14400},  # 4 hours
                'optimize-models': {'timeout': 10800}  # 3 hours
            },
            
            'model_groups': {
                'tree_models': ['xgboost', 'lightgbm', 'catboost'],
                'ensemble_models': ['random_forest', 'extratrees'],
                'linear_models': ['ridge', 'lasso'],
                'neural_models': ['neural_network', 'neural_network_light'],
                'other_models': ['svr'],
                'mixed_fast': ['xgboost', 'lightgbm', 'random_forest'],
                'mixed_comprehensive': ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'neural_network']
            },
            
            'feature_configs': {
                'mg_focused': {
                    'enable_molecular_bands': False,
                    'enable_macro_elements': True,
                    'enable_micro_elements': True,
                    'enable_oxygen_hydrogen': False,
                    'enable_advanced_ratios': True,
                    'enable_spectral_patterns': True,
                    'use_focused_magnesium_features': True
                },
                'mg_comprehensive': {
                    'enable_molecular_bands': True,
                    'enable_macro_elements': True,
                    'enable_micro_elements': True,
                    'enable_oxygen_hydrogen': True,
                    'enable_advanced_ratios': True,
                    'enable_spectral_patterns': True,
                    'use_focused_magnesium_features': True
                },
                'mg_minimal': {
                    'enable_molecular_bands': False,
                    'enable_macro_elements': True,
                    'enable_micro_elements': False,
                    'enable_oxygen_hydrogen': False,
                    'enable_advanced_ratios': True,
                    'enable_spectral_patterns': False,
                    'use_focused_magnesium_features': True
                },
                'mg_interference': {
                    'enable_molecular_bands': False,
                    'enable_macro_elements': True,
                    'enable_micro_elements': True,
                    'enable_oxygen_hydrogen': False,
                    'enable_advanced_ratios': True,
                    'enable_spectral_patterns': True,
                    'enable_interference_correction': True,
                    'use_focused_magnesium_features': True
                },
                'raw_spectral': {
                    'use_raw_spectral_data': True,
                    'enable_molecular_bands': False,
                    'enable_macro_elements': False,
                    'enable_micro_elements': False,
                    'enable_oxygen_hydrogen': False,
                    'use_focused_magnesium_features': False
                }
            },
            
            'concentration_configs': {
                'weights_improved': {
                    'use_sample_weights': True,
                    'weight_method': 'improved',
                    'use_concentration_features': True,
                    'use_data_driven_thresholds': True
                },
                'weights_legacy': {
                    'use_sample_weights': True,
                    'weight_method': 'legacy',
                    'use_concentration_features': True,
                    'use_data_driven_thresholds': False
                },
                'no_weights_conc': {
                    'use_sample_weights': False,
                    'weight_method': 'improved',
                    'use_concentration_features': True,
                    'use_data_driven_thresholds': True
                },
                'no_weights_no_conc': {
                    'use_sample_weights': False,
                    'weight_method': 'improved',
                    'use_concentration_features': False,
                    'use_data_driven_thresholds': False
                }
            },
            
            'dimension_configs': {
                'no_reduction': {
                    'use_dimension_reduction': False
                },
                'pls_conservative': {
                    'use_dimension_reduction': True,
                    'dimension_reduction': {
                        'method': 'pls',
                        'n_components': 10,
                        'pls_scale': True
                    }
                },
                'pls_optimal': {
                    'use_dimension_reduction': True,
                    'dimension_reduction': {
                        'method': 'pls',
                        'n_components': 15,
                        'pls_scale': True
                    }
                },
                'pca_variance': {
                    'use_dimension_reduction': True,
                    'dimension_reduction': {
                        'method': 'pca',
                        'n_components': 0.95
                    }
                },
                'pca_fixed': {
                    'use_dimension_reduction': True,
                    'dimension_reduction': {
                        'method': 'pca',
                        'n_components': 20
                    }
                }
            },
            
            'optimization_configs': {
                'quick': {'trials': 50},
                'standard': {'trials': 200},
                'thorough': {'trials': 500},
                'comprehensive': {'trials': 1000}
            },
            
            'gpu_configs': [True, False]
        }
        
    def create_config_file(self, experiment: Dict, exp_id: str) -> str:
        """Create YAML configuration file for experiment"""
        config_name = experiment['config_name']
        config_file = f"./experiment_configs/{config_name}_{exp_id}.yaml"
        
        # Build complete configuration
        config = {
            **experiment['feature_config'],
            **experiment['concentration_config'],
            **experiment['dimension_config']
        }
        
        # Write YAML file
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
            
        return config_file
        
    def submit_vertex_job(self, experiment: Dict, exp_id: str) -> bool:
        """Submit job to Google Cloud Vertex AI"""
        
        # Create config file if needed
        config_file = None
        if any(experiment.get(key) for key in ['feature_config', 'concentration_config', 'dimension_config']):
            config_file = self.create_config_file(experiment, exp_id)
        
        # Build Python command
        python_cmd = f"python main.py {experiment['training_mode']}"
        
        if experiment.get('strategy'):
            python_cmd += f" --strategy {experiment['strategy']}"
        if experiment.get('models'):
            python_cmd += f" --models {experiment['models']}"
        if experiment.get('gpu'):
            python_cmd += " --gpu"
        if experiment.get('trials'):
            python_cmd += f" --trials {experiment['trials']}"
        if config_file:
            python_cmd += f" --config {config_file}"
            
        # Create job name
        job_name = f"{self.service_name}-{exp_id}"
        
        # Container image
        image_uri = f"{self.region}-docker.pkg.dev/{self.project_id}/{self.repo_name}/{self.image_name}:latest"
        
        # Create Vertex AI job configuration
        job_config = {
            "workerPoolSpecs": [
                {
                    "machineSpec": {
                        "machineType": experiment.get('machine_type', 'n1-highmem-8')
                    },
                    "replicaCount": 1,
                    "containerSpec": {
                        "imageUri": image_uri,
                        "command": ["bash", "-c"],
                        "args": [f"source /app/docker-entrypoint.sh && download_training_data && {python_cmd}"],
                        "env": [
                            {"name": "STORAGE_TYPE", "value": "gcs"},
                            {"name": "STORAGE_BUCKET_NAME", "value": self.bucket_name},
                            {"name": "CLOUD_STORAGE_PREFIX", "value": self.cloud_storage_prefix},
                            {"name": "ENVIRONMENT", "value": "staging"},
                            {"name": "PYTHONPATH", "value": "/app"}
                        ]
                    }
                }
            ],
            "scheduling": {
                "timeout": f"{experiment.get('timeout', 7200)}s"
            }
        }
        
        # Add GPU if needed
        if experiment.get('gpu'):
            job_config["workerPoolSpecs"][0]["machineSpec"].update({
                "acceleratorType": "NVIDIA_TESLA_T4",
                "acceleratorCount": 1
            })
            
        # Write job config to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(job_config, f, indent=2)
            config_file_path = f.name
            
        try:
            # Submit job using gcloud
            cmd = [
                'gcloud', 'ai', 'custom-jobs', 'create',
                '--display-name', job_name,
                '--region', self.region,
                '--config', config_file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Extract job ID from output
                job_id = None
                for line in result.stdout.split('\n'):
                    if 'customJobs/' in line:
                        import re
                        match = re.search(r'projects/\d+/locations/[^/]+/customJobs/\d+', line)
                        if match:
                            job_id = match.group(0)
                            break
                            
                logger.info(f"✓ Job submitted: {job_name}")
                if job_id:
                    logger.info(f"  Job ID: {job_id}")
                    
                # Record in tracker
                with open(self.tracker_file, 'a') as f:
                    f.write(f"{exp_id},{datetime.now()},{experiment['training_mode']},{experiment.get('strategy', '')},{experiment.get('models', '')},{experiment.get('config_name', '')},{experiment.get('gpu', False)},SUCCESS,{job_id or ''},{experiment.get('description', '')}\n")
                    
                return True
            else:
                logger.error(f"✗ Failed to submit: {job_name}")
                logger.error(f"Error: {result.stderr}")
                
                with open(self.tracker_file, 'a') as f:
                    f.write(f"{exp_id},{datetime.now()},{experiment['training_mode']},{experiment.get('strategy', '')},{experiment.get('models', '')},{experiment.get('config_name', '')},{experiment.get('gpu', False)},FAILED,,{result.stderr}\n")
                return False
                
        except Exception as e:
            logger.error(f"Exception submitting job {job_name}: {e}")
            return False
        finally:
            # Clean up temp file
            try:
                os.unlink(config_file_path)
            except:
                pass
                
    def generate_quick_experiments(self) -> List[Dict]:
        """Generate 10 high-impact quick experiments"""
        params = self.define_experiment_parameters()
        
        experiments = [
            {
                'name': 'XGBoost Best Features with Improved Weights',
                'training_mode': 'train',
                'strategy': 'simple_only',
                'models': 'xgboost',
                'feature_config': params['feature_configs']['mg_focused'],
                'concentration_config': params['concentration_configs']['weights_improved'],
                'dimension_config': params['dimension_configs']['no_reduction'],
                'gpu': True,
                'machine_type': 'n1-highmem-8',
                'timeout': 7200,
                'config_name': 'xgboost_mg_focused_weights_improved',
                'description': 'XGBoost with magnesium-focused features and improved weighting'
            },
            {
                'name': 'Multi-Model Tree Ensemble',
                'training_mode': 'train',
                'strategy': 'full_context',
                'models': ','.join(params['model_groups']['tree_models']),
                'feature_config': params['feature_configs']['mg_comprehensive'],
                'concentration_config': params['concentration_configs']['weights_improved'],
                'dimension_config': params['dimension_configs']['no_reduction'],
                'gpu': True,
                'machine_type': 'n1-highmem-8',
                'timeout': 7200,
                'config_name': 'tree_ensemble_comprehensive',
                'description': 'Tree model ensemble with comprehensive features'
            },
            {
                'name': 'Neural Network with PLS',
                'training_mode': 'train',
                'strategy': 'simple_only',
                'models': 'neural_network',
                'feature_config': params['feature_configs']['mg_focused'],
                'concentration_config': params['concentration_configs']['weights_improved'],
                'dimension_config': params['dimension_configs']['pls_optimal'],
                'gpu': True,
                'machine_type': 'n1-highmem-8',
                'timeout': 7200,
                'config_name': 'neural_network_pls',
                'description': 'Neural network with PLS dimensionality reduction'
            },
            {
                'name': 'XGBoost Optimization Full Context',
                'training_mode': 'optimize-xgboost',
                'strategy': 'full_context',
                'models': '',
                'feature_config': params['feature_configs']['mg_comprehensive'],
                'concentration_config': params['concentration_configs']['weights_improved'],
                'dimension_config': params['dimension_configs']['no_reduction'],
                'trials': 300,
                'gpu': True,
                'machine_type': 'n1-highmem-8',
                'timeout': 14400,
                'config_name': 'xgboost_optimize_full',
                'description': 'XGBoost hyperparameter optimization with full context'
            },
            {
                'name': 'Mg-Only Strategy Comparison',
                'training_mode': 'train',
                'strategy': 'Mg_only',
                'models': ','.join(params['model_groups']['mixed_fast']),
                'feature_config': params['feature_configs']['mg_minimal'],
                'concentration_config': params['concentration_configs']['weights_improved'],
                'dimension_config': params['dimension_configs']['no_reduction'],
                'gpu': False,
                'machine_type': 'n1-highmem-4',
                'timeout': 7200,
                'config_name': 'mg_only_minimal',
                'description': 'Mg-only strategy with minimal features'
            },
            {
                'name': 'Interference Corrected Models',
                'training_mode': 'train',
                'strategy': 'simple_only',
                'models': ','.join(['xgboost', 'lightgbm']),
                'feature_config': params['feature_configs']['mg_interference'],
                'concentration_config': params['concentration_configs']['weights_improved'],
                'dimension_config': params['dimension_configs']['no_reduction'],
                'gpu': True,
                'machine_type': 'n1-highmem-8',
                'timeout': 7200,
                'config_name': 'interference_corrected',
                'description': 'Models with interference correction'
            },
            {
                'name': 'Raw Spectral Approach',
                'training_mode': 'train',
                'strategy': 'full_context',
                'models': ','.join(['xgboost', 'neural_network']),
                'feature_config': params['feature_configs']['raw_spectral'],
                'concentration_config': params['concentration_configs']['no_weights_conc'],
                'dimension_config': params['dimension_configs']['pca_variance'],
                'gpu': True,
                'machine_type': 'n1-highmem-8',
                'timeout': 7200,
                'config_name': 'raw_spectral_pca',
                'description': 'Raw spectral data with PCA reduction'
            },
            {
                'name': 'Hyperparameter Tuning Ensemble',
                'training_mode': 'tune',
                'strategy': 'simple_only',
                'models': '',
                'feature_config': params['feature_configs']['mg_focused'],
                'concentration_config': params['concentration_configs']['weights_improved'],
                'dimension_config': params['dimension_configs']['no_reduction'],
                'trials': 200,
                'gpu': True,
                'machine_type': 'n1-highmem-8',
                'timeout': 10800,
                'config_name': 'tune_focused',
                'description': 'Hyperparameter tuning with focused features'
            },
            {
                'name': 'Legacy Weights Comparison',
                'training_mode': 'train',
                'strategy': 'full_context',
                'models': ','.join(['xgboost', 'catboost']),
                'feature_config': params['feature_configs']['mg_focused'],
                'concentration_config': params['concentration_configs']['weights_legacy'],
                'dimension_config': params['dimension_configs']['no_reduction'],
                'gpu': True,
                'machine_type': 'n1-highmem-8',
                'timeout': 7200,
                'config_name': 'legacy_weights',
                'description': 'Comparison using legacy weight method'
            },
            {
                'name': 'Conservative PLS with Tree Models',
                'training_mode': 'train',
                'strategy': 'simple_only',
                'models': ','.join(params['model_groups']['ensemble_models']),
                'feature_config': params['feature_configs']['mg_focused'],
                'concentration_config': params['concentration_configs']['weights_improved'],
                'dimension_config': params['dimension_configs']['pls_conservative'],
                'gpu': False,
                'machine_type': 'n1-highmem-8',
                'timeout': 7200,
                'config_name': 'pls_conservative_trees',
                'description': 'Tree models with conservative PLS reduction'
            }
        ]
        
        return experiments
        
    def generate_comprehensive_experiments(self, max_experiments: int = None) -> List[Dict]:
        """Generate comprehensive experiment matrix"""
        params = self.define_experiment_parameters()
        experiments = []
        
        # Generate combinations for different experiment types
        experiment_templates = [
            {
                'training_mode': 'train',
                'models_key': 'tree_models',
                'timeout': 7200
            },
            {
                'training_mode': 'train', 
                'models_key': 'neural_models',
                'timeout': 7200
            },
            {
                'training_mode': 'optimize-xgboost',
                'models_key': None,
                'timeout': 14400,
                'trials': 300
            },
            {
                'training_mode': 'tune',
                'models_key': None,
                'timeout': 10800,
                'trials': 200
            }
        ]
        
        for template in experiment_templates:
            for strategy in params['strategies']:
                for feature_name, feature_config in params['feature_configs'].items():
                    for conc_name, conc_config in params['concentration_configs'].items():
                        for dim_name, dim_config in params['dimension_configs'].items():
                            for gpu in params['gpu_configs']:
                                
                                # Build experiment
                                exp = {
                                    'name': f"{template['training_mode']}_{strategy}_{feature_name}_{conc_name}_{dim_name}_gpu{gpu}",
                                    'training_mode': template['training_mode'],
                                    'strategy': strategy,
                                    'feature_config': feature_config,
                                    'concentration_config': conc_config,
                                    'dimension_config': dim_config,
                                    'gpu': gpu,
                                    'machine_type': 'n1-highmem-16' if gpu else 'n1-highmem-8',
                                    'timeout': template['timeout'],
                                    'config_name': f"{template['training_mode']}_{feature_name}_{conc_name}_{dim_name}",
                                    'description': f"{template['training_mode']} with {feature_name} features, {conc_name} weighting, {dim_name} reduction, GPU={gpu}"
                                }
                                
                                if template.get('models_key'):
                                    exp['models'] = ','.join(params['model_groups'][template['models_key']])
                                else:
                                    exp['models'] = ''
                                    
                                if template.get('trials'):
                                    exp['trials'] = template['trials']
                                    
                                experiments.append(exp)
                                
                                if max_experiments and len(experiments) >= max_experiments:
                                    return experiments[:max_experiments]
        
        logger.info(f"Generated {len(experiments)} comprehensive experiments")
        return experiments[:max_experiments] if max_experiments else experiments
        
    def run_experiments(self, experiment_type: str = "quick", max_experiments: int = None):
        """Run experiments based on type"""
        
        if experiment_type == "quick":
            experiments = self.generate_quick_experiments()
            logger.info(f"Running {len(experiments)} quick high-impact experiments")
        elif experiment_type == "comprehensive":
            experiments = self.generate_comprehensive_experiments(max_experiments)
            logger.info(f"Running comprehensive experiments (up to {len(experiments)})")
        else:
            logger.error(f"Unknown experiment type: {experiment_type}")
            return
            
        successful = 0
        failed = 0
        
        for i, experiment in enumerate(experiments):
            logger.info(f"\n--- Experiment {i+1}/{len(experiments)}: {experiment['name']} ---")
            logger.info(f"Training Mode: {experiment['training_mode']}")
            logger.info(f"Strategy: {experiment['strategy']}")
            logger.info(f"Models: {experiment.get('models', 'default')}")
            logger.info(f"GPU: {experiment['gpu']}")
            
            exp_id = self.generate_experiment_id()
            
            if self.submit_vertex_job(experiment, exp_id):
                successful += 1
                logger.info(f"Progress: {i+1}/{len(experiments)} submitted ({successful} successful, {failed} failed)")
                
                # Add delay between submissions (except for last one)
                if i < len(experiments) - 1:
                    delay = 15 if experiment_type == "quick" else 20
                    logger.info(f"Waiting {delay} seconds before next submission...")
                    time.sleep(delay)
            else:
                failed += 1
                logger.error(f"Failed to submit experiment {i+1}")
                
        logger.info(f"\nExperiment submission completed!")
        logger.info(f"Total submitted: {successful}")
        logger.info(f"Failed: {failed}")
        
    def show_job_status(self):
        """Show current job status"""
        try:
            logger.info("Checking job status...")
            cmd = [
                'gcloud', 'ai', 'custom-jobs', 'list',
                '--region', self.region,
                '--format', 'table(displayName,state,createTime)'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(result.stdout)
            else:
                logger.error("Could not retrieve job status")
                
        except Exception as e:
            logger.error(f"Error checking job status: {e}")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        command = "help"
    else:
        command = sys.argv[1].lower()
        
    max_experiments = None
    if len(sys.argv) > 2:
        try:
            max_experiments = int(sys.argv[2])
        except ValueError:
            logger.error("Invalid max_experiments value. Must be an integer.")
            sys.exit(1)
        
    runner = ComprehensiveExperimentRunner()
    
    logger.info(f"Starting Comprehensive ML Experiment Runner")
    logger.info(f"Project: {runner.project_id} | Region: {runner.region}")
    
    if command == "quick":
        runner.run_experiments("quick")
    elif command == "comprehensive":
        runner.run_experiments("comprehensive", max_experiments)
    elif command == "status":
        runner.show_job_status()
    elif command == "help":
        print("Usage: python3 run_experiments_comprehensive.py [quick|comprehensive|status|help] [max_experiments]")
        print("")
        print("Commands:")
        print("  quick          - 10 high-impact experiments with comprehensive parameters")
        print("  comprehensive  - Full experiment matrix with all parameter combinations")
        print("  status         - Check job status")
        print("  help           - Show this help")
        print("")
        print("Parameters:")
        print("  max_experiments - Maximum number of experiments to run (for comprehensive mode)")
        print("")
        print("Examples:")
        print("  python3 run_experiments_comprehensive.py quick")
        print("  python3 run_experiments_comprehensive.py comprehensive 50")
        print("  python3 run_experiments_comprehensive.py comprehensive")
        print("")
        print("Features included:")
        print("  - All feature engineering strategies (mg_focused, mg_comprehensive, etc.)")
        print("  - Sample weighting methods (improved, legacy, none)")
        print("  - Dimensionality reduction (PLS, PCA, none)")
        print("  - GPU support for compatible models")
        print("  - Multiple model types (tree, neural, ensemble)")
    else:
        logger.error(f"Unknown command: {command}")
        print("Use 'python3 run_experiments_comprehensive.py help' for usage")
        sys.exit(1)
        
    if command not in ["status", "help"]:
        logger.info("Experiment submission completed!")
        logger.info(f"Monitor with: python3 {sys.argv[0]} status")


if __name__ == "__main__":
    main()