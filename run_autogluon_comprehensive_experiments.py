#!/usr/bin/env python3
"""
Comprehensive AutoGluon Experiment Runner for Magnesium Pipeline
Generates and submits all experiment combinations to Google Cloud Vertex AI
"""

import os
import sys
import time
import json
import yaml
import subprocess
import itertools
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Google Cloud AI Platform API
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic
from google.cloud.aiplatform_v1 import JobServiceClient
from google.cloud.aiplatform_v1.types import CustomJob, JobSpec, WorkerPoolSpec, MachineSpec, ContainerSpec

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./autogluon_experiment_logs/python_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoGluonExperimentRunner:
    def __init__(self):
        self.setup_directories()
        self.load_config()
        self.setup_tracking()
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            './autogluon_experiment_results',
            './autogluon_experiment_logs', 
            './autogluon_experiment_configs'
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
        self.service_name = 'autogluon-experiments'
        
        logger.info(f"Configuration loaded:")
        logger.info(f"  PROJECT_ID: {self.project_id}")
        logger.info(f"  REGION: {self.region}")
        logger.info(f"  BUCKET_NAME: {self.bucket_name}")
        
    def setup_tracking(self):
        """Setup experiment tracking CSV"""
        self.tracker_file = './autogluon_experiment_results/autogluon_tracker.csv'
        if not Path(self.tracker_file).exists():
            with open(self.tracker_file, 'w') as f:
                f.write("experiment_id,timestamp,strategy,config_name,status,job_id,r2_target,notes\n")
                
    def generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pid = os.getpid()
        return f"ag_{timestamp}_{pid}_{int(time.time() * 1000) % 100000}"
        
    def define_experiment_matrix(self) -> Dict[str, List]:
        """Define the complete experiment parameter matrix"""
        return {
            'strategies': ['simple_only', 'full_context', 'Mg_only'],
            'ag_configs': {
                'best_quality_stacked': {
                    'presets': 'best_quality',
                    'time_limit': 14400,
                    'num_bag_folds': 5,
                    'num_bag_sets': 3,
                    'num_stack_levels': 2,
                    'auto_stack': True
                },
                'best_quality_long': {
                    'presets': 'best_quality', 
                    'time_limit': 18000,
                    'num_bag_folds': 7,
                    'num_bag_sets': 2,
                    'num_stack_levels': 3,
                    'auto_stack': True
                },
                'high_quality_optimized': {
                    'presets': 'high_quality',
                    'time_limit': 10800,
                    'num_bag_folds': 5,
                    'num_bag_sets': 2,
                    'num_stack_levels': 2,
                    'auto_stack': True
                },
                'high_quality_fast': {
                    'presets': 'high_quality',
                    'time_limit': 7200,
                    'num_bag_folds': 4,
                    'num_bag_sets': 2,
                    'num_stack_levels': 1,
                    'auto_stack': True
                },
                'good_quality_baseline': {
                    'presets': 'good_quality',
                    'time_limit': 5400,
                    'num_bag_folds': 5,
                    'num_bag_sets': 1,
                    'num_stack_levels': 1,
                    'auto_stack': False
                },
                'good_quality_stacked': {
                    'presets': 'good_quality',
                    'time_limit': 7200,
                    'num_bag_folds': 5,
                    'num_bag_sets': 2,
                    'num_stack_levels': 2,
                    'auto_stack': True
                }
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
            'gpu_configs': [False]
        }
        
    def generate_all_experiments(self) -> List[Dict]:
        """Generate all experiment combinations"""
        matrix = self.define_experiment_matrix()
        
        experiments = []
        
        # Generate all combinations
        for strategy in matrix['strategies']:
            for ag_config_name, ag_config in matrix['ag_configs'].items():
                for feature_name, feature_config in matrix['feature_configs'].items():
                    for conc_name, conc_config in matrix['concentration_configs'].items():
                        for dim_name, dim_config in matrix['dimension_configs'].items():
                            for gpu in matrix['gpu_configs']:
                                experiment = {
                                    'strategy': strategy,
                                    'ag_config_name': ag_config_name,
                                    'ag_config': ag_config,
                                    'feature_name': feature_name,
                                    'feature_config': feature_config,
                                    'concentration_name': conc_name,
                                    'concentration_config': conc_config,
                                    'dimension_name': dim_name,
                                    'dimension_config': dim_config,
                                    'gpu': gpu,
                                    'machine_type': 'n1-highmem-16' if gpu else 'n1-highmem-8'
                                }
                                experiments.append(experiment)
                                
        logger.info(f"Generated {len(experiments)} total experiments")
        return experiments
        
    def create_config_base64(self, experiment: Dict, exp_id: str) -> str:
        """Create configuration as base64-encoded JSON string to avoid shell escaping issues"""
        # Build complete configuration
        config = {
            'autogluon': experiment['ag_config'],
            **experiment['feature_config'],
            **experiment['concentration_config'],
            **experiment['dimension_config']
        }
        
        # Convert config to JSON string and base64 encode it
        import json
        import base64
        config_json = json.dumps(config, separators=(',', ':'))
        config_b64 = base64.b64encode(config_json.encode('utf-8')).decode('utf-8')
        return config_b64
        
    def submit_vertex_job(self, experiment: Dict, exp_id: str) -> bool:
        """Submit job to Google Cloud Vertex AI"""
        
        # Create config as base64-encoded JSON string
        config_b64 = self.create_config_base64(experiment, exp_id)
        
        # Build Python command - embed config as base64
        python_cmd = f"python main.py autogluon --strategy {experiment['strategy']}"
        if experiment['gpu']:
            python_cmd += " --gpu"
        python_cmd += f" --config-base64 {config_b64}"
        python_cmd += " --feature-parallel --data-parallel"
        
        # Create job name
        job_name = f"{self.service_name}-{exp_id}"
        
        # Container image
        image_uri = f"{self.region}-docker.pkg.dev/{self.project_id}/{self.repo_name}/{self.image_name}:latest"
        
        # Create Vertex AI job configuration
        job_config = {
            "workerPoolSpecs": [
                {
                    "machineSpec": {
                        "machineType": experiment['machine_type']
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
                "timeout": f"{experiment['ag_config']['time_limit']}s"
            }
        }
        
        # Add GPU if needed
        if experiment['gpu']:
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
                config_name = f"{experiment['ag_config_name']}_{experiment['feature_name']}_{experiment['concentration_name']}_{experiment['dimension_name']}"
                description = f"{experiment['strategy']} strategy, {experiment['ag_config_name']} preset, {experiment['feature_name']} features, GPU={experiment['gpu']}"
                
                with open(self.tracker_file, 'a') as f:
                    f.write(f"{exp_id},{datetime.now()},{experiment['strategy']},{config_name},SUCCESS,{job_id or ''},0.8,{description}\n")
                    
                return True
            else:
                logger.error(f"✗ Failed to submit: {job_name}")
                logger.error(f"Error: {result.stderr}")
                
                config_name = f"{experiment['ag_config_name']}_{experiment['feature_name']}_{experiment['concentration_name']}_{experiment['dimension_name']}"
                with open(self.tracker_file, 'a') as f:
                    f.write(f"{exp_id},{datetime.now()},{experiment['strategy']},{config_name},FAILED,,0.8,{result.stderr}\n")
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
                
    def run_experiments(self, max_experiments: int = None):
        """Run all experiments up to the specified limit"""
        experiments = self.generate_all_experiments()
        
        if max_experiments is None:
            max_experiments = len(experiments)
        else:
            max_experiments = min(max_experiments, len(experiments))
            
        logger.info(f"Starting comprehensive AutoGluon experiments")
        logger.info(f"Target: R² ≥ 0.8, MAPE < 10%, MAE < 0.05")
        logger.info(f"Running up to {max_experiments} experiments out of {len(experiments)} total")
        
        successful = 0
        failed = 0
        
        for i, experiment in enumerate(experiments[:max_experiments]):
            logger.info(f"\n--- Experiment {i+1}/{max_experiments} ---")
            logger.info(f"Strategy: {experiment['strategy']}")
            logger.info(f"AG Config: {experiment['ag_config_name']}")
            logger.info(f"Features: {experiment['feature_name']}")
            logger.info(f"Weights: {experiment['concentration_name']}")
            logger.info(f"Reduction: {experiment['dimension_name']}")
            logger.info(f"GPU: {experiment['gpu']}")
            
            exp_id = self.generate_experiment_id()
            
            if self.submit_vertex_job(experiment, exp_id):
                successful += 1
                logger.info(f"Progress: {i+1}/{max_experiments} submitted ({successful} successful, {failed} failed)")
                
                # Add delay between submissions (except for last one)
                if i < max_experiments - 1:
                    logger.info("Waiting 20 seconds before next submission...")
                    time.sleep(20)
            else:
                failed += 1
                logger.error(f"Failed to submit experiment {i+1}")
                
        logger.info(f"\nExperiment submission completed!")
        logger.info(f"Total submitted: {successful}")
        logger.info(f"Failed: {failed}")
        
        # Show current job status
        self.show_job_status()
        
    def show_job_status(self):
        """Show current job status"""
        try:
            cmd = [
                'gcloud', 'ai', 'custom-jobs', 'list',
                '--region', self.region,
                '--filter', f'displayName:autogluon-experiments*',
                '--format', 'value(state)'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                states = result.stdout.strip().split('\n') if result.stdout.strip() else []
                running = sum(1 for s in states if 'RUNNING' in s)
                pending = sum(1 for s in states if 'PENDING' in s) 
                succeeded = sum(1 for s in states if 'SUCCEEDED' in s)
                
                logger.info(f"Current jobs - Running: {running}, Pending: {pending}, Succeeded: {succeeded}")
            else:
                logger.warning("Could not retrieve job status")
                
        except Exception as e:
            logger.warning(f"Error checking job status: {e}")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        try:
            max_experiments = int(sys.argv[1])
        except ValueError:
            print(f"Usage: {sys.argv[0]} [max_experiments]")
            print(f"  max_experiments: Maximum number of experiments to run (default: all)")
            sys.exit(1)
    else:
        max_experiments = None
        
    runner = AutoGluonExperimentRunner()
    runner.run_experiments(max_experiments)


if __name__ == "__main__":
    main()