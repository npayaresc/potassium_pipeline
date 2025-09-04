#!/usr/bin/env python3
"""
Comprehensive AutoGluon Experiment Runner for Magnesium Pipeline
Generates and submits all experiment combinations to Google Cloud Vertex AI using Google API
"""

import os
import sys
import time
import json
import yaml
import itertools
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Google Cloud AI Platform API
from google.cloud import aiplatform
from google.cloud.aiplatform_v1 import JobServiceClient
from google.cloud.aiplatform_v1.types import (
    CustomJob,
    WorkerPoolSpec,
    MachineSpec,
    ContainerSpec,
    EnvVar
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoGluonExperimentRunner:
    """Runs comprehensive AutoGluon experiments on Google Cloud Vertex AI"""
    
    def __init__(self):
        """Initialize experiment runner"""
        self.setup_configuration()
        self.setup_paths()
        
    def setup_configuration(self):
        """Load configuration from staging config file"""
        staging_config = "/home/payanico/magnesium_pipeline/config/staging_config.yml"
        
        if not os.path.exists(staging_config):
            logger.error(f"Staging config not found: {staging_config}")
            sys.exit(1)
            
        with open(staging_config, 'r') as f:
            config = yaml.safe_load(f)
            
        # Extract GCP configuration
        gcp_config = config.get('gcp', {})
        self.project_id = gcp_config.get('project_id', 'mapana-ai-models')
        self.region = gcp_config.get('region', 'us-central1')
        
        # Extract storage configuration
        storage_config = config.get('storage', {})
        self.bucket_name = storage_config.get('bucket_name', 'staging-magnesium-data')
        self.cloud_storage_prefix = storage_config.get('prefix', 'magnesium-pipeline-staging')
        
        logger.info("Configuration loaded:")
        logger.info(f"  PROJECT_ID: {self.project_id}")
        logger.info(f"  REGION: {self.region}")
        logger.info(f"  BUCKET_NAME: {self.bucket_name}")
        
        # Service configuration
        self.repo_name = "magnesium-repo"
        self.image_name = "magnesium-pipeline"
        self.service_name = "autogluon-experiments"
        
        # Initialize AI Platform client with default credentials
        try:
            aiplatform.init(project=self.project_id, location=self.region)
        except Exception as e:
            logger.warning(f"Could not initialize AI Platform with default credentials: {e}")
            logger.info("Falling back to subprocess approach for authentication issues...")
        
    def setup_paths(self):
        """Setup output directories and tracking files"""
        self.results_dir = Path("./autogluon_experiment_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.tracker_file = self.results_dir / "autogluon_tracker.csv"
        
        # Initialize tracker file with headers
        if not self.tracker_file.exists():
            with open(self.tracker_file, 'w') as f:
                f.write("experiment_id,timestamp,strategy,config_name,status,job_id,target_score,description\n")
                
    def generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        return f"ag_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{hash(str(time.time())) % 100000}"
        
    def get_experiment_matrix(self) -> Dict:
        """Define the complete experiment parameter matrix"""
        return {
            'strategies': ['simple_only', 'full_context', 'Mg_only'],
            'ag_configs': {
                'best_quality_stacked': {
                    'presets': 'best_quality',
                    'time_limit': 14400,  # 4 hours
                    'num_bag_folds': 5,
                    'num_bag_sets': 3,
                    'num_stack_levels': 2,
                    'auto_stack': True
                },
                'medium_quality_fast': {
                    'presets': 'medium_quality',
                    'time_limit': 7200,  # 2 hours
                    'num_bag_folds': 3,
                    'num_bag_sets': 2,
                    'num_stack_levels': 1,
                    'auto_stack': True
                },
                'good_quality_faster': {
                    'presets': 'good_quality',
                    'time_limit': 3600,  # 1 hour
                    'num_bag_folds': 3,
                    'num_bag_sets': 1,
                    'num_stack_levels': 1,
                    'auto_stack': False
                },
                'optimize_for_inference': {
                    'presets': 'optimize_for_deployment',
                    'time_limit': 5400,  # 1.5 hours
                    'num_bag_folds': 2,
                    'num_bag_sets': 1,
                    'num_stack_levels': 1,
                    'auto_stack': False
                },
                'interpretable': {
                    'presets': 'interpretable',
                    'time_limit': 1800,  # 30 minutes
                    'num_bag_folds': 1,
                    'num_bag_sets': 1,
                    'num_stack_levels': 0,
                    'auto_stack': False
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
                    'enable_macro_elements': False,
                    'enable_micro_elements': False,
                    'enable_oxygen_hydrogen': False,
                    'enable_advanced_ratios': False,
                    'enable_spectral_patterns': False,
                    'use_focused_magnesium_features': True
                },
                'mg_interference': {
                    'enable_molecular_bands': True,
                    'enable_macro_elements': True,
                    'enable_micro_elements': False,
                    'enable_oxygen_hydrogen': True,
                    'enable_advanced_ratios': False,
                    'enable_spectral_patterns': True,
                    'use_focused_magnesium_features': False
                },
                'raw_spectral': {
                    'enable_molecular_bands': False,
                    'enable_macro_elements': False,
                    'enable_micro_elements': False,
                    'enable_oxygen_hydrogen': False,
                    'enable_advanced_ratios': False,
                    'enable_spectral_patterns': False,
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
            'gpu_configs': [False]  # AutoGluon experiments run on CPU only
        }
        
    def generate_all_experiments(self) -> List[Dict]:
        """Generate all experiment combinations"""
        matrix = self.get_experiment_matrix()
        experiments = []
        
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
        """Submit job to Google Cloud Vertex AI using Python API, fallback to gcloud CLI"""
        
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
        
        # Try Google API first, fallback to gcloud CLI
        try:
            # Initialize AI Platform client
            client = JobServiceClient()
            
            # Set up machine spec
            machine_spec = MachineSpec(machine_type=experiment['machine_type'])
            
            # Add GPU if needed
            if experiment['gpu']:
                machine_spec.accelerator_type = "NVIDIA_TESLA_T4"
                machine_spec.accelerator_count = 1
            
            # Set up environment variables
            env_vars = [
                EnvVar(name="STORAGE_TYPE", value="gcs"),
                EnvVar(name="STORAGE_BUCKET_NAME", value=self.bucket_name),
                EnvVar(name="CLOUD_STORAGE_PREFIX", value=self.cloud_storage_prefix),
                EnvVar(name="ENVIRONMENT", value="staging"),
                EnvVar(name="PYTHONPATH", value="/app")
            ]
            
            # Set up container spec
            container_spec = ContainerSpec(
                image_uri=image_uri,
                command=["bash", "-c"],
                args=[f"source /app/docker-entrypoint.sh && download_training_data && {python_cmd}"],
                env=env_vars
            )
            
            # Set up worker pool spec
            worker_pool_spec = WorkerPoolSpec(
                machine_spec=machine_spec,
                replica_count=1,
                container_spec=container_spec
            )
            
            # Set up timeout
            timeout_seconds = experiment['ag_config']['time_limit']
            
            # Create custom job with job spec inline
            custom_job = CustomJob(
                display_name=job_name,
                job_spec={
                    'worker_pool_specs': [worker_pool_spec],
                    'scheduling': {'timeout': f"{timeout_seconds}s"}
                }
            )
            
            # Submit the job
            parent = f"projects/{self.project_id}/locations/{self.region}"
            operation = client.create_custom_job(parent=parent, custom_job=custom_job)
            
            # The operation is returned immediately, job is created
            job_resource = operation
            job_id = job_resource.name
            
            logger.info(f"✓ Job submitted: {job_name}")
            logger.info(f"  Job ID: {job_id}")
            
            # Record in tracker
            config_name = f"{experiment['ag_config_name']}_{experiment['feature_name']}_{experiment['concentration_name']}_{experiment['dimension_name']}"
            description = f"{experiment['strategy']} strategy, {experiment['ag_config_name']} preset, {experiment['feature_name']} features, GPU={experiment['gpu']}"
            
            with open(self.tracker_file, 'a') as f:
                f.write(f"{exp_id},{datetime.now()},{experiment['strategy']},{config_name},SUCCESS,{job_id},0.8,{description}\\n")
                
            return True
            
        except Exception as api_error:
            logger.warning(f"Google API failed: {api_error}")
            logger.info("Falling back to gcloud CLI...")
            
            # Fallback to gcloud CLI approach
            return self.submit_vertex_job_gcloud(experiment, exp_id, python_cmd, job_name, image_uri)
            
    def submit_vertex_job_gcloud(self, experiment: Dict, exp_id: str, python_cmd: str, job_name: str, image_uri: str) -> bool:
        """Submit job using gcloud CLI as fallback"""
        
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
            import subprocess
            
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
                for line in result.stdout.split('\\n'):
                    if 'customJobs/' in line:
                        import re
                        match = re.search(r'projects/\\d+/locations/[^/]+/customJobs/\\d+', line)
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
                    f.write(f"{exp_id},{datetime.now()},{experiment['strategy']},{config_name},SUCCESS,{job_id or ''},0.8,{description}\\n")
                    
                return True
            else:
                logger.error(f"✗ Failed to submit: {job_name}")
                logger.error(f"Error: {result.stderr}")
                
                # Record failure in tracker
                config_name = f"{experiment['ag_config_name']}_{experiment['feature_name']}_{experiment['concentration_name']}_{experiment['dimension_name']}"
                description = f"{experiment['strategy']} strategy, {experiment['ag_config_name']} preset, {experiment['feature_name']} features, GPU={experiment['gpu']}"
                
                with open(self.tracker_file, 'a') as f:
                    f.write(f"{exp_id},{datetime.now()},{experiment['strategy']},{config_name},FAILED,'',{description}\\n")
                    
                return False
                
        except Exception as e:
            logger.error(f"✗ Error with gcloud fallback {job_name}: {e}")
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
        
        if max_experiments:
            experiments = experiments[:max_experiments]
            
        logger.info("Starting comprehensive AutoGluon experiments")
        logger.info("Target: R² ≥ 0.8, MAPE < 10%, MAE < 0.05")
        logger.info(f"Running up to {len(experiments)} experiments out of {len(self.generate_all_experiments())} total")
        
        successful = 0
        failed = 0
        
        for i, experiment in enumerate(experiments, 1):
            logger.info(f"\\n--- Experiment {i}/{len(experiments)} ---")
            logger.info(f"Strategy: {experiment['strategy']}")
            logger.info(f"AG Config: {experiment['ag_config_name']}")
            logger.info(f"Features: {experiment['feature_name']}")
            logger.info(f"Weights: {experiment['concentration_name']}")
            logger.info(f"Reduction: {experiment['dimension_name']}")
            logger.info(f"GPU: {experiment['gpu']}")
            
            exp_id = self.generate_experiment_id()
            
            if self.submit_vertex_job(experiment, exp_id):
                successful += 1
            else:
                failed += 1
                
            logger.info(f"Progress: {i}/{len(experiments)} submitted ({successful} successful, {failed} failed)")
            
            # Small delay between submissions to avoid rate limiting
            if i < len(experiments):
                time.sleep(2)
                
        logger.info("\\nExperiment submission completed!")
        logger.info(f"Total submitted: {successful}")
        logger.info(f"Failed: {failed}")
        
        # Show job status
        self.show_job_status()
        
    def show_job_status(self):
        """Show current job status using Google API"""
        try:
            client = JobServiceClient()
            parent = f"projects/{self.project_id}/locations/{self.region}"
            
            # List recent jobs
            request = {"parent": parent, "page_size": 20}
            response = client.list_custom_jobs(request=request)
            
            running_count = 0
            pending_count = 0 
            succeeded_count = 0
            failed_count = 0
            
            for job in response:
                if 'autogluon-experiments' in job.display_name:
                    state = str(job.state).split('.')[-1]
                    if 'RUNNING' in state:
                        running_count += 1
                    elif 'PENDING' in state:
                        pending_count += 1
                    elif 'SUCCEEDED' in state:
                        succeeded_count += 1
                    elif 'FAILED' in state:
                        failed_count += 1
                        
            logger.info(f"Current jobs - Running: {running_count}, Pending: {pending_count}, Succeeded: {succeeded_count}")
            
        except Exception as e:
            logger.warning(f"Could not get job status: {e}")


def main():
    """Main entry point"""
    import sys
    
    # Parse arguments
    max_experiments = None
    if len(sys.argv) > 1:
        try:
            max_experiments = int(sys.argv[1])
        except ValueError:
            logger.error("Usage: python run_autogluon_experiments_gapi.py [max_experiments]")
            logger.error("  max_experiments: Maximum number of experiments to run (default: all)")
            sys.exit(1)
            
    # Run experiments
    runner = AutoGluonExperimentRunner()
    runner.run_experiments(max_experiments)


if __name__ == "__main__":
    main()