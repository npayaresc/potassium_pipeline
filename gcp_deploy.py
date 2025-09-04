#!/usr/bin/env python3
"""
Google Cloud Platform Deployment Script for Magnesium Pipeline (Python Version)
This script provides comprehensive deployment options for GCP using Google Cloud Python APIs

IMPORTANT: AutoGluon uses Ray for distributed machine learning.
Ray is fully compatible with Vertex AI custom training jobs.
For optimal performance in production, consider multi-node Ray clusters.
"""

import os
import sys
import json
import yaml
import time
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import argparse
import re

# Google Cloud APIs
from google.cloud import storage
from google.cloud.aiplatform_v1 import JobServiceClient
from google.cloud.aiplatform_v1.types import CustomJob, WorkerPoolSpec, MachineSpec, ContainerSpec, EnvVar
from google.api_core import exceptions as gcp_exceptions

# Optional imports - will fall back to gcloud CLI if not available
try:
    from google.cloud import artifactregistry_v1
except ImportError:
    artifactregistry_v1 = None
    
try:
    from google.cloud import run_v2
except ImportError:
    run_v2 = None
    
try:
    from google.cloud import container_v1
except ImportError:
    container_v1 = None
    
try:
    from google.cloud import service_usage_v1
except ImportError:
    service_usage_v1 = None
    
try:
    from google.cloud import resourcemanager_v3
except ImportError:
    resourcemanager_v3 = None

# Setup logging with colors
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}[{record.levelname}]{self.RESET}"
        return super().format(record)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(levelname)s %(message)s'))
logger.addHandler(handler)


class GCPDeployment:
    """Main deployment class for GCP operations"""
    
    def __init__(self):
        """Initialize deployment configuration"""
        self.load_configuration()
        self.setup_clients()
        
    def load_configuration(self):
        """Load configuration from environment and files"""
        # Basic configuration
        self.project_id = os.environ.get('PROJECT_ID', 'mapana-ai-models')
        self.region = os.environ.get('REGION', 'us-central1')
        self.service_name = os.environ.get('SERVICE_NAME', 'magnesium-pipeline')
        self.repo_name = os.environ.get('REPO_NAME', 'magnesium-repo')
        self.image_name = os.environ.get('IMAGE_NAME', 'magnesium-pipeline')
        self.bucket_name = os.environ.get('BUCKET_NAME', f'{self.project_id}-magnesium-data')
        
        # Deployment settings
        self.auto_commit_deployment = os.environ.get('AUTO_COMMIT_DEPLOYMENT', 'true').lower() == 'true'
        self.auto_push_deployment = os.environ.get('AUTO_PUSH_DEPLOYMENT', 'false').lower() == 'true'
        
        # Cloud configuration
        self.cloud_config_file = os.environ.get('CLOUD_CONFIG_FILE', 'config/staging_config.yml')
        self.environment = os.environ.get('ENVIRONMENT', 'production')
        self.use_cloud_config = os.environ.get('USE_CLOUD_CONFIG', 'true').lower() == 'true'
        
        # Training configuration
        self.training_mode = os.environ.get('TRAINING_MODE', '')
        self.use_gpu = os.environ.get('USE_GPU', '')
        self.use_raw_spectral = os.environ.get('USE_RAW_SPECTRAL', '')
        self.models = os.environ.get('MODELS', '')
        self.strategy = os.environ.get('STRATEGY', '')
        self.trials = os.environ.get('TRIALS', '')
        self.timeout = os.environ.get('TIMEOUT', '')
        self.machine_type = os.environ.get('MACHINE_TYPE', 'n1-standard-4')
        self.accelerator_type = os.environ.get('ACCELERATOR_TYPE', 'NVIDIA_TESLA_T4')
        self.accelerator_count = int(os.environ.get('ACCELERATOR_COUNT', '1'))
        
        # Load cloud configuration if available
        self.load_cloud_config()
        
    def load_cloud_config(self):
        """Load configuration from staging config file"""
        if not self.use_cloud_config or not os.path.exists(self.cloud_config_file):
            logger.info("Staging config not available, using defaults")
            self.get_python_config_defaults()
            return
            
        try:
            logger.info(f"Loading configuration from: {self.cloud_config_file}")
            with open(self.cloud_config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            # Load GCP configuration from cloud_providers.gcp
            cloud_providers = config.get('cloud_providers', {})
            gcp_config = cloud_providers.get('gcp', {})
            if gcp_config:
                self.project_id = gcp_config.get('project_id', self.project_id)
                self.region = gcp_config.get('region', self.region)
                
            # Load storage configuration
            storage_config = config.get('storage', {})
            if storage_config:
                self.bucket_name = storage_config.get('bucket_name', self.bucket_name)
                
            # Load app environment
            app_config = config.get('app', {})
            if app_config:
                self.environment = app_config.get('environment', self.environment)
                
            # Load pipeline configuration
            pipeline_config = config.get('pipeline', {})
            if pipeline_config:
                if not self.use_gpu:
                    self.use_gpu = str(pipeline_config.get('enable_gpu', True)).lower()
                if not self.strategy:
                    self.strategy = pipeline_config.get('default_strategy', 'full_context')
                    
            # Load compute configuration
            compute_config = config.get('compute', {})
            if compute_config and not self.use_gpu:
                self.use_gpu = str(compute_config.get('gpu_enabled', True)).lower()
                
            logger.info("Configuration loaded successfully")
            logger.info(f"  Project ID: {self.project_id}")
            logger.info(f"  Region: {self.region}")
            logger.info(f"  Bucket: {self.bucket_name}")
            logger.info(f"  Environment: {self.environment}")
            
        except Exception as e:
            logger.warning(f"Could not load staging config: {e}")
            self.get_python_config_defaults()
            
    def get_python_config_defaults(self):
        """Get default values from Python pipeline_config.py"""
        try:
            sys.path.insert(0, '.')
            from src.config.pipeline_config import config
            
            if not self.use_gpu:
                self.use_gpu = str(config.use_gpu).lower()
            if not self.use_raw_spectral:
                self.use_raw_spectral = str(config.use_raw_spectral_data).lower()
            if not self.strategy:
                self.strategy = 'full_context'
            if not self.training_mode:
                self.training_mode = 'autogluon'
                
            logger.info("Loaded defaults from Python config:")
            logger.info(f"  USE_GPU default: {self.use_gpu}")
            logger.info(f"  USE_RAW_SPECTRAL default: {self.use_raw_spectral}")
            logger.info(f"  STRATEGY default: {self.strategy}")
            logger.info(f"  TRAINING_MODE default: {self.training_mode}")
            
        except Exception as e:
            logger.warning(f"Could not load Python config defaults: {e}")
            # Set hardcoded defaults
            self.use_gpu = self.use_gpu or 'true'
            self.use_raw_spectral = self.use_raw_spectral or 'false'
            self.strategy = self.strategy or 'full_context'
            self.training_mode = self.training_mode or 'autogluon'
            
    def setup_clients(self):
        """Initialize Google Cloud API clients"""
        try:
            self.storage_client = storage.Client(project=self.project_id)
            self.job_client = JobServiceClient()
            
            # Optional clients
            self.artifact_client = artifactregistry_v1.ArtifactRegistryClient() if artifactregistry_v1 else None
            self.run_client = run_v2.ServicesClient() if run_v2 else None
            self.container_client = container_v1.ClusterManagerClient() if container_v1 else None
            self.service_usage_client = service_usage_v1.ServiceUsageClient() if service_usage_v1 else None
            self.resource_manager_client = resourcemanager_v3.ProjectsClient() if resourcemanager_v3 else None
            
            logger.info("Google Cloud API clients initialized")
        except Exception as e:
            logger.warning(f"Could not initialize all API clients: {e}")
            logger.info("Some functionality may fall back to gcloud CLI")
            
    def create_deployment_record(self, deployment_type: str, status: str, stage: str, details: Dict) -> str:
        """Create a deployment record file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        deployment_id = f"{deployment_type}_{timestamp}"
        
        record = {
            'deployment_id': deployment_id,
            'timestamp': timestamp,
            'type': deployment_type,
            'status': status,
            'stage': stage,
            'project_id': self.project_id,
            'region': self.region,
            'details': details
        }
        
        os.makedirs('deployments', exist_ok=True)
        record_file = f"deployments/{deployment_id}.json"
        
        with open(record_file, 'w') as f:
            json.dump(record, f, indent=2)
            
        return record_file
        
    def commit_and_tag_deployment(self, deployment_type: str):
        """Commit and tag deployment if auto-commit is enabled"""
        if not self.auto_commit_deployment:
            return
            
        try:
            # Check if we're in a git repository
            result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.info("Not in a git repository, skipping deployment commit")
                return
                
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            tag = f"deploy-{deployment_type}-{timestamp}"
            
            # Add deployment records
            subprocess.run(['git', 'add', 'deployments/'], check=True)
            
            # Commit
            commit_message = f"🚀 Deployment: {deployment_type} to {self.environment}\\n\\nGenerated with Claude Code\\n\\nCo-Authored-By: Claude <noreply@anthropic.com>"
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            
            # Tag
            subprocess.run(['git', 'tag', '-a', tag, '-m', f"Deployment {deployment_type}"], check=True)
            
            if self.auto_push_deployment:
                subprocess.run(['git', 'push'], check=True)
                subprocess.run(['git', 'push', '--tags'], check=True)
                
            logger.info(f"Deployment committed and tagged: {tag}")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not commit deployment: {e}")
            
    def check_basic_prerequisites(self) -> bool:
        """Check basic prerequisites"""
        logger.info("Checking basic prerequisites...")
        
        # Check if gcloud is installed
        if not shutil.which('gcloud'):
            logger.error("gcloud CLI is not installed. Please install it first.")
            return False
            
        # Check if logged in
        try:
            result = subprocess.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE', 
                                   '--format=value(account)'], 
                                  capture_output=True, text=True, check=True)
            if not result.stdout.strip():
                logger.error("Not logged into gcloud. Run: gcloud auth login")
                return False
        except subprocess.CalledProcessError:
            logger.error("Could not check gcloud authentication")
            return False
            
        # Check if docker is installed
        if not shutil.which('docker'):
            logger.error("Docker is not installed. Please install it first.")
            return False
            
        logger.info("Basic prerequisites check passed!")
        return True
        
    def check_project_configuration(self) -> bool:
        """Check and validate project configuration"""
        if self.project_id == "your-gcp-project-id":
            logger.warning("PROJECT_ID is not configured. Let's set it up...")
            self.list_gcp_projects()
            return False
            
        try:
            # Set the project
            subprocess.run(['gcloud', 'config', 'set', 'project', self.project_id], 
                         check=True, capture_output=True)
            logger.info(f"Project set to: {self.project_id}")
            return True
        except subprocess.CalledProcessError:
            logger.error(f"Could not set project to: {self.project_id}")
            return False
            
    def list_gcp_projects(self):
        """List available GCP projects"""
        try:
            result = subprocess.run(['gcloud', 'projects', 'list', '--format=value(projectId)'], 
                                  capture_output=True, text=True, check=True)
            projects = result.stdout.strip().split('\\n')
            
            logger.info("Available projects:")
            for project in projects:
                if project.strip():
                    logger.info(f"  - {project.strip()}")
                    
            logger.info("\\nTo set a project:")
            logger.info("  export PROJECT_ID=your-chosen-project-id")
            logger.info("Or set it inline:")
            logger.info("  PROJECT_ID=your-chosen-project-id python gcp_deploy.py setup")
            
        except subprocess.CalledProcessError:
            logger.error("Could not list GCP projects")
            
    def update_config_project_id(self, new_project_id: str):
        """Update project ID in configuration files"""
        config_files = [
            'config/cloud_config.yml',
            'config/staging_config.yml',
            'config/production_config.yml'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()
                        
                    # Replace project ID
                    updated_content = re.sub(
                        r'(project_id:\s*)["\']?[^"\'\\n]*["\']?',
                        f'\\1"{new_project_id}"',
                        content
                    )
                    
                    with open(config_file, 'w') as f:
                        f.write(updated_content)
                        
                    logger.info(f"Updated project ID in {config_file}")
                    
                except Exception as e:
                    logger.warning(f"Could not update {config_file}: {e}")
                    
    def enable_required_apis(self):
        """Enable required Google Cloud APIs"""
        required_apis = [
            'aiplatform.googleapis.com',
            'artifactregistry.googleapis.com',
            'storage.googleapis.com',
            'run.googleapis.com',
            'container.googleapis.com',
            'cloudbuild.googleapis.com'
        ]
        
        logger.info("Enabling required Google Cloud APIs...")
        
        for api in required_apis:
            try:
                subprocess.run(['gcloud', 'services', 'enable', api, '--quiet'], 
                             check=True, capture_output=True)
                logger.info(f"✓ Enabled {api}")
            except subprocess.CalledProcessError:
                logger.warning(f"Could not enable {api}")
                
    def setup_project(self):
        """Setup GCP project with required services and resources"""
        logger.info("Setting up GCP project...")
        
        if not self.check_basic_prerequisites():
            return False
            
        if not self.check_project_configuration():
            return False
            
        # Enable required APIs
        self.enable_required_apis()
        
        # Create Artifact Registry repository
        self.create_artifact_repository()
        
        # Create GCS bucket
        self.create_gcs_bucket()
        
        logger.info("Project setup completed!")
        return True
        
    def create_artifact_repository(self):
        """Create Artifact Registry repository"""
        if not self.artifact_client:
            logger.info("Artifact Registry API client not available, using gcloud CLI...")
            try:
                subprocess.run([
                    'gcloud', 'artifacts', 'repositories', 'create', self.repo_name,
                    '--repository-format=docker',
                    f'--location={self.region}',
                    '--description=Docker repository for Magnesium Pipeline',
                    '--quiet'
                ], check=True, capture_output=True)
                logger.info(f"✓ Created Artifact Registry repository: {self.repo_name}")
            except subprocess.CalledProcessError:
                logger.warning("Could not create repository with gcloud")
            return
            
        try:
            parent = f"projects/{self.project_id}/locations/{self.region}"
            repository_id = self.repo_name
            
            repository = {
                "format_": artifactregistry_v1.Repository.Format.DOCKER,
                "description": "Docker repository for Magnesium Pipeline"
            }
            
            request = artifactregistry_v1.CreateRepositoryRequest(
                parent=parent,
                repository_id=repository_id,
                repository=repository
            )
            
            operation = self.artifact_client.create_repository(request=request)
            logger.info(f"Creating repository {repository_id}...")
            
            try:
                result = operation.result(timeout=300)
                logger.info(f"✓ Created Artifact Registry repository: {repository_id}")
            except Exception:
                # Repository might already exist
                logger.info(f"Repository {repository_id} already exists or creation in progress")
                
        except Exception as e:
            logger.warning(f"Could not create Artifact Registry repository: {e}")
            logger.info("Falling back to gcloud command...")
            
            try:
                subprocess.run([
                    'gcloud', 'artifacts', 'repositories', 'create', self.repo_name,
                    '--repository-format=docker',
                    f'--location={self.region}',
                    '--description=Docker repository for Magnesium Pipeline',
                    '--quiet'
                ], check=True, capture_output=True)
                logger.info(f"✓ Created Artifact Registry repository: {self.repo_name}")
            except subprocess.CalledProcessError:
                logger.warning("Could not create repository with gcloud fallback")
                
    def create_gcs_bucket(self):
        """Create Google Cloud Storage bucket"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            
            if bucket.exists():
                logger.info(f"GCS bucket already exists: {self.bucket_name}")
                return
                
            bucket = self.storage_client.create_bucket(
                bucket_or_name=self.bucket_name,
                location=self.region
            )
            logger.info(f"✓ Created GCS bucket: {self.bucket_name}")
            
        except Exception as e:
            logger.warning(f"Could not create GCS bucket: {e}")
            logger.info("Falling back to gcloud command...")
            
            try:
                subprocess.run([
                    'gsutil', 'mb', '-p', self.project_id, '-c', 'STANDARD',
                    '-l', self.region, f'gs://{self.bucket_name}'
                ], check=True, capture_output=True)
                logger.info(f"✓ Created GCS bucket: {self.bucket_name}")
            except subprocess.CalledProcessError:
                logger.warning("Could not create bucket with gsutil fallback")
                
    def check_image_exists(self, image_uri: str) -> bool:
        """Check if Docker image exists in Artifact Registry"""
        try:
            result = subprocess.run([
                'gcloud', 'artifacts', 'docker', 'images', 'describe',
                image_uri, '--quiet'
            ], capture_output=True)
            
            return result.returncode == 0
        except Exception:
            return False
            
    def build_image(self):
        """Build and push Docker image using Cloud Build"""
        logger.info("Building Docker image...")
        
        image_uri = f"{self.region}-docker.pkg.dev/{self.project_id}/{self.repo_name}/{self.image_name}:latest"
        
        # Configure Docker for Artifact Registry
        try:
            subprocess.run([
                'gcloud', 'auth', 'configure-docker',
                f'{self.region}-docker.pkg.dev', '--quiet'
            ], check=True)
        except subprocess.CalledProcessError:
            logger.warning("Could not configure Docker authentication")
            
        # Build with Cloud Build
        try:
            logger.info("Building image with Cloud Build...")
            subprocess.run([
                'gcloud', 'builds', 'submit',
                '--tag', image_uri,
                '--timeout=1h',
                '--machine-type=e2-highcpu-8'
            ], check=True)
            
            logger.info(f"✓ Built and pushed image: {image_uri}")
            return True
            
        except subprocess.CalledProcessError:
            logger.error("Cloud Build failed")
            return False
            
    def build_image_local(self):
        """Build Docker image locally"""
        logger.info("Building Docker image locally...")
        
        image_uri = f"{self.region}-docker.pkg.dev/{self.project_id}/{self.repo_name}/{self.image_name}:latest"
        
        try:
            # Build image
            subprocess.run(['docker', 'build', '-t', image_uri, '.'], check=True)
            
            # Push image
            subprocess.run(['docker', 'push', image_uri], check=True)
            
            logger.info(f"✓ Built and pushed image locally: {image_uri}")
            return True
            
        except subprocess.CalledProcessError:
            logger.error("Local Docker build failed")
            return False
            
    def check_data_exists_in_gcs(self) -> bool:
        """Check if training data exists in GCS"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            prefix = "magnesium-pipeline/data_5278_Phase3/"
            
            blobs = list(bucket.list_blobs(prefix=prefix, max_results=1))
            return len(blobs) > 0
            
        except Exception as e:
            logger.warning(f"Could not check GCS data: {e}")
            return False
            
    def upload_training_data(self):
        """Upload training data to GCS bucket"""
        if self.check_data_exists_in_gcs():
            logger.info("Training data already exists in GCS")
            return True
            
        data_dirs = [
            "data/raw/data_5278_Phase3",
            "data/reference_data"
        ]
        
        existing_dirs = [d for d in data_dirs if os.path.exists(d)]
        
        if not existing_dirs:
            logger.warning("No local training data found to upload")
            return False
            
        logger.info("Uploading training data to GCS...")
        
        try:
            for data_dir in existing_dirs:
                self._upload_directory_to_gcs(data_dir)
                
            logger.info("✓ Training data uploaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload training data: {e}")
            return False
            
    def _upload_directory_to_gcs(self, local_dir: str):
        """Upload a directory to GCS"""
        bucket = self.storage_client.bucket(self.bucket_name)
        
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, ".")
                gcs_path = f"magnesium-pipeline/{relative_path}"
                
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
                logger.info(f"Uploaded: {local_path} -> gs://{self.bucket_name}/{gcs_path}")
                
    def build_training_command(self) -> str:
        """Build the training command string"""
        base_cmd = f"python main.py {self.training_mode}"
        
        # Add GPU flag
        if self.use_gpu.lower() == 'true':
            base_cmd += " --gpu"
            
        # Add raw spectral flag
        if self.use_raw_spectral.lower() == 'true':
            base_cmd += " --raw-spectral"
            
        # Add models
        if self.models:
            models_list = self.models.split(',')
            base_cmd += f" --models {' '.join(models_list)}"
            
        # Add strategy for applicable modes
        if self.training_mode in ['train', 'optimize-xgboost', 'optimize-autogluon', 'optimize-models']:
            if self.strategy:
                base_cmd += f" --strategy {self.strategy}"
                
        # Add trials for optimization modes
        if self.trials and 'optimize' in self.training_mode:
            base_cmd += f" --trials {self.trials}"
            
        # Add timeout for optimization modes
        if self.timeout and 'optimize' in self.training_mode:
            base_cmd += f" --timeout {self.timeout}"
            
        return base_cmd
        
    def deploy_vertex_ai(self):
        """Deploy training job to Vertex AI"""
        logger.info("Deploying training job to Vertex AI...")
        
        if not self.check_basic_prerequisites():
            return False
            
        self.load_cloud_config()
        
        if not self.check_project_configuration():
            return False
            
        # Validate training configuration
        if not self.training_mode:
            logger.error("TRAINING_MODE is required for Vertex AI deployment")
            return False
            
        image_uri = f"{self.region}-docker.pkg.dev/{self.project_id}/{self.repo_name}/{self.image_name}:latest"
        
        # Check if image exists, build if needed
        if not self.check_image_exists(image_uri):
            logger.info("Container image not found. Setting up project and building image...")
            if not self.setup_project():
                return False
            if not self.build_image():
                return False
                
        # Upload training data
        self.upload_training_data()
        
        # Build training command
        training_cmd = self.build_training_command()
        
        logger.info(f"Training command: {training_cmd}")
        
        # Determine timeout based on training mode
        timeouts = {
            'optimize': 86400,  # 24 hours
            'autogluon': 14400,  # 4 hours
            'default': 7200     # 2 hours
        }
        
        job_timeout = timeouts.get(
            next((k for k in timeouts.keys() if k in self.training_mode), 'default'),
            timeouts['default']
        )
        
        # Create job
        job_display_name = f"{self.service_name}-{self.training_mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Create deployment record
        deployment_details = {
            "image_uri": image_uri,
            "machine_type": self.machine_type,
            "accelerator_type": self.accelerator_type,
            "accelerator_count": self.accelerator_count,
            "job_timeout": job_timeout,
            "training_command": training_cmd
        }
        
        deployment_file = self.create_deployment_record("vertex-ai", "pending", "started", deployment_details)
        logger.info(f"Created deployment record: {deployment_file}")
        
        # Submit job
        success = self._submit_vertex_ai_job(job_display_name, image_uri, training_cmd, job_timeout)
        
        if success:
            self.commit_and_tag_deployment("vertex-ai")
            logger.info("✓ Vertex AI deployment completed successfully")
        else:
            logger.error("✗ Vertex AI deployment failed")
            
        return success
        
    def _submit_vertex_ai_job(self, job_name: str, image_uri: str, training_cmd: str, timeout: int) -> bool:
        """Submit job to Vertex AI"""
        try:
            # Set up machine spec
            machine_spec = MachineSpec(machine_type=self.machine_type)
            
            # Add GPU if needed
            if self.use_gpu.lower() == 'true':
                machine_spec.accelerator_type = self.accelerator_type
                machine_spec.accelerator_count = self.accelerator_count
                
            # Environment variables
            env_vars = [
                EnvVar(name="STORAGE_TYPE", value="gcs"),
                EnvVar(name="STORAGE_BUCKET_NAME", value=self.bucket_name),
                EnvVar(name="CLOUD_STORAGE_PREFIX", value="magnesium-pipeline"),
                EnvVar(name="ENVIRONMENT", value=self.environment),
                EnvVar(name="PYTHONPATH", value="/app")
            ]
            
            # Container spec
            container_spec = ContainerSpec(
                image_uri=image_uri,
                command=["bash", "-c"],
                args=[f"source /app/docker-entrypoint.sh && download_training_data && {training_cmd}"],
                env=env_vars
            )
            
            # Worker pool spec
            worker_pool_spec = WorkerPoolSpec(
                machine_spec=machine_spec,
                replica_count=1,
                container_spec=container_spec
            )
            
            # Custom job
            custom_job = CustomJob(
                display_name=job_name,
                job_spec={
                    'worker_pool_specs': [worker_pool_spec],
                    'scheduling': {'timeout': f"{timeout}s"}
                }
            )
            
            # Submit job
            parent = f"projects/{self.project_id}/locations/{self.region}"
            operation = self.job_client.create_custom_job(parent=parent, custom_job=custom_job)
            
            job_resource = operation
            job_id = job_resource.name
            
            logger.info(f"✓ Job submitted: {job_name}")
            logger.info(f"  Job ID: {job_id}")
            
            return True
            
        except Exception as e:
            logger.warning(f"Google API failed: {e}")
            logger.info("Falling back to gcloud CLI...")
            return self._submit_vertex_ai_job_gcloud(job_name, image_uri, training_cmd, timeout)
            
    def _submit_vertex_ai_job_gcloud(self, job_name: str, image_uri: str, training_cmd: str, timeout: int) -> bool:
        """Submit job using gcloud CLI as fallback"""
        job_config = {
            "workerPoolSpecs": [
                {
                    "machineSpec": {
                        "machineType": self.machine_type
                    },
                    "replicaCount": 1,
                    "containerSpec": {
                        "imageUri": image_uri,
                        "command": ["bash", "-c"],
                        "args": [f"source /app/docker-entrypoint.sh && download_training_data && {training_cmd}"],
                        "env": [
                            {"name": "STORAGE_TYPE", "value": "gcs"},
                            {"name": "STORAGE_BUCKET_NAME", "value": self.bucket_name},
                            {"name": "CLOUD_STORAGE_PREFIX", "value": "magnesium-pipeline"},
                            {"name": "ENVIRONMENT", "value": self.environment},
                            {"name": "PYTHONPATH", "value": "/app"}
                        ]
                    }
                }
            ],
            "scheduling": {
                "timeout": f"{timeout}s"
            }
        }
        
        # Add GPU if needed
        if self.use_gpu.lower() == 'true':
            job_config["workerPoolSpecs"][0]["machineSpec"].update({
                "acceleratorType": self.accelerator_type,
                "acceleratorCount": self.accelerator_count
            })
            
        # Write config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(job_config, f, indent=2)
            config_file = f.name
            
        try:
            result = subprocess.run([
                'gcloud', 'ai', 'custom-jobs', 'create',
                '--display-name', job_name,
                '--region', self.region,
                '--config', config_file
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info(f"✓ Job submitted via gcloud: {job_name}")
                return True
            else:
                logger.error(f"Failed to submit job: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            return False
        finally:
            os.unlink(config_file)
            
    def deploy_cloud_run(self):
        """Deploy to Cloud Run"""
        logger.info("Deploying to Cloud Run...")
        
        if not self.check_basic_prerequisites():
            return False
            
        if not self.check_project_configuration():
            return False
            
        image_uri = f"{self.region}-docker.pkg.dev/{self.project_id}/{self.repo_name}/{self.image_name}:latest"
        
        if not self.check_image_exists(image_uri):
            if not self.setup_project():
                return False
            if not self.build_image():
                return False
                
        try:
            subprocess.run([
                'gcloud', 'run', 'deploy', self.service_name,
                '--image', image_uri,
                '--region', self.region,
                '--allow-unauthenticated',
                '--memory', '2Gi',
                '--cpu', '2',
                '--max-instances', '10',
                '--set-env-vars', f'STORAGE_TYPE=gcs,STORAGE_BUCKET_NAME={self.bucket_name},ENVIRONMENT={self.environment}'
            ], check=True)
            
            logger.info(f"✓ Deployed to Cloud Run: {self.service_name}")
            self.commit_and_tag_deployment("cloud-run")
            return True
            
        except subprocess.CalledProcessError:
            logger.error("Cloud Run deployment failed")
            return False
            
    def deploy_inference_service(self):
        """Deploy inference service"""
        logger.info("Deploying inference service...")
        # This would be similar to Cloud Run but with inference-specific configuration
        return self.deploy_cloud_run()
        
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        
        try:
            # List and delete recent jobs
            result = subprocess.run([
                'gcloud', 'ai', 'custom-jobs', 'list',
                '--region', self.region,
                '--filter', f'displayName:{self.service_name}',
                '--format', 'value(name)'
            ], capture_output=True, text=True)
            
            jobs = result.stdout.strip().split('\\n')
            for job in jobs:
                if job.strip():
                    logger.info(f"Found job: {job}")
                    
        except Exception as e:
            logger.warning(f"Could not list jobs for cleanup: {e}")
            
    def upload_logs(self):
        """Upload logs to GCS"""
        logger.info("Uploading logs to GCS...")
        
        try:
            log_files = []
            
            # Find log files
            if os.path.exists('logs'):
                for f in os.listdir('logs'):
                    if f.endswith('.log'):
                        log_files.append(os.path.join('logs', f))
                        
            if not log_files:
                logger.info("No log files found to upload")
                return
                
            bucket = self.storage_client.bucket(self.bucket_name)
            
            for log_file in log_files:
                blob_name = f"magnesium-pipeline/logs/{os.path.basename(log_file)}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(log_file)
                logger.info(f"Uploaded: {log_file} -> gs://{self.bucket_name}/{blob_name}")
                
        except Exception as e:
            logger.error(f"Could not upload logs: {e}")
            
    def generate_cloud_config(self):
        """Generate cloud configuration file"""
        config_template = {
            'gcp': {
                'project_id': self.project_id,
                'region': self.region,
                'bucket_name': self.bucket_name
            },
            'training': {
                'use_gpu': True,
                'use_raw_spectral': False,
                'strategy': 'full_context',
                'mode': 'autogluon',
                'machine_type': 'n1-standard-4'
            },
            'deployment': {
                'auto_commit': True,
                'auto_push': False,
                'environment': 'production'
            }
        }
        
        os.makedirs('config', exist_ok=True)
        config_file = 'config/cloud_config.yml'
        
        with open(config_file, 'w') as f:
            yaml.dump(config_template, f, default_flow_style=False, indent=2)
            
        logger.info(f"Generated cloud config: {config_file}")
        
    def show_help(self):
        """Show help information"""
        help_text = """
Google Cloud Platform Deployment Script for Magnesium Pipeline (Python Version)

USAGE:
    python gcp_deploy.py <command> [options]

COMMANDS:
    setup           Setup GCP project (APIs, repositories, buckets)
    build           Build Docker image using Cloud Build
    build-local     Build Docker image locally
    cloud-run       Deploy to Cloud Run
    inference       Deploy inference service
    vertex-ai       Deploy training job to Vertex AI
    
    # Training command aliases
    train           Deploy standard training to Vertex AI
    tune            Deploy hyperparameter tuning to Vertex AI
    autogluon       Deploy AutoGluon training to Vertex AI
    
    cleanup         Cleanup resources
    upload-logs     Upload logs to GCS
    generate-config Generate cloud configuration template
    list-projects   List available GCP projects
    help            Show this help message

ENVIRONMENT VARIABLES:
    PROJECT_ID              GCP project ID (default: mapana-ai-models)
    REGION                  GCP region (default: us-central1)
    TRAINING_MODE           Training mode (train, autogluon, tune, optimize-*)
    USE_GPU                 Enable GPU (true/false)
    STRATEGY                Feature strategy (full_context, simple_only, Mg_only)
    MACHINE_TYPE            Machine type (default: n1-standard-4)
    
EXAMPLES:
    # Setup project
    python gcp_deploy.py setup
    
    # Build image
    python gcp_deploy.py build
    
    # Deploy training
    TRAINING_MODE=autogluon USE_GPU=true python gcp_deploy.py vertex-ai
    
    # Deploy with specific configuration
    PROJECT_ID=my-project STRATEGY=simple_only python gcp_deploy.py train

For detailed configuration options, see config/staging_config.yml
        """
        print(help_text)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='GCP Deployment Script for Magnesium Pipeline')
    parser.add_argument('command', nargs='?', default='help',
                       help='Command to execute')
    parser.add_argument('--project-id', help='GCP project ID')
    parser.add_argument('--region', help='GCP region')
    parser.add_argument('--training-mode', help='Training mode')
    parser.add_argument('--use-gpu', action='store_true', help='Enable GPU')
    parser.add_argument('--strategy', help='Feature strategy')
    
    args = parser.parse_args()
    
    # Set environment variables from command line args
    if args.project_id:
        os.environ['PROJECT_ID'] = args.project_id
    if args.region:
        os.environ['REGION'] = args.region
    if args.training_mode:
        os.environ['TRAINING_MODE'] = args.training_mode
    if args.use_gpu:
        os.environ['USE_GPU'] = 'true'
    if args.strategy:
        os.environ['STRATEGY'] = args.strategy
    
    # Initialize deployment manager
    deployment = GCPDeployment()
    
    # Execute command
    command = args.command.lower()
    
    if command == 'setup':
        deployment.setup_project()
    elif command == 'build':
        deployment.check_basic_prerequisites()
        deployment.load_cloud_config()
        deployment.check_project_configuration()
        deployment.build_image()
    elif command == 'build-local':
        deployment.check_basic_prerequisites()
        deployment.load_cloud_config()
        deployment.check_project_configuration()
        deployment.build_image_local()
    elif command == 'cloud-run':
        deployment.deploy_cloud_run()
    elif command == 'inference':
        deployment.deploy_inference_service()
    elif command == 'vertex-ai':
        deployment.deploy_vertex_ai()
    elif command == 'train':
        deployment.training_mode = 'train'
        deployment.deploy_vertex_ai()
    elif command == 'tune':
        deployment.training_mode = 'tune'
        deployment.deploy_vertex_ai()
    elif command == 'autogluon':
        deployment.training_mode = 'autogluon'
        deployment.deploy_vertex_ai()
    elif command == 'cleanup':
        deployment.cleanup()
    elif command == 'upload-logs':
        deployment.upload_logs()
    elif command == 'generate-config':
        deployment.generate_cloud_config()
    elif command == 'list-projects':
        deployment.list_gcp_projects()
    elif command == 'help':
        deployment.show_help()
    else:
        logger.error(f"Unknown command: {command}")
        deployment.show_help()
        sys.exit(1)


if __name__ == "__main__":
    main()