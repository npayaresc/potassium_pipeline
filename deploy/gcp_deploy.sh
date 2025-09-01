#!/bin/bash

# Google Cloud Platform Deployment Script for Magnesium Pipeline
# This script provides multiple deployment options for GCP
#
# IMPORTANT: AutoGluon uses Ray for distributed machine learning.
# Ray is fully compatible with Vertex AI custom training jobs.
# For optimal performance in production, consider multi-node Ray clusters.

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"mapana-ai-models"}  # Set to your actual GCP project ID
REGION=${REGION:-"us-central1"}
SERVICE_NAME=${SERVICE_NAME:-"magnesium-pipeline"}
REPO_NAME=${REPO_NAME:-"magnesium-repo"} # New: For Artifact Registry
IMAGE_NAME=${IMAGE_NAME:-"magnesium-pipeline"} # Image name within the repo
BUCKET_NAME=${BUCKET_NAME:-"${PROJECT_ID}-magnesium-data"}
AUTO_COMMIT_DEPLOYMENT=${AUTO_COMMIT_DEPLOYMENT:-"true"}  # Auto-commit and tag deployments
AUTO_PUSH_DEPLOYMENT=${AUTO_PUSH_DEPLOYMENT:-"false"}  # Auto-push commits and tags (set to true for CI/CD)

# Cloud Configuration
CLOUD_CONFIG_FILE=${CLOUD_CONFIG_FILE:-"config/cloud_config.yml"}  # Path to cloud config file
ENVIRONMENT=${ENVIRONMENT:-"production"}  # Environment: development, staging, production
USE_CLOUD_CONFIG=${USE_CLOUD_CONFIG:-"true"}  # Whether to use cloud configuration

# Training Configuration - will be initialized from Python config if not set
TRAINING_MODE=${TRAINING_MODE:-""}  # train, autogluon, tune, optimize-models, optimize-xgboost, optimize-autogluon
USE_GPU=${USE_GPU:-""}  # Enable GPU acceleration (will get default from Python config)
USE_RAW_SPECTRAL=${USE_RAW_SPECTRAL:-""}  # Use raw spectral data (will get default from Python config)
MODELS=${MODELS:-""}  # Specific models to train (comma-separated)
STRATEGY=${STRATEGY:-""}  # Feature strategy: full_context, simple_only, Mg_only (will get default from Python config)
TRIALS=${TRIALS:-""}  # Number of optimization trials
TIMEOUT=${TIMEOUT:-""}  # Timeout in seconds for optimization
MACHINE_TYPE=${MACHINE_TYPE:-"n1-standard-4"}  # Machine type for training
ACCELERATOR_TYPE=${ACCELERATOR_TYPE:-"NVIDIA_TESLA_T4"}  # GPU type
ACCELERATOR_COUNT=${ACCELERATOR_COUNT:-"1"}  # Number of GPUs

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Simple YAML parser for the cloud config file
parse_yaml() {
    local yaml_file="$1"
    local prefix="$2"
    
    if [ ! -f "$yaml_file" ]; then
        log_warn "Cloud config file not found: $yaml_file"
        return 1
    fi
    
    # Parse YAML and create environment variables
    # This is a simplified parser that handles the cloud_config.yml structure
    eval $(sed -e 's/:[^:\/\/]/=/g;s/$//g;s/ *=/=/g' "$yaml_file" | grep -E '^[a-zA-Z_][a-zA-Z0-9_]*=' | sed "s/^/${prefix}_/")
}

get_python_config_defaults() {
    # Get default values from Python pipeline_config.py
    if command -v python3 &> /dev/null; then
        local python_defaults=$(python3 -c "
import sys
import os
sys.path.insert(0, '.')
try:
    from src.config.pipeline_config import config
    
    # Get defaults from Python config
    print(f'PYTHON_USE_GPU={str(config.use_gpu).lower()}')
    print(f'PYTHON_USE_RAW_SPECTRAL={str(config.use_raw_spectral_data).lower()}')
    # Strategy default is not in Python config, will use hardcoded default
    print(f'PYTHON_DEFAULT_STRATEGY=full_context')  # Default strategy when not in YAML
    print(f'PYTHON_TRAINING_MODE=autogluon')  # Default training mode when not in YAML
except Exception as e:
    # If we can't load Python config, return empty (will use bash defaults as last resort)
    pass
" 2>/dev/null | grep -E '^PYTHON_')
        
        if [ -n "$python_defaults" ]; then
            eval "$python_defaults"
            log_info "Loaded defaults from Python config:"
            log_info "  USE_GPU default: ${PYTHON_USE_GPU}"
            log_info "  USE_RAW_SPECTRAL default: ${PYTHON_USE_RAW_SPECTRAL}"
            log_info "  STRATEGY default: ${PYTHON_DEFAULT_STRATEGY}"
            log_info "  TRAINING_MODE default: ${PYTHON_TRAINING_MODE}"
        fi
    fi
}

load_cloud_config() {
    # Save command-line values if they were explicitly set
    local cmd_use_gpu=""
    local cmd_strategy=""
    local cmd_use_raw_spectral=""
    
    if [ -n "$USE_GPU" ]; then
        cmd_use_gpu="$USE_GPU"
    fi
    if [ -n "$STRATEGY" ]; then
        cmd_strategy="$STRATEGY"
    fi
    if [ -n "$USE_RAW_SPECTRAL" ]; then
        cmd_use_raw_spectral="$USE_RAW_SPECTRAL"
    fi
    
    # First, get defaults from Python config (lowest priority)
    get_python_config_defaults
    
    # Apply Python defaults if values are still empty (not set by command line)
    if [ -z "$USE_GPU" ]; then
        USE_GPU="${PYTHON_USE_GPU:-true}"  # Final fallback to true if Python config not available
    fi
    if [ -z "$USE_RAW_SPECTRAL" ]; then
        USE_RAW_SPECTRAL="${PYTHON_USE_RAW_SPECTRAL:-false}"  # Final fallback to false
    fi
    if [ -z "$STRATEGY" ]; then
        STRATEGY="${PYTHON_DEFAULT_STRATEGY:-full_context}"  # Final fallback to full_context
    fi
    if [ -z "$TRAINING_MODE" ] && [ -n "${PYTHON_TRAINING_MODE}" ]; then
        TRAINING_MODE="${PYTHON_TRAINING_MODE}"
    fi
    
    # Second, load from YAML config if available (medium priority)
    if [ "$USE_CLOUD_CONFIG" = "true" ] && [ -f "$CLOUD_CONFIG_FILE" ]; then
        log_info "Loading cloud configuration from: $CLOUD_CONFIG_FILE"
        
        # Read GCP-specific configuration using yq or python if available
        if command -v python3 &> /dev/null; then
            # Use Python to parse YAML more reliably
            local gcp_config=$(python3 -c "
import yaml
import sys
try:
    with open('$CLOUD_CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    app = config.get('app', {})
    if 'environment' in app:
        print(f\"CLOUD_ENVIRONMENT={app.get('environment')}\")
    
    gcp = config.get('cloud_providers', {}).get('gcp', {})
    print(f\"GCP_PROJECT_ID={gcp.get('project_id', '')}\")
    print(f\"GCP_REGION={gcp.get('region', '')}\")
    print(f\"GCP_STORAGE_CLASS={gcp.get('storage_class', '')}\")
    print(f\"GCP_COMPUTE_ZONE={gcp.get('compute_zone', '')}\")
    
    storage = config.get('storage', {})
    print(f\"CLOUD_STORAGE_TYPE={storage.get('type', '')}\")
    print(f\"CLOUD_STORAGE_BUCKET={storage.get('bucket_name', '')}\")
    print(f\"CLOUD_STORAGE_PREFIX={storage.get('prefix', '')}\")
    
    compute = config.get('compute', {})
    # Only output if explicitly set in YAML (not using defaults)
    if 'gpu_enabled' in compute:
        print(f\"CLOUD_GPU_ENABLED={str(compute.get('gpu_enabled')).lower()}\")
    print(f\"CLOUD_GPU_MEMORY_FRACTION={compute.get('gpu_memory_fraction', 0.8)}\")
    
    pipeline = config.get('pipeline', {})
    # Only output if explicitly set in YAML
    if 'default_strategy' in pipeline:
        print(f\"CLOUD_DEFAULT_STRATEGY={pipeline.get('default_strategy')}\")
    print(f\"CLOUD_TIME_LIMIT={pipeline.get('time_limit', '')}\")
    if 'enable_gpu' in pipeline:
        print(f\"CLOUD_ENABLE_GPU={str(pipeline.get('enable_gpu')).lower()}\")
    
    monitoring = config.get('monitoring', {})
    print(f\"CLOUD_LOG_LEVEL={monitoring.get('log_level', '')}\")
    
except Exception as e:
    print(f\"# Error parsing YAML: {e}\", file=sys.stderr)
" 2>/dev/null)
            
            if [ $? -eq 0 ] && [ -n "$gcp_config" ]; then
                eval "$gcp_config"
                
                # Override defaults with cloud config values (command line still has highest priority)
                if [ -n "$CLOUD_ENVIRONMENT" ]; then
                    ENVIRONMENT="$CLOUD_ENVIRONMENT"
                fi
                if [ -n "$GCP_PROJECT_ID" ]; then
                    PROJECT_ID="$GCP_PROJECT_ID"
                fi
                if [ -n "$GCP_REGION" ]; then
                    REGION="$GCP_REGION"
                fi
                if [ -n "$CLOUD_STORAGE_BUCKET" ]; then
                    BUCKET_NAME="$CLOUD_STORAGE_BUCKET"
                fi
                
                # Override training settings from YAML only if not set by command line
                local strategy_source="python-config"
                local gpu_source="python-config"
                
                if [ -n "$CLOUD_DEFAULT_STRATEGY" ] && [ -z "$cmd_strategy" ]; then
                    STRATEGY="$CLOUD_DEFAULT_STRATEGY"
                    strategy_source="yaml"
                elif [ -n "$cmd_strategy" ]; then
                    STRATEGY="$cmd_strategy"
                    strategy_source="command-line"
                fi
                
                if [ -n "$CLOUD_ENABLE_GPU" ] && [ -z "$cmd_use_gpu" ]; then
                    USE_GPU="$CLOUD_ENABLE_GPU"
                    gpu_source="yaml"
                elif [ -n "$cmd_use_gpu" ]; then
                    USE_GPU="$cmd_use_gpu"
                    gpu_source="command-line"
                fi
                
                log_info "Configuration hierarchy applied:"
                log_info "  Environment: $ENVIRONMENT (from: yaml)"
                log_info "  Project ID: $PROJECT_ID"
                log_info "  Region: $REGION"
                log_info "  Bucket: $BUCKET_NAME"
                log_info "  Strategy: $STRATEGY (from: $strategy_source)"
                log_info "  GPU Enabled: $USE_GPU (from: $gpu_source)"
            else
                log_warn "Failed to parse cloud configuration with Python"
                # Restore command-line values if they were set
                if [ -n "$cmd_strategy" ]; then
                    STRATEGY="$cmd_strategy"
                    log_info "  Strategy: $STRATEGY (from command-line)"
                else
                    log_info "  Strategy: $STRATEGY (from Python config)"
                fi
                if [ -n "$cmd_use_gpu" ]; then
                    USE_GPU="$cmd_use_gpu"
                    log_info "  GPU Enabled: $USE_GPU (from command-line)"
                else
                    log_info "  GPU Enabled: $USE_GPU (from Python config)"
                fi
            fi
        else
            log_warn "Python3 not available for YAML parsing"
            # Restore command-line values if they were set
            if [ -n "$cmd_strategy" ]; then
                STRATEGY="$cmd_strategy"
                log_info "  Strategy: $STRATEGY (from command-line)"
            else
                log_info "  Strategy: $STRATEGY (from Python config)"
            fi
            if [ -n "$cmd_use_gpu" ]; then
                USE_GPU="$cmd_use_gpu"
                log_info "  GPU Enabled: $USE_GPU (from command-line)"
            else
                log_info "  GPU Enabled: $USE_GPU (from Python config)"
            fi
        fi
    else
        log_info "Cloud configuration disabled or file not found"
        # Restore command-line values if they were set
        if [ -n "$cmd_strategy" ]; then
            STRATEGY="$cmd_strategy"
            log_info "  Strategy: $STRATEGY (from command-line)"
        else
            log_info "  Strategy: $STRATEGY (from Python config)"
        fi
        if [ -n "$cmd_use_gpu" ]; then
            USE_GPU="$cmd_use_gpu"
            log_info "  GPU Enabled: $USE_GPU (from command-line)"
        else
            log_info "  GPU Enabled: $USE_GPU (from Python config)"
        fi
    fi
}

# Deployment tracking functions
create_deployment_record() {
    local deployment_type="$1"  # vertex-ai, cloud-run, gke
    local deployment_id="$2"
    local deployment_status="$3"  # started, success, failed
    local deployment_details="$4"  # Additional details (JSON string)
    
    local deployment_dir=".deployments"
    mkdir -p "$deployment_dir"
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local deployment_file="$deployment_dir/deployment_${timestamp}_${deployment_type}.json"
    
    cat > "$deployment_file" <<EOF
{
  "deployment_id": "$deployment_id",
  "deployment_type": "$deployment_type",
  "timestamp": "$timestamp",
  "iso_timestamp": "$(date -Iseconds)",
  "status": "$deployment_status",
  "environment": "${ENVIRONMENT}",
  "project_id": "${PROJECT_ID}",
  "region": "${REGION}",
  "training_mode": "${TRAINING_MODE}",
  "strategy": "${STRATEGY}",
  "use_gpu": "${USE_GPU}",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
  "details": $deployment_details
}
EOF
    
    echo "$deployment_file"
}

commit_and_tag_deployment() {
    local deployment_type="$1"
    local deployment_id="$2"
    local deployment_file="$3"
    
    if [ "$AUTO_COMMIT_DEPLOYMENT" != "true" ]; then
        log_info "Auto-commit disabled. Skipping git operations."
        return 0
    fi
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_warn "Not in a git repository. Skipping commit and tag."
        return 1
    fi
    
    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        log_info "Committing deployment changes..."
        
        # Add all changes including the deployment record
        git add -A
        
        # Create commit message
        local commit_message="🚀 Deploy: ${deployment_type} [${deployment_id}]

Deployment Details:
- Type: ${deployment_type}
- Environment: ${ENVIRONMENT}
- Training Mode: ${TRAINING_MODE}
- Strategy: ${STRATEGY}
- GPU Enabled: ${USE_GPU}
- Project: ${PROJECT_ID}
- Region: ${REGION}
- Timestamp: $(date -Iseconds)

Deployment ID: ${deployment_id}"
        
        git commit -m "$commit_message"
        log_success "Created deployment commit: $(git rev-parse --short HEAD)"
    else
        log_info "No uncommitted changes to commit."
    fi
    
    # Create a tag for this deployment
    local tag_name="deploy-${ENVIRONMENT}-${deployment_type}-$(date +%Y%m%d-%H%M%S)"
    local tag_message="Deployment ${deployment_id}
Type: ${deployment_type}
Environment: ${ENVIRONMENT}
Training Mode: ${TRAINING_MODE}"
    
    git tag -a "$tag_name" -m "$tag_message"
    log_success "Created deployment tag: $tag_name"
    
    # Push if enabled
    if [ "$AUTO_PUSH_DEPLOYMENT" = "true" ]; then
        log_info "Pushing deployment commit and tags..."
        git push origin HEAD
        git push origin "$tag_name"
        log_success "Pushed deployment to remote repository"
    else
        log_info "Auto-push disabled. To push manually, run:"
        log_info "  git push origin HEAD"
        log_info "  git push origin $tag_name"
    fi
    
    return 0
}

validate_training_config() {
    local valid_modes=("train" "autogluon" "tune" "optimize-models" "optimize-xgboost" "optimize-autogluon" "optimize-range-specialist")
    local valid_strategies=("full_context" "simple_only" "Mg_only")
    local valid_models=("ridge" "lasso" "random_forest" "xgboost" "lightgbm" "catboost" "extratrees" "neural_network" "neural_network_light" "svr")
    
    # Validate training mode
    if [[ ! " ${valid_modes[@]} " =~ " ${TRAINING_MODE} " ]]; then
        log_error "Invalid TRAINING_MODE: ${TRAINING_MODE}"
        log_error "Valid options: ${valid_modes[*]}"
        exit 1
    fi
    
    # Validate strategy
    if [[ ! " ${valid_strategies[@]} " =~ " ${STRATEGY} " ]]; then
        log_error "Invalid STRATEGY: ${STRATEGY}"
        log_error "Valid options: ${valid_strategies[*]}"
        exit 1
    fi
    
    # Validate models if specified
    if [ -n "${MODELS}" ]; then
        IFS=',' read -ra MODEL_ARRAY <<< "${MODELS}"
        for model in "${MODEL_ARRAY[@]}"; do
            if [[ ! " ${valid_models[@]} " =~ " ${model} " ]]; then
                log_error "Invalid model: ${model}"
                log_error "Valid models: ${valid_models[*]}"
                exit 1
            fi
        done
    fi
    
    # Log configuration summary
    log_info "Training Configuration:"
    log_info "  Mode: ${TRAINING_MODE}"
    log_info "  GPU: ${USE_GPU}"
    log_info "  Raw Spectral: ${USE_RAW_SPECTRAL}"
    log_info "  Strategy: ${STRATEGY}"
    if [ -n "${MODELS}" ]; then
        log_info "  Models: ${MODELS}"
    else
        # Get default models from Python config when not specified
        local default_models=$(python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from src.config.pipeline_config import config
    print(','.join(config.models_to_train))
except Exception as e:
    print('unable to determine')
" 2>/dev/null)
        if [ -n "$default_models" ]; then
            log_info "  Models: ${default_models} (from Python config default)"
        else
            log_info "  Models: Will use defaults from Python config"
        fi
    fi
    if [ -n "${TRIALS}" ]; then
        log_info "  Trials: ${TRIALS}"
    fi
    if [ -n "${TIMEOUT}" ]; then
        log_info "  Timeout: ${TIMEOUT}s"
    fi
    log_info "  Machine: ${MACHINE_TYPE} with ${ACCELERATOR_COUNT}x ${ACCELERATOR_TYPE}"
}

build_training_command() {
    local base_cmd="python main.py ${TRAINING_MODE}"
    
    # Add GPU flag
    if [ "${USE_GPU}" = "true" ]; then
        base_cmd="${base_cmd} --gpu"
    fi
    
    # Add raw spectral flag
    if [ "${USE_RAW_SPECTRAL}" = "true" ]; then
        base_cmd="${base_cmd} --raw-spectral"
    fi
    
    # Add models (convert comma-separated to space-separated)
    if [ -n "${MODELS}" ]; then
        local models_array=(${MODELS//,/ })
        base_cmd="${base_cmd} --models ${models_array[@]}"
    fi
    
    # Add strategy for optimization commands and train command
    case "${TRAINING_MODE}" in
        "train"|"optimize-xgboost"|"optimize-autogluon"|"optimize-models"|"optimize-range-specialist")
            if [ -n "${STRATEGY}" ]; then
                base_cmd="${base_cmd} --strategy ${STRATEGY}"
            fi
            ;;
    esac
    
    # Add trials for optimization commands
    if [ -n "${TRIALS}" ] && [[ "${TRAINING_MODE}" == *"optimize"* ]]; then
        base_cmd="${base_cmd} --trials ${TRIALS}"
    fi
    
    # Add timeout for optimization commands
    if [ -n "${TIMEOUT}" ] && [[ "${TRAINING_MODE}" == *"optimize"* ]]; then
        base_cmd="${base_cmd} --timeout ${TIMEOUT}"
    fi
    
    echo "${base_cmd}"
}

check_basic_prerequisites() {
    log_info "Checking basic prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if logged in
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_error "Not logged into gcloud. Run: gcloud auth login"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    log_info "Basic prerequisites check passed!"
}

check_project_configuration() {
    # Check if PROJECT_ID is set to default placeholder after loading cloud config
    if [ "$PROJECT_ID" = "your-gcp-project-id" ]; then
        log_warn "PROJECT_ID is not configured. Let's set it up..."
        
        log_info "Available projects:"
        local projects_output=$(gcloud projects list --format="table(projectId,name)" 2>/dev/null)
        
        if [ $? -eq 0 ] && [ -n "$projects_output" ]; then
            echo "$projects_output"
            echo
            
            # Interactive project selection
            if [ -t 0 ]; then  # Check if running interactively
                echo -n "Enter your project ID from the list above: "
                read -r selected_project_id
                
                if [ -n "$selected_project_id" ]; then
                    # Validate the entered project ID exists
                    if echo "$projects_output" | grep -q "^$selected_project_id\s"; then
                        export PROJECT_ID="$selected_project_id"
                        log_info "Using project ID: $PROJECT_ID"
                        
                        # Ask if user wants to save to config file
                        echo -n "Save this project ID to $CLOUD_CONFIG_FILE? (y/n): "
                        read -r save_choice
                        if [[ "$save_choice" =~ ^[Yy]$ ]]; then
                            update_config_project_id "$PROJECT_ID"
                        fi
                    else
                        log_error "Invalid project ID: $selected_project_id"
                        exit 1
                    fi
                else
                    log_error "No project ID entered."
                    exit 1
                fi
            else
                log_error "Running in non-interactive mode. Please set PROJECT_ID."
                log_info "Set PROJECT_ID using: export PROJECT_ID=your-actual-project-id"
                log_info "Or update your cloud configuration file: $CLOUD_CONFIG_FILE"
                exit 1
            fi
        else
            log_error "Unable to list projects. Please check your gcloud authentication."
            log_info "Set PROJECT_ID using: export PROJECT_ID=your-actual-project-id"
            exit 1
        fi
    fi
    
    log_info "Project configuration validated: $PROJECT_ID"
}

update_config_project_id() {
    local new_project_id="$1"
    local config_file="$CLOUD_CONFIG_FILE"
    
    if [ ! -f "$config_file" ]; then
        log_warn "Config file $config_file not found. Creating it..."
        generate_cloud_config "$(basename "$config_file" _config.yml)"
    fi
    
    # Update the project_id in the YAML file
    if command -v sed &> /dev/null; then
        # Create a backup
        cp "$config_file" "$config_file.backup"
        
        # Update the project_id line
        sed -i.tmp "s/project_id: \".*\"/project_id: \"$new_project_id\"/" "$config_file" && rm -f "$config_file.tmp"
        
        if [ $? -eq 0 ]; then
            log_info "Updated project_id in $config_file"
            log_info "Backup saved as $config_file.backup"
        else
            log_error "Failed to update config file"
            mv "$config_file.backup" "$config_file"  # Restore backup
        fi
    else
        log_warn "sed not available. Please manually update project_id in $config_file"
    fi
}

setup_project() {
    # Check basic prerequisites first
    check_basic_prerequisites
    
    # Load cloud configuration
    load_cloud_config
    
    # Check project configuration after loading cloud config
    check_project_configuration
    
    log_info "Setting up GCP project..."
    
    # Check if PROJECT_ID looks like a project number (all digits)
    if [[ "$PROJECT_ID" =~ ^[0-9]+$ ]]; then
        log_error "PROJECT_ID appears to be a project number ($PROJECT_ID), not a project ID"
        log_error "Please set PROJECT_ID to your actual project ID (e.g., 'my-project-name')"
        log_error "You can find your project ID by running: gcloud projects list"
        exit 1
    fi
    
    # Set project
    gcloud config set project $PROJECT_ID
    
    # Enable required APIs
    log_info "Enabling required APIs..."
    gcloud services enable cloudbuild.googleapis.com
    gcloud services enable run.googleapis.com
    gcloud services enable container.googleapis.com
    gcloud services enable storage-api.googleapis.com
    gcloud services enable aiplatform.googleapis.com
    gcloud services enable artifactregistry.googleapis.com
    
    # Create storage bucket
    log_info "Creating storage bucket..."
    if ! gsutil ls gs://${BUCKET_NAME} &> /dev/null; then
        gsutil mb -l ${REGION} gs://${BUCKET_NAME}
        log_info "Created bucket: gs://${BUCKET_NAME}"
    else
        log_warn "Bucket gs://${BUCKET_NAME} already exists"
    fi

    # Create Artifact Registry repository
    log_info "Creating Artifact Registry repository..."
    if ! gcloud artifacts repositories describe ${REPO_NAME} --location=${REGION} &> /dev/null; then
        gcloud artifacts repositories create ${REPO_NAME} \
            --repository-format=docker \
            --location=${REGION} \
            --description="Docker repository for Magnesium Pipeline"
        log_info "Created Artifact Registry repo: ${REPO_NAME}"
    else
        log_warn "Artifact Registry repo ${REPO_NAME} already exists"
    fi
}

check_image_exists() {
    local image_uri="$1"
    log_info "Checking if container image exists: ${image_uri}"
    
    # Check if image exists in Artifact Registry
    if gcloud artifacts docker images describe "${image_uri}" --quiet &>/dev/null; then
        log_info "Container image found: ${image_uri}"
        return 0
    else
        log_warn "Container image not found: ${image_uri}"
        return 1
    fi
}

build_image() {
    log_info "Building and pushing container image with caching..."
    local image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
    local cache_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:cache"
    
    # Configure Docker to use gcloud as credential helper
    gcloud auth configure-docker ${REGION}-docker.pkg.dev
    
    # Create a cloudbuild.yaml for better caching
    cat > /tmp/cloudbuild.yaml <<EOF
steps:
  # Build the image using cache
  - name: 'gcr.io/kaniko-project/executor:latest'
    args:
      - '--destination=${image_uri}'
      - '--cache=true'
      - '--cache-ttl=168h'
      - '--cache-repo=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/cache'
      - '--use-new-run'
      - '--snapshot-mode=redo'
      - '--context=dir://.'
      - '--dockerfile=Dockerfile'
timeout: '1200s'
options:
  machineType: 'E2_HIGHCPU_8'
  logging: CLOUD_LOGGING_ONLY
EOF
    
    # Build image with Cloud Build using Kaniko for caching
    log_info "Using Kaniko builder for efficient caching..."
    gcloud builds submit --config=/tmp/cloudbuild.yaml . || {
        log_warn "Kaniko build failed, falling back to standard build..."
        gcloud builds submit --tag ${image_uri} .
    }
    
    log_info "Image built and pushed to ${image_uri}"
}

build_image_local() {
    log_info "Building container image locally with Docker..."
    local image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
    local skip_push=${SKIP_PUSH:-"false"}
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker or use 'build' for Cloud Build."
        exit 1
    fi
    
    # Configure Docker to use gcloud as credential helper
    if [ "$skip_push" != "true" ]; then
        gcloud auth configure-docker ${REGION}-docker.pkg.dev
    fi
    
    # Build locally with Docker (uses local cache)
    log_info "Building with local Docker for faster caching..."
    docker build -t ${image_uri} \
        --cache-from ${image_uri} \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        .
    
    if [ "$skip_push" = "true" ]; then
        log_info "Skipping push (SKIP_PUSH=true). Image built locally: ${image_uri}"
        log_warn "Note: You'll need to push manually before deploying to cloud"
    else
        # Push to registry
        log_info "Pushing image to Artifact Registry..."
        docker push ${image_uri}
        log_info "Image built locally and pushed to ${image_uri}"
    fi
}

check_data_exists_in_gcs() {
    local bucket_prefix="${CLOUD_STORAGE_PREFIX:-magnesium-pipeline}"
    
    log_info "Checking if training data exists in GCS bucket..."
    
    # Check for key data directories (only raw data, not generated directories)
    local required_paths=(
        "gs://${BUCKET_NAME}/${bucket_prefix}/data_5278_Phase3/"
    )
    
    local missing_paths=0
    for path in "${required_paths[@]}"; do
        if ! gsutil ls "${path}" &>/dev/null; then
            log_warn "Missing: ${path}"
            ((missing_paths++))
        else
            log_info "Found: ${path}"
        fi
    done
    
    if [[ $missing_paths -eq 0 ]]; then
        log_info "All required training data found in GCS bucket"
        return 0
    else
        log_warn "Missing ${missing_paths} required data paths in GCS"
        return 1
    fi
}

list_and_verify_gcs_data() {
    local bucket_prefix="${CLOUD_STORAGE_PREFIX:-magnesium-pipeline}"
    
    log_info "======================================"
    log_info "GCS Data Inventory and Verification"
    log_info "======================================"
    log_info "Bucket: gs://${BUCKET_NAME}/${bucket_prefix}/"
    log_info ""
    
    # Define directories to check
    local raw_dir="gs://${BUCKET_NAME}/${bucket_prefix}/data_5278_Phase3/"
    local averaged_dir="gs://${BUCKET_NAME}/${bucket_prefix}/averaged_files_per_sample/"
    local cleansed_dir="gs://${BUCKET_NAME}/${bucket_prefix}/cleansed_files_per_sample/"
    local reference_dir="gs://${BUCKET_NAME}/${bucket_prefix}/reference_data/"
    
    # Check RAW data directory (most important)
    log_info "1. RAW DATA (data_5278_Phase3):"
    log_info "--------------------------------"
    if gsutil ls "${raw_dir}" &>/dev/null; then
        log_success "✓ Directory exists"
        # Count CSV files
        local raw_count=$(gsutil ls "${raw_dir}*.csv.txt" 2>/dev/null | wc -l)
        if [[ $raw_count -gt 0 ]]; then
            log_success "✓ Contains ${raw_count} CSV files"
            # Show sample files
            log_info "  Sample files:"
            gsutil ls "${raw_dir}*.csv.txt" 2>/dev/null | head -5 | while read -r file; do
                log_info "    - $(basename "$file")"
            done
            if [[ $raw_count -gt 5 ]]; then
                log_info "    ... and $((raw_count - 5)) more files"
            fi
        else
            log_error "✗ No CSV files found!"
        fi
    else
        log_error "✗ Directory NOT FOUND - This is critical for training!"
    fi
    log_info ""
    
    # Check REFERENCE data directory (Excel files)
    log_info "2. REFERENCE DATA (Excel files):"
    log_info "--------------------------------"
    if gsutil ls "${reference_dir}" &>/dev/null; then
        log_success "✓ Directory exists"
        # Count Excel files
        local excel_count=$(gsutil ls "${reference_dir}*.xlsx" "${reference_dir}*.xls" 2>/dev/null | wc -l)
        if [[ $excel_count -gt 0 ]]; then
            log_success "✓ Contains ${excel_count} Excel file(s)"
            # List all Excel files (usually just a few)
            log_info "  Excel files:"
            gsutil ls "${reference_dir}*.xlsx" "${reference_dir}*.xls" 2>/dev/null | while read -r file; do
                log_info "    - $(basename "$file")"
            done
        else
            log_error "✗ No Excel files found - Required for ground truth values!"
        fi
    else
        log_warn "⚠ Reference directory not found - Will need to upload Excel files"
    fi
    log_info ""
    
    # Check AVERAGED data directory (GENERATED DURING TRAINING)
    log_info "3. AVERAGED DATA (Generated during training):"
    log_info "--------------------------------"
    if gsutil ls "${averaged_dir}" &>/dev/null; then
        log_success "✓ Directory exists"
        local averaged_count=$(gsutil ls "${averaged_dir}*.csv.txt" 2>/dev/null | wc -l)
        if [[ $averaged_count -gt 0 ]]; then
            log_success "✓ Contains ${averaged_count} averaged files"
        else
            log_info "  Directory exists but empty (will be populated during training)"
        fi
    else
        log_info "  Not present (normal - will be created during training)"
    fi
    log_info ""
    
    # Check CLEANSED data directory (GENERATED DURING TRAINING)
    log_info "4. CLEANSED DATA (Generated during training):"
    log_info "--------------------------------"
    if gsutil ls "${cleansed_dir}" &>/dev/null; then
        log_success "✓ Directory exists"
        local cleansed_count=$(gsutil ls "${cleansed_dir}*.csv.txt" 2>/dev/null | wc -l)
        if [[ $cleansed_count -gt 0 ]]; then
            log_success "✓ Contains ${cleansed_count} cleansed files"
        else
            log_info "  Directory exists but empty (will be populated during training)"
        fi
    else
        log_info "  Not present (normal - will be created during training)"
    fi
    log_info ""
    
    # Summary
    log_info "======================================"
    log_info "SUMMARY:"
    log_info "======================================"
    
    local critical_missing=false
    
    # Check critical components
    if ! gsutil ls "${raw_dir}" &>/dev/null; then
        log_error "CRITICAL: Raw data directory missing!"
        critical_missing=true
    elif [[ $(gsutil ls "${raw_dir}*.csv.txt" 2>/dev/null | wc -l) -eq 0 ]]; then
        log_error "CRITICAL: No raw CSV files found!"
        critical_missing=true
    fi
    
    if ! gsutil ls "${reference_dir}" &>/dev/null || [[ $(gsutil ls "${reference_dir}*.xlsx" "${reference_dir}*.xls" 2>/dev/null | wc -l) -eq 0 ]]; then
        log_warn "WARNING: No reference Excel files found - required for training!"
    fi
    
    if [[ "$critical_missing" == "false" ]]; then
        log_success "✓ Essential data components are present"
        log_info "  Ready for training after reference data is uploaded"
    else
        log_error "✗ Missing critical data components"
        log_info "  Run: $0 upload-data"
    fi
}

upload_training_data() {
    local bucket_prefix="${CLOUD_STORAGE_PREFIX:-magnesium-pipeline}"
    
    # Check if data already exists in GCS
    if check_data_exists_in_gcs; then
        log_info "Training data already exists in GCS bucket. Skipping upload."
        log_info "Use './deploy/gcp_deploy.sh upload-data --force' to re-upload data."
        return 0
    fi
    
    log_info "Uploading training data to GCS bucket..."
    
    # Upload data directories with rsync for efficiency
    if [[ -d "data/raw/data_5278_Phase3" ]]; then
        log_info "Uploading raw spectral data..."
        gsutil -m rsync -r -d "data/raw/data_5278_Phase3" "gs://${BUCKET_NAME}/${bucket_prefix}/data_5278_Phase3/"
    fi
    
    # Note: averaged_files_per_sample and cleansed_files_per_sample are generated during training
    # They don't need to be uploaded beforehand
    
    # Upload reference files (Excel files)
    if ls data/*.xlsx 1> /dev/null 2>&1; then
        log_info "Uploading Excel reference files..."
        gsutil -m cp data/*.xlsx "gs://${BUCKET_NAME}/${bucket_prefix}/"
    fi
    
    if ls data/*.xls 1> /dev/null 2>&1; then
        log_info "Uploading XLS reference files..."
        gsutil -m cp data/*.xls "gs://${BUCKET_NAME}/${bucket_prefix}/"
    fi
    
    # Upload reference_data directory if it exists
    if [[ -d "data/reference_data" ]]; then
        log_info "Uploading reference data directory..."
        gsutil -m rsync -r -d "data/reference_data" "gs://${BUCKET_NAME}/${bucket_prefix}/reference_data/"
    fi
    
    log_info "Training data upload completed to gs://${BUCKET_NAME}/${bucket_prefix}/"
    log_info "Verifying uploaded data structure..."
    gsutil ls -r "gs://${BUCKET_NAME}/${bucket_prefix}/" | head -20
}

force_upload_training_data() {
    log_info "Force uploading training data (ignoring existing data)..."
    upload_training_data
}

deploy_cloud_run() {
    log_info "Deploying to Cloud Run..."
    local image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
    
    # Create deployment details
    local deployment_details=$(cat <<DETAILS
{
  "service_name": "${SERVICE_NAME}",
  "image_uri": "${image_uri}",
  "memory": "8Gi",
  "cpu": "4",
  "timeout": 3600,
  "concurrency": 10,
  "max_instances": 5
}
DETAILS
)
    
    # Create initial deployment record
    local deployment_file=$(create_deployment_record "cloud-run" "pending" "started" "$deployment_details")
    log_info "Created deployment record: $deployment_file"

    # Deploy and capture output
    local deploy_output=$(gcloud run deploy ${SERVICE_NAME} \
        --image ${image_uri} \
        --platform managed \
        --region ${REGION} \
        --memory 8Gi \
        --cpu 4 \
        --timeout 3600 \
        --concurrency 10 \
        --max-instances 5 \
        --allow-unauthenticated \
        --set-env-vars="STORAGE_TYPE=gcs,STORAGE_BUCKET_NAME=${BUCKET_NAME},GPU_ENABLED=false" 2>&1)
    
    local deploy_status=$?
    
    if [ $deploy_status -eq 0 ]; then
        # Get service URL
        SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")
        log_success "Service deployed at: ${SERVICE_URL}"
        
        # Create deployment ID from service URL
        local deployment_id="${SERVICE_NAME}-${REGION}-$(date +%Y%m%d-%H%M%S)"
        
        # Test health endpoint
        log_info "Testing deployment..."
        local health_status="unknown"
        if curl -s "${SERVICE_URL}/health" | grep -q "healthy"; then
            log_success "Deployment test successful!"
            health_status="healthy"
        else
            log_warn "Deployment test failed. Check logs with: gcloud run logs tail ${SERVICE_NAME} --region ${REGION}"
            health_status="unhealthy"
        fi
        
        # Update deployment record with success
        local updated_details=$(cat <<DETAILS
{
  "deployment_id": "$deployment_id",
  "service_name": "${SERVICE_NAME}",
  "service_url": "${SERVICE_URL}",
  "image_uri": "${image_uri}",
  "memory": "8Gi",
  "cpu": "4",
  "timeout": 3600,
  "concurrency": 10,
  "max_instances": 5,
  "health_status": "$health_status"
}
DETAILS
)
        
        # Create final deployment record
        local final_deployment_file=$(create_deployment_record "cloud-run" "$deployment_id" "success" "$updated_details")
        
        # Commit and tag the deployment
        commit_and_tag_deployment "cloud-run" "$deployment_id" "$final_deployment_file"
        
        log_warn "Cloud Run service is publicly accessible. For production, consider using --no-allow-unauthenticated and setting up Cloud IAP or API Gateway."
    else
        log_error "Cloud Run deployment failed"
        log_info "Error output: $deploy_output"
        
        # Update deployment record with failure
        create_deployment_record "cloud-run" "failed" "failed" '{"error": "Deployment failed"}'
    fi
}

deploy_inference_service() {
    log_info "Deploying inference service to Cloud Run..."
    local image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
    local inference_service_name="${SERVICE_NAME}-inference"

    gcloud run deploy ${inference_service_name} \
        --image ${image_uri} \
        --platform managed \
        --region ${REGION} \
        --memory 4Gi \
        --cpu 2 \
        --timeout 900 \
        --concurrency 100 \
        --max-instances 10 \
        --allow-unauthenticated \
        --port 8000 \
        --set-env-vars="STORAGE_TYPE=gcs,STORAGE_BUCKET_NAME=${BUCKET_NAME},CLOUD_STORAGE_PREFIX=${CLOUD_STORAGE_PREFIX:-magnesium-pipeline},INFERENCE_MODE=true,GPU_ENABLED=false" \
        --command="bash" \
        --args="-c","source /app/docker-entrypoint.sh && download_models_for_inference && python api_server.py"
    
    # Get service URL
    INFERENCE_URL=$(gcloud run services describe ${inference_service_name} --region=${REGION} --format="value(status.url)")
    log_info "Inference service deployed at: ${INFERENCE_URL}"
    
    # Test health endpoint
    log_info "Testing inference service..."
    sleep 10  # Give service time to start
    if curl -s "${INFERENCE_URL}/health" | grep -q "healthy"; then
        log_info "Inference service test successful!"
        log_info "API Documentation: ${INFERENCE_URL}/docs"
        log_info "Test prediction: curl -X POST '${INFERENCE_URL}/predict' -F 'file=@your_file.csv.txt' -F 'model_path=model.pkl'"
    else
        log_warn "Inference service test failed. Check logs with: gcloud run logs tail ${inference_service_name} --region ${REGION}"
    fi
}

deploy_gke() {
    log_info "Deploying to GKE..."
    local image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"

    CLUSTER_NAME="${SERVICE_NAME}-cluster"
    
    # Create GKE cluster with GPU nodes
    if ! gcloud container clusters describe ${CLUSTER_NAME} --region=${REGION} &> /dev/null; then
        log_info "Creating GKE cluster with GPU support..."
        gcloud container clusters create ${CLUSTER_NAME} \
            --region=${REGION} \
            --num-nodes=1 \
            --machine-type=n1-standard-4 \
            --enable-autoscaling \
            --min-nodes=0 \
            --max-nodes=3 \
            --enable-autorepair \
            --enable-autoupgrade \
            --disk-size=100GB
        
        # Add GPU node pool
        gcloud container node-pools create gpu-pool \
            --cluster=${CLUSTER_NAME} \
            --region=${REGION} \
            --machine-type=n1-standard-4 \
            --accelerator=type=nvidia-tesla-t4,count=1 \
            --num-nodes=0 \
            --enable-autoscaling \
            --min-nodes=0 \
            --max-nodes=2 \
            --disk-size=100GB \
            --preemptible
    fi
    
    # Get cluster credentials
    gcloud container clusters get-credentials ${CLUSTER_NAME} --region=${REGION}
    
    # Install NVIDIA device plugin
    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
    
    # Create namespace
    kubectl create namespace magnesium-pipeline --dry-run=client -o yaml | kubectl apply -f -
    
    # Create deployment
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${SERVICE_NAME}
  namespace: magnesium-pipeline
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${SERVICE_NAME}
  template:
    metadata:
      labels:
        app: ${SERVICE_NAME}
    spec:
      containers:
      - name: ${SERVICE_NAME}
        image: ${image_uri}
        ports:
        - containerPort: 8000
        env:
        - name: STORAGE_TYPE
          value: "gcs"
        - name: STORAGE_BUCKET
          value: "${BUCKET_NAME}" # Note: Your cloud_manager.py expects STORAGE_BUCKET_NAME
        - name: GPU_ENABLED
          value: "true"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
          requests:
            memory: "4Gi"
            cpu: "2"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
---
apiVersion: v1
kind: Service
metadata:
  name: ${SERVICE_NAME}-service
  namespace: magnesium-pipeline
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  selector:
    app: ${SERVICE_NAME}
EOF
    
    log_info "Deployment created. Waiting for external IP..."
    kubectl wait --namespace=magnesium-pipeline --for=condition=available --timeout=300s deployment/${SERVICE_NAME}
    
    EXTERNAL_IP=$(kubectl get service ${SERVICE_NAME}-service --namespace=magnesium-pipeline -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -n "$EXTERNAL_IP" ]; then
        log_info "Service available at: http://${EXTERNAL_IP}"
    else
        log_warn "External IP not yet assigned. Check with: kubectl get services -n magnesium-pipeline"
    fi
}

deploy_vertex_ai() {
    # Load cloud configuration first
    load_cloud_config
    
    # Validate training configuration
    validate_training_config
    
    log_info "Deploying to Vertex AI for ${TRAINING_MODE} training..."
    
    # Show final model configuration
    if [ -n "${MODELS}" ]; then
        log_info "Models to be trained: ${MODELS}"
    else
        # Get default models from Python config
        local python_models=$(python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from src.config.pipeline_config import config
    models = config.models_to_train
    print(f'Models from Python config: {models}')
    print(f'Number of models: {len(models)}')
    for m in models:
        print(f'  - {m}')
except Exception as e:
    print(f'Error reading models: {e}')
" 2>&1)
        if [ $? -eq 0 ]; then
            echo "$python_models" | while IFS= read -r line; do
                log_info "$line"
            done
        else
            log_warn "Could not determine models from Python config"
        fi
    fi
    
    # Special warnings based on training mode
    case "${TRAINING_MODE}" in
        "autogluon"|"optimize-autogluon")
            log_warn "NOTE: AutoGluon uses Ray for distributed training. Ray is compatible with Vertex AI"
            log_warn "custom training jobs but may require additional configuration for optimal performance."
            log_info "For production AutoGluon deployments, consider using multiple workers or Ray clusters."
            ;;
        *"optimize"*)
            log_info "Running optimization training - this may take significant time and resources."
            ;;
    esac
    
    local image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
    
    # Check if container image exists, build if needed
    if ! check_image_exists "${image_uri}"; then
        log_info "Container image not found. Setting up project and building image..."
        
        # Ensure project setup is complete (repositories, APIs, etc.)
        setup_project
        
        # Build and push the container image
        build_image
        
        # Verify the image was built successfully
        if ! check_image_exists "${image_uri}"; then
            log_error "Failed to build container image. Cannot proceed with training."
            exit 1
        fi
        
        log_info "Container image ready for training!"
    else
        log_info "Using existing container image: ${image_uri}"
    fi
    
    # Upload training data to GCS bucket
    upload_training_data
    
    # Build the training command and capture it without log messages
    local training_cmd=$(build_training_command 2>/dev/null)
    
    # Now log the information about models
    if [ -n "${MODELS}" ]; then
        log_info "Models explicitly specified: ${MODELS}"
    else
        log_info "No models specified - will use Python config defaults"
    fi
    
    local cmd_array=(${training_cmd})  # Convert to array
    
    log_info "Training command: ${training_cmd}"
    
    # Build JSON command array
    # Use bash -c to ensure the entrypoint runs first, then the training command
    # This ensures data is downloaded from GCS before training starts
    local json_cmd="\"bash\", \"-c\", \"source /app/docker-entrypoint.sh && download_training_data && ${training_cmd}\""
    
    # Determine appropriate timeout based on training mode
    local job_timeout=""
    case "${TRAINING_MODE}" in
        *"optimize"*)
            job_timeout="86400"  # 24 hours for optimization
            ;;
        "autogluon")
            job_timeout="14400"  # 4 hours for AutoGluon
            ;;
        *)
            job_timeout="7200"   # 2 hours for standard training
            ;;
    esac

    # Create a unique job name
    local job_display_name="${SERVICE_NAME}-${TRAINING_MODE}-$(date +%Y%m%d-%H%M%S)"
    
    # Create deployment details JSON
    local deployment_details=$(cat <<DETAILS
{
  "image_uri": "${image_uri}",
  "machine_type": "${MACHINE_TYPE}",
  "accelerator_type": "${ACCELERATOR_TYPE}",
  "accelerator_count": ${ACCELERATOR_COUNT},
  "job_timeout": ${job_timeout},
  "training_command": "${cmd}"
}
DETAILS
)
    
    # Create initial deployment record
    local deployment_file=$(create_deployment_record "vertex-ai" "pending" "started" "$deployment_details")
    log_info "Created deployment record: $deployment_file"
    
    # Create a training job and capture the output
    local job_output=$(gcloud ai custom-jobs create \
        --region=${REGION} \
        --display-name="${job_display_name}" \
        --config=- <<EOF
{
  "workerPoolSpecs": [
    {
      "machineSpec": {
        "machineType": "${MACHINE_TYPE}",
        "acceleratorType": "${ACCELERATOR_TYPE}",
        "acceleratorCount": ${ACCELERATOR_COUNT}
      },
      "replicaCount": 1,
      "containerSpec": {
        "imageUri": "${image_uri}",
        "command": [${json_cmd}],
        "env": [
          {
            "name": "STORAGE_TYPE",
            "value": "gcs"
          },
          {
            "name": "STORAGE_BUCKET_NAME", 
            "value": "${BUCKET_NAME}"
          },
          {
            "name": "CLOUD_STORAGE_PREFIX",
            "value": "${CLOUD_STORAGE_PREFIX:-magnesium-pipeline}"
          },
          {
            "name": "ENVIRONMENT",
            "value": "${ENVIRONMENT}"
          },
          {
            "name": "LOG_LEVEL",
            "value": "${CLOUD_LOG_LEVEL:-INFO}"
          },
          {
            "name": "GPU_MEMORY_FRACTION",
            "value": "${CLOUD_GPU_MEMORY_FRACTION:-0.8}"
          },
          {
            "name": "CLOUD_PROVIDER",
            "value": "gcp"
          },
          {
            "name": "GCP_PROJECT_ID",
            "value": "${PROJECT_ID}"
          },
          {
            "name": "GCP_REGION",
            "value": "${REGION}"
          },
          {
            "name": "PYTHONPATH",
            "value": "/app"
          }
        ]
      }
    }
  ],
  "scheduling": {
    "timeout": "${job_timeout}s"
  }
}
EOF
2>&1)
    
    # Extract job ID from the output - handle different formats
    local job_id=""
    
    # Try full path format first
    job_id=$(echo "$job_output" | grep -oP 'projects/[0-9]+/locations/[^/]+/customJobs/[0-9]+' | head -1)
    
    # If not found, try just the numeric job ID
    if [ -z "$job_id" ]; then
        local numeric_id=$(echo "$job_output" | grep -oP 'customJobs/[0-9]+' | head -1)
        if [ -n "$numeric_id" ]; then
            job_id="projects/${PROJECT_ID}/locations/${REGION}/${numeric_id}"
        fi
    fi
    
    # Try extracting from the CustomJob line
    if [ -z "$job_id" ]; then
        job_id=$(echo "$job_output" | grep -oP 'CustomJob \[\K[^\]]+' | head -1)
    fi
    
    # Debug: show what we're trying to match
    echo "[DEBUG] Searching for job ID in output:"
    echo "$job_output" | grep -i "customjob\|job"
    
    if [ -n "$job_id" ]; then
        log_success "Vertex AI training job created: $job_id"
        log_info "Display name: $job_display_name"
        log_info "Timeout: ${job_timeout}s"
        
        # Update deployment record with actual job ID
        local updated_details=$(cat <<DETAILS
{
  "job_id": "$job_id",
  "display_name": "$job_display_name",
  "image_uri": "${image_uri}",
  "machine_type": "${MACHINE_TYPE}",
  "accelerator_type": "${ACCELERATOR_TYPE}",
  "accelerator_count": ${ACCELERATOR_COUNT},
  "job_timeout": ${job_timeout},
  "training_command": "${cmd}"
}
DETAILS
)
        
        # Create final deployment record
        local final_deployment_file=$(create_deployment_record "vertex-ai" "$job_id" "submitted" "$updated_details")
        
        # Commit and tag the deployment
        commit_and_tag_deployment "vertex-ai" "$job_id" "$final_deployment_file"
        
        log_info "Monitor job progress with:"
        log_info "  gcloud ai custom-jobs describe $job_id --region=${REGION}"
        log_info "  gcloud ai custom-jobs stream-logs $job_id --region=${REGION}"
    else
        log_error "Failed to extract job ID from deployment output"
        log_info "Output was: $job_output"
    fi
}

cleanup() {
    log_info "Cleaning up resources..."
    
    read -p "This will delete all created resources. Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Delete Cloud Run service
        gcloud run services delete ${SERVICE_NAME} --region=${REGION} --quiet || true
        
        # Delete GKE cluster
        gcloud container clusters delete ${SERVICE_NAME}-cluster --region=${REGION} --quiet || true
        
        # Delete storage bucket (optional)
        read -p "Delete storage bucket gs://${BUCKET_NAME}? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            gsutil rm -r gs://${BUCKET_NAME} || true
        fi
        
        log_info "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

upload_logs() {
    log_info "Manually uploading logs from recent Vertex AI job..."
    
    # Get the most recent job
    local recent_job=$(gcloud ai custom-jobs list --region=${REGION} --format="value(name)" --limit=1 2>/dev/null)
    
    if [ -n "$recent_job" ]; then
        log_info "Found recent job: $recent_job"
        log_info "Fetching logs..."
        
        # Stream logs to a local file
        local log_file="vertex-ai-logs-$(date +%Y%m%d_%H%M%S).log"
        gcloud ai custom-jobs stream-logs "$recent_job" --region=${REGION} > "$log_file" 2>&1
        
        if [ -s "$log_file" ]; then
            log_success "Logs saved to: $log_file"
            
            # Upload to GCS if configured
            if [ "${STORAGE_TYPE:-gcs}" = "gcs" ] && [ -n "${BUCKET_NAME}" ]; then
                local bucket_prefix="${CLOUD_STORAGE_PREFIX:-magnesium-pipeline}"
                local timestamp=$(date +%Y%m%d_%H%M%S)
                local gcs_path="gs://${BUCKET_NAME}/${bucket_prefix}/manual-logs/${timestamp}/"
                
                log_info "Uploading logs to GCS: $gcs_path"
                gsutil cp "$log_file" "${gcs_path}vertex-ai-training.log"
                
                if [ $? -eq 0 ]; then
                    log_success "Logs uploaded to: ${gcs_path}vertex-ai-training.log"
                else
                    log_error "Failed to upload logs to GCS"
                fi
            else
                log_info "GCS not configured. Logs saved locally only."
            fi
        else
            log_warn "No logs found for job: $recent_job"
        fi
    else
        log_warn "No recent Vertex AI jobs found"
        log_info "To manually fetch logs from a specific job:"
        log_info "  gcloud ai custom-jobs stream-logs JOB_ID --region=${REGION}"
    fi
}

show_help() {
    echo "Google Cloud Platform Deployment Script for Magnesium Pipeline"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Infrastructure Commands:"
    echo "  setup       Setup GCP project and enable APIs"
    echo "  build       Build container image with Cloud Build (Kaniko caching)"
    echo "  build-local Build container image locally with Docker (faster)"
    echo "  cloud-run   Deploy to Cloud Run (serverless)"
    echo "  inference   Deploy inference service to Cloud Run (loads models from GCS)"
    echo "  gke         Deploy to Google Kubernetes Engine"
    echo "  cleanup     Clean up all resources"
    echo "  all         Run setup, build, and cloud-run"
    echo
    echo "Training Commands (Vertex AI):"
    echo "  vertex-ai   Submit training job to Vertex AI (uses TRAINING_MODE)"
    echo "  train       Train standard models on Vertex AI"
    echo "  tune        Hyperparameter tuning on Vertex AI"
    echo "  autogluon   AutoGluon training on Vertex AI"
    echo "  optimize    Multi-model optimization on Vertex AI"
    echo
    echo "Configuration Commands:"
    echo "  config-help         Show detailed cloud configuration help"
    echo "  generate-config ENV Generate cloud config template for environment"
    echo "  list-projects       List available GCP projects"
    echo "  test-config         Test cloud configuration loading"
    echo
    echo "Data Management Commands:"
    echo "  upload-data         Upload training data to GCS (skip if exists)"
    echo "  upload-data --force Force re-upload training data to GCS"
    echo "  check-data          Check if training data exists in GCS"
    echo "  list-data           List training data files in GCS bucket"
    echo "  upload-logs         Manually fetch and upload logs from recent Vertex AI job"
    echo
    echo "Infrastructure Environment Variables:"
    echo "  PROJECT_ID   GCP Project ID (default: from cloud_config.yml or 586323113169)"
    echo "  REGION       GCP Region (default: from cloud_config.yml or us-central1)"
    echo "  SERVICE_NAME Service name (default: magnesium-pipeline)"
    echo "  BUCKET_NAME  Storage bucket name (default: from cloud_config.yml or PROJECT_ID-magnesium-data)"
    echo
    echo "Cloud Configuration Variables:"
    echo "  CLOUD_CONFIG_FILE    Path to cloud config file (default: config/cloud_config.yml)"
    echo "  ENVIRONMENT          Environment setting (default: production)"
    echo "  USE_CLOUD_CONFIG     Whether to use cloud config (default: true)"
    echo
    echo "Training Environment Variables:"
    echo "  TRAINING_MODE     Training mode (default: autogluon)"
    echo "                    Options: train, autogluon, tune, optimize-models,"
    echo "                             optimize-xgboost, optimize-autogluon"
    echo "  USE_GPU           Enable GPU acceleration (default: true)"
    echo "  USE_RAW_SPECTRAL  Use raw spectral data (default: false)"
    echo "  MODELS            Specific models to train (comma-separated)"
    echo "                    Options: ridge,lasso,random_forest,xgboost,lightgbm,"
    echo "                             catboost,extratrees,neural_network,neural_network_light,svr"
    echo "  STRATEGY          Feature strategy (default: full_context)"
    echo "                    Options: full_context, simple_only, Mg_only"
    echo "  TRIALS            Number of optimization trials"
    echo "  TIMEOUT           Timeout in seconds for optimization"
    echo "  MACHINE_TYPE      GCP machine type (default: n1-standard-4)"
    echo "  ACCELERATOR_TYPE  GPU type (default: NVIDIA_TESLA_T4)"
    echo "  ACCELERATOR_COUNT Number of GPUs (default: 1)"
    echo
    echo "Infrastructure Examples:"
    echo "  PROJECT_ID=my-project ./deploy/gcp_deploy.sh all"
    echo "  REGION=europe-west1 ./deploy/gcp_deploy.sh cloud-run"
    echo
    echo "Training Examples:"
    echo "  # Train specific models with GPU"
    echo "  MODELS=xgboost,lightgbm,catboost ./deploy/gcp_deploy.sh train"
    echo
    echo "  # AutoGluon training"
    echo "  ./deploy/gcp_deploy.sh autogluon"
    echo
    echo "  # Multi-model optimization with custom parameters"
    echo "  MODELS=xgboost,lightgbm STRATEGY=simple_only TRIALS=200 ./deploy/gcp_deploy.sh optimize"
    echo
    echo "  # Hyperparameter tuning with raw spectral data"
    echo "  USE_RAW_SPECTRAL=true MODELS=neural_network ./deploy/gcp_deploy.sh tune"
    echo
    echo "  # High-resource optimization"
    echo "  MACHINE_TYPE=n1-standard-8 ACCELERATOR_COUNT=2 TRIALS=500 ./deploy/gcp_deploy.sh optimize"
    echo
    echo "Quick Start Training Examples:"
    echo "  # Basic AutoGluon training (recommended for most users)"
    echo "  ./deploy/gcp_deploy.sh autogluon"
    echo
    echo "  # Train specific models quickly"
    echo "  MODELS=xgboost,lightgbm ./deploy/gcp_deploy.sh train"
    echo
    echo "  # Optimize XGBoost specifically"
    echo "  TRAINING_MODE=optimize-xgboost TRIALS=300 ./deploy/gcp_deploy.sh vertex-ai"
    echo
    echo "  # Full pipeline: setup + train"
    echo "  ./deploy/gcp_deploy.sh setup && ./deploy/gcp_deploy.sh build && ./deploy/gcp_deploy.sh autogluon"
    echo
    echo "Cloud Configuration Examples:"
    echo "  # Use default cloud_config.yml"
    echo "  ./deploy/gcp_deploy.sh autogluon"
    echo
    echo "  # Use custom cloud config file"
    echo "  CLOUD_CONFIG_FILE=config/staging_config.yml ./deploy/gcp_deploy.sh train"
    echo
    echo "  # Override cloud config with environment variables"
    echo "  PROJECT_ID_OVERRIDE=true PROJECT_ID=my-custom-project ./deploy/gcp_deploy.sh autogluon"
    echo
    echo "  # Use staging environment"
    echo "  ENVIRONMENT=staging ./deploy/gcp_deploy.sh optimize"
    echo
    echo "  # Disable cloud config and use only environment variables"
    echo "  USE_CLOUD_CONFIG=false PROJECT_ID=my-project REGION=europe-west1 ./deploy/gcp_deploy.sh train"
    echo
    echo "Cloud Config File Structure (config/cloud_config.yml):"
    echo "  cloud_providers:"
    echo "    gcp:"
    echo "      project_id: \"your-gcp-project-id\""
    echo "      region: \"us-central1\""
    echo "      storage_class: \"STANDARD\""
    echo "  storage:"
    echo "    bucket_name: \"your-bucket-name\""
    echo "    prefix: \"magnesium-pipeline\""
    echo "  pipeline:"
    echo "    default_strategy: \"simple_only\""
    echo "    enable_gpu: true"
}

show_config_help() {
    echo "Cloud Configuration Help for Magnesium Pipeline"
    echo "================================================="
    echo
    echo "The cloud_config.yml file provides environment-agnostic configuration"
    echo "that can be used across different cloud providers and environments."
    echo
    echo "Key Benefits:"
    echo "  - Centralized configuration management"
    echo "  - Environment-specific settings (dev, staging, prod)"
    echo "  - Cloud provider abstraction"
    echo "  - Consistent deployment parameters"
    echo
    echo "Configuration Priority (highest to lowest):"
    echo "  1. Environment variables with _OVERRIDE suffix"
    echo "  2. Direct environment variables"  
    echo "  3. Cloud configuration file values"
    echo "  4. Script defaults"
    echo
    echo "Usage Examples:"
    echo "  # Use default config"
    echo "  ./deploy/gcp_deploy.sh autogluon"
    echo
    echo "  # Use staging config"
    echo "  CLOUD_CONFIG_FILE=config/staging_config.yml ./deploy/gcp_deploy.sh train"
    echo
    echo "  # Override specific values"
    echo "  PROJECT_ID_OVERRIDE=true PROJECT_ID=my-project ./deploy/gcp_deploy.sh optimize"
    echo
    echo "  # Generate new config template"
    echo "  ./deploy/gcp_deploy.sh generate-config production"
    echo
    echo "Environment Variables that can be overridden:"
    echo "  PROJECT_ID, REGION, BUCKET_NAME, STRATEGY, USE_GPU"
    echo "  Add '_OVERRIDE=true' to force override cloud config values"
    echo
    echo "For more examples, run: ./deploy/gcp_deploy.sh help"
}

generate_cloud_config() {
    local env_name="${1:-development}"
    local config_file="config/${env_name}_config.yml"
    
    log_info "Generating cloud configuration template for environment: $env_name"
    
    # Create config directory if it doesn't exist
    mkdir -p "$(dirname "$config_file")"
    
    cat > "$config_file" << EOF
# ${env_name^} Environment Configuration for Magnesium Pipeline
# Generated on $(date)

# General Settings
app:
  name: "magnesium-pipeline-${env_name}"
  version: "1.0.0"
  environment: "${env_name}"
  debug: $([ "$env_name" = "production" ] && echo "false" || echo "true")

# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  workers: $([ "$env_name" = "production" ] && echo "4" || echo "1")
  timeout: $([ "$env_name" = "production" ] && echo "300" || echo "600")
  max_file_size: $([ "$env_name" = "production" ] && echo "100MB" || echo "500MB")
  cors_origins: ["*"]

# Storage Configuration
storage:
  type: "gcs"
  data_path: "/app/data"
  models_path: "/app/models"
  reports_path: "/app/reports"
  logs_path: "/app/logs"
  bucket_name: "${env_name}-magnesium-data"
  credentials_path: null
  prefix: "magnesium-pipeline-${env_name}"

# Compute Configuration
compute:
  gpu_enabled: true
  gpu_memory_fraction: $([ "$env_name" = "production" ] && echo "0.8" || echo "0.9")
  cpu_cores: null
  memory_limit: null

# Pipeline Configuration
pipeline:
  default_strategy: $([ "$env_name" = "production" ] && echo "\"full_context\"" || echo "\"simple_only\"")
  time_limit: $([ "$env_name" = "production" ] && echo "3600" || echo "1800")
  enable_gpu: true
  enable_sample_weights: true

# Monitoring & Logging
monitoring:
  log_level: $([ "$env_name" = "production" ] && echo "\"INFO\"" || echo "\"DEBUG\"")
  enable_metrics: true
  metrics_port: 9090
  health_check_interval: 30

# Security
security:
  api_key_required: $([ "$env_name" = "production" ] && echo "true" || echo "false")
  rate_limiting:
    enabled: $([ "$env_name" = "production" ] && echo "true" || echo "false")
    requests_per_minute: 60

# Cloud Provider Specific Overrides
cloud_providers:
  gcp:
    project_id: "YOUR_GCP_PROJECT_ID"  # TODO: Update this
    region: "us-central1"
    storage_class: "STANDARD"
    compute_zone: "us-central1-a"
    
  aws:
    region: "us-east-1"
    storage_class: $([ "$env_name" = "production" ] && echo "\"STANDARD\"" || echo "\"STANDARD_IA\"")
    instance_type: "ml.m5.large"
    
  azure:
    resource_group: "magnesium-${env_name}-rg"
    location: "East US"
    storage_tier: "Standard"
EOF

    log_info "Generated configuration file: $config_file"
    log_warn "Remember to update YOUR_GCP_PROJECT_ID in the generated file!"
    echo
    echo "To use this configuration:"
    echo "  CLOUD_CONFIG_FILE=$config_file ./deploy/gcp_deploy.sh autogluon"
}

list_gcp_projects() {
    log_info "Listing available GCP projects..."
    
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if logged in
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_error "Not logged into gcloud. Run: gcloud auth login"
        exit 1
    fi
    
    echo
    echo "Available GCP Projects:"
    echo "======================="
    gcloud projects list --format="table(projectId:label='PROJECT_ID',name:label='PROJECT_NAME',projectNumber:label='PROJECT_NUMBER')" 2>/dev/null
    
    echo
    echo "To use a project, set the PROJECT_ID environment variable:"
    echo "  export PROJECT_ID=your-chosen-project-id"
    echo "  ./deploy/gcp_deploy.sh setup"
    echo
    echo "Or set it inline:"
    echo "  PROJECT_ID=your-chosen-project-id ./deploy/gcp_deploy.sh setup"
}

# Main script logic
case "${1:-help}" in
    setup)
        setup_project
        ;;
    build)
        check_basic_prerequisites
        load_cloud_config
        check_project_configuration
        build_image
        ;;
    build-local)
        check_basic_prerequisites
        load_cloud_config
        check_project_configuration
        build_image_local
        ;;
    cloud-run)
        check_basic_prerequisites
        load_cloud_config
        check_project_configuration
        deploy_cloud_run
        ;;
    inference)
        check_basic_prerequisites
        load_cloud_config
        check_project_configuration
        deploy_inference_service
        ;;
    gke)
        check_basic_prerequisites
        load_cloud_config
        check_project_configuration
        deploy_gke
        ;;
    vertex-ai)
        check_basic_prerequisites
        deploy_vertex_ai
        ;;
    # Training command aliases
    train)
        export TRAINING_MODE="train"
        check_basic_prerequisites
        deploy_vertex_ai
        ;;
    tune)
        export TRAINING_MODE="tune"
        check_basic_prerequisites
        deploy_vertex_ai
        ;;
    autogluon)
        export TRAINING_MODE="autogluon"
        check_basic_prerequisites
        deploy_vertex_ai
        ;;
    optimize)
        export TRAINING_MODE="optimize-models"
        check_basic_prerequisites
        deploy_vertex_ai
        ;;
    # Configuration management commands
    config-help)
        show_config_help
        ;;
    generate-config)
        generate_cloud_config "$2"
        ;;
    list-projects)
        list_gcp_projects
        ;;
    test-config)
        check_basic_prerequisites
        log_info "Testing configuration loading..."
        log_info "CLOUD_CONFIG_FILE: $CLOUD_CONFIG_FILE"
        load_cloud_config
        check_project_configuration
        log_info "Configuration test completed successfully!"
        ;;
    cleanup)
        cleanup
        ;;
    all)
        setup_project
        build_image
        deploy_cloud_run
        ;;
    # Data management commands
    upload-data)
        check_basic_prerequisites
        load_cloud_config
        if [[ "$2" == "--force" ]]; then
            force_upload_training_data
        else
            upload_training_data
        fi
        ;;
    check-data)
        check_basic_prerequisites
        load_cloud_config
        check_data_exists_in_gcs
        ;;
    list-data)
        check_basic_prerequisites
        load_cloud_config
        list_and_verify_gcs_data
        ;;
    upload-logs)
        check_basic_prerequisites
        load_cloud_config
        upload_logs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac

log_info "Script completed successfully!"