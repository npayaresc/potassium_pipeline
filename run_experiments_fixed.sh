#!/bin/bash

# Fixed Comprehensive Experiment Script
# Uses gcp_deploy.sh for prerequisite checks but handles Vertex AI deployment directly

set -e

# Colors and logging functions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Results tracking
RESULTS_DIR="./experiment_results"
LOGS_DIR="./experiment_logs"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOGS_DIR/main.log"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOGS_DIR/main.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOGS_DIR/main.log"
}

# Load configuration from staging config
STAGING_CONFIG="/home/payanico/magnesium_pipeline/config/staging_config.yml"

# Parse YAML configuration (simple parser for this config structure)
parse_staging_config() {
    if [ ! -f "$STAGING_CONFIG" ]; then
        log_error "Staging config not found: $STAGING_CONFIG"
        exit 1
    fi
    
    log_info "Loading configuration from: $STAGING_CONFIG"
    
    # Parse the staging config YAML file - specifically from GCP section
    PROJECT_ID=$(grep -A 10 "gcp:" "$STAGING_CONFIG" | grep "project_id:" | head -1 | sed 's/.*project_id: *"\?//' | sed 's/"\? *#.*//' | sed 's/"\?$//')
    REGION=$(grep -A 10 "gcp:" "$STAGING_CONFIG" | grep "region:" | head -1 | sed 's/.*region: *"\?//' | sed 's/"\? *#.*//' | sed 's/"\?$//')
    BUCKET_NAME=$(grep "bucket_name:" "$STAGING_CONFIG" | sed 's/.*bucket_name: *"\?//' | sed 's/"\? *#.*//' | sed 's/"\?$//')
    CLOUD_STORAGE_PREFIX=$(grep "prefix:" "$STAGING_CONFIG" | sed 's/.*prefix: *"\?//' | sed 's/"\? *#.*//' | sed 's/"\?$//')
    
    # Use defaults if not found in config
    PROJECT_ID="${PROJECT_ID:-mapana-ai-models}"
    REGION="${REGION:-us-central1}"
    BUCKET_NAME="${BUCKET_NAME:-staging-magnesium-data}"
    CLOUD_STORAGE_PREFIX="${CLOUD_STORAGE_PREFIX:-magnesium-pipeline-staging}"
    
    log_info "Configuration loaded:"
    log_info "  PROJECT_ID: $PROJECT_ID"
    log_info "  REGION: $REGION"
    log_info "  BUCKET_NAME: $BUCKET_NAME"
    log_info "  CLOUD_STORAGE_PREFIX: $CLOUD_STORAGE_PREFIX"
}

# Parse configuration first
parse_staging_config

# Configuration (loaded from staging config)
REPO_NAME="${REPO_NAME:-magnesium-repo}"
IMAGE_NAME="${IMAGE_NAME:-magnesium-pipeline}"
SERVICE_NAME="${SERVICE_NAME:-magnesium-pipeline}"

generate_experiment_id() {
    echo "exp_$(date +%Y%m%d_%H%M%S)_$$"
}

# Submit Vertex AI job
submit_vertex_job() {
    local exp_id="$1"
    local training_mode="$2"
    local strategy="$3"
    local models="$4"
    local use_gpu="$5"
    local trials="$6"
    local machine_type="$7"
    
    log_info "Submitting experiment: $exp_id"
    log_info "  Mode: $training_mode | Strategy: $strategy | GPU: $use_gpu"
    
    # Build Python command
    local python_cmd="python main.py $training_mode"
    
    if [ -n "$models" ] && [ "$models" != "null" ]; then
        python_cmd="$python_cmd --models $models"
    fi
    
    if [ -n "$strategy" ] && [ "$strategy" != "null" ]; then
        python_cmd="$python_cmd --strategy $strategy"
    fi
    
    if [ "$use_gpu" = "true" ]; then
        python_cmd="$python_cmd --gpu"
    fi
    
    if [ -n "$trials" ] && [ "$trials" != "null" ] && [[ "$training_mode" == *"optimize"* || "$training_mode" == "tune" ]]; then
        python_cmd="$python_cmd --trials $trials"
    fi
    
    # Create unique job name
    local job_name="${SERVICE_NAME}-${exp_id}"
    
    # Container image
    local image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
    
    # Set timeout based on job type
    local timeout="7200"  # 2 hours default
    if [ "$training_mode" = "autogluon" ]; then
        timeout="21600"  # 6 hours for AutoGluon
    elif [[ "$training_mode" == *"optimize"* ]]; then
        timeout="14400"  # 4 hours for optimization
    fi
    
    # Create temporary job config
    local config_file="/tmp/vertex-job-${exp_id}.json"
    
    cat > "$config_file" << EOF
{
  "workerPoolSpecs": [
    {
      "machineSpec": {
        "machineType": "$machine_type"$([ "$use_gpu" = "true" ] && echo ",
        \"acceleratorType\": \"NVIDIA_TESLA_T4\",
        \"acceleratorCount\": 1")
      },
      "replicaCount": 1,
      "containerSpec": {
        "imageUri": "$image_uri",
        "command": ["bash", "-c"],
        "args": ["source /app/docker-entrypoint.sh && download_training_data && $python_cmd"],
        "env": [
          {"name": "STORAGE_TYPE", "value": "gcs"},
          {"name": "STORAGE_BUCKET_NAME", "value": "$BUCKET_NAME"},
          {"name": "CLOUD_STORAGE_PREFIX", "value": "$CLOUD_STORAGE_PREFIX"},
          {"name": "ENVIRONMENT", "value": "staging"},
          {"name": "PYTHONPATH", "value": "/app"}
        ]
      }
    }
  ],
  "scheduling": {
    "timeout": "${timeout}s"
  }
}
EOF
    
    # Submit job (with debug output)
    log_info "Job config file: $config_file"
    log_info "Command: gcloud ai custom-jobs create --display-name=$job_name --region=$REGION --config=$config_file"
    
    local job_id
    if job_id=$(gcloud ai custom-jobs create \
        --display-name="$job_name" \
        --region="$REGION" \
        --config="$config_file" \
        --format="value(name)" 2>&1); then
        
        log_info "✓ Job submitted: $job_name"
        echo "$exp_id,$(date),$job_id,SUCCESS,$training_mode,$strategy,$models" >> "$RESULTS_DIR/tracker.csv"
        rm -f "$config_file"
        return 0
    else
        log_error "✗ Failed to submit: $job_name"
        log_error "Error output: $job_id"
        echo "$exp_id,$(date),,FAILED,$training_mode,$strategy,$models" >> "$RESULTS_DIR/tracker.csv"
        # Don't delete config file for debugging
        log_error "Config file saved for debugging: $config_file"
        return 1
    fi
}

# Check prerequisites using gcp_deploy.sh
check_prerequisites() {
    log_info "Checking prerequisites using gcp_deploy.sh..."
    
    local deploy_script="./deploy/gcp_deploy.sh"
    
    if [ ! -f "$deploy_script" ]; then
        log_error "gcp_deploy.sh not found at $deploy_script"
        exit 1
    fi
    
    log_info "=== Running Prerequisites Checks ==="
    
    # 1. Basic prerequisites check (gcloud, docker, auth) - do this directly
    log_info "Checking basic prerequisites..."
    
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_error "Not logged into gcloud. Run: gcloud auth login"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    log_info "Basic prerequisites check passed!"
    
    # 2. Set the project
    log_info "Setting GCP project to: $PROJECT_ID"
    gcloud config set project "$PROJECT_ID" --quiet
    
    # 3. Check if container image exists
    local image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
    log_info "Checking if container image exists: ${image_uri}"
    
    if gcloud artifacts docker images describe "${image_uri}" --quiet &>/dev/null; then
        log_info "Container image found: ${image_uri}"
    else
        log_error "Container image not found: ${image_uri}"
        log_error "Please build and push the Docker image first using:"
        log_error "  ./deploy/gcp_deploy.sh build-image"
        exit 1
    fi
    
    # 4. Check if training data exists in GCS
    log_info "Checking if training data exists in GCS bucket..."
    local bucket_prefix="${CLOUD_STORAGE_PREFIX:-magnesium-pipeline-staging}"
    local required_path="gs://${BUCKET_NAME}/${bucket_prefix}/data_5278_Phase3/"
    
    if gsutil ls "${required_path}" &>/dev/null; then
        log_info "Training data found: ${required_path}"
    else
        log_error "Training data not found: ${required_path}"
        log_error "Please upload training data first using:"
        log_error "  ./deploy/gcp_deploy.sh upload-data"
        exit 1
    fi
    
    log_info "=== All Prerequisites Passed! ==="
}

# Setup environment after prerequisites check
setup_environment() {
    log_info "Setting up experiment environment..."
    
    # Initialize tracker
    if [ ! -f "$RESULTS_DIR/tracker.csv" ]; then
        echo "experiment,timestamp,job_id,status,mode,strategy,models" > "$RESULTS_DIR/tracker.csv"
    fi
    
    # Enable APIs
    gcloud services enable aiplatform.googleapis.com --quiet 2>/dev/null || true
    gcloud services enable artifactregistry.googleapis.com --quiet 2>/dev/null || true
    
    log_info "Environment setup complete"
}

# ============================================================================
# QUICK HIGH-IMPACT EXPERIMENTS
# ============================================================================

run_quick_experiments() {
    log_info "Running Quick High-Impact Experiments"
    
    # Experiment 1: Best AutoGluon (CPU-only)
    exp_id=$(generate_experiment_id)
    submit_vertex_job "$exp_id" "autogluon" "simple_only" "" "false" "" "n1-highmem-8"
    sleep 10
    
    # Experiment 2: XGBoost Optimization
    exp_id=$(generate_experiment_id)
    submit_vertex_job "$exp_id" "optimize-xgboost" "full_context" "" "true" "300" "n1-highmem-8"
    sleep 10
    
    # Experiment 3: Multi-model training
    exp_id=$(generate_experiment_id)
    submit_vertex_job "$exp_id" "train" "simple_only" "xgboost,lightgbm,catboost" "true" "" "n1-highmem-4"
    sleep 10
    
    # Experiment 4: Neural Network
    exp_id=$(generate_experiment_id)
    submit_vertex_job "$exp_id" "train" "full_context" "neural_network" "true" "" "n1-highmem-8"
    sleep 10
    
    # Experiment 5: Hyperparameter tuning
    exp_id=$(generate_experiment_id)
    submit_vertex_job "$exp_id" "tune" "simple_only" "" "true" "200" "n1-highmem-8"
    
    log_info "Quick experiments submitted!"
}

# ============================================================================
# FEATURE STRATEGY EXPERIMENTS
# ============================================================================

run_feature_experiments() {
    log_info "Running Feature Strategy Experiments"
    
    local strategies=("Mg_only" "simple_only" "full_context")
    local models="xgboost,lightgbm"
    
    for strategy in "${strategies[@]}"; do
        log_info "Testing strategy: $strategy"
        
        # With standard models
        exp_id=$(generate_experiment_id)
        submit_vertex_job "$exp_id" "train" "$strategy" "$models" "true" "" "n1-standard-8"
        sleep 5
        
        # With AutoGluon (CPU-only)
        exp_id=$(generate_experiment_id)
        submit_vertex_job "$exp_id" "autogluon" "$strategy" "" "false" "" "n1-highmem-8"
        sleep 10
    done
    
    log_info "Feature experiments submitted!"
}

# ============================================================================
# OPTIMIZATION EXPERIMENTS
# ============================================================================

run_optimization_experiments() {
    log_info "Running Optimization Experiments"
    
    local strategies=("simple_only" "full_context")
    
    for strategy in "${strategies[@]}"; do
        # XGBoost optimization
        exp_id=$(generate_experiment_id)
        submit_vertex_job "$exp_id" "optimize-xgboost" "$strategy" "" "true" "400" "n1-highmem-8"
        sleep 10
        
        # Multi-model optimization
        exp_id=$(generate_experiment_id)
        submit_vertex_job "$exp_id" "optimize-models" "$strategy" "xgboost,lightgbm,catboost" "true" "300" "n1-highmem-8"
        sleep 10
    done
    
    log_info "Optimization experiments submitted!"
}

# ============================================================================
# AUTOGLUON EXPERIMENTS (CPU-ONLY)
# ============================================================================

run_autogluon_experiments() {
    log_info "Running AutoGluon Experiments (CPU-only)"
    
    local strategies=("simple_only" "full_context")
    local machine_types=("n1-highmem-8" "n1-highmem-16")
    
    for strategy in "${strategies[@]}"; do
        for machine in "${machine_types[@]}"; do
            exp_id=$(generate_experiment_id)
            submit_vertex_job "$exp_id" "autogluon" "$strategy" "" "false" "" "$machine"
            sleep 15
        done
    done
    
    log_info "AutoGluon experiments submitted!"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_info "Starting Fixed Experiment Runner"
    log_info "Project: $PROJECT_ID | Region: $REGION"
    
    # Check all prerequisites first
    check_prerequisites
    
    # Setup experiment environment
    setup_environment
    
    case "${1:-quick}" in
        quick)
            run_quick_experiments
            ;;
        features)
            run_feature_experiments
            ;;
        optimize)
            run_optimization_experiments
            ;;
        autogluon)
            run_autogluon_experiments
            ;;
        all)
            log_info "Running all experiment types..."
            run_quick_experiments
            sleep 30
            run_feature_experiments
            sleep 30
            run_optimization_experiments
            sleep 30
            run_autogluon_experiments
            ;;
        status)
            log_info "Checking job status..."
            gcloud ai custom-jobs list --region="$REGION" --format="table(displayName,state,createTime)"
            ;;
        results)
            log_info "Downloading results..."
            mkdir -p ./cloud_results
            gsutil -m cp -r "gs://$BUCKET_NAME/magnesium-pipeline/reports/*" ./cloud_results/ 2>/dev/null || true
            log_info "Results downloaded to ./cloud_results/"
            ;;
        help)
            echo "Usage: $0 [quick|features|optimize|autogluon|all|status|results|help]"
            echo ""
            echo "Commands:"
            echo "  quick      - 5 high-impact experiments"
            echo "  features   - Feature strategy comparison"
            echo "  optimize   - Model optimization experiments"
            echo "  autogluon  - AutoGluon experiments (CPU-only)"
            echo "  all        - All experiment types"
            echo "  status     - Check job status"
            echo "  results    - Download results"
            echo "  help       - Show this help"
            ;;
        *)
            log_error "Unknown command: $1"
            echo "Use '$0 help' for usage"
            exit 1
            ;;
    esac
    
    log_info "Experiment submission completed!"
    log_info "Monitor with: $0 status"
    log_info "Get results: $0 results"
    
    # Show current job count
    local running_jobs=$(gcloud ai custom-jobs list --region="$REGION" --filter="state=JOB_STATE_RUNNING" --format="value(name)" | wc -l)
    log_info "Current running jobs: $running_jobs"
}

# Run main
main "$@"