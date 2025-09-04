#!/bin/bash

# Cloud Experiment Runner
# Simplified version that calls main.py directly through Vertex AI

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-mapana-ai-models}"
REGION="${REGION:-us-central1}"
BUCKET_NAME="${BUCKET_NAME:-${PROJECT_ID}-magnesium-data}"
REPO_NAME="${REPO_NAME:-magnesium-repo}"
IMAGE_NAME="${IMAGE_NAME:-magnesium-pipeline}"
SERVICE_NAME="${SERVICE_NAME:-magnesium-pipeline}"

# Results tracking
RESULTS_DIR="./experiment_results"
LOGS_DIR="./experiment_logs"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOGS_DIR/cloud_runner.log"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOGS_DIR/cloud_runner.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOGS_DIR/cloud_runner.log"
}

# Function to submit Vertex AI job
submit_vertex_job() {
    local exp_name=$1
    local training_mode=$2
    local strategy=$3
    local models=$4
    local use_gpu=$5
    local trials=${6:-100}
    local machine_type=${7:-"n1-standard-4"}
    local accelerator_type=${8:-"NVIDIA_TESLA_T4"}
    local accelerator_count=${9:-1}
    
    log_info "Submitting Vertex AI job: $exp_name"
    log_info "  Mode: $training_mode | Strategy: $strategy | Models: $models"
    log_info "  Machine: $machine_type | GPU: $use_gpu"
    
    # Build Python command
    local python_cmd="python main.py $training_mode"
    
    if [ -n "$models" ]; then
        # Convert comma-separated models to space-separated for argparse
        MODELS_SPACE_SEPARATED="${models//,/ }"
        python_cmd="$python_cmd --models $MODELS_SPACE_SEPARATED"
    fi
    
    if [ -n "$strategy" ]; then
        python_cmd="$python_cmd --strategy $strategy"
    fi
    
    if [ "$use_gpu" = "true" ]; then
        python_cmd="$python_cmd --gpu"
    fi
    
    if [ -n "$trials" ] && [[ "$training_mode" == *"optimize"* || "$training_mode" == "tune" ]]; then
        python_cmd="$python_cmd --trials $trials"
    fi
    
    # Create unique job name
    local job_name="${SERVICE_NAME}-${exp_name}-$(date +%Y%m%d-%H%M%S)"
    
    # Container image
    local image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
    
    # Create temporary job config
    local config_file="/tmp/vertex-job-${exp_name}.json"
    
    # Set timeout based on job type
    local timeout="7200"  # 2 hours default
    if [ "$training_mode" = "autogluon" ]; then
        timeout="21600"  # 6 hours for AutoGluon
    elif [[ "$training_mode" == *"optimize"* ]]; then
        timeout="14400"  # 4 hours for optimization
    fi
    
    cat > "$config_file" << EOF
{
  "displayName": "$job_name",
  "jobSpec": {
    "workerPoolSpecs": [
      {
        "machineSpec": {
          "machineType": "$machine_type"$([ "$use_gpu" = "true" ] && echo ",
          \"acceleratorType\": \"$accelerator_type\",
          \"acceleratorCount\": $accelerator_count")
        },
        "replicaCount": 1,
        "containerSpec": {
          "imageUri": "$image_uri",
          "command": ["bash", "-c"],
          "args": ["source /app/docker-entrypoint.sh && download_training_data && $python_cmd"],
          "env": [
            {"name": "STORAGE_TYPE", "value": "gcs"},
            {"name": "STORAGE_BUCKET_NAME", "value": "$BUCKET_NAME"},
            {"name": "CLOUD_STORAGE_PREFIX", "value": "magnesium-pipeline"},
            {"name": "ENVIRONMENT", "value": "production"},
            {"name": "PYTHONPATH", "value": "/app"}
          ]
        }
      }
    ],
    "scheduling": {
      "timeout": "${timeout}s"
    }
  }
}
EOF
    
    # Submit job
    log_info "Submitting job with timeout: ${timeout}s"
    
    local job_id
    if job_id=$(gcloud ai custom-jobs create \
        --region="$REGION" \
        --config="$config_file" \
        --format="value(name)" 2>/dev/null); then
        
        log_info "✓ Job submitted successfully: $job_name"
        log_info "  Job ID: $job_id"
        echo "$exp_name,$(date),$job_id,SUBMITTED,$training_mode,$strategy,$models" >> "$RESULTS_DIR/cloud_tracker.csv"
        
        # Clean up
        rm -f "$config_file"
        return 0
    else
        log_error "✗ Failed to submit job: $job_name"
        echo "$exp_name,$(date),,FAILED,$training_mode,$strategy,$models" >> "$RESULTS_DIR/cloud_tracker.csv"
        rm -f "$config_file"
        return 1
    fi
}

# Check prerequisites
check_prerequisites() {
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Please install Google Cloud SDK."
        exit 1
    fi
    
    # Check if logged in
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        log_error "Not logged into gcloud. Run: gcloud auth login"
        exit 1
    fi
    
    # Set project
    gcloud config set project "$PROJECT_ID"
}

# Setup GCP resources
setup_gcp() {
    log_info "Setting up GCP resources..."
    
    # Enable APIs
    gcloud services enable aiplatform.googleapis.com --quiet
    gcloud services enable artifactregistry.googleapis.com --quiet
    gcloud services enable storage-api.googleapis.com --quiet
    
    # Create bucket if it doesn't exist
    if ! gsutil ls "gs://$BUCKET_NAME" &> /dev/null; then
        log_info "Creating GCS bucket: $BUCKET_NAME"
        gsutil mb -l "$REGION" "gs://$BUCKET_NAME"
    fi
    
    # Check if container image exists
    local image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
    if ! gcloud artifacts docker images describe "$image_uri" --quiet &> /dev/null; then
        log_warn "Container image not found. Please build it first:"
        log_warn "  ./deploy/gcp_deploy.sh build"
        return 1
    fi
    
    log_info "GCP setup completed"
    return 0
}

# Helper function for external scripts
submit_single_job() {
    local exp_name="$1"
    local training_mode="$2"
    local strategy="$3"
    local models="$4"
    local use_gpu="$5"
    local trials="$6"
    local machine_type="$7"
    local accelerator_type="$8"
    local accelerator_count="$9"
    
    # Set defaults
    trials=${trials:-100}
    machine_type=${machine_type:-"n1-standard-4"}
    accelerator_type=${accelerator_type:-"NVIDIA_TESLA_T4"}
    accelerator_count=${accelerator_count:-1}
    
    submit_vertex_job "$exp_name" "$training_mode" "$strategy" "$models" "$use_gpu" "$trials" "$machine_type" "$accelerator_type" "$accelerator_count"
}

# Main execution
main() {
    log_info "Starting Cloud Experiments"
    log_info "Project: $PROJECT_ID | Region: $REGION"
    
    # Initialize tracker
    if [ ! -f "$RESULTS_DIR/cloud_tracker.csv" ]; then
        echo "experiment,timestamp,job_id,status,mode,strategy,models" > "$RESULTS_DIR/cloud_tracker.csv"
    fi
    
    check_prerequisites
    setup_gcp || exit 1
    
    # Handle special case for external script calls
    if [ "$1" = "submit_job" ]; then
        shift
        check_prerequisites
        setup_gcp || exit 1
        submit_single_job "$@"
        return $?
    fi
    
    case "${1:-quick}" in
        quick)
            log_info "Running quick high-impact experiments..."
            
            # Experiment 1: Best AutoGluon (CPU only)
            submit_vertex_job "ag-best" "autogluon" "simple_only" "" "false" "" "n1-highmem-8" "" "0"
            sleep 5
            
            # Experiment 2: XGBoost Optimization
            submit_vertex_job "xgb-opt" "optimize-xgboost" "full_context" "" "true" "300" "n1-highmem-8" "NVIDIA_TESLA_T4" "1"
            sleep 5
            
            # Experiment 3: Multi-model training
            submit_vertex_job "multi-train" "train" "simple_only" "xgboost,lightgbm,catboost" "true" "" "n1-highmem-4" "NVIDIA_TESLA_T4" "1"
            ;;
            
        feature-test)
            log_info "Testing different feature strategies..."
            
            submit_vertex_job "feat-mg" "train" "Mg_only" "xgboost,lightgbm" "true" "" "n1-standard-4" "" ""
            sleep 5
            submit_vertex_job "feat-simple" "train" "simple_only" "xgboost,lightgbm" "true" "" "n1-standard-4" "" ""
            sleep 5
            submit_vertex_job "feat-full" "train" "full_context" "xgboost,lightgbm" "true" "" "n1-highmem-4" "NVIDIA_TESLA_T4" "1"
            ;;
            
        optimize)
            log_info "Running optimization experiments..."
            
            # XGBoost optimization with different strategies
            submit_vertex_job "xgb-simple" "optimize-xgboost" "simple_only" "" "true" "400" "n1-highmem-8" "NVIDIA_TESLA_T4" "1"
            sleep 5
            submit_vertex_job "xgb-full" "optimize-xgboost" "full_context" "" "true" "400" "n1-highmem-8" "NVIDIA_TESLA_T4" "1"
            ;;
            
        autogluon)
            log_info "Running AutoGluon experiments..."
            
            submit_vertex_job "ag-simple" "autogluon" "simple_only" "" "false" "" "n1-highmem-8" "" "0"
            sleep 5
            submit_vertex_job "ag-full" "autogluon" "full_context" "" "false" "" "n1-highmem-16" "" "0"
            ;;
            
        status)
            log_info "Checking job status..."
            gcloud ai custom-jobs list --region="$REGION" --format="table(displayName,state,createTime)"
            ;;
            
        results)
            log_info "Downloading results..."
            mkdir -p ./cloud_results
            gsutil -m cp -r "gs://$BUCKET_NAME/magnesium-pipeline/reports/*" ./cloud_results/ 2>/dev/null || true
            gsutil -m cp -r "gs://$BUCKET_NAME/magnesium-pipeline/models/*" ./cloud_results/ 2>/dev/null || true
            log_info "Results downloaded to ./cloud_results/"
            ;;
            
        help)
            echo "Usage: $0 [quick|feature-test|optimize|autogluon|status|results|help]"
            echo ""
            echo "Commands:"
            echo "  quick       - Run 3 high-impact experiments"
            echo "  feature-test - Test different feature strategies"
            echo "  optimize    - Run optimization experiments"
            echo "  autogluon   - Run AutoGluon experiments"
            echo "  status      - Check running job status"
            echo "  results     - Download results from GCS"
            echo "  help        - Show this help"
            ;;
            
        *)
            log_error "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
    
    log_info "Cloud experiment submission completed!"
    log_info "Monitor with: $0 status"
    log_info "Get results: $0 results"
}

# Run main
main "$@"