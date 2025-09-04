#!/bin/bash

# Comprehensive AutoGluon Experiment Script - Simplified and Fixed
# Runs ALL experiments from the original matrix without complex nested loops in functions

set -e

# Colors and logging functions  
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Results tracking
RESULTS_DIR="./autogluon_experiment_results"
LOGS_DIR="./autogluon_experiment_logs"
CONFIGS_DIR="./autogluon_experiment_configs"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR" "$CONFIGS_DIR"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOGS_DIR/main.log"
}

# Load configuration from staging config
STAGING_CONFIG="/home/payanico/magnesium_pipeline/config/staging_config.yml"

# Parse YAML configuration
if [ ! -f "$STAGING_CONFIG" ]; then
    echo "Staging config not found: $STAGING_CONFIG"
    exit 1
fi

log_info "Loading configuration from: $STAGING_CONFIG"

PROJECT_ID=$(grep -A 10 "gcp:" "$STAGING_CONFIG" | grep "project_id:" | head -1 | sed 's/.*project_id: *"\?//' | sed 's/"\? *#.*//' | sed 's/"\?$//')
REGION=$(grep -A 10 "gcp:" "$STAGING_CONFIG" | grep "region:" | head -1 | sed 's/.*region: *"\?//' | sed 's/"\? *#.*//' | sed 's/"\?$//')
BUCKET_NAME=$(grep "bucket_name:" "$STAGING_CONFIG" | sed 's/.*bucket_name: *"\?//' | sed 's/"\? *#.*//' | sed 's/"\?$//')
CLOUD_STORAGE_PREFIX=$(grep "prefix:" "$STAGING_CONFIG" | sed 's/.*prefix: *"\?//' | sed 's/"\? *#.*//' | sed 's/"\?$//')

PROJECT_ID="${PROJECT_ID:-mapana-ai-models}"
REGION="${REGION:-us-central1}"
BUCKET_NAME="${BUCKET_NAME:-staging-magnesium-data}"
CLOUD_STORAGE_PREFIX="${CLOUD_STORAGE_PREFIX:-magnesium-pipeline-staging}"

REPO_NAME="${REPO_NAME:-magnesium-repo}"
IMAGE_NAME="${IMAGE_NAME:-magnesium-pipeline}"
SERVICE_NAME="${SERVICE_NAME:-autogluon-experiments}"

# Experiment tracking
EXPERIMENT_TRACKER="$RESULTS_DIR/autogluon_tracker.csv"
if [ ! -f "$EXPERIMENT_TRACKER" ]; then
    echo "experiment_id,timestamp,strategy,config_name,status,job_id,r2_target,notes" > "$EXPERIMENT_TRACKER"
fi

log_info "Configuration loaded:"
log_info "  PROJECT_ID: $PROJECT_ID"
log_info "  REGION: $REGION"
log_info "  BUCKET_NAME: $BUCKET_NAME"
log_info "  CLOUD_STORAGE_PREFIX: $CLOUD_STORAGE_PREFIX"

generate_experiment_id() {
    echo "ag_$(date +%Y%m%d_%H%M%S)_$$_${RANDOM}"
}

# Submit Vertex AI job
submit_job() {
    local exp_id="$1"
    local strategy="$2" 
    local ag_preset="$3"
    local time_limit="$4"
    local feature_config="$5"
    local weights="$6"
    local dim_reduction="$7"
    local use_gpu="$8"
    local machine_type="$9"
    
    local config_name="${ag_preset}_${feature_config}_${weights}_${dim_reduction}"
    
    log_info "Submitting experiment $exp_id:"
    log_info "  Strategy: $strategy | Preset: $ag_preset | Features: $feature_config"
    log_info "  Weights: $weights | Reduction: $dim_reduction | GPU: $use_gpu"
    
    # Build Python command
    local python_cmd="python main.py autogluon --strategy $strategy"
    if [ "$use_gpu" = "true" ]; then
        python_cmd="$python_cmd --gpu"
    fi
    
    # Create unique job name
    local job_name="${SERVICE_NAME}-${exp_id}"
    
    # Container image
    local image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
    
    # Create temporary job config
    local config_file_vertex="/tmp/vertex-job-${exp_id}.json"
    
    cat > "$config_file_vertex" << EOF
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
    "timeout": "${time_limit}s"
  }
}
EOF
    
    # Submit job
    local job_output job_id
    if job_output=$(gcloud ai custom-jobs create \
        --display-name="$job_name" \
        --region="$REGION" \
        --config="$config_file_vertex" 2>&1); then
        
        job_id=$(echo "$job_output" | grep -oE 'projects/[0-9]+/locations/[^/]+/customJobs/[0-9]+' | head -1)
        
        log_info "✓ Job submitted: $job_name"
        log_info "  Job ID: $job_id"
        
        local description="$strategy strategy, $ag_preset preset, $feature_config features, $weights weighting, $dim_reduction reduction, GPU=$use_gpu"
        echo "$exp_id,$(date),$strategy,$config_name,SUCCESS,$job_id,0.8,$description" >> "$EXPERIMENT_TRACKER"
        rm -f "$config_file_vertex"
        return 0
    else
        log_info "✗ Failed to submit: $job_name"
        echo "$exp_id,$(date),$strategy,$config_name,FAILED,,$job_output" >> "$EXPERIMENT_TRACKER"
        rm -f "$config_file_vertex"
        return 1
    fi
}

# Main execution - direct nested loops without function scope issues
log_info "Starting Comprehensive AutoGluon Experiments"
log_info "Target: R² ≥ 0.8, MAPE < 10%, MAE < 0.05"

# Define experiment matrix
strategies=("simple_only" "full_context" "Mg_only")
ag_configs=("best_quality_stacked" "best_quality_long" "high_quality_optimized" "high_quality_fast" "good_quality_baseline" "good_quality_stacked")
feature_configs=("mg_focused" "mg_comprehensive" "mg_minimal" "mg_interference" "raw_spectral")
concentration_configs=("weights_improved" "weights_legacy" "no_weights_conc" "no_weights_no_conc") 
dimension_configs=("no_reduction" "pls_conservative" "pls_optimal" "pca_variance" "pca_fixed")
gpu_configs=("true" "false")

# Calculate total experiments
total_experiments=$((${#strategies[@]} * ${#ag_configs[@]} * ${#feature_configs[@]} * ${#concentration_configs[@]} * ${#dimension_configs[@]} * ${#gpu_configs[@]}))
log_info "Total experiments to run: $total_experiments"

exp_count=0
max_experiments="${1:-$total_experiments}"

log_info "Running up to $max_experiments experiments..."

# Direct nested loops
for strategy in "${strategies[@]}"; do
    for ag_config in "${ag_configs[@]}"; do
        for feature_config in "${feature_configs[@]}"; do
            for concentration_config in "${concentration_configs[@]}"; do
                for dimension_config in "${dimension_configs[@]}"; do
                    for gpu in "${gpu_configs[@]}"; do
                        # Check limit
                        if [ $exp_count -ge $max_experiments ]; then
                            log_info "Reached experiment limit: $max_experiments"
                            log_info "Experiments submitted: $exp_count"
                            exit 0
                        fi
                        
                        # Set time limit and machine type based on config
                        local time_limit machine_type
                        case $ag_config in
                            "best_quality_stacked") time_limit="14400" ;;
                            "best_quality_long") time_limit="18000" ;;
                            "high_quality_optimized") time_limit="10800" ;;
                            "high_quality_fast") time_limit="7200" ;;
                            "good_quality_baseline") time_limit="5400" ;;
                            "good_quality_stacked") time_limit="7200" ;;
                        esac
                        
                        if [ "$gpu" = "true" ]; then
                            machine_type="n1-highmem-16"
                        else
                            machine_type="n1-highmem-8"  
                        fi
                        
                        # Generate experiment ID
                        exp_id=$(generate_experiment_id)
                        
                        # Submit job
                        submit_job "$exp_id" "$strategy" "$ag_config" "$time_limit" \
                            "$feature_config" "$concentration_config" "$dimension_config" \
                            "$gpu" "$machine_type"
                        
                        ((exp_count++))
                        
                        # Add delay between submissions
                        if [ $exp_count -lt $max_experiments ]; then
                            log_info "Progress: $exp_count/$max_experiments submitted. Waiting 20 seconds..."
                            sleep 20
                        fi
                    done
                done
            done
        done
    done
done

log_info "All experiments completed! Total submitted: $exp_count"

# Show current job count
running_jobs=$(gcloud ai custom-jobs list --region="$REGION" --filter="state=JOB_STATE_RUNNING AND displayName:autogluon-experiments*" --format="value(name)" | wc -l)
pending_jobs=$(gcloud ai custom-jobs list --region="$REGION" --filter="state=JOB_STATE_PENDING AND displayName:autogluon-experiments*" --format="value(name)" | wc -l)
log_info "Current jobs - Running: $running_jobs, Pending: $pending_jobs"