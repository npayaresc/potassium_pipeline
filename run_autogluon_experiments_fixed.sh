#!/bin/bash

# Fixed AutoGluon Experiment Script for Magnesium Pipeline
# This version properly iterates through all experiment configurations

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

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOGS_DIR/main.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOGS_DIR/main.log"
}

# Load configuration from staging config
STAGING_CONFIG="/home/payanico/magnesium_pipeline/config/staging_config.yml"

# Parse YAML configuration
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
SERVICE_NAME="${SERVICE_NAME:-autogluon-experiments}"

# Experiment tracking
EXPERIMENT_TRACKER="$RESULTS_DIR/autogluon_tracker.csv"
if [ ! -f "$EXPERIMENT_TRACKER" ]; then
    echo "experiment_id,timestamp,strategy,config_name,status,job_id,r2_target,notes" > "$EXPERIMENT_TRACKER"
fi

generate_experiment_id() {
    echo "ag_$(date +%Y%m%d_%H%M%S)_$$_${RANDOM}"
}

# Submit Vertex AI job
submit_autogluon_job() {
    local exp_id="$1"
    local strategy="$2"
    local config_name="$3"
    local use_gpu="$4"
    local machine_type="$5"
    local notes="$6"
    
    log_info "Submitting AutoGluon experiment: $exp_id"
    log_info "  Strategy: $strategy | Config: $config_name | GPU: $use_gpu | Machine: $machine_type"
    
    # Build Python command
    local python_cmd="python main.py autogluon --strategy $strategy"
    
    if [ "$use_gpu" = "true" ]; then
        python_cmd="$python_cmd --gpu"
    fi
    
    # Add custom config if generated
    local config_file="$CONFIGS_DIR/${config_name}_${strategy}_${exp_id}.yaml"
    if [ -f "$config_file" ]; then
        python_cmd="$python_cmd --config $config_file"
    fi
    
    # Create unique job name
    local job_name="${SERVICE_NAME}-${exp_id}"
    
    # Container image
    local image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
    
    # Set timeout based on config (AutoGluon needs longer time)
    local timeout="18000"  # 5 hours for AutoGluon experiments
    
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
    "timeout": "${timeout}s"
  }
}
EOF
    
    # Submit job
    local job_output
    local job_id
    if job_output=$(gcloud ai custom-jobs create \
        --display-name="$job_name" \
        --region="$REGION" \
        --config="$config_file_vertex" 2>&1); then
        
        # Extract job ID from the output - look for the customJobs ID
        job_id=$(echo "$job_output" | grep -oE 'projects/[0-9]+/locations/[^/]+/customJobs/[0-9]+' | head -1)
        
        log_info "✓ Job submitted: $job_name"
        log_info "  Job ID: $job_id"
        echo "$exp_id,$(date),$strategy,$config_name,SUCCESS,$job_id,0.8,$notes" >> "$EXPERIMENT_TRACKER"
        rm -f "$config_file_vertex"
        return 0
    else
        log_error "✗ Failed to submit: $job_name"
        log_error "Error output: $job_output"
        echo "$exp_id,$(date),$strategy,$config_name,FAILED,,$notes" >> "$EXPERIMENT_TRACKER"
        rm -f "$config_file_vertex"
        return 1
    fi
}

# Generate custom configuration file
generate_config() {
    local config_name="$1"
    local strategy="$2"
    local exp_id="$3"
    local autogluon_params="$4"
    local feature_params="$5"
    local sample_params="$6"
    local dim_params="$7"
    
    local config_file="$CONFIGS_DIR/${config_name}_${strategy}_${exp_id}.yaml"
    
    cat > "$config_file" << EOF
# Generated AutoGluon Configuration for Magnesium Prediction
# Experiment: $exp_id
# Strategy: $strategy
# Target: R² ≥ 0.8, MAPE < 10%, MAE < 0.05

# AutoGluon Configuration
$autogluon_params

# Feature Engineering Configuration
$feature_params

# Sample Weighting Configuration
$sample_params

# Dimensionality Reduction Configuration
$dim_params
EOF
    
    # Return config file path (remove echo to prevent issues with set -e)
    return 0
}

# Define ALL experiment configurations from original script
run_all_experiments() {
    log_info "=== Running ALL AutoGluon Experiments ==="
    local exp_count=0
    local max_experiments="${1:-9999}"  # Run all experiments unless limited
    
    # Define experiment configurations as arrays
    # Each element is: "strategy|ag_preset|ag_time|feature_set|weights|dim_reduction|gpu|machine|description"
    
    # Generate all experiment combinations from original script matrix
    local strategies=("simple_only" "full_context" "Mg_only")
    local ag_configs=("best_quality_stacked" "best_quality_long" "high_quality_optimized" "high_quality_fast" "good_quality_baseline" "good_quality_stacked")
    local feature_configs=("mg_focused" "mg_comprehensive" "mg_minimal" "mg_interference" "raw_spectral")
    local concentration_configs=("weights_improved" "weights_legacy" "no_weights_conc" "no_weights_no_conc")
    local dimension_configs=("no_reduction" "pls_conservative" "pls_optimal" "pca_variance" "pca_fixed")
    local gpu_configs=("true" "false")
    local machine_types_gpu=("n1-highmem-16")
    local machine_types_cpu=("n1-highmem-8")

    local experiments=()
    
    # Generate comprehensive experiment matrix
    for strategy in "${strategies[@]}"; do
        for ag_config in "${ag_configs[@]}"; do
            for feature_config in "${feature_configs[@]}"; do
                for concentration_config in "${concentration_configs[@]}"; do
                    for dimension_config in "${dimension_configs[@]}"; do
                        for gpu in "${gpu_configs[@]}"; do
                            # Set time limit and machine type based on config
                            local time_limit machine_type
                            case $ag_config in
                                "best_quality_stacked")
                                    time_limit="14400"
                                    ;;
                                "best_quality_long")
                                    time_limit="18000"
                                    ;;
                                "high_quality_optimized")
                                    time_limit="10800"
                                    ;;
                                "high_quality_fast")
                                    time_limit="7200"
                                    ;;
                                "good_quality_baseline")
                                    time_limit="5400"
                                    ;;
                                "good_quality_stacked")
                                    time_limit="7200"
                                    ;;
                            esac
                            
                            if [ "$gpu" = "true" ]; then
                                machine_type="n1-highmem-16"
                            else
                                machine_type="n1-highmem-8"
                            fi
                            
                            # Create experiment description
                            local description="$strategy strategy, $ag_config preset, $feature_config features, $concentration_config weighting, $dimension_config reduction, GPU=$gpu"
                            
                            experiments+=("$strategy|$ag_config|$time_limit|$feature_config|$concentration_config|$dimension_config|$gpu|$machine_type|$description")
                        done
                    done
                done
            done
        done
    done
    
    # Process experiments
    for exp_config in "${experiments[@]}"; do
        if [ $exp_count -ge $max_experiments ]; then
            log_warn "Reached maximum experiment limit: $max_experiments"
            break
        fi
        
        # Parse configuration
        IFS='|' read -r strategy ag_preset ag_time feature_set weights dim_reduction use_gpu machine description <<< "$exp_config"
        
        # Generate experiment ID
        exp_id=$(generate_experiment_id)
        
        # Create configuration name
        config_name="${ag_preset}_${feature_set}_${weights}_${dim_reduction}"
        
        # Build AutoGluon parameters based on preset
        local ag_params=""
        case $ag_preset in
            "best_quality_stacked")
                ag_params="autogluon:
  presets: \"best_quality\"
  time_limit: $ag_time
  num_bag_folds: 5
  num_bag_sets: 3
  num_stack_levels: 2
  auto_stack: true"
                ;;
            "best_quality_long")
                ag_params="autogluon:
  presets: \"best_quality\"
  time_limit: $ag_time
  num_bag_folds: 7
  num_bag_sets: 2
  num_stack_levels: 3
  auto_stack: true"
                ;;
            "high_quality_optimized")
                ag_params="autogluon:
  presets: \"high_quality\"
  time_limit: $ag_time
  num_bag_folds: 5
  num_bag_sets: 2
  num_stack_levels: 2
  auto_stack: true"
                ;;
            "high_quality_fast")
                ag_params="autogluon:
  presets: \"high_quality\"
  time_limit: $ag_time
  num_bag_folds: 4
  num_bag_sets: 2
  num_stack_levels: 1
  auto_stack: true"
                ;;
            "good_quality_baseline")
                ag_params="autogluon:
  presets: \"good_quality\"
  time_limit: $ag_time
  num_bag_folds: 5
  num_bag_sets: 1
  num_stack_levels: 1
  auto_stack: false"
                ;;
            "good_quality_stacked")
                ag_params="autogluon:
  presets: \"good_quality\"
  time_limit: $ag_time
  num_bag_folds: 5
  num_bag_sets: 2
  num_stack_levels: 2
  auto_stack: true"
                ;;
        esac
        
        # Build feature parameters based on feature_set
        local feature_params=""
        case $feature_set in
            mg_focused)
                feature_params="enable_molecular_bands: false
enable_macro_elements: true
enable_micro_elements: true
enable_oxygen_hydrogen: false
enable_advanced_ratios: true
enable_spectral_patterns: true
use_focused_magnesium_features: true"
                ;;
            mg_comprehensive)
                feature_params="enable_molecular_bands: true
enable_macro_elements: true
enable_micro_elements: true
enable_oxygen_hydrogen: true
enable_advanced_ratios: true
enable_spectral_patterns: true
use_focused_magnesium_features: true"
                ;;
            mg_minimal)
                feature_params="enable_molecular_bands: false
enable_macro_elements: true
enable_micro_elements: false
enable_oxygen_hydrogen: false
enable_advanced_ratios: true
enable_spectral_patterns: false
use_focused_magnesium_features: true"
                ;;
            mg_interference)
                feature_params="enable_molecular_bands: false
enable_macro_elements: true
enable_micro_elements: true
enable_oxygen_hydrogen: false
enable_advanced_ratios: true
enable_spectral_patterns: true
enable_interference_correction: true
use_focused_magnesium_features: true"
                ;;
            raw_spectral)
                feature_params="use_raw_spectral_data: true
enable_molecular_bands: false
enable_macro_elements: false
enable_micro_elements: false
enable_oxygen_hydrogen: false
use_focused_magnesium_features: false"
                ;;
        esac
        
        # Build sample weighting parameters
        local sample_params=""
        case $weights in
            weights_improved)
                sample_params="use_sample_weights: true
weight_method: \"improved\"
use_concentration_features: true
use_data_driven_thresholds: true"
                ;;
            weights_legacy)
                sample_params="use_sample_weights: true
weight_method: \"legacy\"
use_concentration_features: true
use_data_driven_thresholds: false"
                ;;
            no_weights_conc)
                sample_params="use_sample_weights: false
weight_method: \"improved\"
use_concentration_features: true
use_data_driven_thresholds: true"
                ;;
            no_weights_no_conc)
                sample_params="use_sample_weights: false
weight_method: \"improved\"
use_concentration_features: false
use_data_driven_thresholds: false"
                ;;
        esac
        
        # Build dimension reduction parameters
        local dim_params=""
        case $dim_reduction in
            no_reduction)
                dim_params="use_dimension_reduction: false"
                ;;
            pls_optimal)
                dim_params="use_dimension_reduction: true
dimension_reduction:
  method: \"pls\"
  n_components: 15
  pls_scale: true"
                ;;
            pls_conservative)
                dim_params="use_dimension_reduction: true
dimension_reduction:
  method: \"pls\"
  n_components: 10
  pls_scale: true"
                ;;
            pca_variance)
                dim_params="use_dimension_reduction: true
dimension_reduction:
  method: \"pca\"
  n_components: 0.95"
                ;;
            pca_fixed)
                dim_params="use_dimension_reduction: true
dimension_reduction:
  method: \"pca\"
  n_components: 20"
                ;;
        esac
        
        # Generate config file
        generate_config "$config_name" "$strategy" "$exp_id" \
            "$ag_params" "$feature_params" "$sample_params" "$dim_params"
        
        # Submit job
        submit_autogluon_job "$exp_id" "$strategy" "$config_name" "$use_gpu" "$machine" "$description"
        
        ((exp_count++))
        
        # Add delay between submissions
        if [ $exp_count -lt $max_experiments ]; then
            log_info "Waiting 20 seconds before next submission... ($exp_count/$max_experiments submitted)"
            sleep 20
        fi
    done
    
    log_info "Completed: $exp_count experiments submitted"
}

# Main execution
main() {
    log_info "Starting Fixed AutoGluon Experiment Submission"
    log_info "Target: R² ≥ 0.8, MAPE < 10%, MAE < 0.05"
    log_info "Project: $PROJECT_ID | Region: $REGION"
    
    case "${1:-help}" in
        run)
            local max_exp="${2:-9999}"
            log_info "Running up to $max_exp comprehensive experiments..."
            run_all_experiments "$max_exp"
            ;;
        status)
            log_info "Checking AutoGluon job status..."
            gcloud ai custom-jobs list --region="$REGION" --filter="displayName:autogluon-experiments*" --format="table(displayName,state,createTime)"
            ;;
        results)
            log_info "Downloading AutoGluon results..."
            mkdir -p ./autogluon_cloud_results
            gsutil -m cp -r "gs://$BUCKET_NAME/$CLOUD_STORAGE_PREFIX/reports/*autogluon*" ./autogluon_cloud_results/ 2>/dev/null || true
            log_info "Results downloaded to ./autogluon_cloud_results/"
            ;;
        help)
            echo "Usage: $0 [run|status|results|help] [max_experiments]"
            echo ""
            echo "Commands:"
            echo "  run [N]    - Run up to N comprehensive experiments (default: ALL)"
            echo "  status     - Check job status"
            echo "  results    - Download results from GCS"
            echo "  help       - Show this help"
            echo ""
            echo "Total experiment matrix: 3 strategies × 6 AutoGluon configs × 5 feature configs × 4 weight configs × 5 dimension configs × 2 GPU configs = 1800 experiments"
            echo ""
            echo "Examples:"
            echo "  $0 run       - Run ALL experiments (1800 total)"
            echo "  $0 run 100   - Run first 100 experiments"
            echo "  $0 run 50    - Run first 50 experiments"
            echo "  $0 status    - Check status of all jobs"
            ;;
        *)
            log_error "Unknown command: $1"
            echo "Use '$0 help' for usage"
            exit 1
            ;;
    esac
    
    log_info "AutoGluon experiments submission completed!"
    
    # Show current job count
    local running_jobs=$(gcloud ai custom-jobs list --region="$REGION" --filter="state=JOB_STATE_RUNNING AND displayName:autogluon-experiments*" --format="value(name)" | wc -l)
    local pending_jobs=$(gcloud ai custom-jobs list --region="$REGION" --filter="state=JOB_STATE_PENDING AND displayName:autogluon-experiments*" --format="value(name)" | wc -l)
    log_info "Current AutoGluon jobs - Running: $running_jobs, Pending: $pending_jobs"
}

# Run main
main "$@"