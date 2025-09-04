#!/bin/bash

# Comprehensive AutoGluon Experiment Script for Magnesium Pipeline
# Target Performance: R² ≥ 0.8, MAPE < 10%, MAE < 0.05
# 
# This script systematically explores all relevant configuration options
# to achieve optimal magnesium concentration prediction performance

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
    echo "ag_$(date +%Y%m%d_%H%M%S)_$$"
}

# ============================================================================
# EXPERIMENT CONFIGURATION MATRICES
# ============================================================================

# Feature Strategies (Impact: HIGH)
STRATEGIES=("Mg_only" "simple_only" "full_context")

# AutoGluon Configurations (Impact: VERY HIGH)
declare -A AUTOGLUON_CONFIGS=(
    # Best Quality Configurations (for R² ≥ 0.8)
    ["best_quality_stacked"]="presets=best_quality,time_limit=14400,num_bag_folds=5,num_bag_sets=3,num_stack_levels=2,auto_stack=true"
    ["best_quality_long"]="presets=best_quality,time_limit=18000,num_bag_folds=7,num_bag_sets=2,num_stack_levels=3,auto_stack=true"
    
    # High Quality Configurations (balanced performance/time)
    ["high_quality_optimized"]="presets=high_quality,time_limit=10800,num_bag_folds=5,num_bag_sets=2,num_stack_levels=2,auto_stack=true"
    ["high_quality_fast"]="presets=high_quality,time_limit=7200,num_bag_folds=4,num_bag_sets=2,num_stack_levels=1,auto_stack=true"
    
    # Good Quality Configurations (faster baseline)
    ["good_quality_baseline"]="presets=good_quality,time_limit=5400,num_bag_folds=5,num_bag_sets=1,num_stack_levels=1,auto_stack=false"
    ["good_quality_stacked"]="presets=good_quality,time_limit=7200,num_bag_folds=5,num_bag_sets=2,num_stack_levels=2,auto_stack=true"
)

# Feature Engineering Configurations (Impact: HIGH for magnesium)
declare -A FEATURE_CONFIGS=(
    # Magnesium-focused configurations
    ["mg_focused"]="enable_molecular_bands=false,enable_macro_elements=true,enable_micro_elements=true,enable_oxygen_hydrogen=false,enable_advanced_ratios=true,enable_spectral_patterns=true,use_focused_magnesium_features=true"
    ["mg_comprehensive"]="enable_molecular_bands=true,enable_macro_elements=true,enable_micro_elements=true,enable_oxygen_hydrogen=true,enable_advanced_ratios=true,enable_spectral_patterns=true,use_focused_magnesium_features=true"
    ["mg_minimal"]="enable_molecular_bands=false,enable_macro_elements=true,enable_micro_elements=false,enable_oxygen_hydrogen=false,enable_advanced_ratios=true,enable_spectral_patterns=false,use_focused_magnesium_features=true"
    ["mg_interference_corrected"]="enable_molecular_bands=false,enable_macro_elements=true,enable_micro_elements=true,enable_oxygen_hydrogen=false,enable_advanced_ratios=true,enable_spectral_patterns=true,enable_interference_correction=true,use_focused_magnesium_features=true"
    
    # Raw spectral configurations
    ["raw_spectral"]="use_raw_spectral_data=true,enable_molecular_bands=false,enable_macro_elements=false,enable_micro_elements=false,enable_oxygen_hydrogen=false,use_focused_magnesium_features=false"
)

# Sample Weighting and Concentration Features (Impact: MEDIUM-HIGH)
declare -A CONCENTRATION_CONFIGS=(
    ["weights_improved"]="use_sample_weights=true,weight_method=improved,use_concentration_features=true,use_data_driven_thresholds=true"
    ["weights_legacy"]="use_sample_weights=true,weight_method=legacy,use_concentration_features=true,use_data_driven_thresholds=false"
    ["no_weights_conc"]="use_sample_weights=false,weight_method=improved,use_concentration_features=true,use_data_driven_thresholds=true"
    ["no_weights_no_conc"]="use_sample_weights=false,weight_method=improved,use_concentration_features=false,use_data_driven_thresholds=false"
)

# Dimensionality Reduction Configurations (Impact: MEDIUM)
declare -A DIMENSION_CONFIGS=(
    ["no_reduction"]="use_dimension_reduction=false"
    ["pls_conservative"]="use_dimension_reduction=true,method=pls,n_components=10,pls_scale=true"
    ["pls_optimal"]="use_dimension_reduction=true,method=pls,n_components=15,pls_scale=true"
    ["pca_variance"]="use_dimension_reduction=true,method=pca,n_components=0.95"
    ["pca_fixed"]="use_dimension_reduction=true,method=pca,n_components=20"
)

# GPU Configurations
GPU_CONFIGS=("true" "false")

# Machine Types (based on AutoGluon requirements)
MACHINE_TYPES=("n1-highmem-8" "n1-highmem-16" "n1-standard-8")

# ============================================================================
# EXPERIMENT SUBMISSION FUNCTIONS
# ============================================================================

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
    local autogluon_config="$4"
    local feature_config="$5"
    local concentration_config="$6"
    local dimension_config="$7"
    
    local config_file="$CONFIGS_DIR/${config_name}_${strategy}_${exp_id}.yaml"
    
    cat > "$config_file" << EOF
# Generated AutoGluon Configuration for Magnesium Prediction
# Experiment: $exp_id
# Target: R² ≥ 0.8, MAPE < 10%, MAE < 0.05

# AutoGluon Configuration
autogluon:
EOF
    
    # Parse and add AutoGluon config
    IFS=',' read -ra AG_PARAMS <<< "$autogluon_config"
    for param in "${AG_PARAMS[@]}"; do
        IFS='=' read -ra PARAM_PAIR <<< "$param"
        key="${PARAM_PAIR[0]}"
        value="${PARAM_PAIR[1]}"
        
        case $key in
            "presets"|"fold_fitting_strategy"|"weight_method")
                echo "  $key: \"$value\"" >> "$config_file"
                ;;
            "auto_stack"|"dynamic_stacking"|"pls_scale"|"use_sample_weights"|"use_concentration_features"|"use_data_driven_thresholds"|"use_raw_spectral_data"|"use_dimension_reduction"|"enable_molecular_bands"|"enable_macro_elements"|"enable_micro_elements"|"enable_oxygen_hydrogen"|"enable_advanced_ratios"|"enable_spectral_patterns"|"enable_interference_correction"|"use_focused_magnesium_features")
                echo "  $key: $value" >> "$config_file"
                ;;
            "method")
                echo "  $key: \"$value\"" >> "$config_file"
                ;;
            *)
                echo "  $key: $value" >> "$config_file"
                ;;
        esac
    done
    
    # Add feature configuration
    echo "" >> "$config_file"
    echo "# Feature Engineering Configuration" >> "$config_file"
    IFS=',' read -ra FEATURE_PARAMS <<< "$feature_config"
    for param in "${FEATURE_PARAMS[@]}"; do
        IFS='=' read -ra PARAM_PAIR <<< "$param"
        key="${PARAM_PAIR[0]}"
        value="${PARAM_PAIR[1]}"
        
        case $key in
            "use_raw_spectral_data"|"enable_molecular_bands"|"enable_macro_elements"|"enable_micro_elements"|"enable_oxygen_hydrogen"|"enable_advanced_ratios"|"enable_spectral_patterns"|"enable_interference_correction"|"use_focused_magnesium_features")
                echo "$key: $value" >> "$config_file"
                ;;
        esac
    done
    
    # Add concentration configuration
    echo "" >> "$config_file"
    echo "# Sample Weighting Configuration" >> "$config_file"
    IFS=',' read -ra CONC_PARAMS <<< "$concentration_config"
    for param in "${CONC_PARAMS[@]}"; do
        IFS='=' read -ra PARAM_PAIR <<< "$param"
        key="${PARAM_PAIR[0]}"
        value="${PARAM_PAIR[1]}"
        
        case $key in
            "weight_method")
                echo "$key: \"$value\"" >> "$config_file"
                ;;
            "use_sample_weights"|"use_concentration_features"|"use_data_driven_thresholds")
                echo "$key: $value" >> "$config_file"
                ;;
        esac
    done
    
    # Add dimensionality reduction configuration
    echo "" >> "$config_file"
    echo "# Dimensionality Reduction Configuration" >> "$config_file"
    IFS=',' read -ra DIM_PARAMS <<< "$dimension_config"
    for param in "${DIM_PARAMS[@]}"; do
        IFS='=' read -ra PARAM_PAIR <<< "$param"
        key="${PARAM_PAIR[0]}"
        value="${PARAM_PAIR[1]}"
        
        case $key in
            "method")
                echo "dimension_reduction:" >> "$config_file"
                echo "  $key: \"$value\"" >> "$config_file"
                ;;
            "use_dimension_reduction"|"pls_scale")
                if [ "$key" = "use_dimension_reduction" ]; then
                    echo "$key: $value" >> "$config_file"
                else
                    echo "  $key: $value" >> "$config_file"
                fi
                ;;
            "n_components")
                echo "  $key: $value" >> "$config_file"
                ;;
        esac
    done
    
    echo "$config_file"
}

# ============================================================================
# EXPERIMENT PHASES
# ============================================================================

# Phase 1: High-Impact Configurations (Best chance for R² ≥ 0.8)
run_phase1_high_impact() {
    log_info "=== PHASE 1: High-Impact Configurations ==="
    local phase_count=0
    
    # Best strategies with best AutoGluon configs
    for strategy in "simple_only" "full_context"; do
        for ag_config_name in "best_quality_stacked" "best_quality_long"; do
            for feature_config_name in "mg_focused" "mg_comprehensive"; do
                for concentration_config_name in "weights_improved" "no_weights_conc"; do
                    for dimension_config_name in "no_reduction" "pls_optimal"; do
                        for use_gpu in "true" "false"; do
                            exp_id=$(generate_experiment_id)
                            config_name="phase1_${ag_config_name}_${feature_config_name}_${concentration_config_name}_${dimension_config_name}"
                            
                            # Generate config file
                            generate_config "$config_name" "$strategy" "$exp_id" \
                                "${AUTOGLUON_CONFIGS[$ag_config_name]}" \
                                "${FEATURE_CONFIGS[$feature_config_name]}" \
                                "${CONCENTRATION_CONFIGS[$concentration_config_name]}" \
                                "${DIMENSION_CONFIGS[$dimension_config_name]}"
                            
                            # Select machine type based on GPU
                            local machine_type
                            if [ "$use_gpu" = "true" ]; then
                                machine_type="n1-highmem-16"
                            else
                                machine_type="n1-highmem-8"
                            fi
                            
                            submit_autogluon_job "$exp_id" "$strategy" "$config_name" "$use_gpu" "$machine_type" "Phase1: High-impact config targeting R²≥0.8"
                            
                            ((phase_count++))
                            # Add delay between submissions
                            sleep 15
                            
                            # Limit phase 1 experiments
                            if [ $phase_count -ge 16 ]; then
                                log_warn "Phase 1 limited to 16 experiments to manage resources"
                                return
                            fi
                        done
                    done
                done
            done
        done
    done
    
    log_info "Phase 1 completed: $phase_count experiments submitted"
}

# Phase 2: Optimization Experiments (Medium-impact configurations)
run_phase2_optimization() {
    log_info "=== PHASE 2: Optimization Experiments ==="
    local phase_count=0
    
    # Test different AutoGluon configurations with best feature settings
    for strategy in "Mg_only" "simple_only"; do
        for ag_config_name in "high_quality_optimized" "good_quality_stacked"; do
            for feature_config_name in "mg_focused" "mg_interference_corrected"; do
                for concentration_config_name in "weights_improved" "weights_legacy"; do
                    exp_id=$(generate_experiment_id)
                    config_name="phase2_${ag_config_name}_${feature_config_name}_${concentration_config_name}"
                    
                    generate_config "$config_name" "$strategy" "$exp_id" \
                        "${AUTOGLUON_CONFIGS[$ag_config_name]}" \
                        "${FEATURE_CONFIGS[$feature_config_name]}" \
                        "${CONCENTRATION_CONFIGS[$concentration_config_name]}" \
                        "${DIMENSION_CONFIGS["no_reduction"]}"
                    
                    submit_autogluon_job "$exp_id" "$strategy" "$config_name" "true" "n1-highmem-8" "Phase2: Optimization experiments"
                    
                    ((phase_count++))
                    sleep 10
                    
                    if [ $phase_count -ge 12 ]; then
                        break 4
                    fi
                done
            done
        done
    done
    
    log_info "Phase 2 completed: $phase_count experiments submitted"
}

# Phase 3: Raw Spectral and Alternative Approaches
run_phase3_alternatives() {
    log_info "=== PHASE 3: Alternative Approaches ==="
    local phase_count=0
    
    # Raw spectral experiments
    for strategy in "Mg_only" "simple_only" "full_context"; do
        for ag_config_name in "best_quality_stacked" "high_quality_optimized"; do
            exp_id=$(generate_experiment_id)
            config_name="phase3_raw_spectral_${ag_config_name}"
            
            generate_config "$config_name" "$strategy" "$exp_id" \
                "${AUTOGLUON_CONFIGS[$ag_config_name]}" \
                "${FEATURE_CONFIGS["raw_spectral"]}" \
                "${CONCENTRATION_CONFIGS["no_weights_no_conc"]}" \
                "${DIMENSION_CONFIGS["no_reduction"]}"
            
            submit_autogluon_job "$exp_id" "$strategy" "$config_name" "true" "n1-highmem-16" "Phase3: Raw spectral approach"
            
            ((phase_count++))
            sleep 10
            
            if [ $phase_count -ge 6 ]; then
                break 2
            fi
        done
    done
    
    log_info "Phase 3 completed: $phase_count experiments submitted"
}

# Quick test experiment (single high-probability success)
run_quick_test() {
    log_info "=== QUICK TEST: Single High-Probability Experiment ==="
    
    exp_id=$(generate_experiment_id)
    config_name="quick_test_best_config"
    strategy="simple_only"
    
    generate_config "$config_name" "$strategy" "$exp_id" \
        "${AUTOGLUON_CONFIGS["best_quality_stacked"]}" \
        "${FEATURE_CONFIGS["mg_focused"]}" \
        "${CONCENTRATION_CONFIGS["weights_improved"]}" \
        "${DIMENSION_CONFIGS["no_reduction"]}"
    
    submit_autogluon_job "$exp_id" "$strategy" "$config_name" "true" "n1-highmem-16" "Quick test: Best probability config for R²≥0.8"
    
    log_info "Quick test submitted: $exp_id"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_info "Starting AutoGluon Comprehensive Experiments"
    log_info "Target: R² ≥ 0.8, MAPE < 10%, MAE < 0.05"
    log_info "Project: $PROJECT_ID | Region: $REGION"
    
    case "${1:-quick}" in
        quick)
            run_quick_test
            ;;
        phase1)
            run_phase1_high_impact
            ;;
        phase2)
            run_phase2_optimization
            ;;
        phase3)
            run_phase3_alternatives
            ;;
        all)
            log_info "Running all experiment phases..."
            run_phase1_high_impact
            sleep 60
            run_phase2_optimization  
            sleep 60
            run_phase3_alternatives
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
            echo "Usage: $0 [quick|phase1|phase2|phase3|all|status|results|help]"
            echo ""
            echo "Commands:"
            echo "  quick      - Single high-probability experiment for R² ≥ 0.8"
            echo "  phase1     - High-impact configurations (16 experiments)"
            echo "  phase2     - Optimization experiments (12 experiments)"
            echo "  phase3     - Alternative approaches (6 experiments)"
            echo "  all        - All experiment phases (34 experiments total)"
            echo "  status     - Check job status"
            echo "  results    - Download results from GCS"
            echo "  help       - Show this help"
            echo ""
            echo "Target Performance:"
            echo "  R² ≥ 0.8 (coefficient of determination)"
            echo "  MAPE < 10% (mean absolute percentage error)"
            echo "  MAE < 0.05 (mean absolute error)"
            ;;
        *)
            log_error "Unknown command: $1"
            echo "Use '$0 help' for usage"
            exit 1
            ;;
    esac
    
    log_info "AutoGluon experiments submission completed!"
    log_info "Monitor with: $0 status"
    log_info "Get results: $0 results"
    
    # Show current job count
    local running_jobs=$(gcloud ai custom-jobs list --region="$REGION" --filter="state=JOB_STATE_RUNNING AND displayName:autogluon-experiments*" --format="value(name)" | wc -l)
    log_info "Current running AutoGluon jobs: $running_jobs"
}

# Run main
main "$@"