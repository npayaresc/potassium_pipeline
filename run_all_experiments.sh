#!/bin/bash

# Comprehensive Experiment Automation Script for Magnesium Pipeline
# This script systematically explores ALL configuration options from pipeline_config.py
# Target: R² ≥ 0.8, MAPE < 10%, MAE < 0.04

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# GCP Settings
PROJECT_ID="${PROJECT_ID:-mapana-ai-models}"
REGION="${REGION:-us-central1}"
BUCKET_NAME="${BUCKET_NAME:-${PROJECT_ID}-magnesium-data}"
RESULTS_DIR="./experiment_results"
LOGS_DIR="./experiment_logs"
CONFIGS_DIR="./experiment_configs"

# Create directories
mkdir -p "$RESULTS_DIR" "$LOGS_DIR" "$CONFIGS_DIR"

# Experiment tracking file
EXPERIMENT_TRACKER="$RESULTS_DIR/experiment_tracker.csv"
if [ ! -f "$EXPERIMENT_TRACKER" ]; then
    echo "experiment_id,timestamp,phase,config_file,status,r2,mape,mae,notes" > "$EXPERIMENT_TRACKER"
fi

# ============================================================================
# ALL CONFIGURATION OPTIONS FROM pipeline_config.py
# ============================================================================

# Feature Engineering Strategies
FEATURE_STRATEGIES=("Mg_only" "simple_only" "full_context")

# Feature Configuration Flags
MOLECULAR_BANDS_OPTIONS=("true" "false")
MACRO_ELEMENTS_OPTIONS=("true" "false")
MICRO_ELEMENTS_OPTIONS=("true" "false")
OXYGEN_HYDROGEN_OPTIONS=("true" "false")
ADVANCED_RATIOS_OPTIONS=("true" "false")
SPECTRAL_PATTERNS_OPTIONS=("true" "false")
INTERFERENCE_CORRECTION_OPTIONS=("true" "false")

# Magnesium Feature Methods
FOCUSED_MG_FEATURES_OPTIONS=("true" "false")

# Raw Spectral Data Mode
RAW_SPECTRAL_OPTIONS=("true" "false")

# Dimensionality Reduction Methods
DIM_REDUCTION_OPTIONS=("false" "true")
DIM_REDUCTION_METHODS=("pca" "pls" "autoencoder" "feature_clustering" "vae" "denoising_ae" "sparse_ae")
DIM_REDUCTION_COMPONENTS=(10 15 20 30 50 "0.95" "0.97" "0.99")

# Sample Weighting Methods
SAMPLE_WEIGHT_OPTIONS=("true" "false")
SAMPLE_WEIGHT_METHODS=("legacy" "improved" "weighted_r2" "distribution_based" "hybrid")

# Concentration Features
CONCENTRATION_FEATURES_OPTIONS=("true" "false")

# Outlier Detection Settings
OUTLIER_METHODS=("SAM" "MAD")
OUTLIER_THRESHOLDS=(0.7 0.75 0.8 0.85 0.9 0.95)
MAX_OUTLIER_PERCENTAGES=(20 30 40 50)

# Wavelength Standardization
WAVELENGTH_STD_OPTIONS=("true" "false")
WAVELENGTH_INTERP_METHODS=("linear" "cubic" "nearest")

# Cross-validation Folds
CV_FOLDS=(3 5 10)

# Training Modes
TRAINING_MODES=("train" "tune" "autogluon")

# Models to Train
MODEL_SETS=(
    "ridge,lasso,elastic_net"
    "random_forest,gradient_boost,extratrees"
    "xgboost,lightgbm,catboost"
    "svr"
    "neural_network,neural_network_light"
    "xgboost,lightgbm,catboost,random_forest,extratrees"
)

# Optimization Objectives (for tuning)
OBJECTIVE_FUNCTIONS=("r2" "robust" "robust_v2" "mape_focused" "distribution_based" "quantile_weighted" "balanced_mae" "hybrid_weighted")

# AutoGluon Specific Settings
AUTOGLUON_PRESETS=("good_quality" "best_quality" "high_quality")
AUTOGLUON_STACK_LEVELS=(0 1 2 3)
AUTOGLUON_BAG_FOLDS=(3 5 8)
AUTOGLUON_BAG_SETS=(1 2 3)

# Machine Types and GPU Configurations
MACHINE_CONFIGS=(
    "n1-standard-4:none:0"
    "n1-standard-8:none:0"
    "n1-highmem-8:NVIDIA_TESLA_T4:1"
    "n1-highmem-16:NVIDIA_TESLA_T4:1"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1" | tee -a "$LOGS_DIR/main.log"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOGS_DIR/main.log"
}

generate_experiment_id() {
    echo "exp_$(date +%Y%m%d_%H%M%S)_$$"
}

# Function to create a Python config file
create_python_config() {
    local config_name=$1
    local config_file="$CONFIGS_DIR/${config_name}.yaml"
    
    cat > "$config_file" << EOF
# Auto-generated configuration for experiment: $config_name
# Generated: $(date)

# Feature Engineering
feature_strategies: ["$FEATURE_STRATEGY"]
use_raw_spectral_data: $USE_RAW_SPECTRAL
use_focused_magnesium_features: $FOCUSED_MG_FEATURES

# Enhanced Features
enable_molecular_bands: $ENABLE_MOLECULAR_BANDS
enable_macro_elements: $ENABLE_MACRO_ELEMENTS
enable_micro_elements: $ENABLE_MICRO_ELEMENTS
enable_oxygen_hydrogen: $ENABLE_OXYGEN_HYDROGEN
enable_advanced_ratios: $ENABLE_ADVANCED_RATIOS
enable_spectral_patterns: $ENABLE_SPECTRAL_PATTERNS
enable_interference_correction: $ENABLE_INTERFERENCE_CORRECTION

# Dimensionality Reduction
use_dimension_reduction: $USE_DIM_REDUCTION
dimension_reduction:
  method: "$DIM_REDUCTION_METHOD"
  n_components: $DIM_REDUCTION_COMPONENTS

# Sample Weighting
use_sample_weights: $USE_SAMPLE_WEIGHTS
sample_weight_method: "$SAMPLE_WEIGHT_METHOD"
use_concentration_features: $USE_CONCENTRATION_FEATURES

# Data Processing
outlier_method: "$OUTLIER_METHOD"
outlier_threshold: $OUTLIER_THRESHOLD
max_outlier_percentage: $MAX_OUTLIER_PERCENTAGE
enable_wavelength_standardization: $ENABLE_WAVELENGTH_STD
wavelength_interpolation_method: "$WAVELENGTH_INTERP_METHOD"

# Training Configuration
models_to_train: [$MODELS_TO_TRAIN]
cv_folds: $CV_FOLDS
use_gpu: $USE_GPU

# Optimization Settings (for tuning)
tuner:
  objective_function_name: "$OBJECTIVE_FUNCTION"
  n_trials: $TRIALS
  timeout: $TIMEOUT

# AutoGluon Settings
autogluon:
  presets: "$AUTOGLUON_PRESET"
  ag_args_fit:
    num_bag_folds: $AUTOGLUON_BAG_FOLDS
    num_bag_sets: $AUTOGLUON_BAG_SETS
    num_stack_levels: $AUTOGLUON_STACK_LEVELS
  time_limit: $AUTOGLUON_TIME_LIMIT
EOF
    
    echo "$config_file"
}

# Function to run an experiment
run_experiment() {
    local exp_id=$1
    local phase=$2
    local config_file=$3
    local training_mode=$4
    
    log_info "Starting experiment: $exp_id (Phase: $phase)"
    
    # Set machine configuration
    IFS=':' read -r machine gpu gpu_count <<< "$MACHINE_CONFIG"
    
    export TRAINING_MODE="$training_mode"
    export MACHINE_TYPE="$machine"
    export ACCELERATOR_TYPE="$gpu"
    export ACCELERATOR_COUNT="$gpu_count"
    export CONFIG_FILE="$config_file"
    
    if [ "$gpu" != "none" ]; then
        export USE_GPU="true"
    else
        export USE_GPU="false"
    fi
    
    # Log experiment details
    cat >> "$LOGS_DIR/${exp_id}.log" << EOF
Experiment ID: $exp_id
Phase: $phase
Config File: $config_file
Training Mode: $training_mode
Machine Type: $machine
GPU: $gpu ($gpu_count)
Strategy: $FEATURE_STRATEGY
Models: $MODELS_TO_TRAIN
Timestamp: $(date)
================================================================================
EOF
    
    # Build Python command
    local python_cmd="python main.py $training_mode"
    
    if [ -n "$MODELS_TO_TRAIN" ]; then
        # Convert comma-separated models to space-separated for argparse
        MODELS_SPACE_SEPARATED="${MODELS_TO_TRAIN//,/ }"
        python_cmd="$python_cmd --models $MODELS_SPACE_SEPARATED"
    fi
    
    if [ -n "$FEATURE_STRATEGY" ]; then
        python_cmd="$python_cmd --strategy $FEATURE_STRATEGY"
    fi
    
    if [ "$USE_GPU" = "true" ]; then
        python_cmd="$python_cmd --gpu"
    fi
    
    if [ -n "$TRIALS" ] && [[ "$training_mode" == *"optimize"* || "$training_mode" == "tune" ]]; then
        python_cmd="$python_cmd --trials $TRIALS"
    fi
    
    # Submit Vertex AI job
    log_info "Executing: $python_cmd"
    
    if ./run_cloud_experiments.sh submit_job "$exp_id" "$training_mode" "$FEATURE_STRATEGY" "$MODELS_TO_TRAIN" "$USE_GPU" "$TRIALS" "$machine" "$gpu" "$gpu_count" >> "$LOGS_DIR/${exp_id}.log" 2>&1; then
        echo "$exp_id,$(date),${phase},${config_file},SUCCESS,,,," >> "$EXPERIMENT_TRACKER"
        log_info "Experiment $exp_id completed successfully"
        return 0
    else
        echo "$exp_id,$(date),${phase},${config_file},FAILED,,,," >> "$EXPERIMENT_TRACKER"
        log_error "Experiment $exp_id failed"
        return 1
    fi
}

# ============================================================================
# PHASE 1: FEATURE ENGINEERING EXPLORATION
# ============================================================================

run_phase1_feature_strategies() {
    log_info "PHASE 1: Feature Engineering Strategies"
    
    for strategy in "${FEATURE_STRATEGIES[@]}"; do
        for raw_spectral in "${RAW_SPECTRAL_OPTIONS[@]}"; do
            for focused_mg in "${FOCUSED_MG_FEATURES_OPTIONS[@]}"; do
                
                exp_id=$(generate_experiment_id)
                
                # Set configuration
                export FEATURE_STRATEGY="$strategy"
                export USE_RAW_SPECTRAL="$raw_spectral"
                export FOCUSED_MG_FEATURES="$focused_mg"
                export ENABLE_MOLECULAR_BANDS="false"
                export ENABLE_MACRO_ELEMENTS="true"
                export ENABLE_MICRO_ELEMENTS="false"
                export ENABLE_OXYGEN_HYDROGEN="false"
                export ENABLE_ADVANCED_RATIOS="true"
                export ENABLE_SPECTRAL_PATTERNS="true"
                export ENABLE_INTERFERENCE_CORRECTION="false"
                export USE_DIM_REDUCTION="false"
                export DIM_REDUCTION_METHOD="pca"
                export DIM_REDUCTION_COMPONENTS="30"
                export USE_SAMPLE_WEIGHTS="true"
                export SAMPLE_WEIGHT_METHOD="distribution_based"
                export USE_CONCENTRATION_FEATURES="true"
                export OUTLIER_METHOD="SAM"
                export OUTLIER_THRESHOLD="0.85"
                export MAX_OUTLIER_PERCENTAGE="30"
                export ENABLE_WAVELENGTH_STD="false"
                export WAVELENGTH_INTERP_METHOD="linear"
                export MODELS_TO_TRAIN="xgboost,lightgbm,catboost"
                export CV_FOLDS="5"
                export OBJECTIVE_FUNCTION="distribution_based"
                export TRIALS="100"
                export TIMEOUT="7200"
                export MACHINE_CONFIG="n1-highmem-8:NVIDIA_TESLA_T4:1"
                
                config_file=$(create_python_config "${exp_id}_phase1_${strategy}_raw${raw_spectral}_focused${focused_mg}")
                run_experiment "$exp_id" "phase1_features" "$config_file" "train"
                
                sleep 5  # Brief pause between experiments
            done
        done
    done
}

# ============================================================================
# PHASE 2: ENHANCED FEATURES EXPLORATION
# ============================================================================

run_phase2_enhanced_features() {
    log_info "PHASE 2: Enhanced Features Combinations"
    
    # Test different feature combinations
    local feature_combinations=(
        "true:false:false:false"   # Molecular bands only
        "false:true:false:false"    # Macro elements only
        "false:false:true:false"    # Micro elements only
        "false:false:false:true"    # Oxygen/Hydrogen only
        "true:true:false:false"     # Molecular + Macro
        "true:true:true:false"      # Molecular + Macro + Micro
        "true:true:true:true"       # All features
        "false:false:false:false"   # No enhanced features
    )
    
    for combo in "${feature_combinations[@]}"; do
        IFS=':' read -r molecular macro micro oxygen <<< "$combo"
        
        exp_id=$(generate_experiment_id)
        
        export FEATURE_STRATEGY="full_context"
        export USE_RAW_SPECTRAL="false"
        export FOCUSED_MG_FEATURES="true"
        export ENABLE_MOLECULAR_BANDS="$molecular"
        export ENABLE_MACRO_ELEMENTS="$macro"
        export ENABLE_MICRO_ELEMENTS="$micro"
        export ENABLE_OXYGEN_HYDROGEN="$oxygen"
        export ENABLE_ADVANCED_RATIOS="true"
        export ENABLE_SPECTRAL_PATTERNS="true"
        export ENABLE_INTERFERENCE_CORRECTION="true"
        export USE_DIM_REDUCTION="false"
        export USE_SAMPLE_WEIGHTS="true"
        export SAMPLE_WEIGHT_METHOD="distribution_based"
        export USE_CONCENTRATION_FEATURES="true"
        export OUTLIER_METHOD="SAM"
        export OUTLIER_THRESHOLD="0.85"
        export MAX_OUTLIER_PERCENTAGE="30"
        export ENABLE_WAVELENGTH_STD="false"
        export WAVELENGTH_INTERP_METHOD="linear"
        export MODELS_TO_TRAIN="xgboost,lightgbm"
        export CV_FOLDS="5"
        export MACHINE_CONFIG="n1-highmem-8:NVIDIA_TESLA_T4:1"
        
        config_file=$(create_python_config "${exp_id}_phase2_mol${molecular}_mac${macro}_mic${micro}_oxy${oxygen}")
        run_experiment "$exp_id" "phase2_enhanced" "$config_file" "train"
        
        sleep 5
    done
}

# ============================================================================
# PHASE 3: DIMENSIONALITY REDUCTION
# ============================================================================

run_phase3_dimension_reduction() {
    log_info "PHASE 3: Dimensionality Reduction Methods"
    
    for method in "${DIM_REDUCTION_METHODS[@]}"; do
        for components in "${DIM_REDUCTION_COMPONENTS[@]}"; do
            
            # Skip invalid combinations
            if [[ "$method" == "feature_clustering" && "$components" == "0."* ]]; then
                continue
            fi
            
            exp_id=$(generate_experiment_id)
            
            export FEATURE_STRATEGY="full_context"
            export USE_RAW_SPECTRAL="false"
            export FOCUSED_MG_FEATURES="true"
            export ENABLE_MOLECULAR_BANDS="false"
            export ENABLE_MACRO_ELEMENTS="true"
            export ENABLE_MICRO_ELEMENTS="false"
            export ENABLE_OXYGEN_HYDROGEN="false"
            export ENABLE_ADVANCED_RATIOS="true"
            export ENABLE_SPECTRAL_PATTERNS="true"
            export ENABLE_INTERFERENCE_CORRECTION="false"
            export USE_DIM_REDUCTION="true"
            export DIM_REDUCTION_METHOD="$method"
            export DIM_REDUCTION_COMPONENTS="$components"
            export USE_SAMPLE_WEIGHTS="true"
            export SAMPLE_WEIGHT_METHOD="distribution_based"
            export USE_CONCENTRATION_FEATURES="true"
            export OUTLIER_METHOD="SAM"
            export OUTLIER_THRESHOLD="0.85"
            export MAX_OUTLIER_PERCENTAGE="30"
            export ENABLE_WAVELENGTH_STD="false"
            export WAVELENGTH_INTERP_METHOD="linear"
            export MODELS_TO_TRAIN="xgboost,neural_network"
            export CV_FOLDS="5"
            export MACHINE_CONFIG="n1-highmem-8:NVIDIA_TESLA_T4:1"
            
            config_file=$(create_python_config "${exp_id}_phase3_${method}_${components}")
            run_experiment "$exp_id" "phase3_dimred" "$config_file" "train"
            
            sleep 5
        done
    done
}

# ============================================================================
# PHASE 4: SAMPLE WEIGHTING STRATEGIES
# ============================================================================

run_phase4_sample_weighting() {
    log_info "PHASE 4: Sample Weighting Methods"
    
    for weight_method in "${SAMPLE_WEIGHT_METHODS[@]}"; do
        for use_weights in "${SAMPLE_WEIGHT_OPTIONS[@]}"; do
            
            exp_id=$(generate_experiment_id)
            
            export FEATURE_STRATEGY="simple_only"
            export USE_RAW_SPECTRAL="false"
            export FOCUSED_MG_FEATURES="true"
            export ENABLE_MOLECULAR_BANDS="false"
            export ENABLE_MACRO_ELEMENTS="true"
            export ENABLE_MICRO_ELEMENTS="false"
            export ENABLE_OXYGEN_HYDROGEN="false"
            export ENABLE_ADVANCED_RATIOS="true"
            export ENABLE_SPECTRAL_PATTERNS="true"
            export ENABLE_INTERFERENCE_CORRECTION="false"
            export USE_DIM_REDUCTION="false"
            export USE_SAMPLE_WEIGHTS="$use_weights"
            export SAMPLE_WEIGHT_METHOD="$weight_method"
            export USE_CONCENTRATION_FEATURES="true"
            export OUTLIER_METHOD="SAM"
            export OUTLIER_THRESHOLD="0.85"
            export MAX_OUTLIER_PERCENTAGE="30"
            export ENABLE_WAVELENGTH_STD="false"
            export WAVELENGTH_INTERP_METHOD="linear"
            export MODELS_TO_TRAIN="xgboost,lightgbm,catboost"
            export CV_FOLDS="5"
            export MACHINE_CONFIG="n1-highmem-8:NVIDIA_TESLA_T4:1"
            
            config_file=$(create_python_config "${exp_id}_phase4_weights_${weight_method}_${use_weights}")
            run_experiment "$exp_id" "phase4_weights" "$config_file" "train"
            
            sleep 5
        done
    done
}

# ============================================================================
# PHASE 5: OUTLIER DETECTION STRATEGIES
# ============================================================================

run_phase5_outlier_detection() {
    log_info "PHASE 5: Outlier Detection Methods"
    
    for method in "${OUTLIER_METHODS[@]}"; do
        for threshold in "${OUTLIER_THRESHOLDS[@]}"; do
            for max_percent in "${MAX_OUTLIER_PERCENTAGES[@]}"; do
                
                exp_id=$(generate_experiment_id)
                
                export FEATURE_STRATEGY="simple_only"
                export USE_RAW_SPECTRAL="false"
                export FOCUSED_MG_FEATURES="true"
                export ENABLE_MOLECULAR_BANDS="false"
                export ENABLE_MACRO_ELEMENTS="true"
                export ENABLE_MICRO_ELEMENTS="false"
                export ENABLE_OXYGEN_HYDROGEN="false"
                export ENABLE_ADVANCED_RATIOS="true"
                export ENABLE_SPECTRAL_PATTERNS="true"
                export ENABLE_INTERFERENCE_CORRECTION="false"
                export USE_DIM_REDUCTION="false"
                export USE_SAMPLE_WEIGHTS="true"
                export SAMPLE_WEIGHT_METHOD="distribution_based"
                export USE_CONCENTRATION_FEATURES="true"
                export OUTLIER_METHOD="$method"
                export OUTLIER_THRESHOLD="$threshold"
                export MAX_OUTLIER_PERCENTAGE="$max_percent"
                export ENABLE_WAVELENGTH_STD="false"
                export WAVELENGTH_INTERP_METHOD="linear"
                export MODELS_TO_TRAIN="xgboost"
                export CV_FOLDS="5"
                export MACHINE_CONFIG="n1-standard-8:none:0"
                
                config_file=$(create_python_config "${exp_id}_phase5_outlier_${method}_${threshold}_${max_percent}")
                run_experiment "$exp_id" "phase5_outlier" "$config_file" "train"
                
                sleep 5
            done
        done
    done
}

# ============================================================================
# PHASE 6: HYPERPARAMETER TUNING
# ============================================================================

run_phase6_hyperparameter_tuning() {
    log_info "PHASE 6: Hyperparameter Tuning with Different Objectives"
    
    for objective in "${OBJECTIVE_FUNCTIONS[@]}"; do
        
        exp_id=$(generate_experiment_id)
        
        export FEATURE_STRATEGY="simple_only"
        export USE_RAW_SPECTRAL="false"
        export FOCUSED_MG_FEATURES="true"
        export ENABLE_MOLECULAR_BANDS="false"
        export ENABLE_MACRO_ELEMENTS="true"
        export ENABLE_MICRO_ELEMENTS="false"
        export ENABLE_OXYGEN_HYDROGEN="false"
        export ENABLE_ADVANCED_RATIOS="true"
        export ENABLE_SPECTRAL_PATTERNS="true"
        export ENABLE_INTERFERENCE_CORRECTION="false"
        export USE_DIM_REDUCTION="false"
        export USE_SAMPLE_WEIGHTS="true"
        export SAMPLE_WEIGHT_METHOD="distribution_based"
        export USE_CONCENTRATION_FEATURES="true"
        export OUTLIER_METHOD="SAM"
        export OUTLIER_THRESHOLD="0.85"
        export MAX_OUTLIER_PERCENTAGE="30"
        export ENABLE_WAVELENGTH_STD="false"
        export WAVELENGTH_INTERP_METHOD="linear"
        export MODELS_TO_TRAIN="xgboost,lightgbm,catboost,extratrees"
        export CV_FOLDS="5"
        export OBJECTIVE_FUNCTION="$objective"
        export TRIALS="300"
        export TIMEOUT="14400"
        export MACHINE_CONFIG="n1-highmem-16:NVIDIA_TESLA_T4:1"
        
        config_file=$(create_python_config "${exp_id}_phase6_tune_${objective}")
        run_experiment "$exp_id" "phase6_tuning" "$config_file" "tune"
        
        sleep 10
    done
}

# ============================================================================
# PHASE 7: AUTOGLUON ENSEMBLE LEARNING
# ============================================================================

run_phase7_autogluon() {
    log_info "PHASE 7: AutoGluon Ensemble Learning"
    
    for preset in "${AUTOGLUON_PRESETS[@]}"; do
        for stack_levels in "${AUTOGLUON_STACK_LEVELS[@]}"; do
            for bag_folds in "${AUTOGLUON_BAG_FOLDS[@]}"; do
                
                exp_id=$(generate_experiment_id)
                
                export FEATURE_STRATEGY="simple_only"
                export USE_RAW_SPECTRAL="false"
                export FOCUSED_MG_FEATURES="true"
                export ENABLE_MOLECULAR_BANDS="false"
                export ENABLE_MACRO_ELEMENTS="true"
                export ENABLE_MICRO_ELEMENTS="false"
                export ENABLE_OXYGEN_HYDROGEN="false"
                export ENABLE_ADVANCED_RATIOS="true"
                export ENABLE_SPECTRAL_PATTERNS="true"
                export ENABLE_INTERFERENCE_CORRECTION="false"
                export USE_DIM_REDUCTION="false"
                export USE_SAMPLE_WEIGHTS="true"
                export SAMPLE_WEIGHT_METHOD="improved"
                export USE_CONCENTRATION_FEATURES="true"
                export OUTLIER_METHOD="SAM"
                export OUTLIER_THRESHOLD="0.85"
                export MAX_OUTLIER_PERCENTAGE="30"
                export ENABLE_WAVELENGTH_STD="false"
                export WAVELENGTH_INTERP_METHOD="linear"
                export AUTOGLUON_PRESET="$preset"
                export AUTOGLUON_STACK_LEVELS="$stack_levels"
                export AUTOGLUON_BAG_FOLDS="$bag_folds"
                export AUTOGLUON_BAG_SETS="2"
                export AUTOGLUON_TIME_LIMIT="21600"
                export CV_FOLDS="5"
                export MACHINE_CONFIG="n1-highmem-16:none:0"
                
                config_file=$(create_python_config "${exp_id}_phase7_ag_${preset}_stack${stack_levels}_bag${bag_folds}")
                run_experiment "$exp_id" "phase7_autogluon" "$config_file" "autogluon"
                
                sleep 10
            done
        done
    done
}

# ============================================================================
# PHASE 8: BEST COMBINATIONS
# ============================================================================

run_phase8_best_combinations() {
    log_info "PHASE 8: Best Configuration Combinations"
    
    # Based on results from previous phases, run optimized combinations
    local best_configs=(
        "simple_only:false:true:false:true:false:false:true:true:false:pls:30:true:distribution_based"
        "full_context:false:true:true:true:false:false:true:true:false:pca:0.95:true:improved"
        "Mg_only:true:true:false:false:false:false:false:false:false:none:0:false:none"
    )
    
    for config in "${best_configs[@]}"; do
        IFS=':' read -r strategy raw focused mol macro micro oxy adv_ratio spectral interf dim_method dim_comp use_weight weight_method <<< "$config"
        
        exp_id=$(generate_experiment_id)
        
        export FEATURE_STRATEGY="$strategy"
        export USE_RAW_SPECTRAL="$raw"
        export FOCUSED_MG_FEATURES="$focused"
        export ENABLE_MOLECULAR_BANDS="$mol"
        export ENABLE_MACRO_ELEMENTS="$macro"
        export ENABLE_MICRO_ELEMENTS="$micro"
        export ENABLE_OXYGEN_HYDROGEN="$oxy"
        export ENABLE_ADVANCED_RATIOS="$adv_ratio"
        export ENABLE_SPECTRAL_PATTERNS="$spectral"
        export ENABLE_INTERFERENCE_CORRECTION="$interf"
        
        if [ "$dim_method" != "none" ]; then
            export USE_DIM_REDUCTION="true"
            export DIM_REDUCTION_METHOD="$dim_method"
            export DIM_REDUCTION_COMPONENTS="$dim_comp"
        else
            export USE_DIM_REDUCTION="false"
            export DIM_REDUCTION_METHOD="pca"
            export DIM_REDUCTION_COMPONENTS="30"
        fi
        
        export USE_SAMPLE_WEIGHTS="$use_weight"
        export SAMPLE_WEIGHT_METHOD="$weight_method"
        export USE_CONCENTRATION_FEATURES="true"
        export OUTLIER_METHOD="SAM"
        export OUTLIER_THRESHOLD="0.85"
        export MAX_OUTLIER_PERCENTAGE="30"
        export ENABLE_WAVELENGTH_STD="false"
        export WAVELENGTH_INTERP_METHOD="linear"
        export MODELS_TO_TRAIN="xgboost,lightgbm,catboost,neural_network"
        export CV_FOLDS="5"
        export TRIALS="500"
        export TIMEOUT="28800"
        export MACHINE_CONFIG="n1-highmem-32:NVIDIA_TESLA_V100:2"
        
        config_file=$(create_python_config "${exp_id}_phase8_best_${strategy}")
        
        # Run with optimize-models for best results
        export TRAINING_MODE="optimize-models"
        run_experiment "$exp_id" "phase8_best" "$config_file" "optimize-models"
        
        sleep 10
    done
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_info "Starting Comprehensive Experiment Automation"
    log_info "Project: $PROJECT_ID | Region: $REGION"
    
    # Check prerequisites
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Please install Google Cloud SDK."
        exit 1
    fi
    
    # Setup GCP if needed
    log_info "Setting up GCP project..."
    ./deploy/gcp_deploy.sh setup 2>&1 | grep -v "\[CONFIG\]" || true
    
    # Build container if needed
    log_info "Building container image..."
    ./deploy/gcp_deploy.sh build-local 2>&1 | grep -v "\[CONFIG\]" || true
    
    # Upload data if needed
    log_info "Uploading training data..."
    ./deploy/gcp_deploy.sh upload-data 2>&1 | grep -v "\[CONFIG\]" || true
    
    # Run experiments based on command line argument
    case "${1:-all}" in
        phase1)
            run_phase1_feature_strategies
            ;;
        phase2)
            run_phase2_enhanced_features
            ;;
        phase3)
            run_phase3_dimension_reduction
            ;;
        phase4)
            run_phase4_sample_weighting
            ;;
        phase5)
            run_phase5_outlier_detection
            ;;
        phase6)
            run_phase6_hyperparameter_tuning
            ;;
        phase7)
            run_phase7_autogluon
            ;;
        phase8)
            run_phase8_best_combinations
            ;;
        quick)
            # Quick high-impact experiments only
            log_info "Running quick high-impact experiments"
            export MACHINE_CONFIG="n1-highmem-16:NVIDIA_TESLA_T4:1"
            
            # Best known configuration
            exp_id=$(generate_experiment_id)
            export FEATURE_STRATEGY="simple_only"
            export USE_RAW_SPECTRAL="false"
            export FOCUSED_MG_FEATURES="true"
            export ENABLE_MOLECULAR_BANDS="false"
            export ENABLE_MACRO_ELEMENTS="true"
            export ENABLE_MICRO_ELEMENTS="false"
            export ENABLE_OXYGEN_HYDROGEN="false"
            export ENABLE_ADVANCED_RATIOS="true"
            export ENABLE_SPECTRAL_PATTERNS="true"
            export ENABLE_INTERFERENCE_CORRECTION="false"
            export USE_DIM_REDUCTION="false"
            export USE_SAMPLE_WEIGHTS="true"
            export SAMPLE_WEIGHT_METHOD="distribution_based"
            export USE_CONCENTRATION_FEATURES="true"
            export OUTLIER_METHOD="SAM"
            export OUTLIER_THRESHOLD="0.85"
            export MAX_OUTLIER_PERCENTAGE="30"
            export AUTOGLUON_PRESET="best_quality"
            export AUTOGLUON_STACK_LEVELS="2"
            export AUTOGLUON_BAG_FOLDS="5"
            export AUTOGLUON_TIME_LIMIT="14400"
            
            config_file=$(create_python_config "${exp_id}_quick_best")
            run_experiment "$exp_id" "quick" "$config_file" "autogluon"
            ;;
        all)
            # Run all phases sequentially
            run_phase1_feature_strategies
            run_phase2_enhanced_features
            run_phase3_dimension_reduction
            run_phase4_sample_weighting
            run_phase5_outlier_detection
            run_phase6_hyperparameter_tuning
            run_phase7_autogluon
            run_phase8_best_combinations
            ;;
        *)
            echo "Usage: $0 [phase1|phase2|phase3|phase4|phase5|phase6|phase7|phase8|quick|all]"
            echo ""
            echo "Phases:"
            echo "  phase1 - Feature engineering strategies"
            echo "  phase2 - Enhanced features exploration"
            echo "  phase3 - Dimensionality reduction methods"
            echo "  phase4 - Sample weighting strategies"
            echo "  phase5 - Outlier detection methods"
            echo "  phase6 - Hyperparameter tuning objectives"
            echo "  phase7 - AutoGluon ensemble configurations"
            echo "  phase8 - Best combination runs"
            echo "  quick  - Quick high-impact experiment"
            echo "  all    - Run all phases (default)"
            exit 1
            ;;
    esac
    
    log_info "Experiment automation completed!"
    log_info "Results tracker: $EXPERIMENT_TRACKER"
    log_info "Logs directory: $LOGS_DIR"
    log_info "Configs directory: $CONFIGS_DIR"
    
    # Generate summary report
    ./generate_experiment_report.sh
}

# Run main function
main "$@"