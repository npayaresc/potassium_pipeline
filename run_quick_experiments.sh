#!/bin/bash

# Quick High-Impact Experiments Script
# Runs the most promising configurations based on pipeline analysis

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-mapana-ai-models}"
REGION="${REGION:-us-central1}"
BUCKET_NAME="${BUCKET_NAME:-${PROJECT_ID}-magnesium-data}"
RESULTS_DIR="./experiment_results"
LOGS_DIR="./experiment_logs"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# HIGH-IMPACT EXPERIMENTS
# ============================================================================

run_experiment() {
    local exp_name=$1
    local training_mode=$2
    local strategy=$3
    local models=$4
    local use_gpu=$5
    local additional_params=$6
    
    log_info "Starting experiment: $exp_name"
    log_info "  Mode: $training_mode"
    log_info "  Strategy: $strategy"
    log_info "  Models: $models"
    log_info "  GPU: $use_gpu"
    
    export TRAINING_MODE="$training_mode"
    export STRATEGY="$strategy"
    export MODELS="$models"
    export USE_GPU="$use_gpu"
    
    # Parse additional parameters
    if [ -n "$additional_params" ]; then
        eval "$additional_params"
    fi
    
    # Log file for this experiment
    local log_file="$LOGS_DIR/${exp_name}_$(date +%Y%m%d_%H%M%S).log"
    
    # Run the experiment
    log_info "Executing: ./deploy/gcp_deploy.sh $training_mode"
    if ./deploy/gcp_deploy.sh "$training_mode" > "$log_file" 2>&1; then
        log_info "✓ Experiment $exp_name completed successfully"
        echo "$exp_name,$(date),SUCCESS" >> "$RESULTS_DIR/quick_tracker.csv"
    else
        log_error "✗ Experiment $exp_name failed - check $log_file"
        echo "$exp_name,$(date),FAILED" >> "$RESULTS_DIR/quick_tracker.csv"
    fi
    
    sleep 5
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_info "Starting Quick High-Impact Experiments"
    log_info "Project: $PROJECT_ID | Region: $REGION"
    
    # Initialize tracker
    if [ ! -f "$RESULTS_DIR/quick_tracker.csv" ]; then
        echo "experiment,timestamp,status" > "$RESULTS_DIR/quick_tracker.csv"
    fi
    
    # Check prerequisites
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Please install Google Cloud SDK."
        exit 1
    fi
    
    log_info "Setting up GCP environment..."
    ./deploy/gcp_deploy.sh setup 2>&1 | grep -v "\[CONFIG\]" || true
    ./deploy/gcp_deploy.sh build-local 2>&1 | grep -v "\[CONFIG\]" || true
    ./deploy/gcp_deploy.sh upload-data 2>&1 | grep -v "\[CONFIG\]" || true
    
    # ========================================================================
    # EXPERIMENT 1: Best AutoGluon Configuration
    # Based on best historical performance
    # ========================================================================
    
    run_experiment \
        "exp1_autogluon_best" \
        "autogluon" \
        "simple_only" \
        "" \
        "false" \
        "export MACHINE_TYPE=n1-highmem-16; export TIMEOUT=21600"
    
    # ========================================================================
    # EXPERIMENT 2: Optimized XGBoost
    # High-performing single model
    # ========================================================================
    
    run_experiment \
        "exp2_xgboost_optimized" \
        "optimize-xgboost" \
        "full_context" \
        "" \
        "true" \
        "export TRIALS=500; export MACHINE_TYPE=n1-highmem-8; export ACCELERATOR_TYPE=NVIDIA_TESLA_T4; export ACCELERATOR_COUNT=1"
    
    # ========================================================================
    # EXPERIMENT 3: Multi-Model Optimization
    # Best ensemble of gradient boosting models
    # ========================================================================
    
    run_experiment \
        "exp3_multimodel_opt" \
        "optimize-models" \
        "simple_only" \
        "xgboost,lightgbm,catboost" \
        "true" \
        "export TRIALS=400; export MACHINE_TYPE=n1-highmem-8; export ACCELERATOR_TYPE=NVIDIA_TESLA_T4"
    
    # ========================================================================
    # EXPERIMENT 4: Neural Network with PLS
    # Advanced dimension reduction + neural network
    # ========================================================================
    
    run_experiment \
        "exp4_nn_pls" \
        "train" \
        "full_context" \
        "neural_network" \
        "true" \
        "export USE_DIM_REDUCTION=true; export DIM_REDUCTION_METHOD=pls; export DIM_REDUCTION_COMPONENTS=30; export MACHINE_TYPE=n1-highmem-8; export ACCELERATOR_TYPE=NVIDIA_TESLA_T4"
    
    # ========================================================================
    # EXPERIMENT 5: Raw Spectral with XGBoost
    # Direct spectral intensities
    # ========================================================================
    
    run_experiment \
        "exp5_raw_spectral_xgb" \
        "train" \
        "full_context" \
        "xgboost" \
        "true" \
        "export USE_RAW_SPECTRAL=true; export MACHINE_TYPE=n1-standard-8"
    
    # ========================================================================
    # EXPERIMENT 6: Best Feature Combination
    # Optimized feature engineering settings
    # ========================================================================
    
    run_experiment \
        "exp6_best_features" \
        "train" \
        "simple_only" \
        "xgboost,lightgbm" \
        "true" \
        "export ENABLE_MACRO_ELEMENTS=true; export ENABLE_ADVANCED_RATIOS=true; export ENABLE_SPECTRAL_PATTERNS=true; export USE_SAMPLE_WEIGHTS=true; export SAMPLE_WEIGHT_METHOD=distribution_based; export MACHINE_TYPE=n1-highmem-8"
    
    # ========================================================================
    # EXPERIMENT 7: Hyperparameter Tuning with MAPE Focus
    # Optimize for percentage error
    # ========================================================================
    
    run_experiment \
        "exp7_tune_mape" \
        "tune" \
        "simple_only" \
        "" \
        "true" \
        "export OBJECTIVE_FUNCTION=mape_focused; export TRIALS=300; export MACHINE_TYPE=n1-highmem-8; export ACCELERATOR_TYPE=NVIDIA_TESLA_T4"
    
    # ========================================================================
    # EXPERIMENT 8: Extended AutoGluon with Stacking
    # Maximum ensemble complexity
    # ========================================================================
    
    run_experiment \
        "exp8_autogluon_stack" \
        "autogluon" \
        "full_context" \
        "" \
        "false" \
        "export AUTOGLUON_PRESET=best_quality; export AUTOGLUON_STACK_LEVELS=2; export AUTOGLUON_BAG_FOLDS=5; export TIMEOUT=28800; export MACHINE_TYPE=n1-highmem-16"
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    log_info "========================================="
    log_info "Quick experiments completed!"
    log_info "========================================="
    
    # Show summary
    echo ""
    echo "Experiment Summary:"
    echo "==================="
    cat "$RESULTS_DIR/quick_tracker.csv" | column -t -s','
    
    # Count results
    total=$(tail -n +2 "$RESULTS_DIR/quick_tracker.csv" | wc -l)
    successful=$(grep ",SUCCESS" "$RESULTS_DIR/quick_tracker.csv" | wc -l)
    failed=$(grep ",FAILED" "$RESULTS_DIR/quick_tracker.csv" | wc -l)
    
    echo ""
    echo "Statistics:"
    echo "  Total: $total"
    echo "  Successful: $successful"
    echo "  Failed: $failed"
    
    if [ $successful -gt 0 ]; then
        echo ""
        log_info "Check GCS for results: gs://${BUCKET_NAME}/magnesium-pipeline/reports/"
        log_info "Download results: gsutil -m cp -r gs://${BUCKET_NAME}/magnesium-pipeline/reports/* ./cloud_results/"
    fi
}

# Handle command line arguments
case "${1:-run}" in
    run)
        main
        ;;
    status)
        # Check status of running jobs
        log_info "Checking Vertex AI job status..."
        gcloud ai custom-jobs list --region="$REGION" --filter="state=JOB_STATE_RUNNING" --format="table(displayName,state,createTime)"
        ;;
    results)
        # Download and analyze results
        log_info "Downloading results from GCS..."
        mkdir -p ./cloud_results
        gsutil -m cp -r "gs://${BUCKET_NAME}/magnesium-pipeline/reports/*" ./cloud_results/ 2>/dev/null || true
        
        if [ -f "./generate_experiment_report.sh" ]; then
            ./generate_experiment_report.sh
        fi
        ;;
    help)
        echo "Usage: $0 [run|status|results|help]"
        echo ""
        echo "Commands:"
        echo "  run     - Run quick high-impact experiments (default)"
        echo "  status  - Check status of running Vertex AI jobs"
        echo "  results - Download and analyze results"
        echo "  help    - Show this help message"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac