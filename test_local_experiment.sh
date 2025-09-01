#!/bin/bash

# Test Local Experiment Script
# Tests experiments locally before deploying to cloud

set -e

# Configuration
RESULTS_DIR="./local_test_results"
LOGS_DIR="./local_test_logs"
CONFIGS_DIR="./local_test_configs"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR" "$CONFIGS_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOGS_DIR/test.log"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOGS_DIR/test.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOGS_DIR/test.log"
}

# Test function
test_experiment() {
    local exp_name=$1
    local training_mode=$2
    local strategy=$3
    local models=$4
    local use_gpu=$5
    
    log_info "Testing experiment: $exp_name"
    log_info "  Mode: $training_mode"
    log_info "  Strategy: $strategy"
    log_info "  Models: $models"
    log_info "  GPU: $use_gpu"
    
    # Build command
    local cmd="python main.py $training_mode"
    
    if [ -n "$models" ]; then
        cmd="$cmd --models $models"
    fi
    
    if [ -n "$strategy" ]; then
        cmd="$cmd --strategy $strategy"
    fi
    
    if [ "$use_gpu" = "true" ]; then
        cmd="$cmd --gpu"
    fi
    
    # Create log file
    local log_file="$LOGS_DIR/${exp_name}_$(date +%Y%m%d_%H%M%S).log"
    
    log_info "Executing: $cmd"
    
    # Run the command
    if timeout 300 $cmd > "$log_file" 2>&1; then
        log_info "✓ Test $exp_name completed successfully"
        return 0
    else
        log_error "✗ Test $exp_name failed - check $log_file"
        echo "Last 10 lines of error log:"
        tail -10 "$log_file"
        return 1
    fi
}

# Main function
main() {
    log_info "Starting Local Experiment Tests"
    
    # Test basic training
    test_experiment "basic_train" "train" "simple_only" "xgboost" "false"
    
    if [ $? -eq 0 ]; then
        log_info "Basic test passed! Pipeline is working correctly."
        
        # Test with GPU (if available)
        if command -v nvidia-smi &> /dev/null; then
            log_info "GPU detected, testing GPU mode..."
            test_experiment "gpu_train" "train" "simple_only" "xgboost" "true"
        fi
        
        # Test AutoGluon
        log_info "Testing AutoGluon..."
        test_experiment "autogluon_test" "autogluon" "simple_only" "" "false"
        
        log_info "All local tests completed!"
        log_info "You can now run the cloud experiments with confidence."
        
    else
        log_error "Basic test failed. Check your environment and dependencies."
        exit 1
    fi
}

# Run main function
main "$@"