#!/bin/bash
# Generic SHAP analysis script for any trained model
#
# Usage:
#   ./run_shap_analysis.sh <model_path>
#   ./run_shap_analysis.sh --latest <pattern>
#   ./run_shap_analysis.sh --latest ridge
#   ./run_shap_analysis.sh --latest catboost
#   ./run_shap_analysis.sh --latest autogluon

set -e  # Exit on error

# Default parameters
MAX_SAMPLES=500
BACKGROUND_SAMPLES=50
EXPLAIN_SAMPLES=200
TOP_N=30

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Function to find latest model matching pattern
find_latest_model() {
    local pattern=$1
    local model_path=""

    # Try different patterns
    if [ -f "$pattern" ]; then
        # Direct path provided
        model_path="$pattern"
    elif [ "$pattern" == "latest" ] || [ -z "$pattern" ]; then
        # Find latest model
        model_path=$(ls -t models/*.pkl 2>/dev/null | grep -v "autogluon" | head -1)
    else
        # Pattern matching (e.g., "ridge", "catboost", "xgboost")
        model_path=$(ls -t models/*${pattern}*.pkl 2>/dev/null | head -1)
    fi

    echo "$model_path"
}

# Function to extract strategy from model path
extract_strategy() {
    local model_path=$1
    local filename=$(basename "$model_path")

    # Extract strategy from filename patterns:
    # Regular models: <strategy>_<model>_<timestamp>.pkl
    #   simple_only_ridge_20251006_024858.pkl -> simple_only
    #   full_context_xgboost_20251006_000443.pkl -> full_context
    #   K_only_catboost_20251006_123456.pkl -> K_only
    #
    # Optimized models: optimized_<model>_<strategy>_<timestamp>.pkl
    #   optimized_lightgbm_full_context_20251006_025844.pkl -> full_context
    #   optimized_xgboost_simple_only_20251006_000443.pkl -> simple_only

    # Match strategy anywhere in filename
    if [[ $filename =~ (K_only|simple_only|full_context) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        # Default to simple_only if can't determine
        print_warning "Could not determine strategy from filename, defaulting to simple_only" >&2
        echo "simple_only"
    fi
}

# Parse command line arguments
if [ "$1" == "--latest" ]; then
    MODEL_PATH=$(find_latest_model "$2")
elif [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    cat << EOF
SHAP Analysis Script - Analyze feature importance for any trained model

Usage:
  $0 <model_path>                    # Analyze specific model
  $0 --latest [pattern]              # Analyze latest model matching pattern
  $0 -h, --help                      # Show this help

Examples:
  $0 models/simple_only_ridge_20251006_024858.pkl
  $0 --latest ridge
  $0 --latest catboost
  $0 --latest xgboost
  $0 --latest                        # Latest non-autogluon model

Parameters (edit script to change):
  MAX_SAMPLES=$MAX_SAMPLES           # Samples for data extraction
  BACKGROUND_SAMPLES=$BACKGROUND_SAMPLES  # Background samples for SHAP
  EXPLAIN_SAMPLES=$EXPLAIN_SAMPLES        # Samples to explain
  TOP_N=$TOP_N                       # Top features to display

Output:
  - SHAP importance CSV in models/
  - Visualizations in models/shap_analysis/
  - Feature names loaded from .feature_names.json

EOF
    exit 0
else
    MODEL_PATH=$(find_latest_model "$1")
fi

# Validate model path
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    print_error "Model not found: $MODEL_PATH"
    echo ""
    echo "Available models:"
    ls -lht models/*.pkl 2>/dev/null | head -10 || echo "  No models found"
    exit 1
fi

print_header "SHAP FEATURE IMPORTANCE ANALYSIS"

print_info "Model: $MODEL_PATH"

# Check if feature_names.json exists
FEATURE_NAMES_FILE="${MODEL_PATH%.pkl}.feature_names.json"
if [ -f "$FEATURE_NAMES_FILE" ]; then
    print_success "Found feature names file: $(basename $FEATURE_NAMES_FILE)"
    # Extract feature count
    FEATURE_COUNT=$(python3 -c "import json; f=json.load(open('$FEATURE_NAMES_FILE')); print(f['feature_count'])")
    print_info "Features: $FEATURE_COUNT"
else
    print_warning "No feature names file found (will use generic names)"
fi

# Extract strategy from model path
STRATEGY=$(extract_strategy "$MODEL_PATH")
print_info "Strategy: $STRATEGY"

# Data file for SHAP
DATA_FILE="data/training_data_${STRATEGY}_for_shap_${MAX_SAMPLES}.csv"

# Step 1: Extract training data if not exists or is old
print_header "Step 1: Extract Training Data"

if [ -f "$DATA_FILE" ]; then
    # Check if data file is older than model
    if [ "$DATA_FILE" -ot "$MODEL_PATH" ]; then
        print_warning "Data file is older than model, regenerating..."
        EXTRACT_DATA=true
    else
        print_success "Using existing data file: $DATA_FILE"
        EXTRACT_DATA=false
    fi
else
    print_info "Data file not found, extracting..."
    EXTRACT_DATA=true
fi

if [ "$EXTRACT_DATA" = true ]; then
    print_info "Extracting $MAX_SAMPLES samples of training data (strategy: $STRATEGY)..."

    # Pass feature names file if it exists (to match feature selection)
    if [ -f "$FEATURE_NAMES_FILE" ]; then
        uv run python extract_training_data_for_shap.py \
            --strategy "$STRATEGY" \
            --max-samples $MAX_SAMPLES \
            --output "$DATA_FILE" \
            --feature-names-file "$FEATURE_NAMES_FILE"
    else
        uv run python extract_training_data_for_shap.py \
            --strategy "$STRATEGY" \
            --max-samples $MAX_SAMPLES \
            --output "$DATA_FILE"
    fi

    if [ $? -eq 0 ]; then
        print_success "Data extraction complete"
    else
        print_error "Data extraction failed"
        exit 1
    fi
fi

# Step 2: Run SHAP analysis
print_header "Step 2: Run SHAP Analysis"

print_info "Parameters:"
echo "  - Background samples: $BACKGROUND_SAMPLES"
echo "  - Explain samples: $EXPLAIN_SAMPLES"
echo "  - Top features: $TOP_N"

uv run python analyze_feature_importance.py "$MODEL_PATH" \
    --data "$DATA_FILE" \
    --top-n $TOP_N \
    --background-samples $BACKGROUND_SAMPLES \
    --explain-samples $EXPLAIN_SAMPLES

if [ $? -eq 0 ]; then
    print_header "Analysis Complete!"

    MODEL_STEM=$(basename "$MODEL_PATH" .pkl)

    print_success "Outputs generated:"
    echo "  - SHAP importance: models/${MODEL_STEM}_shap_importance.csv"
    echo "  - Visualizations: models/shap_analysis/"

    # Show what visualizations were created
    if [ -d "models/shap_analysis" ]; then
        echo ""
        print_info "Visualization files:"
        ls -lh models/shap_analysis/${MODEL_STEM}_shap*.png 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'
    fi

    echo ""
    print_success "SHAP analysis completed successfully!"
else
    print_error "SHAP analysis failed"
    exit 1
fi
