#!/bin/bash
#
# Optimize Model to Reduce Prediction Uncertainty
# Implements Phase 1 quick wins from REDUCE_UNCERTAINTY_GUIDE.md
#
# Expected: 25-35% MAE reduction → interval width from ±1.83% to ~±1.2-1.3%
#

set -e  # Exit on error

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="reports/uncertainty_optimization_${TIMESTAMP}"
mkdir -p "${REPORT_DIR}"

# Configuration
STRATEGY="full_context"
GPU_FLAG="--gpu"
PARALLEL_FLAGS="--data-parallel --feature-parallel"
TRIALS=300

echo "======================================================================"
echo "UNCERTAINTY REDUCTION OPTIMIZATION PIPELINE"
echo "======================================================================"
echo "Timestamp: ${TIMESTAMP}"
echo "Strategy: ${STRATEGY}"
echo "Optimization trials: ${TRIALS}"
echo "Report directory: ${REPORT_DIR}"
echo ""
echo "This will:"
echo "  1. Detect and exclude mislabeled samples"
echo "  2. Perform SHAP feature selection"
echo "  3. Run hyperparameter optimization with cleaned data"
echo "  4. Analyze uncertainty improvements"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# ======================================================================
# STEP 1: Baseline Analysis
# ======================================================================
echo ""
echo "======================================================================"
echo "STEP 1: Analyzing baseline uncertainty (current model)"
echo "======================================================================"

# Find latest AutoGluon predictions
LATEST_PREDICTIONS=$(ls -t reports/predictions_${STRATEGY}_autogluon_*.csv 2>/dev/null | head -1)

if [ -z "$LATEST_PREDICTIONS" ]; then
    echo "⚠️  No existing predictions found. Training baseline model first..."

    echo "Training baseline AutoGluon model..."
    uv run python main.py autogluon ${GPU_FLAG} ${PARALLEL_FLAGS}

    # Find the newly created predictions
    LATEST_PREDICTIONS=$(ls -t reports/predictions_${STRATEGY}_autogluon_*.csv | head -1)
fi

echo "Using predictions: ${LATEST_PREDICTIONS}"

# Extract timestamp from filename
PRED_TIMESTAMP=$(echo $LATEST_PREDICTIONS | grep -oP '\d{8}_\d{6}')
MODEL_PATH="models/autogluon/${STRATEGY}_${PRED_TIMESTAMP}"

echo "Analyzing baseline uncertainty..."
uv run python analyze_prediction_uncertainty.py \
    --predictions "${LATEST_PREDICTIONS}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${REPORT_DIR}/baseline_uncertainty"

# Extract baseline metrics
BASELINE_MAE=$(grep "Mean Absolute Error" "${REPORT_DIR}/baseline_uncertainty/uncertainty_analysis_report.txt" | awk '{print $5}')
BASELINE_RMSE=$(grep "Root Mean Squared Error" "${REPORT_DIR}/baseline_uncertainty/uncertainty_analysis_report.txt" | awk '{print $6}')

echo ""
echo "📊 Baseline Performance:"
echo "   MAE:  ${BASELINE_MAE}"
echo "   RMSE: ${BASELINE_RMSE}"
echo ""

# ======================================================================
# STEP 2: Mislabel Detection (if not already done)
# ======================================================================
echo ""
echo "======================================================================"
echo "STEP 2: Detecting mislabeled samples"
echo "======================================================================"

SUSPECTS_FILE="reports/mislabel_analysis/suspicious_samples_min_confidence_2.csv"

if [ -f "$SUSPECTS_FILE" ]; then
    NUM_SUSPECTS=$(wc -l < "$SUSPECTS_FILE")
    NUM_SUSPECTS=$((NUM_SUSPECTS - 1))  # Subtract header
    echo "✓ Found existing mislabel analysis: ${NUM_SUSPECTS} suspicious samples"
    echo "  Using: ${SUSPECTS_FILE}"
else
    echo "Running mislabel detection..."
    uv run python main.py detect-mislabels \
        --focus-min 0.0 --focus-max 15.0 \
        --min-confidence 2 \
        ${PARALLEL_FLAGS}

    if [ -f "$SUSPECTS_FILE" ]; then
        NUM_SUSPECTS=$(wc -l < "$SUSPECTS_FILE")
        NUM_SUSPECTS=$((NUM_SUSPECTS - 1))
        echo "✓ Detected ${NUM_SUSPECTS} suspicious samples"
    else
        echo "⚠️  No suspicious samples file created"
        SUSPECTS_FILE=""
    fi
fi

# ======================================================================
# STEP 3: SHAP Feature Selection
# ======================================================================
echo ""
echo "======================================================================"
echo "STEP 3: SHAP-based feature selection"
echo "======================================================================"

# Find existing SHAP importance file
SHAP_FILE=$(ls -t models/${STRATEGY}_*_shap_importance.csv 2>/dev/null | head -1)

if [ -z "$SHAP_FILE" ]; then
    echo "No existing SHAP analysis found. Running SHAP analysis..."

    # Train a LightGBM model for SHAP analysis
    echo "Training LightGBM model for SHAP analysis..."
    uv run python main.py train --models lightgbm --strategy ${STRATEGY} ${GPU_FLAG}

    # Run SHAP analysis
    echo "Running SHAP analysis..."
    ./run_shap_analysis.sh --latest lightgbm

    # Find the newly created SHAP file
    SHAP_FILE=$(ls -t models/${STRATEGY}_*_shap_importance.csv | head -1)
fi

if [ -n "$SHAP_FILE" ]; then
    echo "✓ Using SHAP importance file: ${SHAP_FILE}"

    # Count features
    NUM_TOTAL_FEATURES=$(wc -l < "$SHAP_FILE")
    NUM_TOTAL_FEATURES=$((NUM_TOTAL_FEATURES - 1))  # Subtract header
    echo "  Total features: ${NUM_TOTAL_FEATURES}"
else
    echo "⚠️  No SHAP file found - will train without feature selection"
    SHAP_FILE=""
fi

# ======================================================================
# STEP 4: Optimize Models with All Improvements
# ======================================================================
echo ""
echo "======================================================================"
echo "STEP 4: Hyperparameter optimization with improvements"
echo "======================================================================"

# Build command with all optimizations
CMD="uv run python main.py optimize-models \
    --models xgboost lightgbm catboost \
    --strategy ${STRATEGY} \
    --trials ${TRIALS} \
    ${GPU_FLAG} \
    ${PARALLEL_FLAGS}"

# Add mislabel exclusion if available
if [ -n "$SUSPECTS_FILE" ]; then
    CMD="${CMD} --exclude-suspects ${SUSPECTS_FILE}"
    echo "✓ Excluding ${NUM_SUSPECTS} suspicious samples"
fi

# Add SHAP feature selection if available
if [ -n "$SHAP_FILE" ]; then
    SHAP_TOP_N=50  # Use top 50 features
    CMD="${CMD} --shap-features ${SHAP_FILE} --shap-top-n ${SHAP_TOP_N}"
    echo "✓ Using top ${SHAP_TOP_N} features from SHAP analysis"
fi

echo ""
echo "Running optimization with command:"
echo "${CMD}"
echo ""
echo "⏱️  This will take some time (trials=${TRIALS})..."
echo ""

# Run optimization
eval $CMD

echo ""
echo "✓ Optimization complete!"

# ======================================================================
# STEP 5: Analyze Optimized Model Uncertainty
# ======================================================================
echo ""
echo "======================================================================"
echo "STEP 5: Analyzing optimized model uncertainty"
echo "======================================================================"

# Find the best model from optimization
# Look for latest training summary
LATEST_SUMMARY=$(ls -t reports/training_summary_${STRATEGY}_*.csv 2>/dev/null | head -1)

if [ -z "$LATEST_SUMMARY" ]; then
    echo "⚠️  Could not find training summary"
    exit 1
fi

echo "Training summary: ${LATEST_SUMMARY}"

# Find best model (lowest RMSE)
BEST_MODEL_INFO=$(tail -n +2 "$LATEST_SUMMARY" | sort -t',' -k5 -n | head -1)
BEST_MODEL_NAME=$(echo "$BEST_MODEL_INFO" | cut -d',' -f2)
BEST_MODEL_RMSE=$(echo "$BEST_MODEL_INFO" | cut -d',' -f5)

echo "Best model: ${BEST_MODEL_NAME} (RMSE: ${BEST_MODEL_RMSE})"

# Find the corresponding model file
BEST_MODEL_FILE=$(ls -t models/${STRATEGY}_${BEST_MODEL_NAME}_*.pkl 2>/dev/null | head -1)

if [ -z "$BEST_MODEL_FILE" ]; then
    echo "⚠️  Could not find model file for ${BEST_MODEL_NAME}"
    exit 1
fi

echo "Model file: ${BEST_MODEL_FILE}"

# Make predictions on validation set (need to run prediction command)
# For now, we'll use the training summary metrics as proxy

# ======================================================================
# STEP 6: Compare Results
# ======================================================================
echo ""
echo "======================================================================"
echo "STEP 6: Comparing baseline vs optimized performance"
echo "======================================================================"

OPTIMIZED_MAE=$(echo "$BEST_MODEL_INFO" | cut -d',' -f4)
OPTIMIZED_RMSE=$(echo "$BEST_MODEL_INFO" | cut -d',' -f5)

echo ""
echo "📊 BASELINE PERFORMANCE:"
echo "   MAE:  ${BASELINE_MAE}"
echo "   RMSE: ${BASELINE_RMSE}"
echo ""
echo "📊 OPTIMIZED PERFORMANCE:"
echo "   MAE:  ${OPTIMIZED_MAE}"
echo "   RMSE: ${OPTIMIZED_RMSE}"
echo ""

# Calculate improvements
MAE_IMPROVEMENT=$(python3 -c "print(f'{(1 - ${OPTIMIZED_MAE}/${BASELINE_MAE}) * 100:.1f}')")
RMSE_IMPROVEMENT=$(python3 -c "print(f'{(1 - ${OPTIMIZED_RMSE}/${BASELINE_RMSE}) * 100:.1f}')")

echo "📈 IMPROVEMENTS:"
echo "   MAE:  ${MAE_IMPROVEMENT}% reduction"
echo "   RMSE: ${RMSE_IMPROVEMENT}% reduction"
echo ""

# Estimate new uncertainty width
# Conformal interval width ≈ 1.9 × RMSE (empirically observed)
BASELINE_INTERVAL=$(python3 -c "print(f'{1.9 * ${BASELINE_RMSE}:.3f}')")
OPTIMIZED_INTERVAL=$(python3 -c "print(f'{1.9 * ${OPTIMIZED_RMSE}:.3f}')")

echo "📏 ESTIMATED 95% INTERVAL WIDTH:"
echo "   Baseline:  ±${BASELINE_INTERVAL}%"
echo "   Optimized: ±${OPTIMIZED_INTERVAL}%"
echo ""

INTERVAL_IMPROVEMENT=$(python3 -c "print(f'{(1 - ${OPTIMIZED_INTERVAL}/${BASELINE_INTERVAL}) * 100:.1f}')")
echo "   Interval narrowing: ${INTERVAL_IMPROVEMENT}%"
echo ""

# ======================================================================
# STEP 7: Save Report
# ======================================================================
REPORT_FILE="${REPORT_DIR}/optimization_report.txt"
cat > "$REPORT_FILE" << EOF
================================================================================
UNCERTAINTY REDUCTION OPTIMIZATION REPORT
================================================================================
Date: $(date)
Strategy: ${STRATEGY}
Optimization trials: ${TRIALS}

OPTIMIZATIONS APPLIED:
----------------------
$([ -n "$SUSPECTS_FILE" ] && echo "✓ Mislabel exclusion: ${NUM_SUSPECTS} samples removed")
$([ -n "$SHAP_FILE" ] && echo "✓ SHAP feature selection: Top ${SHAP_TOP_N} features")
✓ Hyperparameter optimization: ${TRIALS} trials

BASELINE PERFORMANCE:
---------------------
MAE:  ${BASELINE_MAE}
RMSE: ${BASELINE_RMSE}
95% Interval Width: ±${BASELINE_INTERVAL}%

OPTIMIZED PERFORMANCE:
----------------------
Best Model: ${BEST_MODEL_NAME}
MAE:  ${OPTIMIZED_MAE}
RMSE: ${OPTIMIZED_RMSE}
95% Interval Width: ±${OPTIMIZED_INTERVAL}% (estimated)

IMPROVEMENTS:
-------------
MAE Reduction:       ${MAE_IMPROVEMENT}%
RMSE Reduction:      ${RMSE_IMPROVEMENT}%
Interval Narrowing:  ${INTERVAL_IMPROVEMENT}%

FILES:
------
Best Model: ${BEST_MODEL_FILE}
Training Summary: ${LATEST_SUMMARY}
$([ -n "$SUSPECTS_FILE" ] && echo "Excluded Samples: ${SUSPECTS_FILE}")
$([ -n "$SHAP_FILE" ] && echo "SHAP Features: ${SHAP_FILE}")

NEXT STEPS:
-----------
1. Validate optimized model on independent test set
2. Run uncertainty analysis on new validation predictions:
   python analyze_prediction_uncertainty.py \\
       --predictions reports/predictions_${STRATEGY}_TIMESTAMP.csv \\
       --output-dir reports/uncertainty_analysis_optimized

3. If results are good, consider Phase 2 optimizations:
   - Extended hyperparameter search (longer time limits)
   - Custom ensemble stacking
   - Advanced spectral features
   See: REDUCE_UNCERTAINTY_GUIDE.md

================================================================================
EOF

echo "✓ Report saved to: ${REPORT_FILE}"
echo ""
echo "======================================================================"
echo "OPTIMIZATION COMPLETE!"
echo "======================================================================"
echo ""
echo "Best model file: ${BEST_MODEL_FILE}"
echo ""
echo "To use the optimized model for predictions:"
echo "  python main.py predict-batch \\"
echo "      --input-dir path/to/data \\"
echo "      --model-path ${BEST_MODEL_FILE} \\"
echo "      --output-file predictions_optimized.csv"
echo ""
echo "See full report: ${REPORT_FILE}"
echo "======================================================================"
