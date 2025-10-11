#!/bin/bash
# Quick SHAP analysis on latest CatBoost model using existing cleansed data

MODEL_PATH="models/optimized_catboost_simple_only_20251006_000443.pkl"

# Extract 500 samples of training data first (much faster than full dataset)
echo "Extracting training data for SHAP..."
uv run python extract_training_data_for_shap.py --strategy simple_only --max-samples 500 --output data/training_data_simple_only_for_shap_500.csv

# Run SHAP analysis
echo "Running SHAP feature importance analysis..."
uv run python analyze_feature_importance.py "$MODEL_PATH" --data data/training_data_simple_only_for_shap_500.csv --top-n 30 --background-samples 50 --explain-samples 200

echo "SHAP analysis complete! Check models/shap_analysis/ for visualizations."
