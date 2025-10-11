# SHAP Feature Importance Analysis Guide

## Overview

This guide explains how to perform SHAP (SHapley Additive exPlanations) feature importance analysis on any trained model in the potassium pipeline.

**New in this version:**
- ✅ Works with ANY model type (Ridge, XGBoost, LightGBM, CatBoost, Random Forest, etc.)
- ✅ **Supports optimized models** from hyperparameter tuning (`optimized_lightgbm_full_context_...`)
- ✅ **Works with all strategies**: K_only, simple_only, full_context
- ✅ Automatically loads feature names from `.feature_names.json` files
- ✅ **Respects feature selection** (only analyzes features actually used by the model)
- ✅ Auto-detects model strategy and extracts corresponding training data
- ✅ Colored terminal output with progress indicators

## Quick Start

### Analyze Latest Model by Type

```bash
# Analyze latest Ridge model
./run_shap_analysis.sh --latest ridge

# Analyze latest XGBoost model
./run_shap_analysis.sh --latest xgboost

# Analyze latest CatBoost model
./run_shap_analysis.sh --latest catboost

# Analyze latest LightGBM model (including optimized)
./run_shap_analysis.sh --latest lightgbm

# Analyze latest model (any type)
./run_shap_analysis.sh --latest
```

### Analyze Specific Model

```bash
# Regular model
./run_shap_analysis.sh models/simple_only_ridge_20251006_024858.pkl

# Optimized model (works automatically!)
./run_shap_analysis.sh models/optimized_lightgbm_full_context_20251006_025844.pkl
```

## How It Works

### Workflow

1. **Model Detection**
   - Script detects model path (explicit or pattern match)
   - Extracts strategy from filename (e.g., `simple_only` from `simple_only_ridge_20251006_024858.pkl`)
   - Checks for corresponding `.feature_names.json` file

2. **Feature Name Loading**
   - If `.feature_names.json` exists, loads the EXACT feature names used by the model
   - Respects feature selection (e.g., if model uses 133 features after selection, uses those 133 names)
   - Falls back to generic names if file doesn't exist

3. **Training Data Extraction**
   - Extracts training data using the detected strategy
   - Uses same feature engineering pipeline as training
   - Extracts descriptive feature names from pipeline (K_I_simple_peak_area, etc.)
   - Caches data to `data/training_data_<strategy>_for_shap_<samples>.csv`
   - Reuses cached data if it's newer than the model

4. **SHAP Analysis**
   - Automatically selects appropriate SHAP explainer:
     - TreeExplainer for tree-based models (XGBoost, LightGBM, CatBoost, Random Forest) - FAST
     - KernelExplainer for linear models (Ridge, Lasso) - SLOW
     - DeepExplainer for neural networks
   - Calculates SHAP values for sampled data
   - Generates importance rankings and visualizations

### Output Files

```
models/
├── <model_name>_shap_importance.csv          # Feature importance table
└── shap_analysis/
    ├── <model_name>_shap_summary.png         # Beeswarm plot (feature values vs SHAP)
    ├── <model_name>_shap_bar.png             # Bar plot (global importance)
    └── <model_name>_shap_custom_bar.png      # Color-coded by category
```

## Configuration Parameters

Edit the script to change these defaults:

```bash
MAX_SAMPLES=500              # Samples for data extraction
BACKGROUND_SAMPLES=50        # Background samples for SHAP
EXPLAIN_SAMPLES=200          # Samples to explain
TOP_N=30                     # Top features to display
```

### Performance vs. Accuracy Trade-offs

| Model Type | Explainer | Speed | Recommended EXPLAIN_SAMPLES |
|------------|-----------|-------|------------------------------|
| XGBoost/LightGBM/CatBoost | TreeExplainer | ⚡ Fast | 500-1000 |
| Random Forest/ExtraTrees | TreeExplainer | ⚡ Fast | 500-1000 |
| Ridge/Lasso/SVR | KernelExplainer | 🐌 Slow | 100-200 |
| Neural Network | DeepExplainer | ⚡ Fast | 500-1000 |

**Note:** KernelExplainer is model-agnostic but slow (O(n²) complexity). For linear models, consider using fewer samples.

## Understanding the Output

### 1. SHAP Importance CSV

```csv
feature,shap_importance,%_of_total,cumulative_%
K_I_simple_peak_area,0.123456,15.2,15.2
K_I_simple_peak_height,0.098765,12.1,27.3
...
```

- **shap_importance**: Mean absolute SHAP value (average impact on predictions)
- **%_of_total**: Percentage of total SHAP importance
- **cumulative_%**: Cumulative importance (e.g., top 10 features = 60%)

### 2. Summary Plot (Beeswarm)

- Each point represents a sample
- X-axis: SHAP value (impact on prediction)
- Color: Feature value (red = high, blue = low)
- Shows how feature values affect predictions

### 3. Bar Plot

- Global feature importance
- Mean absolute SHAP values
- Simple ranking of features

### 4. Custom Bar Plot

Color-coded by feature category:
- 🔵 Blue: Physics-informed features (FWHM, gamma, asymmetry, etc.)
- 🟠 Orange: Peak area features
- 🟢 Green: Ratio features
- 🔴 Red: Other features

## Feature Selection Integration

**Important:** The SHAP analysis now respects feature selection!

Example:
```
Original features: 267
After feature selection: 133
SHAP analysis: Uses the exact 133 features from .feature_names.json
```

This ensures SHAP analyzes only the features that actually contribute to the model, making the analysis more accurate and meaningful.

## Advanced Usage

### Manual Analysis with Custom Parameters

```bash
# Extract training data manually
uv run python extract_training_data_for_shap.py \
    --strategy simple_only \
    --max-samples 1000 \
    --output data/my_custom_data.csv

# Run SHAP analysis with custom settings
uv run python analyze_feature_importance.py \
    models/simple_only_ridge_20251006_024858.pkl \
    --data data/my_custom_data.csv \
    --top-n 50 \
    --background-samples 100 \
    --explain-samples 500
```

### Skip Visualizations (Faster)

```bash
uv run python analyze_feature_importance.py \
    models/simple_only_ridge_20251006_024858.pkl \
    --data data/training_data_simple_only_for_shap_500.csv \
    --no-plots
```

## Interpreting Results

### What SHAP Values Mean

- **Positive SHAP value**: Feature pushes prediction higher
- **Negative SHAP value**: Feature pushes prediction lower
- **Large absolute value**: Strong impact on prediction
- **Near zero**: Minimal impact on prediction

### Feature Categories in LIBS Analysis

1. **Physics-Informed Features** (Most interpretable)
   - FWHM: Line width (temperature/pressure effects)
   - Gamma: Stark broadening (electron density)
   - Asymmetry: Line shape (self-absorption)
   - Kurtosis: Peak sharpness

2. **Peak Features** (Intermediate interpretability)
   - Peak area: Total emission intensity
   - Peak height: Maximum intensity
   - Baseline: Background level

3. **Ratio Features** (High robustness)
   - K/C ratio: Potassium relative to carbon
   - K/H ratio: Potassium relative to hydrogen
   - Cross-element interactions

### Common Insights

- **Top feature importance:** Often K (potassium) peaks dominate for potassium prediction
- **Physics features:** FWHM and gamma often rank high (capture physical properties)
- **Cumulative importance:** Top 30-50 features typically cover 80-90% of importance

## Troubleshooting

### "Failed to load feature names from .feature_names.json"

**Cause:** Model was trained before feature name saving was implemented.
**Solution:** Retrain the model or use generic feature names.

### "SHAP analysis is very slow"

**Cause:** Using KernelExplainer (linear models) with too many samples.
**Solution:** Reduce EXPLAIN_SAMPLES to 100-200 in the script.

### "Feature name count mismatch"

**Cause:** Training data has different number of features than model expects.
**Solution:** Ensure the strategy matches the model (check model filename).

### "No models found matching pattern"

**Cause:** No trained models exist or pattern doesn't match.
**Solution:** Train a model first using `uv run python main.py train` or check pattern.

## Performance Tips

1. **Use tree-based models for SHAP**: XGBoost/LightGBM/CatBoost are 100x faster than Ridge/Lasso
2. **Start with fewer samples**: Use 100 samples first, increase if needed
3. **Reuse cached data**: Don't delete the `data/training_data_*_for_shap_*.csv` files
4. **Use `--no-plots`**: If you only need the CSV output, skip visualizations

## Examples

### Example 1: Quick Analysis of Latest XGBoost Model

```bash
./run_shap_analysis.sh --latest xgboost
```

**Output:**
```
========================================
SHAP FEATURE IMPORTANCE ANALYSIS
========================================

ℹ Model: models/simple_only_xgboost_20251006_012345.pkl
✓ Found feature names file: simple_only_xgboost_20251006_012345.feature_names.json
ℹ Features: 133
ℹ Strategy: simple_only

========================================
Step 1: Extract Training Data
========================================

✓ Using existing data file: data/training_data_simple_only_for_shap_500.csv

========================================
Step 2: Run SHAP Analysis
========================================

ℹ Parameters:
  - Background samples: 50
  - Explain samples: 200
  - Top features: 30

...

✓ SHAP analysis completed successfully!
```

### Example 2: Detailed Analysis with More Samples

Edit `run_shap_analysis.sh`:
```bash
MAX_SAMPLES=1000
EXPLAIN_SAMPLES=500
TOP_N=50
```

Then run:
```bash
./run_shap_analysis.sh models/full_context_catboost_20251006_123456.pkl
```

### Example 3: Compare Feature Importance Across Models

```bash
# Analyze multiple models
./run_shap_analysis.sh --latest ridge
./run_shap_analysis.sh --latest xgboost
./run_shap_analysis.sh --latest catboost

# Compare the CSV files
cat models/simple_only_ridge_*_shap_importance.csv | head -20
cat models/simple_only_xgboost_*_shap_importance.csv | head -20
cat models/simple_only_catboost_*_shap_importance.csv | head -20
```

## Migration from Old Script

**Old workflow (CatBoost only):**
```bash
./run_shap_on_catboost.sh
```

**New workflow (any model):**
```bash
./run_shap_analysis.sh --latest catboost
```

**Key improvements:**
- ✅ Works with any model type (not just CatBoost)
- ✅ Loads feature names from `.feature_names.json` (descriptive names)
- ✅ Auto-detects strategy from filename
- ✅ Better error handling and progress feedback
- ✅ Color-coded output for easier reading

## References

- **SHAP Documentation**: https://shap.readthedocs.io/
- **SHAP Paper**: Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions"
- **SHAP GitHub**: https://github.com/slundberg/shap

## Summary

The new SHAP analysis workflow:
1. **Supports all model types** - Ridge, XGBoost, LightGBM, CatBoost, Random Forest, Neural Networks
2. **Uses saved feature names** - Loads from `.feature_names.json` for accurate interpretation
3. **Auto-detects strategy** - No manual configuration needed
4. **Respects feature selection** - Analyzes only features actually used by the model
5. **Provides rich visualizations** - Summary plots, bar charts, and custom categorized plots
6. **Is fully automated** - One command to run entire analysis

**Next steps:**
- Run SHAP on your trained models to understand feature importance
- Compare feature importance across different model types
- Use insights to improve feature engineering and model performance
