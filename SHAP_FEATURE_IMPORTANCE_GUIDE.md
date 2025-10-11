# SHAP-Based Feature Importance Analysis Guide

## Overview

The pipeline now uses **SHAP (SHapley Additive exPlanations)** for feature importance analysis instead of traditional tree-based importance. This provides more robust and reliable results for spectral data analysis.

## Why SHAP Over Tree-Based Importance?

### Traditional Tree Importance Problems:
- ❌ **Biased toward high-cardinality features**
- ❌ **Inconsistent across different tree structures**
- ❌ **Misleading with correlated features** (critical issue for spectral data!)
- ❌ **No individual prediction explanations**
- ❌ **Model-specific only (tree models)**

### SHAP Advantages:
- ✅ **Mathematically consistent** (based on game theory)
- ✅ **Handles correlated features correctly** (essential for LIBS spectra!)
- ✅ **Works with all model types** (XGBoost, LightGBM, CatBoost, Neural Networks, AutoGluon)
- ✅ **Individual prediction explanations** (understand why specific predictions were made)
- ✅ **Feature interaction analysis**
- ✅ **Publication-ready visualizations**

### Why This Matters for LIBS Spectral Analysis:

1. **Spectral features are highly correlated**:
   - Neighboring wavelengths have similar intensities
   - Physics-informed features (FWHM, gamma, asymmetry, kurtosis) are interdependent
   - Traditional importance can incorrectly attribute importance

2. **Multiple model types**:
   - Your pipeline uses XGBoost, LightGBM, CatBoost, Neural Networks, and AutoGluon
   - SHAP provides consistent importance across all models

3. **Interpretability**:
   - Understand which physical processes drive predictions
   - Identify which spectral regions are most informative
   - Debug model behavior on specific samples

## Installation

SHAP is now included in the dependencies. To install:

```bash
uv sync
# or
uv pip install shap
```

## Usage

### Basic Usage

```bash
# Analyze feature importance for a trained model
python analyze_feature_importance.py <model_path> --data <training_data.csv>
```

### Examples

```bash
# Example 1: Analyze latest optimized XGBoost model
python analyze_feature_importance.py \
  models/optimized_xgboost_simple_only_20250106.pkl \
  --data data/training_data.csv

# Example 2: With custom parameters
python analyze_feature_importance.py \
  models/my_model.pkl \
  --data data/training_data.csv \
  --top-n 50 \
  --background-samples 200 \
  --explain-samples 1000

# Example 3: Skip visualization plots (faster)
python analyze_feature_importance.py \
  models/my_model.pkl \
  --data data/training_data.csv \
  --no-plots
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `model_path` | Path to trained model file | Latest optimized model |
| `--data` | Path to training data CSV (REQUIRED) | None |
| `--top-n` | Number of top features to display | 30 |
| `--background-samples` | Background samples for SHAP | 100 |
| `--explain-samples` | Samples to explain | 500 |
| `--no-plots` | Skip visualization plots | False |

## Output Files

The analysis generates several output files in the model directory:

### 1. CSV File: `<model>_shap_importance.csv`
Contains detailed SHAP importance for all features:
- `feature`: Feature name
- `shap_importance`: Mean absolute SHAP value
- `%_of_total`: Percentage of total importance
- `cumulative_%`: Cumulative percentage

### 2. Visualization Files (in `shap_analysis/` subdirectory):

#### a. Summary Plot (Beeswarm): `<model>_shap_summary.png`
- Shows SHAP values for all features
- Each dot is a sample
- Color indicates feature value (red=high, blue=low)
- Position shows SHAP value (impact on prediction)

**Interpretation:**
- Features at top are most important
- Wide spread = high impact variation
- Color pattern shows relationship direction

#### b. Bar Plot: `<model>_shap_bar.png`
- Simple bar chart of mean absolute SHAP values
- Quick overview of global importance

#### c. Custom Categorical Bar Plot: `<model>_shap_custom_bar.png`
- Color-coded by feature category:
  - **Blue**: Physics-informed features (FWHM, gamma, asymmetry, kurtosis)
  - **Orange**: Peak area features
  - **Green**: Ratio features
  - **Red**: Other features

## Console Output

The script provides comprehensive analysis in the console:

### 1. Model Type Detection
```
🔍 Model Type Detected: xgboost
✓ Using TreeExplainer (optimized for tree ensembles)
```

### 2. Dataset Statistics
```
📊 Dataset Statistics:
   Total samples: 1234
   Total features: 456
   Background samples for SHAP: 100
   Samples to explain: 500
```

### 3. Top Features
```
🏆 Top 30 Most Important Features (SHAP):
Rank   Feature                      SHAP Value      % of Total   Cumulative %
1      K_I_peak_0                   0.123456        15.23        15.23
2      K_I_fwhm_0                   0.098765        12.18        27.41
...
```

### 4. Feature Categories
```
📂 Feature Categories (SHAP-based):

Physics-Informed: 84 features (45.2% of total SHAP importance)
  #2   K_I_fwhm_0                                  0.098765 (12.18%)
  #5   K_I_asymmetry_0                             0.076543 (9.45%)
  #8   K_I_kurtosis_0                              0.065432 (8.07%)
  ...
```

### 5. Cumulative Milestones
```
📊 Cumulative SHAP Importance Milestones:
  Top  12 features cover 50% of total SHAP importance
  Top  28 features cover 80% of total SHAP importance
  Top  45 features cover 90% of total SHAP importance
  Top  67 features cover 95% of total SHAP importance
```

## Interpreting SHAP Values

### Global Importance (What We Calculate)
- **Mean |SHAP value|** = Average absolute impact on model output
- Higher values = more important features
- Summing to explain total model complexity

### Physical Interpretation for LIBS

#### High SHAP Importance Indicates:
1. **Strong signal correlation** with potassium concentration
2. **Consistent physical process** across samples
3. **Low noise/high reproducibility**

#### Feature Category Insights:

**Physics-Informed Features:**
- High importance → Physical process is key
  - High `fwhm` importance → Stark broadening matters
  - High `asymmetry` importance → Self-absorption is significant
  - High `kurtosis` importance → Peak shape quality is critical

**Peak Area Features:**
- Direct intensity correlation
- Classic LIBS quantification

**Ratio Features:**
- Normalization importance
- Matrix effect compensation

## Advanced: Programmatic Usage

You can also use the SHAP analyzer in your own Python scripts:

```python
from analyze_feature_importance import analyze_shap_importance
import pandas as pd
import numpy as np

# Load your data
df = pd.read_csv("data/training_data.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
feature_names = list(df.columns[:-1])

# Run analysis
importance_df = analyze_shap_importance(
    model_path="models/my_model.pkl",
    X_data=X,
    y_data=y,
    feature_names=feature_names,
    top_n=30,
    save_plots=True,
    background_samples=100,
    explain_samples=500
)

# Use results
top_10_features = importance_df.head(10)['feature'].tolist()
print(f"Top 10 features: {top_10_features}")
```

## Supported Model Types

The analyzer automatically detects and uses the appropriate SHAP explainer:

| Model Type | SHAP Explainer | Speed | Notes |
|------------|----------------|-------|-------|
| XGBoost | TreeExplainer | ⚡⚡⚡ Fast | Optimized for tree models |
| LightGBM | TreeExplainer | ⚡⚡⚡ Fast | Optimized for tree models |
| CatBoost | TreeExplainer | ⚡⚡⚡ Fast | Optimized for tree models |
| RandomForest | TreeExplainer | ⚡⚡⚡ Fast | Sklearn tree models |
| ExtraTrees | TreeExplainer | ⚡⚡⚡ Fast | Sklearn tree models |
| Neural Networks (PyTorch) | DeepExplainer | ⚡⚡ Medium | Specialized for deep learning |
| AutoGluon | KernelExplainer | ⚡ Slow | Model-agnostic |
| Other sklearn | KernelExplainer | ⚡ Slow | Model-agnostic fallback |

## Performance Tips

1. **Reduce sample sizes** for faster analysis:
   ```bash
   --background-samples 50 --explain-samples 200
   ```

2. **Skip plots** if you only need importance values:
   ```bash
   --no-plots
   ```

3. **Use TreeExplainer models** (XGBoost, LightGBM, CatBoost) for fastest results

4. **Sample your data** if dataset is very large (>10k samples)

## Troubleshooting

### "SHAP not installed"
```bash
uv pip install shap
```

### "Data file is required"
You must provide training data for SHAP analysis:
```bash
python analyze_feature_importance.py model.pkl --data training_data.csv
```

### "Failed to calculate SHAP values"
- Check model compatibility
- Try reducing `--explain-samples`
- Check for NaN/Inf values in data

### Plots not generating
- Install matplotlib: `uv pip install matplotlib`
- Check disk space
- Try `--no-plots` to skip visualization

## Comparison with Old Method

### Old Method (Tree-based importance):
```python
importance = model.feature_importances_  # ❌ Model-specific, biased
```

### New Method (SHAP):
```python
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)
importance = np.abs(shap_values.values).mean(axis=0)  # ✅ Consistent, unbiased
```

**Key Difference:** SHAP measures actual impact on predictions, not just split frequency in trees.

## References

1. **SHAP Paper**: Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions"
2. **SHAP Documentation**: https://shap.readthedocs.io/
3. **TreeExplainer Paper**: Lundberg et al. (2020), "From local explanations to global understanding"

## Next Steps

After running feature importance analysis, you can:

1. **Feature Selection**: Remove low-importance features for faster training
2. **Physics Interpretation**: Investigate why certain physics features are important
3. **Model Debugging**: Understand predictions on specific samples
4. **Publication**: Use SHAP plots in papers (properly cite SHAP library)

## Questions?

Refer to:
- SHAP documentation: https://shap.readthedocs.io/
- This project's main README
- Feature engineering documentation in `src/features/`
