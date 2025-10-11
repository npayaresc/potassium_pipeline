# Feature Names Saving Implementation

## Overview

Every trained model now automatically saves feature names to a `.feature_names.json` file alongside the model `.pkl` file. This ensures you always know which features went into each model, accounting for feature selection and dimension reduction.

## Implementation Details

### Pipeline Structure

The typical saved pipeline structure is:
```python
Pipeline([
    ('features', feature_pipeline),      # Feature engineering (LIBS spectral → engineered features)
    ('model', model_pipeline)            # Contains: feature_selection → dimension_reduction → regressor
])
```

### Feature Extraction Logic

The implementation handles three scenarios:

#### 1. **Feature Selection Only**
- Example: `SpectralFeatureSelector` filters from 500 → 100 features
- **Result**: Saves the 100 selected feature names
- Feature names are extracted using `get_selected_features()` method

#### 2. **Dimension Reduction Present**
- Example: PCA reduces 500 features → 50 principal components
- **Result**: Cannot preserve original feature names (PCA creates new features)
- Saves metadata: `{'dimension_reduction': {'method': 'PCAReducer', 'n_components': 50}}`

#### 3. **No Feature Filtering**
- No feature_selection or dimension_reduction steps
- **Result**: Saves all engineered features from the feature engineering pipeline

### Code Flow

```python
# 1. Check for nested 'model' pipeline
if step_name == 'model' and hasattr(transformer, 'steps'):

    # 2. Check for dimension reduction (prevents feature name extraction)
    if 'dimension_reduction' in nested_pipeline.steps:
        return None  # Can't preserve feature names after transformation

    # 3. Check for feature selection (preserves feature names)
    if 'feature_selection' in nested_pipeline.steps:
        return feature_selector.get_selected_features()  # Returns filtered list

# 4. Fallback to feature engineering pipeline
return feature_pipeline.get_feature_names_out()
```

## Output Format

### Example JSON File: `models/simple_only_xgboost_20251006_000443.feature_names.json`

```json
{
  "model_name": "xgboost",
  "model_path": "models/simple_only_xgboost_20251006_000443.pkl",
  "feature_count": 127,
  "feature_names": [
    "K_I_404_amplitude_0",
    "K_I_404_asymmetry_0",
    "K_I_404_fwhm_0",
    "K_I_404_quality_0",
    "K_I_404_kurtosis_0",
    ...
  ],
  "pipeline_steps": ["features", "model"],
  "strategy": "simple_only",
  "timestamp": "20251006_000443",
  "model_type": "standard",
  "transformations": {
    "feature_selection": {
      "method": "SpectralFeatureSelector",
      "n_features_selected": 127
    },
    "dimension_reduction": null
  }
}
```

### Example with Dimension Reduction

```json
{
  "model_name": "ridge",
  "model_path": "models/simple_only_ridge_pca_20251006_000443.pkl",
  "feature_count": null,
  "feature_names": null,
  "pipeline_steps": ["features", "model"],
  "strategy": "simple_only",
  "timestamp": "20251006_000443",
  "model_type": "standard",
  "transformations": {
    "feature_selection": null,
    "dimension_reduction": {
      "method": "PCA",
      "n_components": 50
    }
  }
}
```

## Files Modified

### Core Implementation
- **`src/models/feature_name_saver.py`** (NEW)
  - `save_feature_names()` - Main function for standard models
  - `save_feature_names_for_nn()` - Function for neural network models
  - `_extract_feature_names()` - Handles nested pipeline extraction
  - `_extract_from_nested_pipeline()` - Extracts from model pipeline
  - `_extract_transformation_metadata()` - Captures transformation details

### Integration Points
- **`src/models/model_trainer.py`**
  - `save_pipeline()` - Calls `save_feature_names()`
  - `save_neural_network_pipeline()` - Calls `save_feature_names_for_nn()`

- **`src/models/optimize_all_models.py`**
  - `optimize_single_model()` - Calls `save_feature_names()`
  - `optimize_model()` - Calls `save_feature_names()`

## Validation

To verify feature names are correctly extracted:

1. **With Feature Selection**:
```bash
python main.py train --models xgboost
# Check: models/simple_only_xgboost_*.feature_names.json
# Should contain: selected feature names (if feature selection enabled)
```

2. **With Dimension Reduction**:
```bash
python main.py train --models ridge_pca
# Check: models/simple_only_ridge_pca_*.feature_names.json
# Should contain: null for feature_names, transformation metadata for PCA
```

3. **Verify Counts Match**:
```python
import json
import joblib

# Load model
model = joblib.load('models/simple_only_xgboost_20251006_000443.pkl')

# Load feature names
with open('models/simple_only_xgboost_20251006_000443.feature_names.json') as f:
    feature_info = json.load(f)

# Get model's expected input features
# For models with feature selection, check the selector
if hasattr(model.named_steps['model'], 'steps'):
    for step_name, step in model.named_steps['model'].steps:
        if step_name == 'feature_selection':
            actual_features = step.get_selected_features()
            print(f"Feature count matches: {len(actual_features) == feature_info['feature_count']}")
            print(f"Feature names match: {actual_features == feature_info['feature_names']}")
```

## Benefits

1. **Transparency**: Always know which features went into each model
2. **Reproducibility**: Can verify feature engineering consistency across runs
3. **Debugging**: Identify when feature selection is working correctly
4. **Analysis**: Compare which features different models use
5. **SHAP Compatibility**: Feature names ensure SHAP analysis uses correct labels

## Future Enhancements

1. Add support for feature importance extraction alongside feature names
2. Save feature statistics (mean, std) for each feature
3. Create utility function to compare feature sets across models
4. Integrate with SHAP analysis to auto-label features
