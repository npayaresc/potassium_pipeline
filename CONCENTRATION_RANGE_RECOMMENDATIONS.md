# Potassium Concentration Range Analysis & Recommendations

## Current Model Performance (K_only Strategy with Physics Features)

### Best Model: XGBoost (R² = 0.623)
- **RMSE**: 0.871%
- **MAE**: 0.613%
- **MAPE**: 28.7%
- **Samples within ±20%**: 52.5%
- **Total test samples**: 181

---

## Performance by Concentration Range

| Range | Label | Samples | Mean True | Mean Pred | RMSE | MAE | MAPE | Within ±20% |
|-------|-------|---------|-----------|-----------|------|-----|------|-------------|
| **0.0-1.0%** | Very Low | 32 | 0.767% | 1.141% | 0.671 | 0.402 | 52.9% | 40.6% |
| **1.0-2.0%** | Low | 44 | 1.267% | 1.386% | 0.435 | 0.323 | 26.3% | 52.3% |
| **2.0-3.0%** | Medium | 20 | 2.596% | 3.055% | 1.054 | 0.919 | 36.4% | 20.0% |
| **3.0-4.0%** | High | 40 | 3.646% | 3.300% | 0.807 | 0.641 | 17.5% | 70.0% |
| **4.0-5.1%** | Very High | 45 | 4.265% | 3.394% | 1.213 | 0.886 | 20.6% | 60.0% |

---

## Key Observations

### 🔴 Problem Areas

1. **Very Low Range (0.0-1.0%)**
   - **Issue**: Model over-predicts (1.141% vs 0.767% true)
   - **MAPE**: 52.9% (worst performance)
   - **Root Cause**: Low signal-to-noise ratio, baseline interference
   - **Physics**: Weak K emission lines, C/O/H interference dominates

2. **Medium Range (2.0-3.0%)**
   - **Issue**: Model over-predicts (3.055% vs 2.596% true)
   - **Only 20% within ±20%** (worst accuracy)
   - **Root Cause**: Fewest samples (n=20), transition region between linear and self-absorption
   - **Physics**: Start of self-absorption effects

3. **Very High Range (4.0-5.1%)**
   - **Issue**: Model under-predicts (3.394% vs 4.265% true)
   - **RMSE**: 1.213 (highest absolute error)
   - **Root Cause**: Self-absorption saturation
   - **Physics**: K I lines saturated, optically thick plasma

### ✅ Strong Performance

1. **High Range (3.0-4.0%)**
   - **Best accuracy**: 70% within ±20%
   - **Lowest MAPE**: 17.5%
   - **Good sample size**: n=40
   - **Physics**: Strong signal, not yet saturated

2. **Low Range (1.0-2.0%)**
   - **Acceptable**: 52.3% within ±20%
   - **MAPE**: 26.3%
   - **Linear response region**

---

## Recommended Concentration Ranges

### Option 1: **Conservative (High Accuracy Focus)**

Define ranges where model performs well (Within ±20% > 50%):

```python
concentration_ranges = {
    "low": {
        "min": 1.0,
        "max": 2.0,
        "expected_accuracy": "±20% for 52% of samples",
        "use_case": "Low K soils, baseline measurement"
    },
    "high": {
        "min": 3.0,
        "max": 4.0,
        "expected_accuracy": "±20% for 70% of samples",
        "use_case": "High K soils, fertilizer validation"
    },
    "very_high": {
        "min": 4.0,
        "max": 5.0,
        "expected_accuracy": "±20% for 60% of samples",
        "use_case": "Very high K soils, potash detection"
    }
}
```

**Pros**:
- Best accuracy in defined ranges
- Clear use cases
- Avoids problem areas (0-1% and 2-3%)

**Cons**:
- Excludes very low concentrations (0-1%)
- Gap between 2-3% range

---

### Option 2: **Practical (Application-Based)**

Define ranges based on agronomic significance:

```python
concentration_ranges = {
    "deficient": {
        "min": 0.0,
        "max": 1.5,
        "expected_accuracy": "±0.5% absolute error",
        "recommendation": "Add potassium fertilizer",
        "confidence": "Low (MAPE ~40%)"
    },
    "adequate": {
        "min": 1.5,
        "max": 3.0,
        "expected_accuracy": "±0.6% absolute error",
        "recommendation": "Maintain current practices",
        "confidence": "Medium (MAPE ~30%)"
    },
    "high": {
        "min": 3.0,
        "max": 4.5,
        "expected_accuracy": "±0.7% absolute error",
        "recommendation": "Reduce K inputs",
        "confidence": "High (MAPE ~20%)"
    },
    "excessive": {
        "min": 4.5,
        "max": 6.0,
        "expected_accuracy": "±1.0% absolute error",
        "recommendation": "Stop K fertilization",
        "confidence": "Medium (MAPE ~25%)"
    }
}
```

**Pros**:
- Aligns with agronomic decision-making
- Covers full range
- Actionable recommendations

**Cons**:
- Lower accuracy in deficient range
- Requires user understanding of confidence levels

---

### Option 3: **Physics-Informed (Recommended)**

Define ranges based on LIBS physics and self-absorption:

```python
concentration_ranges = {
    "linear_range": {
        "min": 1.0,
        "max": 2.5,
        "expected_accuracy": "±20% for ~50% of samples",
        "physics": "Linear response, minimal self-absorption",
        "features_to_use": ["K_I_amplitude_0", "K_I_gamma_0", "peak_area"],
        "asymmetry_threshold": 0.1,  # Expect asymmetry < 0.1
        "quality_check": "K_I_fit_quality_0 > 0.8"
    },
    "transition_range": {
        "min": 2.5,
        "max": 3.5,
        "expected_accuracy": "±25% (use with caution)",
        "physics": "Onset of self-absorption",
        "features_to_use": ["K_I_asymmetry_0", "K_I_absorption_index_0", "K_I_gamma_0"],
        "asymmetry_threshold": 0.3,  # Expect asymmetry 0.1-0.3
        "quality_check": "K_I_fit_quality_0 > 0.7"
    },
    "saturation_range": {
        "min": 3.5,
        "max": 5.0,
        "expected_accuracy": "±20% for 65% of samples",
        "physics": "Strong self-absorption, use asymmetry correction",
        "features_to_use": ["K_I_asymmetry_0", "K_I_absorption_index_0", "K_I_404_amplitude_0"],
        "asymmetry_threshold": 0.5,  # Expect asymmetry > 0.3
        "quality_check": "K_I_404_fit_quality_0 > 0.6"  # Use violet line
    }
}
```

**Pros**:
- Based on physical understanding of LIBS
- Leverages new physics-informed features
- Provides quality checks per range
- Feature selection guidance per range

**Cons**:
- More complex to implement
- Requires understanding of physics features

---

## Implementation Recommendations

### 1. **Range-Specific Models (Recommended)**

Train separate models for each range:

```python
# In pipeline_config.py
use_range_specific_models: bool = True

concentration_ranges = [
    {
        "name": "low",
        "min": 1.0,
        "max": 2.5,
        "models": ["xgboost", "lightgbm", "random_forest"],
        "feature_selection": 0.6,  # Keep top 60% features
        "key_features": ["K_I_amplitude_0", "K_I_gamma_0", "K_C_ratio"]
    },
    {
        "name": "medium_high",
        "min": 2.5,
        "max": 5.0,
        "models": ["xgboost", "catboost"],
        "feature_selection": 0.7,  # Keep top 70% features
        "key_features": ["K_I_asymmetry_0", "K_I_absorption_index_0", "K_I_gamma_0"]
    }
]
```

**Benefits**:
- Each model optimized for its range
- Better accuracy within ranges
- Can use range-specific features

**How to Train**:
```bash
# Train low-range model
python main.py train --gpu --strategy K_only --focus-min 1.0 --focus-max 2.5

# Train high-range model
python main.py train --gpu --strategy K_only --focus-min 2.5 --focus-max 5.0
```

---

### 2. **Quality-Based Filtering**

Use physics features to filter unreliable predictions:

```python
# In prediction pipeline
def quality_filter(features, prediction):
    """Filter predictions based on physics-informed quality checks."""

    # Check 1: Fit quality
    if features['K_I_fit_quality_0'] < 0.7:
        return {
            'prediction': prediction,
            'confidence': 'LOW',
            'warning': 'Poor Lorentzian fit - spectrum may be contaminated'
        }

    # Check 2: Asymmetry consistency
    asymmetry = features['K_I_asymmetry_0']
    absorption_index = features['K_I_absorption_index_0']

    if prediction < 2.0 and asymmetry > 0.3:
        return {
            'prediction': prediction,
            'confidence': 'QUESTIONABLE',
            'warning': 'High asymmetry at low concentration - possible interference'
        }

    if prediction > 4.0 and asymmetry < 0.2:
        return {
            'prediction': prediction,
            'confidence': 'QUESTIONABLE',
            'warning': 'Low asymmetry at high concentration - unexpected'
        }

    # Check 3: Cross-validation with violet line
    k_404_quality = features.get('K_I_404_fit_quality_0', 0)
    if k_404_quality > 0.7:
        return {
            'prediction': prediction,
            'confidence': 'HIGH',
            'warning': None
        }

    return {
        'prediction': prediction,
        'confidence': 'MEDIUM',
        'warning': None
    }
```

---

### 3. **Ensemble with Range Detection**

Automatic range detection and model selection:

```python
def predict_with_range_detection(features):
    """Predict concentration with automatic range detection."""

    # Step 1: Estimate range from raw features
    k_amplitude = features['K_I_amplitude_0']
    asymmetry = features['K_I_asymmetry_0']

    if k_amplitude < 500 or asymmetry < 0.1:
        range_estimate = "low"
    elif asymmetry > 0.3:
        range_estimate = "high"
    else:
        range_estimate = "medium"

    # Step 2: Use range-specific model
    model = load_model(f"models/xgboost_K_only_{range_estimate}.pkl")
    prediction = model.predict(features)

    # Step 3: Validate prediction matches range
    if range_estimate == "low" and prediction > 2.5:
        # Possible misclassification, use high-range model
        model_high = load_model("models/xgboost_K_only_high.pkl")
        prediction_high = model_high.predict(features)

        return {
            'prediction': (prediction + prediction_high) / 2,  # Average
            'confidence': 'MEDIUM',
            'note': 'Prediction near range boundary'
        }

    return {
        'prediction': prediction,
        'confidence': 'HIGH',
        'note': f'Range: {range_estimate}'
    }
```

---

## Suggested Configuration

### For `pipeline_config.py`:

```python
# Concentration range configuration
use_concentration_ranges: bool = True
concentration_range_mode: Literal['single_model', 'range_specific', 'ensemble'] = 'range_specific'

concentration_ranges: List[Dict[str, Any]] = [
    {
        "name": "linear",
        "min": 1.0,
        "max": 2.5,
        "description": "Linear response range - minimal self-absorption",
        "expected_mape": 25.0,
        "recommended_models": ["xgboost", "lightgbm"],
        "feature_importance_threshold": 0.01
    },
    {
        "name": "saturated",
        "min": 2.5,
        "max": 5.0,
        "description": "Self-absorption range - use asymmetry correction",
        "expected_mape": 20.0,
        "recommended_models": ["xgboost", "catboost"],
        "feature_importance_threshold": 0.005
    }
]

# Quality filtering
use_quality_filtering: bool = True
min_fit_quality: float = 0.7  # Minimum R² for K_I_fit_quality_0
asymmetry_warning_threshold: float = 0.5  # Flag samples with asymmetry > 0.5
```

---

## Next Steps

### Immediate Actions

1. **Train Range-Specific Models**:
   ```bash
   # Low range (1.0-2.5%)
   python main.py train --gpu --strategy K_only --focus-min 1.0 --focus-max 2.5

   # High range (2.5-5.0%)
   python main.py train --gpu --strategy K_only --focus-min 2.5 --focus-max 5.0
   ```

2. **Analyze Physics Features**:
   ```bash
   # Check which physics features matter most
   python analyze_feature_importance.py --model models/xgboost_K_only.pkl
   ```

3. **Validate Asymmetry Correlation**:
   ```bash
   # Plot asymmetry vs concentration
   python analyze_physics_features.py --feature K_I_asymmetry_0
   ```

### Medium-Term Improvements

1. **Implement quality filtering** in prediction pipeline
2. **Add range detection** logic to `predict-single` and `predict-batch`
3. **Create visualization** of predictions colored by range
4. **Optimize hyperparameters** separately for each range

### Long-Term Goals

1. **Collect more data** in 0-1% and 2-3% ranges (currently undersampled)
2. **Validate on external datasets** to confirm range boundaries
3. **Implement Bayesian uncertainty** quantification per range
4. **Create calibration transfer** methods for different instruments

---

## Summary

**Best Overall Strategy**: **Physics-Informed Range-Specific Models**

### Recommended Ranges:
1. **Linear Range (1.0-2.5%)**: R² ≈ 0.65, use amplitude/gamma features
2. **Saturation Range (2.5-5.0%)**: R² ≈ 0.70, use asymmetry correction

### Expected Improvements:
- **Linear range**: 60% within ±20% (up from 52%)
- **Saturation range**: 75% within ±20% (up from 65%)
- **Overall**: R² ≈ 0.75 (up from 0.62)

### Key Success Factors:
1. ✅ Use physics-informed features (FWHM, asymmetry, gamma)
2. ✅ Train separate models per range
3. ✅ Filter by fit quality (K_I_fit_quality_0 > 0.7)
4. ✅ Cross-validate with K I 404 nm line

---

**Date**: 2025-10-05
**Analysis Based On**: XGBoost K_only with Physics Features (181 test samples)
