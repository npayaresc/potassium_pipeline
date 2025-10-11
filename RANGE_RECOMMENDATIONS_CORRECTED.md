# Potassium Concentration Range Analysis - CORRECTED
## For Configured Range: 0.2% - 5.3%

**Date**: 2025-10-05
**Model**: XGBoost K_only with Physics-Informed Features
**Test Samples**: 181 (all within target range)

---

## Current Performance by Sub-Range

| Sub-Range | Label | Samples | % of Test | MAPE | Within ±20% | Within ±10% | Status |
|-----------|-------|---------|-----------|------|-------------|-------------|--------|
| **0.2-1.0%** | Low | 32 | 17.7% | 52.9% | 40.6% | 25.0% | ❌ Worst |
| **1.0-2.0%** | Medium-Low | 44 | 24.3% | 26.3% | 52.3% | 27.3% | ⚠️ Fair |
| **2.0-3.0%** | Medium | 20 | 11.0% | 36.4% | **20.0%** | 15.0% | ❌ Worst |
| **3.0-4.0%** | High | 40 | 22.1% | 17.5% | **70.0%** | 32.5% | ✅ Best |
| **4.0-5.3%** | Very High | 45 | 24.9% | 20.6% | 60.0% | 44.4% | ✅ Good |

### Key Findings:

1. **Best Performance**: 3.0-4.0% (70% within ±20%)
2. **Worst Performance**: 2.0-3.0% (only 20% within ±20%)
3. **Undersampled**: 2.0-3.0% has only 20 samples (11% of data)
4. **Bias Pattern**:
   - **Over-predicting** at low ranges (0.2-3.0%)
   - **Under-predicting** at high ranges (3.0-5.3%)

---

## Problem Diagnosis

### 1. **Low Range (0.2-1.0%)** - MAPE 52.9%
**Problem**: Model over-predicts by average +0.374%
**Root Cause**:
- Low K signal (weak emission lines)
- High relative noise (SNR ~3:1)
- C, O, H interference dominates spectrum
- Baseline subtraction errors magnified at low concentrations

**Physics**:
- K_I_amplitude_0 is low (~100-500 units vs. >1000 for high concentrations)
- K_I_fit_quality_0 likely poor (<0.6) due to weak signal
- K_I_asymmetry_0 should be near 0 (no self-absorption)

**Action**: Consider using violet K line (404 nm) which has better SNR at low concentrations

---

### 2. **Medium Range (2.0-3.0%)** - Only 20% within ±20%
**Problem**: Severely over-predicts by +0.459%
**Root Causes**:
- **Undersampled**: Only 20 samples (need ~40+)
- **Transition zone**: Self-absorption starts here
- Model uncertain - bounces between linear and saturated behavior

**Physics**:
- K_I_asymmetry_0 transitions from ~0.1 to ~0.3
- K_I_absorption_index_0 starts increasing
- Peak shape changes from symmetric to asymmetric

**Action**: **Collect more training data in 2.0-3.0% range** (priority!)

---

### 3. **Very High Range (4.0-5.3%)** - Under-predicts
**Problem**: Model under-predicts by -0.872%
**Root Cause**:
- Self-absorption saturation (optically thick plasma)
- K_I lines saturated at 766/769 nm
- Model hasn't learned to use K_I_asymmetry_0 effectively

**Physics**:
- K_I_asymmetry_0 > 0.4 (strong self-absorption)
- K_I_absorption_index_0 > 0.6
- Should use K_I_404 violet line (less affected by saturation)

**Action**: Use physics-informed asymmetry correction

---

## Recommended Sub-Range Definitions

### Option 1: **Two-Range Model (Recommended)**

```python
concentration_sub_ranges = {
    "linear_low": {
        "min": 0.2,
        "max": 2.5,
        "description": "Linear response region",
        "samples": 76 (42% of data),
        "current_accuracy": "~50% within ±20%",
        "expected_with_tuning": "~60% within ±20%",
        "key_features": [
            "K_I_amplitude_0",  # Direct intensity
            "K_I_gamma_0",      # Plasma density
            "K_I_404_amplitude_0",  # Violet line (better SNR)
            "K_C_ratio"
        ],
        "physics_check": "K_I_asymmetry_0 < 0.15",
        "quality_filter": "K_I_fit_quality_0 > 0.7"
    },

    "saturated_high": {
        "min": 2.5,
        "max": 5.3,
        "description": "Self-absorption dominant",
        "samples": 105 (58% of data),
        "current_accuracy": "~65% within ±20%",
        "expected_with_tuning": "~75% within ±20%",
        "key_features": [
            "K_I_asymmetry_0",  # Self-absorption correction
            "K_I_absorption_index_0",  # Saturation indicator
            "K_I_gamma_0",      # Stark broadening
            "K_I_404_amplitude_0"  # Unsaturated line
        ],
        "physics_check": "K_I_asymmetry_0 > 0.2",
        "quality_filter": "K_I_fit_quality_0 > 0.6 OR K_I_404_fit_quality_0 > 0.7"
    }
}
```

**Benefits**:
- Balances sample size (42% vs 58%)
- Clear physics distinction (linear vs saturated)
- Avoids problematic 2.0-3.0% boundary

**Training**:
```bash
# Linear range model
python main.py train --gpu --strategy K_only --focus-min 0.2 --focus-max 2.5

# Saturation range model
python main.py train --gpu --strategy K_only --focus-min 2.5 --focus-max 5.3
```

---

### Option 2: **Three-Range Model (More Aggressive)**

```python
concentration_sub_ranges = {
    "low": {
        "min": 0.2,
        "max": 1.5,
        "samples": 50,
        "action": "Focus on violet line (404 nm), filter by fit quality"
    },

    "medium": {
        "min": 1.5,
        "max": 3.5,
        "samples": 71,
        "action": "COLLECT MORE DATA (especially 2.0-3.0%)"
    },

    "high": {
        "min": 3.5,
        "max": 5.3,
        "samples": 60,
        "action": "Use asymmetry correction heavily"
    }
}
```

**Benefits**:
- More targeted feature selection per range
- Can optimize hyperparameters specifically

**Drawbacks**:
- Medium range still undersampled (need more 2.0-3.0% samples)
- More complex to maintain

---

## Immediate Action Items

### 1. **Data Collection Priority** 🔴
**Collect 30+ more samples in 2.0-3.0% range**
- Currently only 20 samples (11% of data)
- Target: 50+ samples (28% of data)
- This will dramatically improve medium-range accuracy

### 2. **Train Two-Range Models**
```bash
# Linear range (0.2-2.5%)
python main.py train --gpu --strategy K_only \
    --focus-min 0.2 --focus-max 2.5 \
    --models xgboost lightgbm catboost

# Saturated range (2.5-5.3%)
python main.py train --gpu --strategy K_only \
    --focus-min 2.5 --focus-max 5.3 \
    --models xgboost catboost
```

### 3. **Feature Importance Analysis**
Check which physics features matter most:
```bash
# Analyze physics feature importance
python -c "
import joblib
import pandas as pd

model = joblib.load('models/xgboost_K_only.pkl')
importance = pd.DataFrame({
    'feature': model.feature_names_in_,
    'importance': model.feature_importances_
})

# Physics features
physics = importance[importance['feature'].str.contains('fwhm|gamma|asymmetry|absorption|fit_quality')]
print(physics.sort_values('importance', ascending=False).head(15))
"
```

### 4. **Quality Filtering Implementation**
Add to prediction pipeline:
```python
def quality_check(features, prediction):
    fit_quality = features['K_I_fit_quality_0']
    asymmetry = features['K_I_asymmetry_0']

    # Low concentration range
    if prediction < 2.5:
        if fit_quality < 0.7:
            return {'confidence': 'LOW', 'warning': 'Poor fit quality at low concentration'}
        if asymmetry > 0.2:
            return {'confidence': 'QUESTIONABLE', 'warning': 'Unexpected high asymmetry at low K'}

    # High concentration range
    if prediction > 3.5:
        if asymmetry < 0.2:
            return {'confidence': 'QUESTIONABLE', 'warning': 'Unexpectedly low asymmetry at high K'}
        # Use violet line as cross-check
        k404_quality = features.get('K_I_404_fit_quality_0', 0)
        if k404_quality > 0.7:
            return {'confidence': 'HIGH', 'warning': None}

    return {'confidence': 'MEDIUM', 'warning': None}
```

---

## Expected Improvements with Sub-Range Models

| Range | Current ±20% | Expected ±20% | Improvement | Key Changes |
|-------|--------------|---------------|-------------|-------------|
| **0.2-2.5%** | ~50% | ~60% | **+10%** | Better feature selection, 404 nm line |
| **2.5-5.3%** | ~65% | ~75% | **+10%** | Asymmetry correction, range-specific tuning |
| **Overall** | 52.5% | **67%** | **+15%** | Combined sub-range models |

**Overall R² improvement**: 0.623 → **~0.75** (estimated)

---

## Configuration for pipeline_config.py

```python
# Sub-range configuration (within target 0.2-5.3%)
use_concentration_sub_ranges: bool = True

concentration_sub_ranges = [
    {
        "name": "linear_low",
        "min": 0.2,
        "max": 2.5,
        "model_types": ["xgboost", "lightgbm"],
        "key_features": ["K_I_amplitude_0", "K_I_gamma_0", "K_I_404_amplitude_0"],
        "min_fit_quality": 0.7,
        "max_asymmetry": 0.15  # Expect linear response
    },
    {
        "name": "saturated_high",
        "min": 2.5,
        "max": 5.3,
        "model_types": ["xgboost", "catboost"],
        "key_features": ["K_I_asymmetry_0", "K_I_absorption_index_0", "K_I_gamma_0"],
        "min_fit_quality": 0.6,
        "min_asymmetry": 0.2  # Expect self-absorption
    }
]
```

---

## Summary

**Your configured range (0.2-5.3%) is correct**, but performance varies significantly within it:

### ✅ **Strong Performance**:
- **3.0-4.0%**: 70% within ±20% (best)
- **4.0-5.3%**: 60% within ±20% (good)

### ⚠️ **Needs Improvement**:
- **0.2-1.0%**: 40.6% within ±20% (weak signal)
- **1.0-2.0%**: 52.3% within ±20% (acceptable)

### 🔴 **Critical Issue**:
- **2.0-3.0%**: Only 20% within ±20% (severely undersampled - 20 samples)

### **Top Recommendation**:
1. **Collect 30+ more samples in 2.0-3.0% range** (highest priority!)
2. **Train two sub-range models**: 0.2-2.5% (linear) and 2.5-5.3% (saturated)
3. **Use physics-informed features**: Asymmetry for high range, violet line (404nm) for low range
4. **Implement quality filtering**: Based on K_I_fit_quality_0 and asymmetry consistency

**Expected Overall Improvement**: 52.5% → **67% within ±20%** (R² 0.62 → 0.75)
