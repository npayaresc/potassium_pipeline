# Uncertainty Quantification Guide

## 📊 Overview

Uncertainty quantification provides prediction intervals like:
- **"Predicted 2.34% K ± 0.15% (68% CI: 2.19% - 2.49%)"**

This is critical for production LIBS systems to:
1. **Know when to trust predictions** (narrow intervals = high confidence)
2. **Identify when to re-measure** (wide intervals = low confidence)
3. **Meet quality standards** (report predictions with confidence)
4. **Detect outliers** (true value outside interval = potential issue)

---

## 🚀 Quick Start

### Option 1: Add to Existing Training (Simple)

```python
from src.models.uncertainty import UncertaintyQuantifier

# After training your model
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)

# Initialize and calibrate uncertainty
quantifier = UncertaintyQuantifier(
    method='conformal',      # Distribution-free, always works
    confidence_level=0.68    # 1-sigma (68% confidence)
)
quantifier.fit(y_val, y_val_pred)

# Make predictions with intervals on test set
predictions, lower, upper = quantifier.predict_with_intervals(model, X_test)

print(f"Prediction: {predictions[0]:.2f}% ± {(upper[0]-lower[0])/2:.2f}%")
```

### Option 2: Complete Training with Uncertainty

```python
from src.models.uncertainty_trainer import train_model_with_uncertainty

# Train model with full uncertainty support
results = train_model_with_uncertainty(
    model=xgb_model,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
    model_name="XGBoost_K_only",
    uncertainty_method='conformal',
    save_dir=Path("reports/uncertainty")
)

# Results include:
# - Trained model
# - Calibrated uncertainty quantifier
# - Metrics with uncertainty coverage
# - CSV report with all predictions
# - Visualization plot
```

### Option 3: Ensemble Uncertainty (Best)

```python
from src.models.uncertainty_trainer import train_ensemble_with_uncertainty

# Train multiple models
models = [XGBRegressor(), LGBMRegressor(), CatBoostRegressor()]
names = ['XGBoost', 'LightGBM', 'CatBoost']

results = train_ensemble_with_uncertainty(
    models=models,
    model_names=names,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
    save_dir=Path("reports/uncertainty")
)

# Ensemble provides:
# - Better predictions (average of models)
# - Better uncertainty (variance across models)
# - More robust intervals
```

---

## 📋 Uncertainty Methods

### 1. Conformal Prediction (RECOMMENDED)

**What it is:**
- Distribution-free method (no assumptions)
- Uses validation residuals to calibrate intervals
- Provably valid coverage guarantees

**Advantages:**
- ✅ Works with any model
- ✅ No special training required
- ✅ Fast and simple
- ✅ Statistically guaranteed coverage

**How it works:**
1. Compute residuals on validation set: `|y_true - y_pred|`
2. Find quantile q at confidence level (e.g., 68th percentile)
3. Prediction interval: `prediction ± q`

**Use when:**
- Default choice for most cases
- Single model predictions
- Want guaranteed coverage

```python
quantifier = UncertaintyQuantifier(method='conformal', confidence_level=0.68)
```

### 2. Ensemble Uncertainty

**What it is:**
- Uses variance across multiple models
- More models = better uncertainty estimate

**Advantages:**
- ✅ Captures model uncertainty
- ✅ Often tighter intervals than conformal
- ✅ Best if you train multiple models anyway

**How it works:**
1. Train N models on same data
2. Get N predictions for each sample
3. Interval = mean ± z * std_dev

**Use when:**
- Training multiple models
- Want to capture epistemic uncertainty
- Have computational resources

```python
quantifier = UncertaintyQuantifier(method='ensemble')
results = quantifier.predict_with_intervals(model, X, models_ensemble=[model1, model2, model3])
```

### 3. Combined Method (BEST)

**What it is:**
- Combines conformal + ensemble
- Uses conformal calibration with ensemble variance

**Advantages:**
- ✅ Best of both worlds
- ✅ Statistical guarantees (conformal)
- ✅ Tighter intervals (ensemble)
- ✅ Most informative

**Use when:**
- Training ensemble anyway
- Want best uncertainty estimates
- Production deployment

```python
quantifier = UncertaintyQuantifier(method='combined')
```

---

## 🎯 Confidence Levels

### Standard Levels

| Confidence | Meaning | Use Case |
|------------|---------|----------|
| **68%** (1σ) | ±1 standard deviation | **Default** - good balance |
| **90%** | High confidence | Quality control |
| **95%** | Very high confidence | Regulatory compliance |
| **99%** | Extremely high confidence | Critical applications |

### Examples

**68% CI (1-sigma):**
```python
quantifier = UncertaintyQuantifier(confidence_level=0.68)
# Output: "2.34% ± 0.15% (68% CI: 2.19% - 2.49%)"
# Interpretation: ~68% chance true value is in [2.19, 2.49]
```

**95% CI:**
```python
quantifier = UncertaintyQuantifier(confidence_level=0.95)
# Output: "2.34% ± 0.29% (95% CI: 2.05% - 2.63%)"
# Interpretation: ~95% chance true value is in [2.05, 2.63]
```

**Recommendation:** Start with 68% (1σ), increase for critical decisions.

---

## 📊 Evaluation Metrics

### Coverage
**What:** Fraction of true values within prediction intervals
**Target:** Should equal confidence level (e.g., 68% coverage for 68% CI)
**Good:** Within ±5% of target (e.g., 63-73% for 68% CI)

```python
metrics = quantifier.compute_prediction_quality(y_true, predictions, lower, upper)
print(f"Coverage: {metrics['coverage']:.1%}")  # Should be ~68%
```

### Calibration Error
**What:** `|coverage - target_coverage|`
**Target:** < 0.05 (well-calibrated)
**Interpretation:**
- 0.00-0.05: Excellent calibration
- 0.05-0.10: Acceptable
- > 0.10: Needs recalibration

### Interval Width
**What:** Mean/median of `(upper - lower)`
**Interpretation:**
- Narrower = more confident predictions
- Wider = more uncertain predictions
- Compare across models (narrower is better, if coverage is good)

### Relative Width
**What:** `interval_width / |prediction|`
**Interpretation:**
- < 10%: Very precise
- 10-20%: Good precision
- 20-50%: Moderate precision
- > 50%: Low precision (wide intervals)

---

## 🧪 Example Workflow

### Complete Training Pipeline

```python
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from src.models.uncertainty_trainer import train_model_with_uncertainty

# 1. Split data into train/val/test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42
)

# 2. Train with uncertainty
model = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6)

results = train_model_with_uncertainty(
    model=model,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
    model_name="XGBoost_full",
    uncertainty_method='conformal',
    confidence_level=0.68,
    save_dir=Path("reports/uncertainty")
)

# 3. Review metrics
print("Prediction Metrics:")
print(f"  R²: {results['metrics']['r2']:.4f}")
print(f"  MAE: {results['metrics']['mae']:.4f}")
print(f"  RMSE: {results['metrics']['rmse']:.4f}")
print()
print("Uncertainty Metrics:")
print(f"  Coverage: {results['metrics']['coverage']:.1%} (target: 68%)")
print(f"  Mean interval width: {results['metrics']['mean_interval_width']:.4f}%")
print(f"  Calibration error: {results['metrics']['calibration_error']:.3f}")
print(f"  Well calibrated: {results['metrics']['well_calibrated']}")

# 4. Review saved files
# - XGBoost_full_uncertainty.pkl (quantifier for production)
# - XGBoost_full_uncertainty_report.csv (all predictions with intervals)
# - XGBoost_full_uncertainty_plot.png (visualization)
```

### Production Predictions

```python
from src.models.uncertainty import UncertaintyQuantifier
from src.models.uncertainty_trainer import predict_with_uncertainty
import pickle

# Load trained model
with open("models/xgboost_best.pkl", "rb") as f:
    model = pickle.load(f)

# Load calibrated uncertainty quantifier
quantifier = UncertaintyQuantifier.load("reports/uncertainty/XGBoost_full_uncertainty.pkl")

# Make predictions on new data
results = predict_with_uncertainty(
    model=model,
    uncertainty_quantifier=quantifier,
    X=X_new,
    sample_ids=['Sample_001', 'Sample_002', 'Sample_003'],
    output_path=Path("predictions/with_uncertainty.csv")
)

# View results
for _, row in results.head().iterrows():
    print(f"{row['sample_id']}: {row['formatted_prediction']}")

# Output:
# Sample_001: 2.34% ± 0.15% (68% CI: 2.19% - 2.49%)
# Sample_002: 1.87% ± 0.22% (68% CI: 1.65% - 2.09%)
# Sample_003: 3.12% ± 0.09% (68% CI: 3.03% - 3.21%)
```

### Decision Making with Uncertainty

```python
# Flag samples that need re-measurement (wide intervals)
high_uncertainty = results[results['relative_uncertainty_pct'] > 15]
print(f"Samples needing re-measurement: {len(high_uncertainty)}")

# Flag samples with very tight intervals (high confidence)
high_confidence = results[results['relative_uncertainty_pct'] < 5]
print(f"High-confidence samples: {len(high_confidence)}")

# Compare prediction to decision threshold with uncertainty
threshold = 2.0  # e.g., fertilizer application threshold
for _, row in results.iterrows():
    if row['lower_bound'] > threshold:
        print(f"{row['sample_id']}: DEFINITELY above threshold (confident)")
    elif row['upper_bound'] < threshold:
        print(f"{row['sample_id']}: DEFINITELY below threshold (confident)")
    else:
        print(f"{row['sample_id']}: UNCERTAIN - interval straddles threshold, re-measure!")
```

---

## 📈 Advanced Usage

### Ensemble of Best Models

```python
from src.models.uncertainty_trainer import train_ensemble_with_uncertainty

# Use your top 3 models from optimization
models = [
    XGBRegressor(**best_xgb_params),
    LGBMRegressor(**best_lgbm_params),
    CatBoostRegressor(**best_catboost_params)
]

ensemble_results = train_ensemble_with_uncertainty(
    models=models,
    model_names=['XGBoost_opt', 'LightGBM_opt', 'CatBoost_opt'],
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
    confidence_level=0.68,
    save_dir=Path("reports/uncertainty")
)

# Often gives +2-5% R² over best single model
# Plus better uncertainty estimates!
```

### Different Confidence Levels for Different Uses

```python
# Conservative intervals for critical decisions (95%)
quantifier_95 = UncertaintyQuantifier(method='conformal', confidence_level=0.95)
quantifier_95.fit(y_val, y_val_pred)
_, lower_95, upper_95 = quantifier_95.predict_with_intervals(model, X_test)

# Standard intervals for reporting (68%)
quantifier_68 = UncertaintyQuantifier(method='conformal', confidence_level=0.68)
quantifier_68.fit(y_val, y_val_pred)
_, lower_68, upper_68 = quantifier_68.predict_with_intervals(model, X_test)

# Report both
for i in range(len(predictions)):
    print(f"Sample {i}:")
    print(f"  68% CI: [{lower_68[i]:.2f}, {upper_68[i]:.2f}]")
    print(f"  95% CI: [{lower_95[i]:.2f}, {upper_95[i]:.2f}]")
```

---

## 🎯 Integration with Model Trainer

To fully integrate into your existing `model_trainer.py`, add this after model training:

```python
# In model_trainer.py, after training each model

from src.models.uncertainty import UncertaintyQuantifier

# After model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)

# Calibrate uncertainty
uncertainty_quantifier = UncertaintyQuantifier(
    method='conformal',
    confidence_level=0.68
)
uncertainty_quantifier.fit(y_val, y_val_pred)

# Get test predictions with intervals
y_test_pred, lower, upper = uncertainty_quantifier.predict_with_intervals(
    model, X_test
)

# Add to metrics
uncertainty_metrics = uncertainty_quantifier.compute_prediction_quality(
    y_test, y_test_pred, lower, upper
)
all_metrics.update(uncertainty_metrics)

# Save quantifier with model
uncertainty_quantifier.save(model_dir / f"{model_name}_uncertainty.pkl")
```

---

## 📊 Expected Results

### For Your Pipeline (720 samples, K_only)

**Typical Uncertainty Values:**
- **Low K (<0.3%):** Relative uncertainty ~15-20% (higher due to low signal)
- **Mid K (0.3-1.0%):** Relative uncertainty ~8-12% (optimal range)
- **High K (>1.0%):** Relative uncertainty ~10-15% (self-absorption effects)

**Expected Coverage:**
- Well-calibrated: 63-73% (target 68%)
- Under-calibrated: <60% (intervals too narrow)
- Over-calibrated: >75% (intervals too wide, conservative)

**Expected Interval Widths:**
- Mean interval width: ~0.15-0.25% K
- Median relative width: ~10-15%

---

## ✅ Best Practices

1. **Always validate coverage** on held-out test set
2. **Recalibrate periodically** as you collect more data
3. **Use ensemble** if training multiple models anyway
4. **Save quantifier** with each trained model for production
5. **Report intervals** with all predictions
6. **Flag wide intervals** for re-measurement
7. **Use 68% CI** as default, 95% for critical decisions

---

## 🎉 Summary

**What you get:**
- Prediction intervals: "2.34% ± 0.15%"
- Coverage guarantees (68%, 90%, 95%)
- Automatic calibration
- Multiple methods (conformal, ensemble, combined)
- Production-ready tools

**Why it matters:**
- Know when to trust predictions
- Identify samples needing re-measurement
- Meet quality standards
- Scientific credibility

**Implementation:**
- ✅ `src/models/uncertainty.py` - Core module
- ✅ `src/models/uncertainty_trainer.py` - Training integration
- ✅ This guide - Complete documentation

**Next steps:**
1. Try example workflow above
2. Add to your model training
3. Review uncertainty reports
4. Tune for your use case

---

**Ready to use! Start with conformal prediction (simple, guaranteed coverage).**
