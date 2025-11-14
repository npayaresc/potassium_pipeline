# Reducing Prediction Uncertainty: Optimization Strategies

## Understanding the Problem

Your current **95% conformal interval width is ±1.83%**, which means for a prediction of 8.0%, the interval is [6.17%, 9.83%].

**Key insight**: Conformal prediction intervals are **calibrated to your model's actual errors**. The only way to reduce interval width is to:

1. **Reduce prediction errors** (lower MAE/RMSE)
2. **Reduce error variance** (more consistent errors)
3. **Use adaptive methods** (different intervals for different regions)

Current performance:
- MAE: 0.76%
- RMSE: 0.97%
- Std Dev: 0.96%
- 95% conformal width: ±1.83% (≈ 1.9 × std dev)

**Target**: Reduce MAE to 0.5% and std dev to 0.7% → interval width ~±1.4%

---

## Strategy 1: Improve Model Performance (Highest Impact)

### 1.1 Advanced Hyperparameter Optimization

**Current state**: You're using AutoGluon with `extreme` preset, which is already quite good.

**Optimization approach**: Use longer time limits and focused optimization on best model types.

```bash
# Option A: Extend AutoGluon training time
python main.py autogluon --gpu \
    --data-parallel --feature-parallel

# Then edit src/config/pipeline_config.py:
# time_limit: 18000  # 5 hours instead of 3
# num_trials: 150    # More trials
```

**Specific model optimization** (use your existing optimize-models command):

```bash
# Focus on best performing models with more trials
python main.py optimize-models \
    --models xgboost lightgbm catboost \
    --strategy full_context \
    --trials 500 \
    --gpu \
    --data-parallel --feature-parallel
```

**Expected improvement**: 5-15% reduction in MAE

### 1.2 Neural Network Optimization

Your pipeline has neural network support. Try optimizing it specifically:

```bash
# Optimize neural network with more trials
python main.py optimize-models \
    --models neural_network neural_network_light \
    --strategy full_context \
    --trials 200 \
    --gpu \
    --data-parallel --feature-parallel
```

**Key neural network parameters to tune**:
- Hidden layer sizes
- Dropout rates (currently 0.24 for full, 0.3 for light)
- Learning rate with scheduler
- Batch size optimization

**Expected improvement**: 10-20% if neural networks outperform current best

### 1.3 Stacked/Blended Ensembles

AutoGluon already does stacking, but you can create custom ensembles:

```python
# Create custom_ensemble_trainer.py
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
import joblib

def create_stacked_ensemble(model_paths, X_train, y_train, X_test):
    """
    Create a stacked ensemble from multiple trained models.

    Strategy: Use predictions from multiple models as features for meta-learner
    """
    # Load models
    models = [joblib.load(path) for path in model_paths]

    # Get base model predictions
    train_predictions = np.column_stack([
        model.predict(X_train) for model in models
    ])
    test_predictions = np.column_stack([
        model.predict(X_test) for model in models
    ])

    # Train meta-learner (Ridge regression works well)
    meta_learner = Ridge(alpha=1.0)
    meta_learner.fit(train_predictions, y_train)

    # Final predictions
    final_predictions = meta_learner.predict(test_predictions)

    return final_predictions, meta_learner

# Usage
model_paths = [
    'models/full_context_xgboost_best.pkl',
    'models/full_context_lightgbm_best.pkl',
    'models/full_context_catboost_best.pkl',
    'models/full_context_neural_network_best.pkl',
]

predictions, meta_model = create_stacked_ensemble(
    model_paths, X_train, y_train, X_test
)
```

**Expected improvement**: 5-10% if base models are diverse

---

## Strategy 2: Feature Engineering Improvements (High Impact)

### 2.1 SHAP-Based Feature Selection

You already have SHAP analysis tools. Use them to identify the most important features:

```bash
# Step 1: Train initial model
python main.py train --models lightgbm --strategy full_context --gpu

# Step 2: Run SHAP analysis
./run_shap_analysis.sh --latest lightgbm

# Step 3: Train with top features only
python main.py optimize-models \
    --models xgboost lightgbm catboost \
    --strategy full_context \
    --shap-features models/full_context_lightgbm_*_shap_importance.csv \
    --shap-top-n 50 \
    --trials 300 \
    --gpu
```

**Why this works**: Removing noisy/irrelevant features often improves generalization.

**Expected improvement**: 5-20% (can be dramatic if many features are noise)

### 2.2 Advanced Spectral Features

Add domain-specific features based on spectroscopy physics:

```python
# Add to src/features/feature_engineering.py

class AdvancedSpectralFeatures(BaseEstimator, TransformerMixin):
    """
    Advanced spectral features for potassium prediction.

    Based on spectroscopy domain knowledge:
    - Peak ratios (relative intensities)
    - Peak width analysis
    - Baseline stability
    - Inter-peak correlations
    """

    def __init__(self, potassium_regions):
        self.potassium_regions = potassium_regions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []

        # 1. Peak ratio features (most informative)
        # Ratio of primary peak to secondary peak
        for i, region in enumerate(self.potassium_regions):
            for j, other_region in enumerate(self.potassium_regions[i+1:]):
                ratio_feature = (
                    X[f'{region.element}_peak_intensity'] /
                    (X[f'{other_region.element}_peak_intensity'] + 1e-6)
                )
                features.append(ratio_feature)

        # 2. Normalized intensities (reduces instrument drift)
        total_intensity = X[[col for col in X.columns if 'intensity' in col]].sum(axis=1)
        for region in self.potassium_regions:
            normalized = X[f'{region.element}_peak_intensity'] / (total_intensity + 1e-6)
            features.append(normalized)

        # 3. Peak shape features (width/height ratio indicates matrix effects)
        for region in self.potassium_regions:
            if f'{region.element}_peak_width' in X.columns:
                shape_ratio = (
                    X[f'{region.element}_peak_width'] /
                    (X[f'{region.element}_peak_height'] + 1e-6)
                )
                features.append(shape_ratio)

        # 4. Cross-region correlations (multi-line validation)
        # High correlation = more confident measurement
        if len(self.potassium_regions) >= 2:
            region1 = self.potassium_regions[0]
            region2 = self.potassium_regions[1]
            intensity_product = (
                X[f'{region1.element}_peak_intensity'] *
                X[f'{region2.element}_peak_intensity']
            )
            features.append(intensity_product)

        return pd.DataFrame(features).T
```

**Expected improvement**: 10-25% if domain features capture physical relationships

### 2.3 Interaction Features

Create interaction terms between top features:

```python
from sklearn.preprocessing import PolynomialFeatures

# After SHAP analysis, create interactions among top 20 features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interactions = poly.fit_transform(X_top20_features)

# Add to pipeline
# Be careful: can add many features, use with feature selection
```

**Expected improvement**: 5-15% if interactions are meaningful

---

## Strategy 3: Data Quality Improvements (High Impact)

### 3.1 Use Mislabel Detection to Clean Training Data

You already have mislabel detection! Use it to improve training:

```bash
# Step 1: Detect mislabeled samples
python main.py detect-mislabels \
    --focus-min 0.0 --focus-max 15.0 \
    --min-confidence 2 \
    --data-parallel --feature-parallel

# Step 2: Train WITHOUT suspicious samples
python main.py optimize-models \
    --models xgboost lightgbm catboost \
    --strategy full_context \
    --trials 300 \
    --gpu \
    --exclude-suspects reports/mislabel_analysis/suspicious_samples_min_confidence_2.csv
```

**Why this works**: Mislabeled samples increase error variance and confuse the model.

**Expected improvement**: 10-30% if significant mislabeling exists

### 3.2 Outlier Removal in Spectral Data

Add more aggressive outlier detection during data cleaning:

```python
# Modify src/data_management/data_cleansing.py

class SpectralOutlierDetector:
    """Remove spectral outliers using ensemble methods."""

    def __init__(self, contamination=0.05):
        from sklearn.ensemble import IsolationForest
        from sklearn.covariance import EllipticEnvelope

        self.iso_forest = IsolationForest(contamination=contamination, random_state=42)
        self.elliptic = EllipticEnvelope(contamination=contamination)

    def detect_outliers(self, X_spectral):
        """
        Use multiple methods - sample is outlier if flagged by 2+ methods.
        """
        # Method 1: Isolation Forest
        iso_outliers = self.iso_forest.fit_predict(X_spectral)

        # Method 2: Elliptic Envelope (assumes Gaussian)
        elliptic_outliers = self.elliptic.fit_predict(X_spectral)

        # Method 3: Statistical outliers (Mahalanobis distance)
        from scipy.spatial.distance import mahalanobis
        from scipy.stats import chi2

        mean = X_spectral.mean(axis=0)
        cov = np.cov(X_spectral.T)
        inv_cov = np.linalg.pinv(cov)

        mahal_distances = [
            mahalanobis(x, mean, inv_cov)
            for x in X_spectral
        ]
        threshold = chi2.ppf(0.95, df=X_spectral.shape[1])
        mahal_outliers = np.array(mahal_distances) > threshold

        # Consensus: flagged by 2+ methods
        consensus_outliers = (
            (iso_outliers == -1).astype(int) +
            (elliptic_outliers == -1).astype(int) +
            mahal_outliers.astype(int)
        ) >= 2

        return consensus_outliers
```

**Expected improvement**: 5-15% if outliers are present

### 3.3 Increase Training Data Size

More data = better generalization = tighter intervals

**If possible:**
- Collect more samples, especially in high-uncertainty ranges (low and high concentrations)
- Use data augmentation (carefully, for spectral data)
- Combine datasets from multiple runs/instruments (with calibration)

**Expected improvement**: 10-20% with 50% more data

---

## Strategy 4: Reduce Heteroscedasticity (Medium Impact)

### 4.1 Concentration-Weighted Loss Functions

Train models to have more consistent errors across concentration range:

```python
# Custom loss function for XGBoost/LightGBM

def concentration_weighted_mse(y_true, y_pred):
    """
    MSE with inverse variance weighting by concentration.

    Goal: Make errors equally distributed across concentration range.
    """
    residuals = y_true - y_pred

    # Weight by inverse of concentration-specific variance
    # From your uncertainty analysis:
    weights = np.ones_like(y_true)
    for i, y in enumerate(y_true):
        if y < 5.5:
            weights[i] = 1.0 / (0.86 ** 2)  # Low range
        elif y < 7.3:
            weights[i] = 1.0 / (0.98 ** 2)  # Low-mid range
        elif y < 8.3:
            weights[i] = 1.0 / (0.79 ** 2)  # Mid range (best)
        elif y < 9.4:
            weights[i] = 1.0 / (0.88 ** 2)  # Mid-high range
        else:
            weights[i] = 1.0 / (0.98 ** 2)  # High range

    # Normalize weights
    weights = weights / weights.mean()

    # Weighted MSE
    return np.mean(weights * residuals ** 2)

# Use in XGBoost
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    objective=concentration_weighted_mse,
    # ... other parameters
)
```

**Expected improvement**: 10-20% in reducing error variance

### 4.2 Quantile Regression

Train models to predict specific quantiles (e.g., 5th and 95th percentiles) directly:

```python
from sklearn.ensemble import GradientBoostingRegressor

# Train 3 models: median, lower, upper
model_median = GradientBoostingRegressor(loss='quantile', alpha=0.5)
model_lower = GradientBoostingRegressor(loss='quantile', alpha=0.05)
model_upper = GradientBoostingRegressor(loss='quantile', alpha=0.95)

model_median.fit(X_train, y_train)
model_lower.fit(X_train, y_train)
model_upper.fit(X_train, y_train)

# Predictions with built-in intervals
pred_median = model_median.predict(X_test)
pred_lower = model_lower.predict(X_test)
pred_upper = model_upper.predict(X_test)
```

**Advantage**: Prediction intervals are learned directly, can be tighter than conformal

**Expected improvement**: 5-15% for heteroscedastic data

### 4.3 Use Sample Weights During Training

Already partially implemented in your AutoGluon config. Enhance it:

```python
# In src/config/pipeline_config.py, update AutoGluonConfig:

class AutoGluonConfig(BaseModel):
    use_improved_config: bool = True  # ENABLE THIS
    weight_method: Literal["legacy", "improved"] = "improved"

    # Add inverse variance weighting
    use_inverse_variance_weighting: bool = True
```

Then train:
```bash
python main.py autogluon --gpu --data-parallel --feature-parallel
```

**Expected improvement**: 5-10% in reducing heteroscedasticity

---

## Strategy 5: Locally Adaptive Conformal Prediction (Medium Impact)

Instead of one global interval, use different intervals for different concentration ranges.

### 5.1 Concentration-Binned Conformal Intervals

```python
# Add to analyze_prediction_uncertainty.py

class AdaptiveConformalPredictor:
    """Conformal prediction with concentration-specific intervals."""

    def __init__(self, predictions_df, n_bins=5):
        self.predictions_df = predictions_df
        self.n_bins = n_bins
        self.bin_intervals = {}

        # Create concentration bins
        self.predictions_df['bin'] = pd.qcut(
            self.predictions_df['PredictedValue'],
            q=n_bins,
            labels=False,
            duplicates='drop'
        )

        # Calculate conformal threshold per bin
        for bin_id in range(n_bins):
            bin_data = self.predictions_df[
                self.predictions_df['bin'] == bin_id
            ]

            # Calculate nonconformity scores for this bin
            nonconf_scores = bin_data['abs_residual'].values

            # 95% quantile for this bin
            alpha = 0.05
            n = len(nonconf_scores)
            q_level = (1 - alpha) * (1 + 1/n)
            threshold = np.quantile(nonconf_scores, q_level)

            # Store bin info
            self.bin_intervals[bin_id] = {
                'range': (bin_data['PredictedValue'].min(),
                         bin_data['PredictedValue'].max()),
                'threshold': threshold,
                'n_samples': len(bin_data)
            }

    def predict_interval(self, predicted_value, alpha=0.05):
        """Get adaptive interval for a prediction."""
        # Find which bin this prediction falls into
        for bin_id, info in self.bin_intervals.items():
            if info['range'][0] <= predicted_value <= info['range'][1]:
                threshold = info['threshold']
                return (predicted_value - threshold,
                        predicted_value + threshold)

        # Fallback to closest bin
        closest_bin = min(
            self.bin_intervals.keys(),
            key=lambda b: min(
                abs(predicted_value - self.bin_intervals[b]['range'][0]),
                abs(predicted_value - self.bin_intervals[b]['range'][1])
            )
        )
        threshold = self.bin_intervals[closest_bin]['threshold']
        return (predicted_value - threshold, predicted_value + threshold)

# Usage
adaptive_cp = AdaptiveConformalPredictor(predictions_df, n_bins=5)
interval = adaptive_cp.predict_interval(8.0)
print(f"Adaptive interval for 8.0%: [{interval[0]:.2f}, {interval[1]:.2f}]")
```

**Advantage**: Tighter intervals in low-uncertainty regions

**Expected improvement**: 10-30% narrower intervals in best-performing ranges (7-9%)

---

## Strategy 6: Calibration and Post-Processing (Low-Medium Impact)

### 6.1 Isotonic Calibration

Already partially implemented. Ensure it's enabled:

```python
# Calibrate predictions to reduce systematic bias
from sklearn.isotonic import IsotonicRegression

calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(y_pred_train, y_train)

# Calibrated predictions
y_pred_calibrated = calibrator.transform(y_pred_test)
```

**Expected improvement**: 5-10% if systematic bias exists

### 6.2 Temperature Scaling (for Neural Networks)

```python
class TemperatureScaling:
    """Calibrate neural network predictions."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, y_true):
        """Learn optimal temperature on validation set."""
        from scipy.optimize import minimize

        def loss(T):
            scaled = logits / T
            mse = np.mean((scaled - y_true) ** 2)
            return mse

        result = minimize(loss, x0=[1.0], bounds=[(0.1, 10.0)])
        self.temperature = result.x[0]

    def transform(self, logits):
        return logits / self.temperature
```

**Expected improvement**: 5-10% for neural network predictions

---

## Recommended Optimization Roadmap

### Phase 1: Quick Wins (1-2 days, ~20-30% improvement potential)

1. **Clean training data** using mislabel detection
   ```bash
   python main.py detect-mislabels --min-confidence 2
   python main.py optimize-models --models xgboost lightgbm catboost \
       --exclude-suspects reports/mislabel_analysis/suspicious_samples_min_confidence_2.csv \
       --trials 300 --gpu
   ```

2. **SHAP feature selection** - remove noisy features
   ```bash
   ./run_shap_analysis.sh --latest lightgbm
   python main.py optimize-models --shap-features models/*_shap_importance.csv \
       --shap-top-n 50 --trials 300 --gpu
   ```

3. **Enable sample weighting** in AutoGluon config

**Expected total**: 25-35% MAE reduction → interval width from ±1.83% to ~±1.2-1.3%

### Phase 2: Advanced Optimization (3-5 days, ~10-20% additional improvement)

4. **Extended hyperparameter search** (longer time limits)
5. **Custom ensemble stacking** (combine best models)
6. **Advanced spectral features** (domain knowledge)
7. **Concentration-weighted loss functions**

**Expected total**: Additional 10-20% → interval width ~±1.0-1.1%

### Phase 3: Cutting Edge (1-2 weeks, ~5-15% additional improvement)

8. **Quantile regression** for direct interval prediction
9. **Locally adaptive conformal prediction**
10. **Neural architecture search** for optimal network
11. **Collect more training data** in high-uncertainty ranges

**Expected total**: Additional 5-15% → interval width ~±0.9-1.0%

---

## Monitoring Improvements

After each optimization, run:

```bash
# Re-analyze uncertainty
python analyze_prediction_uncertainty.py \
    --predictions reports/predictions_STRATEGY_MODEL_TIMESTAMP.csv \
    --output-dir reports/uncertainty_analysis_optimized

# Compare
echo "=== BEFORE ==="
grep "Mean Absolute Error" reports/uncertainty_analysis_20251113/uncertainty_analysis_report.txt

echo "=== AFTER ==="
grep "Mean Absolute Error" reports/uncertainty_analysis_optimized/uncertainty_analysis_report.txt
```

Track key metrics:
- **MAE** (main target: reduce to <0.5%)
- **RMSE** (target: <0.7%)
- **Std Dev** (target: <0.7%)
- **95% interval width** (target: <±1.4%)
- **Coverage** (must stay at ~95%)

---

## Theoretical Limits

**Best-case scenario** with your current data:
- MAE: ~0.4-0.5% (limited by measurement noise)
- 95% interval width: ~±1.0-1.2%

**To go beyond this**, you would need:
- Better instrumentation (lower spectral noise)
- More precise reference measurements
- Larger sample size in extreme ranges
- Multi-modal data (combine LIBS with other techniques)

---

## Quick Test

Want to see if optimization will help? Run a quick test:

```bash
# 1. Baseline (current)
python main.py train --models lightgbm --strategy full_context --gpu

# 2. With mislabel exclusion
python main.py train --models lightgbm --strategy full_context --gpu \
    --exclude-suspects reports/mislabel_analysis/suspicious_samples_min_confidence_2.csv

# 3. Compare MAE/RMSE in reports/training_summary_*.csv
```

If MAE drops by >10%, you know cleaning mislabeled data will significantly reduce uncertainty!

---

*Remember: Conformal intervals are honest - they reflect your model's true performance. There's no shortcut except improving the model!*
