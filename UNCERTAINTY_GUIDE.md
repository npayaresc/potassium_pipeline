# Prediction Uncertainty Guide

## Overview

This guide explains how to calculate and report prediction uncertainty for your AutoGluon potassium concentration predictions. Understanding uncertainty is crucial for production deployments where you need to know how confident you can be in each prediction.

## Key Findings from Validation Set Analysis

**Overall Performance:**
- **MAE**: 0.76% (mean absolute error)
- **RMSE**: 0.97% (root mean squared error)
- **Std Dev**: 0.96% (standard deviation of errors)
- **Coverage Range**: 2.44% - 13.07% potassium

**95% Confidence Interval**: ±1.83% (using conformal prediction)
**99% Confidence Interval**: ±3.03% (using conformal prediction)

## Four Complementary Approaches to Uncertainty

### 1. **Conformal Prediction Intervals** (Recommended for Production)

**What it is**: Mathematically guaranteed prediction intervals with calibrated coverage.

**When to use**: Default choice for reporting uncertainty to end users.

**How to use**:
```python
import pandas as pd
import numpy as np

# Load the uncertainty lookup table
uncertainty_table = pd.read_csv('reports/uncertainty_analysis_20251113/uncertainty_lookup_table.csv')

def get_prediction_interval(predicted_value, confidence='95'):
    """
    Get prediction interval for a given prediction.

    Args:
        predicted_value: Your model's prediction (e.g., 8.5% potassium)
        confidence: '95' or '99' for confidence level

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Find nearest prediction in lookup table
    idx = (uncertainty_table['predicted_value'] - predicted_value).abs().idxmin()
    row = uncertainty_table.iloc[idx]

    if confidence == '95':
        return (row['conformal_95_lower'], row['conformal_95_upper'])
    else:  # 99
        return (row['conformal_99_lower'], row['conformal_99_upper'])

# Example usage
prediction = 8.5  # 8.5% potassium
lower, upper = get_prediction_interval(prediction, confidence='95')
print(f"Prediction: {prediction:.2f}% ± [{lower:.2f}%, {upper:.2f}%]")
print(f"Interval width: {upper - lower:.2f}%")
```

**Interpretation**:
- For a prediction of 8.0%, the 95% confidence interval is [6.17%, 9.83%]
- This means: "We are 95% confident the true value lies within this range"
- Empirical validation coverage: 95.21% (very close to target 95%)

### 2. **Concentration-Specific Uncertainty**

**What it is**: Different concentration ranges have different error characteristics.

**When to use**: When you want to adjust confidence based on concentration range.

**Key findings by range**:

| Concentration Range | MAE | Std Dev | 95% Interval Width | Interpretation |
|---------------------|-----|---------|-------------------|----------------|
| **2.4% - 5.5%** (Low) | 0.71% | 0.86% | 3.18% | Moderate uncertainty, slight positive bias |
| **5.5% - 7.3%** (Low-Mid) | 0.82% | 0.98% | 3.55% | Highest uncertainty |
| **7.4% - 8.3%** (Mid) | 0.65% | 0.79% | 2.65% | **Best performance** |
| **8.3% - 9.4%** (Mid-High) | 0.72% | 0.88% | 3.13% | Good performance |
| **9.4% - 13.1%** (High) | 0.88% | 0.98% | 3.19% | Higher uncertainty, slight negative bias |

**How to use**:
```python
def get_concentration_specific_uncertainty(predicted_value):
    """Return MAE and std dev for the concentration range."""
    if predicted_value < 5.5:
        return {'mae': 0.71, 'std': 0.86, 'range': 'Low (2.4-5.5%)'}
    elif predicted_value < 7.3:
        return {'mae': 0.82, 'std': 0.98, 'range': 'Low-Mid (5.5-7.3%)'}
    elif predicted_value < 8.3:
        return {'mae': 0.65, 'std': 0.79, 'range': 'Mid (7.4-8.3%) - Best'}
    elif predicted_value < 9.4:
        return {'mae': 0.72, 'std': 0.88, 'range': 'Mid-High (8.3-9.4%)'}
    else:
        return {'mae': 0.88, 'std': 0.98, 'range': 'High (9.4-13.1%)'}

# Example
pred = 7.8
uncertainty = get_concentration_specific_uncertainty(pred)
print(f"Prediction: {pred:.2f}%")
print(f"Range: {uncertainty['range']}")
print(f"Expected error: ±{uncertainty['mae']:.2f}% (MAE)")
print(f"95% approximate interval: [{pred - 2*uncertainty['std']:.2f}%, {pred + 2*uncertainty['std']:.2f}%]")
```

### 3. **Heteroscedastic Uncertainty** (Concentration-Dependent)

**What it is**: Uncertainty that varies smoothly with predicted concentration.

**When to use**: When you need smooth uncertainty estimates across the entire range.

**How to use**:
```python
def get_heteroscedastic_uncertainty(predicted_value, uncertainty_table):
    """
    Get concentration-dependent standard deviation.

    Returns the expected standard deviation for a given prediction level.
    """
    # Find nearest prediction in lookup table
    idx = (uncertainty_table['predicted_value'] - predicted_value).abs().idxmin()
    row = uncertainty_table.iloc[idx]

    return {
        'std': row['std_prediction'],
        'mean_bias': row['mean_correction'],
        'interval_95': (predicted_value - 2*row['std_prediction'],
                        predicted_value + 2*row['std_prediction'])
    }

# Example
prediction = 8.5
uncert = get_heteroscedastic_uncertainty(prediction, uncertainty_table)
print(f"Prediction: {prediction:.2f}%")
print(f"Expected std: ±{uncert['std']:.2f}%")
print(f"Bias correction: {uncert['mean_bias']:.3f}%")
print(f"95% interval (±2σ): [{uncert['interval_95'][0]:.2f}%, {uncert['interval_95'][1]:.2f}%]")
```

**Key insight**: The heteroscedastic model shows that:
- Uncertainty ranges from 0.10% to 1.21% std dev across concentration range
- Generally lower uncertainty in mid-range (7-9%)
- Higher uncertainty at extremes (<5% and >10%)

### 4. **Bootstrap Confidence Intervals**

**What it is**: Resampling-based uncertainty estimates for overall model performance.

**When to use**: For reporting aggregate model uncertainty, not individual predictions.

**Key findings**:
- **Overall Std Dev**: 0.96% [95% CI: 0.88%, 1.05%]
- **Overall MAE**: 0.76% [95% CI: 0.70%, 0.82%]

**Interpretation**:
- We are 95% confident the model's true standard deviation is between 0.88% and 1.05%
- This reflects the model's general reliability, not a specific prediction

## Recommended Production Workflow

### Simple Approach (Quick Implementation)

```python
import pandas as pd

# Load uncertainty table once at startup
uncertainty_table = pd.read_csv('reports/uncertainty_analysis_20251113/uncertainty_lookup_table.csv')

def predict_with_uncertainty(model, X, uncertainty_table):
    """
    Make prediction with uncertainty estimate.

    Returns:
        Dictionary with prediction and 95% confidence interval
    """
    # Make prediction
    prediction = model.predict(X)

    # Get conformal interval
    idx = (uncertainty_table['predicted_value'] - prediction).abs().idxmin()
    row = uncertainty_table.iloc[idx]

    return {
        'prediction': float(prediction),
        'lower_95': float(row['conformal_95_lower']),
        'upper_95': float(row['conformal_95_upper']),
        'uncertainty': float((row['conformal_95_upper'] - row['conformal_95_lower']) / 2)
    }

# Example usage
result = predict_with_uncertainty(model, X_new, uncertainty_table)
print(f"Potassium: {result['prediction']:.2f}% ± {result['uncertainty']:.2f}%")
print(f"95% CI: [{result['lower_95']:.2f}%, {result['upper_95']:.2f}%]")
```

### Advanced Approach (Multi-Method)

```python
import pandas as pd
import numpy as np

class PotassiumUncertaintyEstimator:
    """Comprehensive uncertainty estimation for potassium predictions."""

    def __init__(self, uncertainty_table_path):
        self.uncertainty_table = pd.read_csv(uncertainty_table_path)

        # Concentration-specific statistics from validation analysis
        self.concentration_bins = {
            'low': {'range': (0, 5.5), 'mae': 0.71, 'std': 0.86},
            'low_mid': {'range': (5.5, 7.3), 'mae': 0.82, 'std': 0.98},
            'mid': {'range': (7.3, 8.3), 'mae': 0.65, 'std': 0.79},
            'mid_high': {'range': (8.3, 9.4), 'mae': 0.72, 'std': 0.88},
            'high': {'range': (9.4, 15), 'mae': 0.88, 'std': 0.98}
        }

    def estimate_uncertainty(self, prediction, method='conformal', confidence=95):
        """
        Estimate prediction uncertainty using specified method.

        Args:
            prediction: Predicted potassium concentration (%)
            method: 'conformal' (default), 'concentration_specific', or 'heteroscedastic'
            confidence: 95 or 99 (for conformal method)

        Returns:
            Dictionary with uncertainty estimates
        """
        # Find nearest entry in lookup table
        idx = (self.uncertainty_table['predicted_value'] - prediction).abs().idxmin()
        row = self.uncertainty_table.iloc[idx]

        result = {'prediction': prediction, 'method': method}

        if method == 'conformal':
            if confidence == 95:
                result['lower'] = row['conformal_95_lower']
                result['upper'] = row['conformal_95_upper']
            else:  # 99
                result['lower'] = row['conformal_99_lower']
                result['upper'] = row['conformal_99_upper']
            result['interval_width'] = result['upper'] - result['lower']
            result['confidence'] = confidence

        elif method == 'concentration_specific':
            # Find concentration bin
            for bin_name, bin_info in self.concentration_bins.items():
                if bin_info['range'][0] <= prediction < bin_info['range'][1]:
                    result['bin'] = bin_name
                    result['mae'] = bin_info['mae']
                    result['std'] = bin_info['std']
                    # 95% interval: ±2 std
                    result['lower'] = prediction - 2 * bin_info['std']
                    result['upper'] = prediction + 2 * bin_info['std']
                    break

        elif method == 'heteroscedastic':
            result['std'] = row['std_prediction']
            result['bias_correction'] = row['mean_correction']
            result['corrected_prediction'] = prediction + row['mean_correction']
            # 95% interval: ±2 std
            result['lower'] = prediction - 2 * row['std_prediction']
            result['upper'] = prediction + 2 * row['std_prediction']

        return result

    def comprehensive_report(self, prediction):
        """Generate comprehensive uncertainty report using all methods."""
        print(f"{'='*70}")
        print(f"PREDICTION UNCERTAINTY REPORT")
        print(f"{'='*70}")
        print(f"Predicted Potassium Concentration: {prediction:.2f}%\n")

        # Conformal 95%
        conf_95 = self.estimate_uncertainty(prediction, 'conformal', 95)
        print(f"95% Conformal Prediction Interval:")
        print(f"  [{conf_95['lower']:.2f}%, {conf_95['upper']:.2f}%]")
        print(f"  Interval width: ±{conf_95['interval_width']/2:.2f}%\n")

        # Concentration-specific
        conc_spec = self.estimate_uncertainty(prediction, 'concentration_specific')
        print(f"Concentration-Specific Uncertainty:")
        print(f"  Bin: {conc_spec.get('bin', 'unknown')}")
        print(f"  Expected MAE: {conc_spec.get('mae', 0):.2f}%")
        print(f"  Std Dev: {conc_spec.get('std', 0):.2f}%")
        print(f"  95% interval: [{conc_spec['lower']:.2f}%, {conc_spec['upper']:.2f}%]\n")

        # Heteroscedastic
        hetero = self.estimate_uncertainty(prediction, 'heteroscedastic')
        print(f"Heteroscedastic Model:")
        print(f"  Predicted Std: {hetero['std']:.2f}%")
        print(f"  Bias correction: {hetero['bias_correction']:.3f}%")
        print(f"  95% interval: [{hetero['lower']:.2f}%, {hetero['upper']:.2f}%]")
        print(f"{'='*70}")

# Example usage
estimator = PotassiumUncertaintyEstimator('reports/uncertainty_analysis_20251113/uncertainty_lookup_table.csv')

# Quick estimate
result = estimator.estimate_uncertainty(8.5, method='conformal', confidence=95)
print(f"Prediction: {result['prediction']:.2f}% [{result['lower']:.2f}%, {result['upper']:.2f}%]")

# Comprehensive report
estimator.comprehensive_report(8.5)
```

## Interpretation Guidelines

### For Scientists/Analysts
- **Use conformal intervals** for hypothesis testing and decision making
- Report: "Predicted K = 8.5% ± 1.8% (95% CI: [6.7%, 10.3%])"
- The conformal method provides mathematically guaranteed coverage

### For Production Systems
- **Use concentration-specific uncertainty** for quality control thresholds
- Flag predictions with high uncertainty (>1.0% std) for manual review
- Apply different acceptance criteria by concentration range

### For End Users
- **Simple reporting**: "Potassium: 8.5% ± 1.8%"
- Add context: "This prediction has [low/medium/high] uncertainty based on concentration range"
- Provide recommendation: "Confidence level suitable for [screening/quantitative analysis]"

## Quality Control Recommendations

**Flag for Review If:**
1. Prediction interval width > 3% (indicates high uncertainty)
2. Prediction falls in low-confidence ranges (<5% or >10%)
3. Heteroscedastic std > 1.0% (upper 10% of uncertainty distribution)

**High Confidence Predictions:**
- Mid-range concentrations (7-9%)
- Interval width < 2%
- Heteroscedastic std < 0.8%

## Files Generated

1. **`uncertainty_analysis_report.txt`**: Complete statistical report
2. **`uncertainty_lookup_table.csv`**: Interpolation table for all uncertainty methods
3. **`residual_statistics_by_concentration.csv`**: Detailed statistics per concentration bin
4. **`uncertainty_analysis_plots.png`**: Comprehensive visualization
5. **`concentration_binned_uncertainty.png`**: Uncertainty by concentration range

## Re-running Analysis

To update uncertainty estimates with new validation data:

```bash
python analyze_prediction_uncertainty.py \
    --predictions reports/predictions_STRATEGY_autogluon_TIMESTAMP.csv \
    --model-path models/autogluon/STRATEGY_TIMESTAMP \
    --output-dir reports/uncertainty_analysis_TIMESTAMP
```

## Further Reading

- **Conformal Prediction**: Provides distribution-free prediction intervals
- **Heteroscedasticity**: When error variance changes with predicted value
- **Bootstrap Methods**: Resampling for uncertainty quantification
- **Isotonic Regression**: Monotonic calibration for concentration-dependent effects

---

*Last updated: 2025-11-13*
*Based on validation set: 313 samples, concentration range 2.4-13.1% potassium*
