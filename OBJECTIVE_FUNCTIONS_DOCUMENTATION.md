# Objective Functions and Sample Weighting Documentation

## Overview

This document explains the rationale behind the objective functions and sample weighting strategies used in the magnesium prediction pipeline, based on spectroscopic domain knowledge and machine learning best practices.

## Spectroscopic Background

### Magnesium Detection Challenges

1. **Low Concentrations**: Magnesium typically appears in plant material at 0.1-0.5% dry matter, making detection challenging
2. **Spectral Interference**: P lines can be interfered with by other elements (N, K, Ca) 
3. **Matrix Effects**: Soil and plant matrix composition affects spectral intensity
4. **Signal-to-Noise Ratio**: Low P concentrations result in weak spectral signals

### Why Weighted Approaches Matter

**Agricultural Relevance**: 
- Low P (0.1-0.2%): Critical deficiency range requiring accurate detection
- Medium P (0.2-0.4%): Optimal growing range  
- High P (0.4-0.5%): Excess range, less common but agriculturally significant

**Spectroscopic Considerations**:
- Weaker signals at low concentrations need higher model sensitivity
- Baseline correction becomes more critical at low concentrations
- Spectral line broadening affects peak detection consistency

## Objective Function Categories

### 1. Distribution-Based Weighting (`distribution_based`)

**Rationale**: Emphasizes underrepresented concentration ranges using inverse density weighting.

**Spectroscopic Justification**:
- Accounts for natural sampling bias (more samples at common P levels)
- Ensures model learns rare but important concentration ranges
- Mimics analytical chemistry approach of calibrating across full range

**Configuration Parameters**:
```python
distribution_bins: [0, 20, 40, 60, 80, 100]  # Percentile bins
kde_epsilon: 1e-8  # Smoothing for density estimation
min_sample_weight: 0.2  # Prevents extreme downweighting
max_sample_weight: 5.0  # Prevents extreme upweighting
```

### 2. Hybrid Weighted (`hybrid_weighted`)

**Rationale**: Combines distribution-based weighting with domain knowledge about critical P ranges.

**Agricultural Justification**:
- Low P (bottom quartile): Multiplier 1.3 - Critical for deficiency detection
- High P (top quartile): Multiplier 1.2 - Important for excess detection
- Medium P: Standard weighting - Most common, well-represented range

**Configuration Parameters**:
```python
low_range_modifier: 1.3  # Emphasis on low P concentrations
high_range_modifier: 1.2  # Emphasis on high P concentrations
```

### 3. Weighted R² (`weighted_r2`)

**Rationale**: Applies higher weights to extreme quartiles for balanced performance across concentration ranges.

**Statistical Justification**:
- Bottom quartile (25%): Weight 2.0 - Deficiency detection priority
- Top quartile (75%): Weight 1.5 - Toxicity/excess detection
- Middle quartiles: Weight 1.0 - Standard accuracy expectations

### 4. Quantile Weighted (`quantile_weighted`)

**Rationale**: Ensures good R² performance within each concentration quintile.

**Analytical Chemistry Parallel**:
- Similar to validation across calibration range in analytical methods
- Prevents models that excel at one range while failing at others
- Ensures consistent analytical performance across working range

**Configuration Parameters**:
```python
n_quantiles: 5  # Number of concentration bands to evaluate
min_quantile_samples: 2  # Minimum samples per band for valid evaluation
```

### 5. MAPE-Focused (`mape_focused`)

**Rationale**: Minimizes Mean Absolute Percentage Error, critical for low concentrations where small absolute errors become large relative errors.

**Spectroscopic Justification**:
- At 0.1% P, a 0.05% error = 50% relative error (unacceptable)
- At 0.4% P, a 0.05% error = 12.5% relative error (acceptable)
- MAPE weighting addresses this concentration-dependent error tolerance

**Configuration Parameters**:
```python
mape_low_percentile: 33.0   # Lower third of data range
mape_high_percentile: 66.0  # Upper third of data range
low_mape_weight: 0.7        # Higher emphasis on low concentration accuracy
medium_mape_weight: 0.3     # Lower emphasis on medium concentrations
```

## Sample Weighting Methods

### Legacy Method
Based on fixed concentration ranges derived from agricultural magnesium guidelines:

```python
legacy_ranges: [
    (0.1, 0.15, 2.5),  # Very low P - Critical deficiency
    (0.15, 0.20, 2.0), # Low P - Deficiency likely  
    (0.20, 0.30, 1.2), # Medium-low P - Moderate emphasis
    (0.30, 0.40, 1.5), # Medium-high P - Moderate emphasis
    (0.40, 0.50, 2.5)  # High P - Excess/toxicity risk
]
```

### Improved Method
Uses data-driven percentiles with agriculturally-informed weights:

```python
improved_percentiles: [10, 25, 50, 75, 90]  # Data-driven boundaries
improved_weights: [3.0, 2.2, 1.8, 1.0, 1.5, 2.5]  # Agricultural importance
```

### Distribution-Based Method
Automatically weights based on sample frequency in concentration space.

## Configuration Recommendations

### For Research/Method Development:
```python
use_data_driven_thresholds: True
objective_function_name: 'distribution_based'
sample_weight_method: 'distribution_based'
```

### For Agricultural Applications:
```python
use_data_driven_thresholds: False  # Use agronomic thresholds
objective_function_name: 'hybrid_weighted'
sample_weight_method: 'hybrid'
low_concentration_threshold: 0.15  # Below this = deficiency
high_concentration_threshold: 0.35  # Above this = excess risk
```

### For Analytical Validation:
```python
objective_function_name: 'quantile_weighted'
sample_weight_method: 'weighted_r2'
n_quantiles: 5  # Validate across 5 concentration bands
```

## Validation and Monitoring

### Automatic Validation
The `ObjectiveConfig.validate_concentration_ranges()` method automatically:

1. **Checks threshold validity** against actual data range
2. **Identifies empty concentration bins** in legacy ranges  
3. **Recommends data-driven alternatives** when fixed thresholds fail
4. **Warns about narrow data ranges** that may not benefit from complex weighting

### Example Validation Output:
```python
{
    'data_stats': {'min': 0.12, 'max': 0.48, 'mean': 0.28, 'std': 0.09},
    'fixed_thresholds_valid': True,
    'warnings': [],
    'recommendations': ['Consider fewer bins for narrow range (<0.1)']
}
```

## Best Practices

1. **Start with distribution_based** for initial model development
2. **Use hybrid_weighted** for production agricultural models
3. **Validate thresholds** against each new dataset using built-in validation
4. **Monitor performance** across concentration quartiles using quantile_weighted
5. **Adjust weights** based on analytical requirements and error tolerance

## References

- Agricultural magnesium deficiency thresholds: Soil Science Society of America
- Spectroscopic interference patterns: Applied Spectroscopy literature
- LIBS matrix effects: Journal of Analytical Atomic Spectrometry
- Machine learning sample weighting: Elements of Statistical Learning, Hastie et al.