# Physics-Informed Features - Quick Reference

## New Features (6 per peak)

### 1. FWHM (Full Width at Half Maximum)
**Feature Name**: `{element}_fwhm_{peak_index}`
**Example**: `K_I_fwhm_0` (K I line at 766.49 nm)

**Physical Meaning**: Total peak broadening (temperature + electron density + instrumental)
**Units**: nm (nanometers)
**Typical Range**: 0.1 - 2.0 nm for LIBS
**Interpretation**:
- **< 0.3 nm**: Narrow peak, low temperature/density
- **0.3 - 1.0 nm**: Normal LIBS plasma
- **> 1.0 nm**: Broad peak, high temperature/density or poor resolution

**Use in ML**: Helps distinguish concentration from plasma conditions

---

### 2. Gamma (Stark Broadening)
**Feature Name**: `{element}_gamma_{peak_index}`
**Example**: `K_I_gamma_0`

**Physical Meaning**: Lorentzian width parameter (electron density indicator)
**Units**: nm (HWHM - Half Width at Half Maximum)
**Typical Range**: 0.05 - 1.0 nm
**Relationship**: FWHM = 2 × gamma (for Lorentzian)

**Interpretation**:
- **Higher gamma** → Higher electron density → More collisions
- **Linear with concentration** (better than raw intensity)

**Use in ML**: Primary indicator of plasma electron density, correlates with analyte concentration

---

### 3. Fit Quality (R²)
**Feature Name**: `{element}_fit_quality_{peak_index}`
**Example**: `K_I_fit_quality_0`

**Physical Meaning**: How well Lorentzian fits the observed peak
**Units**: Dimensionless (0 to 1)
**Typical Range**: 0.5 - 1.0 for good data

**Interpretation**:
- **> 0.9**: Excellent fit, reliable features
- **0.7 - 0.9**: Good fit, acceptable
- **0.5 - 0.7**: Fair fit, use with caution
- **< 0.5**: Poor fit, data quality issue

**Use in ML**: Quality control - weight or filter samples by fit quality

---

### 4. Peak Asymmetry
**Feature Name**: `{element}_asymmetry_{peak_index}`
**Example**: `K_I_asymmetry_0`

**Physical Meaning**: Self-absorption indicator (reabsorption of emitted light)
**Units**: Dimensionless (-1 to +1)
**Typical Range**: -0.3 to +0.5

**Interpretation**:
- **+0.3 to +1.0**: Strong right-skew, high self-absorption (HIGH concentration)
- **-0.1 to +0.1**: Symmetric, optically thin plasma (LINEAR range)
- **-0.5 to -0.3**: Left-skew (rare, indicates interference)

**Use in ML**: Corrects non-linearity at high concentrations, identifies saturation

---

### 5. Amplitude
**Feature Name**: `{element}_amplitude_{peak_index}`
**Example**: `K_I_amplitude_0`

**Physical Meaning**: Peak height (maximum intensity above baseline)
**Units**: Arbitrary intensity units
**Typical Range**: Depends on spectrometer, typically 100 - 10,000

**Interpretation**:
- **Higher amplitude** → More emission → Higher concentration (in linear range)
- **Saturates** at high concentration due to self-absorption

**Use in ML**: Alternative to peak area, less affected by fitting errors

---

### 6. Absorption Index
**Feature Name**: `{element}_absorption_index_{peak_index}`
**Example**: `K_I_absorption_index_0`

**Physical Meaning**: Combined broadening + asymmetry (self-absorption strength)
**Units**: nm (dimensionless × nm)
**Formula**: FWHM × |Asymmetry|
**Typical Range**: 0 - 2.0

**Interpretation**:
- **< 0.1**: Minimal self-absorption (linear response)
- **0.1 - 0.5**: Moderate self-absorption (slight saturation)
- **> 0.5**: Strong self-absorption (non-linear, saturated)

**Use in ML**: Single feature for absorption correction, helps models detect saturation

---

## Feature Naming Convention

### Pattern
```
{element}_{parameter}_{peak_index}
```

### Examples
- `K_I_fwhm_0` - FWHM of K I first peak (766.49 nm)
- `K_I_fwhm_1` - FWHM of K I second peak (769.90 nm)
- `K_I_404_asymmetry_0` - Asymmetry of K I violet line (404.41 nm)
- `CA_II_393_fit_quality_0` - Fit quality of Ca II line (393.37 nm)

---

## Most Important Features for Potassium

### Primary K Line (766.49 nm)
1. **K_I_fwhm_0** - Broadening of strongest line
2. **K_I_asymmetry_0** - Self-absorption (critical for high concentrations)
3. **K_I_fit_quality_0** - Data quality check
4. **K_I_gamma_0** - Plasma density indicator
5. **K_I_absorption_index_0** - Overall absorption strength

### Secondary K Line (769.90 nm)
6. **K_I_fwhm_1** - Confirms broadening trend
7. **K_I_asymmetry_1** - Cross-validation of self-absorption

### Violet K Line (404.41 nm)
8. **K_I_404_asymmetry_0** - High-energy line (less affected by self-absorption)
9. **K_I_404_fit_quality_0** - Confirms K detection

---

## Recommended Feature Combinations

### For Self-Absorption Correction
```python
# Identify high-concentration samples
high_absorption = (K_I_asymmetry_0 > 0.3) & (K_I_absorption_index_0 > 0.5)

# Use different model or correction for these samples
```

### For Quality Filtering
```python
# Keep only high-quality fits
good_quality = (K_I_fit_quality_0 > 0.8) & (K_I_404_fit_quality_0 > 0.7)
```

### For Concentration Prediction
```python
# Best features for linear range (low concentration)
features = ['K_I_amplitude_0', 'K_I_gamma_0', 'K_I_fwhm_0']

# Additional features for non-linear range (high concentration)
features += ['K_I_asymmetry_0', 'K_I_absorption_index_0']
```

---

## Interpretation Tips

### High Concentration (> 0.4% K)
**Expect**:
- High asymmetry (> 0.3)
- High absorption index (> 0.5)
- Broader FWHM (> 1.0 nm)

**Model should**:
- Use asymmetry to correct saturation
- Rely less on amplitude, more on gamma

### Low Concentration (< 0.2% K)
**Expect**:
- Low asymmetry (< 0.1)
- Low absorption index (< 0.1)
- Narrower FWHM (< 0.8 nm)

**Model should**:
- Use amplitude directly (linear range)
- Gamma provides plasma normalization

### Poor Data Quality
**Indicators**:
- Low fit quality (< 0.6)
- Extreme asymmetry (< -0.5 or > 0.8)
- Very broad FWHM (> 3.0 nm)

**Action**:
- Filter out or down-weight these samples
- Check for spectral interference

---

## Feature Engineering Recipes

### 1. Self-Absorption Corrected Intensity
```python
# Correct amplitude for self-absorption
corrected_amplitude = K_I_amplitude_0 / (1 + K_I_absorption_index_0)
```

### 2. Concentration-Range Indicator
```python
# Binary feature: high vs. low concentration range
high_range = (K_I_asymmetry_0 > 0.2).astype(int)
```

### 3. Data Quality Score
```python
# Combined quality metric (0 to 1)
quality_score = (K_I_fit_quality_0 + K_I_404_fit_quality_0) / 2
```

### 4. Multi-Line Consistency
```python
# Check if both K lines show similar asymmetry
asymmetry_consistent = abs(K_I_asymmetry_0 - K_I_asymmetry_1) < 0.2
```

---

## Quick Diagnostic

### Problem: Low R² at high concentrations
**Check**: `K_I_asymmetry_0`
**If high** (> 0.3): Self-absorption is causing saturation
**Solution**: Include asymmetry features in model

### Problem: High variance in predictions
**Check**: `K_I_fit_quality_0`
**If variable**: Poor data quality or spectral interference
**Solution**: Filter samples with fit_quality < 0.7

### Problem: Model overfits training data
**Check**: Feature count
**If too many** (> 400): Dimensionality curse
**Solution**: Feature selection to keep top 50-60%

---

## Expected Feature Importance (Top 10)

Based on LIBS physics and potassium chemistry:

1. **K_I_peak_0** (original area) - Primary signal
2. **K_I_amplitude_0** - Peak height
3. **K_I_gamma_0** - Plasma density indicator
4. **K_I_asymmetry_0** - Self-absorption correction
5. **K_I_fwhm_0** - Broadening indicator
6. **K_I_absorption_index_0** - Saturation indicator
7. **K_I_fit_quality_0** - Quality weight
8. **K_I_404_asymmetry_0** - Cross-validation
9. **K_C_ratio** (original) - Normalization
10. **K_I_fwhm_1** - Consistency check

---

**Reference**: PHYSICS_INFORMED_FEATURES_IMPLEMENTATION.md for full details
