# Prediction Uncertainty - Quick Reference

## Current Status

**Your AutoGluon model validation performance:**
- MAE: 0.76%
- RMSE: 0.97%
- **95% Confidence Interval Width: ±1.83%**

**Example**: For a prediction of 8.0%, the interval is [6.17%, 9.83%]

---

## How to Reduce Uncertainty (Narrow the Intervals)

### Quick Answer
Run the automated optimization script:

```bash
./optimize_for_uncertainty_reduction.sh
```

This implements **Phase 1 Quick Wins**:
1. ✓ Detects and excludes mislabeled samples
2. ✓ Performs SHAP feature selection (top 50 features)
3. ✓ Runs hyperparameter optimization (300 trials)
4. ✓ Analyzes uncertainty improvements

**Expected result**: 25-35% MAE reduction → interval width from ±1.83% to **±1.2-1.3%**

---

## Manual Optimization Steps

If you prefer step-by-step control:

### Step 1: Clean Training Data (Highest Impact)
```bash
# Detect mislabeled samples
python main.py detect-mislabels --min-confidence 2 --data-parallel --feature-parallel

# Train without suspicious samples
python main.py optimize-models \
    --models xgboost lightgbm catboost \
    --strategy full_context \
    --trials 300 \
    --gpu \
    --exclude-suspects reports/mislabel_analysis/suspicious_samples_min_confidence_2.csv
```

**Expected improvement**: 10-30% MAE reduction

### Step 2: SHAP Feature Selection
```bash
# Run SHAP analysis
./run_shap_analysis.sh --latest lightgbm

# Train with top features
python main.py optimize-models \
    --models xgboost lightgbm catboost \
    --strategy full_context \
    --trials 300 \
    --gpu \
    --shap-features models/full_context_lightgbm_*_shap_importance.csv \
    --shap-top-n 50
```

**Expected improvement**: 5-20% MAE reduction

### Step 3: Extended AutoGluon Training
```bash
# Edit src/config/pipeline_config.py:
# time_limit: 18000  # 5 hours instead of 3

# Run AutoGluon
python main.py autogluon --gpu --data-parallel --feature-parallel
```

**Expected improvement**: 5-15% MAE reduction

---

## Analyzing Improvements

After any optimization, re-analyze uncertainty:

```bash
python analyze_prediction_uncertainty.py \
    --predictions reports/predictions_full_context_MODEL_TIMESTAMP.csv \
    --output-dir reports/uncertainty_analysis_optimized
```

Compare with baseline:
```bash
# Baseline
grep "Mean Absolute Error" reports/uncertainty_analysis_20251113/uncertainty_analysis_report.txt

# Optimized
grep "Mean Absolute Error" reports/uncertainty_analysis_optimized/uncertainty_analysis_report.txt
```

---

## Optimization Roadmap

### Phase 1: Quick Wins (1-2 days)
- Mislabel exclusion
- SHAP feature selection
- Extended hyperparameter search

**Target**: ±1.2-1.3% interval width (30% improvement)

### Phase 2: Advanced (3-5 days)
- Custom ensemble stacking
- Advanced spectral features
- Concentration-weighted loss functions

**Target**: ±1.0-1.1% interval width (45% improvement)

### Phase 3: Cutting Edge (1-2 weeks)
- Quantile regression
- Locally adaptive conformal prediction
- Neural architecture search
- Collect more training data

**Target**: ±0.9-1.0% interval width (50% improvement)

---

## Key Insight

**Conformal prediction intervals are honest** - they reflect your model's true performance. The only way to narrow them is to:

1. **Reduce prediction errors** (lower MAE/RMSE)
2. **Reduce error variance** (more consistent errors)
3. **Use adaptive methods** (different intervals for different regions)

You cannot "cheat" conformal intervals - they have guaranteed coverage!

---

## Theoretical Limits

With your current data and instrumentation:
- **Best achievable MAE**: ~0.4-0.5%
- **Best achievable interval width**: ~±1.0-1.2%

To go beyond this, you would need:
- Better instrumentation (lower spectral noise)
- More precise reference measurements
- Larger training dataset
- Multi-modal data sources

---

## Documentation Files

1. **`REDUCE_UNCERTAINTY_GUIDE.md`** - Comprehensive 400+ line guide with:
   - All optimization strategies explained in detail
   - Code examples for each approach
   - Expected improvements for each method
   - Implementation roadmap

2. **`UNCERTAINTY_GUIDE.md`** - How to use uncertainty estimates:
   - Four complementary approaches
   - Production workflow examples
   - Interpretation guidelines

3. **`reports/uncertainty_analysis_20251113/`** - Current uncertainty analysis:
   - `uncertainty_lookup_table.csv` - Use this for production
   - `uncertainty_analysis_report.txt` - Full statistical report
   - Visualization plots

---

## Quick Test

Want to see if optimization will help? Run a quick test:

```bash
# Baseline
python main.py train --models lightgbm --strategy full_context --gpu

# With mislabel exclusion
python main.py train --models lightgbm --strategy full_context --gpu \
    --exclude-suspects reports/mislabel_analysis/suspicious_samples_min_confidence_2.csv

# Compare MAE in reports/training_summary_*.csv
```

If MAE drops by >10%, optimization will significantly reduce uncertainty!

---

## Need Help?

- **Getting started**: Run `./optimize_for_uncertainty_reduction.sh`
- **Understanding methods**: Read `REDUCE_UNCERTAINTY_GUIDE.md`
- **Using uncertainty in production**: Read `UNCERTAINTY_GUIDE.md`
- **Current uncertainty**: Check `reports/uncertainty_analysis_20251113/`

---

*Remember: Every 10% reduction in MAE translates to ~10% narrower prediction intervals!*
