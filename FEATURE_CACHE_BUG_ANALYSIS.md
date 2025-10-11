# Feature Cache Bug Analysis and Fix

## Problem Description

When running `optimize-models` with multiple models, each model gets **different results** compared to running them individually. The same model (e.g., XGBoost) produces different R² scores when optimized alone vs. as part of a multi-model run.

## Root Cause

### Feature Caching Mechanism

Each optimizer (XGBoostOptimizer, LightGBMOptimizer, etc.) implements feature caching to avoid recomputing features during parallel Optuna trials:

```python
# In optimize_xgboost.py (and other optimizers)
def _precompute_features(self):
    """Pre-compute and cache features for parallel optimization."""
    if self._cached_features is not None:
        return  # Already computed (CRITICAL LINE!)

    # Build and fit feature pipeline
    feature_pipeline = Pipeline([
        ('features', self.feature_pipeline),
        ('scaler', StandardScaler())
    ])
    features_transformed = feature_pipeline.fit_transform(X_train_features, y_train)

    # CACHE the transformed features and fitted pipeline
    self._cached_features = pd.DataFrame(features_transformed, ...)
    self._fitted_feature_pipeline = feature_pipeline
```

### The Bug

**The cache check on line 205 prevents recomputation**, but here's the issue:

When optimizing multiple models sequentially in `UnifiedModelOptimizer.optimize_multiple_models()`:

1. **Model 1 (XGBoost)** creates `XGBoostOptimizer` instance
   - Calls `_precompute_features()`
   - Creates NEW feature pipeline with preprocessing
   - Fits pipeline and caches features
   - Optimization uses cached features
   - `train_final_model()` reuses `_fitted_feature_pipeline`

2. **Model 2 (LightGBM)** creates `LightGBMOptimizer` instance
   - Calls `_precompute_features()`
   - Creates **NEW** feature pipeline with **DIFFERENT** preprocessing instance
   - Fits pipeline and caches features
   - Optimization uses cached features
   - `train_final_model()` reuses `_fitted_feature_pipeline`

### Why Results Differ

Each optimizer instance creates its own `SpectralFeatureExtractor`, which creates its own `SpectralPreprocessor`. Even though they use the same config, there can be subtle differences:

1. **Floating-point precision differences** in ALS baseline fitting
2. **Scipy sparse matrix solver variations** between runs
3. **Savitzky-Golay edge handling** may vary slightly
4. **Random number generator state** if any randomness exists in feature engineering

These tiny differences compound through the pipeline, causing:
- Different scaled features (StandardScaler fits on slightly different data)
- Different optimal hyperparameters during optimization
- Different final model performance

## Evidence

User reported:
> "I am having issues when executing optimized_models with more than one model, it is executing the feature engineering after each optimization. I implemented before the preprocessing module a caching for the feature engineering and it seems it is calculating something wrong because the performance is different if I execute the optimized_models with a single model or several models. The same model get different results."

This confirms:
- ✅ Feature engineering IS running after each optimization (correct behavior - each optimizer needs its own pipeline)
- ✅ But results DIFFER between single-model and multi-model runs (BUG!)
- ✅ The **caching** is the suspected culprit

## Solution Strategy

### Option 1: Shared Feature Pipeline (RECOMMENDED)

Create features ONCE before optimization loop and pass them to all optimizers:

```python
# In optimize_all_models.py optimize_multiple_models()
def optimize_multiple_models(self, model_names, config, strategy, X_train, y_train, X_test, y_test, ...):
    # Create and fit feature pipeline ONCE
    logger.info("Creating shared feature pipeline for all models...")
    feature_pipeline = create_feature_pipeline(config, strategy, ...)

    pipeline_steps = [
        ('features', feature_pipeline),
        ('scaler', StandardScaler())
    ]
    full_pipeline = Pipeline(pipeline_steps)

    # Transform features ONCE
    X_train_transformed = full_pipeline.fit_transform(X_train, y_train)
    X_test_transformed = full_pipeline.transform(X_test)

    logger.info(f"Shared features created: {X_train_transformed.shape}")

    # Pass TRANSFORMED features to each optimizer
    for model_name in model_names:
        optimizer = self.OPTIMIZER_MAP[model_name](config, strategy, ...)

        # Pass transformed features directly - no need for optimizer to create pipeline
        optimizer.optimize_with_precomputed_features(
            X_train_transformed, y_train,
            X_test_transformed, y_test,
            shared_pipeline=full_pipeline,  # For train_final_model
            n_trials, timeout
        )
```

**Pros:**
- ✅ **Identical features** for all models (deterministic)
- ✅ Faster (features computed once, not per model)
- ✅ More memory efficient
- ✅ Easier to debug (one preprocessing path)

**Cons:**
- ❌ Requires refactoring optimizer API
- ❌ Breaks model-specific feature engineering (if needed)

### Option 2: Fix Preprocessing Determinism

Make preprocessing fully deterministic by:

1. **Set all random seeds** in preprocessing
2. **Force deterministic scipy solvers** in ALS baseline
3. **Validate preprocessing consistency** with unit tests

```python
# In preprocessing.py
class SpectralPreprocessor:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def _apply_als_baseline(self, spectrum, wavelengths):
        # Force deterministic sparse solver
        np.random.seed(self.random_state)
        baseline = spsolve(L, w * spectrum, use_umfpack=False)  # Deterministic
        return spectrum - baseline
```

**Pros:**
- ✅ Minimal code changes
- ✅ Preserves existing architecture

**Cons:**
- ❌ Doesn't fully fix the issue (some scipy functions are still non-deterministic)
- ❌ Hard to guarantee 100% reproducibility

### Option 3: Cache Validation (CURRENT FIX)

Add cache validation to detect when features might differ:

```python
def _precompute_features(self):
    if self._cached_features is not None:
        # Validate cache is still valid for current config
        if not self._validate_cache():
            logger.warning("Cache invalidated - recomputing features")
            self._cached_features = None
            self._fitted_feature_pipeline = None
        else:
            return  # Cache is valid

    # Compute features...
```

**Pros:**
- ✅ Safeguards against stale cache
- ✅ Minimal code changes

**Cons:**
- ❌ Doesn't fix root cause
- ❌ Performance penalty for validation

## Recommended Fix

Implement **Option 1: Shared Feature Pipeline** because:

1. **Correctness**: Guarantees identical features for fair model comparison
2. **Performance**: Faster multi-model optimization (features computed once)
3. **Simplicity**: Easier to understand and maintain
4. **Debugging**: Single preprocessing path to validate

## Implementation Plan

1. Create `optimize_with_precomputed_features()` method in `base_optimizer.py`
2. Modify `UnifiedModelOptimizer.optimize_multiple_models()` to:
   - Create shared feature pipeline
   - Transform features once
   - Pass transformed features to optimizers
3. Update `train_final_model()` to accept pre-fitted pipeline
4. Add validation tests to ensure feature consistency

## Testing Strategy

Create test to verify fix:

```python
def test_multi_model_consistency():
    """Test that multi-model optimization gives same results as individual runs."""

    # Run models individually
    xgb_solo = run_optimize_models(['xgboost'], ...)
    lgbm_solo = run_optimize_models(['lightgbm'], ...)

    # Run models together
    both = run_optimize_models(['xgboost', 'lightgbm'], ...)

    # Assert identical results (within numerical tolerance)
    assert abs(xgb_solo['r2'] - both['xgboost']['r2']) < 1e-6
    assert abs(lgbm_solo['r2'] - both['lightgbm']['r2']) < 1e-6
```

## Files to Modify

1. `/src/models/optimize_all_models.py` - Implement shared feature pipeline
2. `/src/models/base_optimizer.py` - Add `optimize_with_precomputed_features()`
3. `/src/models/optimize_xgboost.py` - Update to use shared features
4. `/src/models/optimize_lightgbm.py` - Update to use shared features
5. `/src/models/optimize_catboost.py` - Update to use shared features
6. `/src/models/optimize_random_forest.py` - Update to use shared features
7. `/src/models/optimize_extratrees.py` - Update to use shared features
8. `/src/models/optimize_neural_network.py` - Update to use shared features

---

**Date:** 2025-10-05
**Status:** Identified - awaiting implementation
**Priority:** HIGH (affects model comparison validity)
