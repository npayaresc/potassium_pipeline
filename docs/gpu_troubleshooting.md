# GPU AutoGluon Troubleshooting Guide

## 🐛 Common Issue: CatBoost/LightGBM Exceptions with GPU + best_quality

**Problem**: When using `--gpu` flag with AutoGluon's `best_quality` preset, CatBoost and sometimes LightGBM throw exceptions, but `extreme_quality` works fine.

**Root Cause**: The `best_quality` preset uses more aggressive hyperparameters that can cause:
- GPU memory conflicts
- Version compatibility issues between CatBoost/LightGBM and CUDA
- Race conditions in parallel GPU training

## ✅ Automatic Solution (Implemented)

The pipeline now automatically switches to GPU-safe configurations when GPU is enabled:

### 1. **Automatic Preset Switching**
```python
# When --gpu is used, automatically switches to:
presets = 'extreme_quality'  # Instead of 'best_quality'
```

### 2. **GPU-Safe Hyperparameters**
Less aggressive, more stable hyperparameters for:
- **CatBoost**: Lower depth, fewer iterations, higher regularization
- **LightGBM**: Fewer leaves, conservative learning rates
- **XGBoost**: Stable GPU configurations

### 3. **Smart Model Exclusion**
Can automatically exclude problematic models when GPU is enabled.

## 🛠️ Manual Configuration Options

### Option 1: Force Specific Preset for GPU
```python
# In pipeline_config.py
class AutoGluonConfig(BaseModel):
    gpu_safe_preset: str = 'extreme_quality'  # or 'good_quality'
```

### Option 2: Exclude Problematic Models
```python
# Add models that consistently fail with your GPU setup
gpu_excluded_models: List[str] = ['CAT']  # Exclude CatBoost entirely
# or
gpu_excluded_models: List[str] = ['CAT', 'GBM']  # Exclude both CatBoost and LightGBM
```

### Option 3: Custom GPU-Safe Hyperparameters
```python
# Override the gpu_safe_hyperparameters in config
gpu_safe_hyperparameters = {
    'CAT': [
        # Very conservative CatBoost settings
        {'iterations': 200, 'learning_rate': 0.1, 'depth': 4, 'l2_leaf_reg': 5.0}
    ],
    'GBM': [
        # Very conservative LightGBM settings  
        {'num_boost_round': 200, 'learning_rate': 0.1, 'num_leaves': 15}
    ]
}
```

## 🔍 Diagnostic Steps

### Step 1: Check GPU Compatibility
```bash
# Verify CUDA and GPU libraries
python -c "
import torch
print('PyTorch CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)

try:
    import lightgbm as lgb
    print('LightGBM version:', lgb.__version__)
except ImportError:
    print('LightGBM not available')

try:
    import catboost as cb  
    print('CatBoost version:', cb.__version__)
except ImportError:
    print('CatBoost not available')
"
```

### Step 2: Test Individual Models
```bash
# Test with only specific models
python main.py autogluon --gpu  # Should now work automatically

# If still failing, try excluding models:
# Edit pipeline_config.py:
# gpu_excluded_models: List[str] = ['CAT']  # Exclude CatBoost
```

### Step 3: Monitor GPU Memory
```bash
# While training is running, monitor GPU usage
nvidia-smi -l 1
```

## 🎯 Recommended GPU Configurations

### For Stable GPU Training (Default)
```python
# Current automatic configuration when --gpu is used:
presets = 'extreme_quality'  # Stable, well-tested
hyperparameters = gpu_safe_hyperparameters  # Conservative settings
```

### For Maximum Performance (If stable)
```python
# If your system is stable with aggressive settings:
gpu_safe_preset = 'best_quality'  # Keep original preset
gpu_excluded_models = []  # Don't exclude any models
```

### For Minimal GPU Usage
```python
# If you have GPU memory constraints:
gpu_excluded_models = ['CAT', 'GBM']  # CPU-only boosting
presets = 'good_quality'  # Even more conservative
```

## 🔧 Environment-Specific Fixes

### CUDA Memory Issues
```bash
# Set environment variables before running
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
```

### CatBoost GPU Issues
```bash
# If CatBoost consistently fails:
pip install --upgrade catboost
# or 
pip install catboost[gpu]
```

### LightGBM GPU Issues
```bash
# If LightGBM consistently fails:
pip install --upgrade lightgbm
# or rebuild with GPU support:
pip install lightgbm --install-option=--gpu
```

## 📊 Performance Comparison

| Configuration | Stability | Speed | Accuracy | Memory Usage |
|---------------|-----------|-------|----------|--------------|
| `extreme_quality` (GPU) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| `best_quality` (GPU) | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| `best_quality` (CPU) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## 🚀 Quick Fixes

### If you're still getting exceptions:

1. **Quick Fix 1**: Use the automatic solution (already implemented)
   ```bash
   python main.py autogluon --gpu
   # Now automatically uses extreme_quality + safe hyperparameters
   ```

2. **Quick Fix 2**: Exclude problematic models
   ```python
   # In pipeline_config.py, add:
   gpu_excluded_models: List[str] = ['CAT']
   ```

3. **Quick Fix 3**: Force extreme_quality always
   ```python
   # In pipeline_config.py:
   presets: str = 'extreme_quality'  # Change from 'best_quality'
   ```

4. **Quick Fix 4**: CPU fallback
   ```bash
   python main.py autogluon  # Remove --gpu flag
   ```

## 📞 Further Support

If you're still experiencing issues:

1. Check the training logs for specific error messages
2. Try the diagnostic steps above
3. Consider your GPU memory and CUDA version compatibility
4. Use the conservative configurations provided

The automatic GPU-safe configuration should resolve most CatBoost/LightGBM exceptions while maintaining good performance.