# Dimension Reduction Guide

This guide explains how to use the modular dimension reduction system in the magnesium pipeline.

## Overview

The pipeline now supports multiple dimensionality reduction strategies that can be easily configured without code changes:

- **PCA** (Principal Component Analysis) - Linear, variance-preserving
- **PLS** (Partial Least Squares) - Supervised, target-aware reduction
- **Autoencoder** - Non-linear, neural network-based
- **Feature Clustering** - Groups similar features, selects representatives

## Configuration

### In `pipeline_config.py`

For standard models:
```python
# Enable dimension reduction
use_dimension_reduction: bool = True

# Configure the method
dimension_reduction: DimensionReductionConfig = DimensionReductionConfig(
    method='pls',  # Options: 'pca', 'pls', 'autoencoder', 'feature_clustering'
    n_components=10,  # Number of components (or variance for PCA)
    pls_params={'scale': True}
)
```

For AutoGluon:
```python
autogluon: AutoGluonConfig = AutoGluonConfig(
    use_dimension_reduction=True,
    dimension_reduction=DimensionReductionConfig(
        method='autoencoder',
        n_components=12,
        autoencoder_params={
            'hidden_layers': [64, 32],
            'epochs': 100,
            'device': 'auto'  # Uses GPU if available
        }
    )
)
```

## Method-Specific Parameters

### PCA
```python
dimension_reduction = DimensionReductionConfig(
    method='pca',
    n_components=0.95,  # Keep 95% variance (float) or fixed components (int)
    pca_params={
        'random_state': 42,
        'whiten': False
    }
)
```

### PLS (Partial Least Squares)
```python
dimension_reduction = DimensionReductionConfig(
    method='pls',
    n_components=10,  # Must be integer
    pls_params={
        'scale': True,  # Scale features before PLS
        'max_iter': 500
    }
)
```

**Note:** PLS requires target values during fitting, making it supervised.

### Autoencoder
```python
dimension_reduction = DimensionReductionConfig(
    method='autoencoder',
    n_components=8,  # Bottleneck layer size
    autoencoder_params={
        'hidden_layers': [100, 50, 20],  # Encoder architecture
        'epochs': 150,
        'batch_size': 32,
        'learning_rate': 0.001,
        'device': 'auto'  # 'cpu', 'cuda', or 'auto'
    }
)
```

### Feature Clustering
```python
dimension_reduction = DimensionReductionConfig(
    method='feature_clustering',
    n_components='auto',  # Auto-select using silhouette score
    clustering_params={
        'method': 'kmeans',
        'n_init': 10
    }
)
```

## Usage Examples

### Example 1: Reduce from 100 to 10 features using PLS
```bash
# Edit pipeline_config.py to set:
# dimension_reduction.method = 'pls'
# dimension_reduction.n_components = 10

python main.py train --gpu
```

### Example 2: Use autoencoder for non-linear reduction
```bash
# Edit pipeline_config.py to set:
# dimension_reduction.method = 'autoencoder'
# dimension_reduction.n_components = 12

python main.py autogluon --gpu
```

### Example 3: Compare different methods
```python
# Run the test script
python test_dimension_reduction.py
```

## Benefits of Each Method

### PCA
- **Pros:** Fast, interpretable, preserves variance
- **Cons:** Linear only, unsupervised
- **Best for:** General purpose, when you need interpretability

### PLS
- **Pros:** Supervised, considers target variable, good for regression
- **Cons:** Requires target values, linear
- **Best for:** When prediction accuracy is paramount

### Autoencoder
- **Pros:** Non-linear, can capture complex patterns
- **Cons:** Slower, requires tuning, less interpretable
- **Best for:** Complex spectral data with non-linear relationships

### Feature Clustering
- **Pros:** Preserves feature interpretability, reduces redundancy
- **Cons:** May lose fine-grained information
- **Best for:** When features are highly correlated in groups

## Backward Compatibility

The system maintains backward compatibility with models using the `_pca` suffix:
- `xgboost_pca` still works and uses PCA with 90% variance
- New models should use the dimension_reduction configuration instead

## Performance Tips

1. **For small datasets (< 1000 samples):**
   - Use PCA or PLS with conservative reduction (retain 90%+ variance)
   - Avoid complex autoencoders

2. **For high-dimensional spectral data:**
   - Try PLS first (it's target-aware)
   - Feature clustering can identify redundant wavelengths

3. **For non-linear relationships:**
   - Use autoencoders with appropriate architecture
   - Ensure sufficient training data

4. **GPU acceleration:**
   - Autoencoders benefit from GPU when available
   - Set `device: 'auto'` to automatically use GPU

## Monitoring

The pipeline logs dimension reduction details:
```
INFO - Applying pls dimension reduction
INFO - PLS fitted: 100 → 10 components
INFO - Features after PLSReducer: 100 → 10
```

## Troubleshooting

1. **PLS fails with "requires target values":**
   - PLS needs y values during fit
   - The pipeline handles this automatically

2. **Autoencoder loss not decreasing:**
   - Reduce learning rate
   - Increase epochs
   - Simplify architecture

3. **Feature clustering selects too few clusters:**
   - Set n_components to fixed value instead of 'auto'
   - Adjust clustering parameters