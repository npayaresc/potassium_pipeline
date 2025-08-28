# Advanced Autoencoder Usage Guide

## ✅ Complete Integration Status

The pipeline now includes **4 sophisticated autoencoder variants** based on research of state-of-the-art implementations for spectral data:

## 🧠 Available Advanced Autoencoders

### 1. **Variational Autoencoder (VAE)**
- **Purpose**: Probabilistic encoding with uncertainty quantification
- **Best for**: When you need uncertainty estimates and generative capabilities
- **Based on**: Modern VAE research for tabular data

### 2. **Denoising Autoencoder**
- **Purpose**: Robust feature extraction from noisy spectral data
- **Best for**: LIBS data with measurement noise or artifacts
- **Based on**: Spectroscopy denoising research

### 3. **Sparse Autoencoder**
- **Purpose**: Interpretable feature selection with sparsity constraints
- **Best for**: Peak detection and wavelength selection
- **Based on**: Spectral peak detection literature

### 4. **Standard Autoencoder** (Enhanced)
- **Purpose**: Non-linear dimension reduction
- **Best for**: General spectral compression
- **Enhanced**: Improved architecture for spectral data

## 🔧 Configuration Examples

### VAE for Uncertainty Quantification
```python
# In pipeline_config.py
dimension_reduction = DimensionReductionConfig(
    method='vae',
    n_components=8,
    vae_params={
        'hidden_layers': [128, 64],
        'epochs': 150,
        'beta': 1.0,  # Standard VAE
        'device': 'auto'
    }
)
```

### Denoising AE for Noisy LIBS Data
```python
dimension_reduction = DimensionReductionConfig(
    method='denoising_ae',
    n_components=10,
    denoising_ae_params={
        'hidden_layers': [128, 64],
        'epochs': 100,
        'noise_factor': 0.2,  # Adjust based on noise level
        'device': 'auto'
    }
)
```

### Sparse AE for Peak Selection
```python
dimension_reduction = DimensionReductionConfig(
    method='sparse_ae',
    n_components=15,
    sparse_ae_params={
        'hidden_layers': [256, 128],
        'epochs': 150,
        'sparsity_param': 0.01,  # Very sparse for peak selection
        'beta': 5.0,  # Strong sparsity enforcement
        'device': 'auto'
    }
)
```

## 📊 Research-Based Features

### VAE (β-VAE Variant)
- **Reparameterization trick** for gradient flow
- **KL divergence regularization** for structured latent space
- **β parameter** for controllable disentanglement
- **Probabilistic encoding** with mean and variance

### Denoising Autoencoder
- **Gaussian noise injection** during training
- **Clean target reconstruction** from noisy inputs
- **Robust feature learning** for measurement artifacts
- **Higher dropout rates** for regularization

### Sparse Autoencoder
- **L1 sparsity penalty** with KL divergence
- **Sigmoid bottleneck** for activation constraints
- **Interpretable components** for wavelength selection
- **Configurable sparsity level** (ρ parameter)

## 🚀 Usage in Your Pipeline

### Current Configuration (PLS)
```python
# Your current setup - supervised reduction
dimension_reduction = DimensionReductionConfig(
    method='pls',
    n_components=8,
    pls_params={'scale': True}
)
```

### Switch to VAE for Uncertainty
```python
# For probabilistic features with uncertainty
dimension_reduction = DimensionReductionConfig(
    method='vae',
    n_components=8,  # Same dimensionality
    vae_params={
        'hidden_layers': [64, 32],
        'epochs': 100,
        'beta': 1.0
    }
)
```

### Switch to Denoising for Robustness
```python
# For noisy spectral measurements
dimension_reduction = DimensionReductionConfig(
    method='denoising_ae',
    n_components=8,
    denoising_ae_params={
        'hidden_layers': [64, 32],
        'epochs': 100,
        'noise_factor': 0.2
    }
)
```

## 🎯 When to Use Each Method

### VAE (Variational Autoencoder)
✅ **Use when:**
- Need uncertainty estimates for predictions
- Want to generate synthetic spectra
- Require probabilistic latent representations
- Data has inherent variability

❌ **Avoid when:**
- Training time is critical (slower than standard AE)
- Only need deterministic features
- Dataset is very small (< 500 samples)

### Denoising Autoencoder
✅ **Use when:**
- LIBS measurements have noise/artifacts
- Spectra contain baseline drift
- Need robust features for varying conditions
- Want to improve signal-to-noise ratio

❌ **Avoid when:**
- Data is already very clean
- Computational resources are limited
- Need interpretable linear features

### Sparse Autoencoder
✅ **Use when:**
- Need interpretable spectral features
- Want automatic wavelength selection
- Spectra have distinct peaks
- Feature selection is important

❌ **Avoid when:**
- All wavelengths are equally important
- Need dense feature representations
- Computational efficiency is critical

## 🔬 Performance Characteristics

| Method | Training Speed | Memory Usage | Interpretability | Noise Robustness |
|--------|----------------|--------------|------------------|------------------|
| PLS | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Standard AE | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| VAE | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| Denoising AE | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Sparse AE | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## 🧪 Testing the Implementation

Run the comprehensive test:
```bash
python test_advanced_autoencoders.py
```

Or quick test with your config:
```python
from src.features.dimension_reduction import DimensionReductionFactory

# Test VAE
reducer = DimensionReductionFactory.create_reducer('vae', {
    'n_components': 8,
    'epochs': 50,
    'device': 'cpu'
})
```

## 📚 Research Foundation

These implementations are based on:

1. **VAE**: Kingma & Welling (2013) - Auto-Encoding Variational Bayes
2. **β-VAE**: Higgins et al. (2017) - β-VAE: Learning Basic Visual Concepts
3. **Denoising AE**: Vincent et al. (2008) - Extracting and Composing Robust Features
4. **Sparse AE**: Ng (2011) - Sparse Autoencoder Tutorial
5. **Spectral Applications**: Recent papers on hyperspectral data compression

## ✨ Benefits for Magnesium Prediction

- **Better feature-to-sample ratios** with sophisticated reduction
- **Noise robustness** for LIBS measurement variations
- **Uncertainty quantification** for quality control
- **Interpretable features** for spectral analysis
- **Non-linear patterns** not captured by PCA/PLS