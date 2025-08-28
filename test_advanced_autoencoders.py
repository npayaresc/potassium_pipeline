#!/usr/bin/env python3
"""
Test and compare advanced autoencoder implementations for spectral data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
import time

from src.features.dimension_reduction import DimensionReductionFactory

def generate_spectral_data(n_samples=500, n_wavelengths=100, n_peaks=5, noise_level=0.1):
    """Generate synthetic spectral data with Gaussian peaks."""
    wavelengths = np.linspace(200, 800, n_wavelengths)
    spectra = np.zeros((n_samples, n_wavelengths))
    
    # Generate random peaks for each sample
    for i in range(n_samples):
        for _ in range(n_peaks):
            center = np.random.uniform(250, 750)
            height = np.random.uniform(0.5, 2.0)
            width = np.random.uniform(5, 20)
            peak = height * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
            spectra[i] += peak
    
    # Add noise
    spectra += np.random.normal(0, noise_level, spectra.shape)
    
    # Create target variable (e.g., concentration correlated with peak intensities)
    y = np.sum(spectra[:, 40:60], axis=1) * 0.1 + np.random.normal(0, 0.05, n_samples)
    
    return spectra, y

def test_autoencoder_variants():
    """Test different autoencoder variants on spectral data."""
    print("="*70)
    print("ADVANCED AUTOENCODER COMPARISON FOR SPECTRAL DATA")
    print("="*70)
    
    # Generate synthetic spectral data
    print("\nGenerating synthetic spectral data...")
    X, y = generate_spectral_data(n_samples=500, n_wavelengths=100, n_peaks=5, noise_level=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Data shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    
    # Test configurations
    test_configs = [
        {
            'name': 'Standard Autoencoder',
            'method': 'autoencoder',
            'params': {
                'n_components': 10,
                'hidden_layers': [64, 32],
                'epochs': 50,
                'device': 'cpu'
            }
        },
        {
            'name': 'Variational Autoencoder (VAE)',
            'method': 'vae',
            'params': {
                'n_components': 10,
                'hidden_layers': [64, 32],
                'epochs': 50,
                'beta': 1.0,
                'device': 'cpu'
            }
        },
        {
            'name': 'Denoising Autoencoder',
            'method': 'denoising_ae',
            'params': {
                'n_components': 10,
                'hidden_layers': [64, 32],
                'epochs': 50,
                'noise_factor': 0.2,
                'device': 'cpu'
            }
        },
        {
            'name': 'Sparse Autoencoder',
            'method': 'sparse_ae',
            'params': {
                'n_components': 10,
                'hidden_layers': [64, 32],
                'epochs': 50,
                'sparsity_param': 0.05,
                'beta': 3.0,
                'device': 'cpu'
            }
        }
    ]
    
    # Also compare with traditional methods
    traditional_configs = [
        {
            'name': 'PCA (95% variance)',
            'method': 'pca',
            'params': {'n_components': 0.95}
        },
        {
            'name': 'PLS (10 components)',
            'method': 'pls',
            'params': {'n_components': 10}
        }
    ]
    
    all_configs = test_configs + traditional_configs
    results = []
    
    print("\n" + "-"*70)
    
    for config in all_configs:
        print(f"\nTesting: {config['name']}")
        print("-" * 50)
        
        try:
            # Create reducer
            start_time = time.time()
            reducer = DimensionReductionFactory.create_reducer(
                method=config['method'],
                params=config['params']
            )
            
            # Fit and transform
            if config['method'] == 'pls':
                X_train_reduced = reducer.fit_transform(X_train, y_train)
            else:
                X_train_reduced = reducer.fit_transform(X_train)
            
            fit_time = time.time() - start_time
            
            # Transform test data
            start_time = time.time()
            X_test_reduced = reducer.transform(X_test)
            transform_time = time.time() - start_time
            
            print(f"Reduced dimensions: {X_train.shape[1]} → {X_train_reduced.shape[1]}")
            print(f"Fit time: {fit_time:.2f}s")
            print(f"Transform time: {transform_time:.3f}s")
            
            # Train a simple model on reduced data
            model = Ridge(alpha=1.0)
            model.fit(X_train_reduced, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_reduced)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"R² score: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")
            
            # Calculate reconstruction error for autoencoders
            if 'autoencoder' in config['method'] or config['method'] == 'vae':
                # For autoencoders, we can check reconstruction quality
                # This would require modifying the reducer classes to expose reconstruction
                recon_error = "N/A"  # Placeholder
            else:
                recon_error = "N/A"
            
            results.append({
                'Method': config['name'],
                'Original Features': X_train.shape[1],
                'Reduced Features': X_train_reduced.shape[1],
                'Fit Time (s)': f"{fit_time:.2f}",
                'Transform Time (s)': f"{transform_time:.3f}",
                'R² Score': r2,
                'RMSE': rmse,
                'Reconstruction Error': recon_error
            })
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'Method': config['name'],
                'Original Features': X_train.shape[1],
                'Reduced Features': "Error",
                'Fit Time (s)': "Error",
                'Transform Time (s)': "Error",
                'R² Score': 0.0,
                'RMSE': 999.0,
                'Reconstruction Error': "Error"
            })
    
    # Display results
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Additional analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    # Best R² score
    best_r2_idx = results_df['R² Score'].idxmax()
    print(f"\nBest R² Score: {results_df.iloc[best_r2_idx]['Method']} "
          f"({results_df.iloc[best_r2_idx]['R² Score']:.4f})")
    
    # Best RMSE
    best_rmse_idx = results_df['RMSE'].idxmin()
    print(f"Best RMSE: {results_df.iloc[best_rmse_idx]['Method']} "
          f"({results_df.iloc[best_rmse_idx]['RMSE']:.4f})")
    
    # Print recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR SPECTRAL DATA")
    print("="*70)
    print("""
1. **Clean Spectral Data**: 
   - Use standard Autoencoder or VAE for balanced performance
   - VAE provides uncertainty estimates useful for quality control

2. **Noisy Spectral Data**:
   - Denoising Autoencoder excels at handling measurement noise
   - Increase noise_factor parameter for very noisy data

3. **Peak-heavy Spectra**:
   - Sparse Autoencoder can identify important spectral peaks
   - Useful when only certain wavelengths contain relevant information

4. **Small Datasets** (< 1000 samples):
   - PLS often performs best due to supervised learning
   - Avoid deep architectures in autoencoders

5. **Large Datasets** (> 5000 samples):
   - Deep autoencoders can capture complex spectral patterns
   - GPU acceleration becomes beneficial

6. **Interpretability Required**:
   - Sparse Autoencoder provides interpretable features
   - PCA/PLS components have clear linear interpretations
    """)

def visualize_encodings():
    """Visualize the learned representations from different autoencoders."""
    print("\n" + "="*70)
    print("VISUALIZING ENCODED REPRESENTATIONS")
    print("="*70)
    
    # Generate data
    X, y = generate_spectral_data(n_samples=200, n_wavelengths=100, n_peaks=5, noise_level=0.1)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    methods = [
        ('PCA', 'pca', {'n_components': 2}),
        ('Standard AE', 'autoencoder', {'n_components': 2, 'hidden_layers': [32], 'epochs': 50}),
        ('VAE', 'vae', {'n_components': 2, 'hidden_layers': [32], 'epochs': 50}),
        ('Denoising AE', 'denoising_ae', {'n_components': 2, 'hidden_layers': [32], 'epochs': 50}),
        ('Sparse AE', 'sparse_ae', {'n_components': 2, 'hidden_layers': [32], 'epochs': 50}),
        ('PLS', 'pls', {'n_components': 2})
    ]
    
    for idx, (name, method, params) in enumerate(methods):
        try:
            # Create and fit reducer
            reducer = DimensionReductionFactory.create_reducer(method, params)
            
            if method == 'pls':
                X_reduced = reducer.fit_transform(X, y)
            else:
                X_reduced = reducer.fit_transform(X)
            
            # Plot
            scatter = axes[idx].scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', alpha=0.6)
            axes[idx].set_title(f'{name} Encoding')
            axes[idx].set_xlabel('Component 1')
            axes[idx].set_ylabel('Component 2')
            plt.colorbar(scatter, ax=axes[idx])
            
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                          transform=axes[idx].transAxes, ha='center')
            axes[idx].set_title(f'{name} (Failed)')
    
    plt.tight_layout()
    plt.savefig('autoencoder_encodings_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: autoencoder_encodings_comparison.png")

if __name__ == "__main__":
    # Test all autoencoder variants
    test_autoencoder_variants()
    
    # Create visualization
    try:
        visualize_encodings()
    except Exception as e:
        print(f"\nVisualization failed: {e}")
    
    print("\n✅ Advanced autoencoder testing complete!")