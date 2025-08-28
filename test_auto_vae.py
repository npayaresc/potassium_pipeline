#!/usr/bin/env python3
"""
Test script for automatic VAE component selection.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import logging
from src.features.advanced_autoencoders import VAEReducer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_auto_vae():
    """Test automatic VAE component selection."""
    print("Testing automatic VAE component selection...")
    
    # Create synthetic spectral data
    np.random.seed(42)
    n_samples, n_features = 100, 50
    
    # Create data with known structure (first 10 components have most variance)
    latent_components = 8
    W = np.random.randn(n_features, latent_components)
    Z = np.random.randn(n_samples, latent_components)
    X = Z @ W.T + 0.1 * np.random.randn(n_samples, n_features)  # Add small noise
    
    print(f"Generated synthetic data: {X.shape}")
    print(f"True latent dimensions: {latent_components}")
    
    # Test automatic selection with elbow method
    print("\n1. Testing elbow method...")
    vae_elbow = VAEReducer(
        n_components='auto',
        epochs=50,  # Reduce for testing
        auto_components_method='elbow',
        auto_components_range=(3, 15),
        device='cpu'  # Force CPU for testing
    )
    
    vae_elbow.fit(X)
    elbow_components = vae_elbow.get_n_components()
    
    # Test transformation
    X_transformed = vae_elbow.transform(X)
    print(f"Elbow method selected: {elbow_components} components")
    print(f"Transformed shape: {X_transformed.shape}")
    
    # Test automatic selection with reconstruction threshold method
    print("\n2. Testing reconstruction threshold method...")
    vae_threshold = VAEReducer(
        n_components='auto',
        epochs=50,  # Reduce for testing
        auto_components_method='reconstruction_threshold',
        auto_components_range=(3, 15),
        device='cpu'  # Force CPU for testing
    )
    
    vae_threshold.fit(X)
    threshold_components = vae_threshold.get_n_components()
    
    X_transformed_threshold = vae_threshold.transform(X)
    print(f"Threshold method selected: {threshold_components} components")
    print(f"Transformed shape: {X_transformed_threshold.shape}")
    
    # Test fixed components for comparison
    print("\n3. Testing fixed components...")
    vae_fixed = VAEReducer(
        n_components=10,
        epochs=50,
        device='cpu'
    )
    
    vae_fixed.fit(X)
    fixed_components = vae_fixed.get_n_components()
    
    X_transformed_fixed = vae_fixed.transform(X)
    print(f"Fixed method used: {fixed_components} components")
    print(f"Transformed shape: {X_transformed_fixed.shape}")
    
    print(f"\nSummary:")
    print(f"True latent dimensions: {latent_components}")
    print(f"Elbow method selected: {elbow_components}")
    print(f"Threshold method selected: {threshold_components}")
    print(f"Fixed method used: {fixed_components}")
    
    # Check if selections are reasonable
    reasonable_range = range(latent_components - 3, latent_components + 3)
    elbow_reasonable = elbow_components in reasonable_range
    threshold_reasonable = threshold_components in reasonable_range
    
    print(f"\nReasonableness check (within ±3 of true {latent_components}):")
    print(f"Elbow method: {'✓' if elbow_reasonable else '✗'}")
    print(f"Threshold method: {'✓' if threshold_reasonable else '✗'}")
    
    return True

if __name__ == "__main__":
    test_auto_vae()