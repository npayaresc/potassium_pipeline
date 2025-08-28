#!/usr/bin/env python3
"""
Test script for the modular dimension reduction system.

Usage:
    python test_dimension_reduction.py
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

from src.features.dimension_reduction import (
    DimensionReductionFactory, 
    PCAReducer, 
    PLSReducer, 
    AutoencoderReducer, 
    FeatureClusteringReducer
)

def test_dimension_reduction():
    """Test different dimension reduction strategies."""
    
    # Create synthetic spectral-like data
    print("Creating synthetic data...")
    X, y = make_regression(n_samples=500, n_features=100, n_informative=20, 
                          noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Data shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}\n")
    
    # Test configurations
    test_configs = [
        {
            'name': 'PCA (95% variance)',
            'method': 'pca',
            'params': {'n_components': 0.95}
        },
        {
            'name': 'PCA (10 components)',
            'method': 'pca',
            'params': {'n_components': 10}
        },
        {
            'name': 'PLS (10 components)',
            'method': 'pls',
            'params': {'n_components': 10, 'scale': True}
        },
        {
            'name': 'Autoencoder (8 components)',
            'method': 'autoencoder',
            'params': {
                'n_components': 8,
                'hidden_layers': [50, 20],
                'epochs': 50,
                'batch_size': 32,
                'device': 'cpu'
            }
        },
        {
            'name': 'Feature Clustering (12 clusters)',
            'method': 'feature_clustering',
            'params': {'n_components': 12}
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\nTesting: {config['name']}")
        print("-" * 50)
        
        # Create reducer
        reducer = DimensionReductionFactory.create_reducer(
            method=config['method'],
            params=config['params']
        )
        
        # Fit and transform
        if config['method'] == 'pls':
            # PLS requires y values
            X_train_reduced = reducer.fit_transform(X_train, y_train)
        else:
            X_train_reduced = reducer.fit_transform(X_train)
        
        X_test_reduced = reducer.transform(X_test)
        
        print(f"Reduced dimensions: {X_train.shape[1]} → {X_train_reduced.shape[1]}")
        
        # Train a simple model on reduced data
        model = Ridge(alpha=1.0)
        model.fit(X_train_reduced, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_reduced)
        r2 = r2_score(y_test, y_pred)
        
        print(f"R² score: {r2:.4f}")
        
        results.append({
            'Method': config['name'],
            'Original Features': X_train.shape[1],
            'Reduced Features': X_train_reduced.shape[1],
            'R² Score': r2
        })
    
    # Display results
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Test factory methods
    print("\n" + "="*60)
    print("Available dimension reduction methods:")
    print(DimensionReductionFactory.get_available_methods())
    
    # Test save/load functionality
    print("\nTesting save/load functionality...")
    reducer = PCAReducer(n_components=5)
    reducer.fit(X_train)
    
    # Save and load
    reducer.save('test_reducer.pkl')
    loaded_reducer = PCAReducer.load('test_reducer.pkl')
    
    # Verify they produce same results
    X_test_original = reducer.transform(X_test[:5])
    X_test_loaded = loaded_reducer.transform(X_test[:5])
    
    print(f"Save/load test passed: {np.allclose(X_test_original, X_test_loaded)}")
    
    # Clean up
    import os
    if os.path.exists('test_reducer.pkl'):
        os.remove('test_reducer.pkl')

if __name__ == "__main__":
    test_dimension_reduction()