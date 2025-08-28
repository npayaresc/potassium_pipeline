#!/usr/bin/env python3
"""
Test script to verify PCA logging functionality
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create synthetic spectral data similar to what the neural network would receive
np.random.seed(42)
n_samples = 100
n_features = 98  # Same as full_context strategy

# Generate correlated spectral features (like real spectral data)
X = np.random.randn(n_samples, n_features)
# Add correlation between adjacent features (spectral bands are correlated)
for i in range(1, n_features):
    X[:, i] = 0.7 * X[:, i-1] + 0.3 * X[:, i]

y = np.random.rand(n_samples) * 0.3 + 0.2  # Target values between 0.2-0.5

print(f"Original synthetic data shape: {X.shape}")

# Test PCA pipeline construction
model_steps = [('pca', PCA(n_components=0.95))]
model_steps.append(('scaler', StandardScaler()))  # Add scaler as placeholder

pipeline = Pipeline(model_steps)

# Fit the pipeline
logger.info(f"Original feature count: {X.shape[1]}")
pipeline.fit(X, y)

# Log PCA results
if 'pca' in pipeline.named_steps:
    pca_component = pipeline.named_steps['pca']
    reduced_features = pca_component.n_components_
    variance_retained = pca_component.explained_variance_ratio_.sum()
    logger.info(f"PCA applied: {X.shape[1]} → {reduced_features} features "
               f"(retained {variance_retained:.1%} variance)")
    
    # Show individual component variances
    logger.info(f"Top 5 principal components explain: {pca_component.explained_variance_ratio_[:5]}")
    
    # Transform data to see final dimension
    X_transformed = pca_component.transform(X)
    logger.info(f"Transformed data shape: {X_transformed.shape}")

print("\nPCA logging test completed successfully!")