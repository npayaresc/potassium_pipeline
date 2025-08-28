#!/usr/bin/env python3
"""
Quick test to verify the feature name extraction fix.
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.features.concentration_features import MinimalConcentrationFeatures

def test_minimal_concentration_features():
    """Test that MinimalConcentrationFeatures correctly reports feature names."""
    
    # Create test data (after RawSpectralTransformer processing)
    np.random.seed(42)
    n_samples = 10
    n_features = 95  # Typical output from RawSpectralTransformer
    
    # Simulate the output from RawSpectralTransformer (numeric features)
    feature_names = [f"P_wl_{210 + i * 0.1:.2f}nm" for i in range(n_features)]
    test_data = pd.DataFrame(
        np.random.rand(n_samples, n_features),
        columns=feature_names
    )
    
    # Target concentrations
    y = np.random.uniform(0.2, 0.5, n_samples)
    
    # Create the transformer
    transformer = MinimalConcentrationFeatures()
    
    # Fit and transform
    transformer.fit(test_data, y)
    X_transformed = transformer.transform(test_data)
    
    # Get feature names
    input_feature_names = test_data.columns
    output_feature_names = transformer.get_feature_names_out(input_feature_names)
    
    print(f"Input features: {len(input_feature_names)} - {list(input_feature_names)}")
    print(f"Transformed data shape: {X_transformed.shape}")
    print(f"Output feature names: {len(output_feature_names)} - {list(output_feature_names)}")
    print(f"Expected concentration features: {transformer.feature_names_out_}")
    
    # Verify the counts match
    if len(output_feature_names) == X_transformed.shape[1]:
        print("✅ SUCCESS: Feature names count matches transformed data shape!")
        return True
    else:
        print(f"❌ FAIL: Feature names count ({len(output_feature_names)}) != data shape ({X_transformed.shape[1]})")
        return False

if __name__ == "__main__":
    success = test_minimal_concentration_features()
    exit(0 if success else 1)