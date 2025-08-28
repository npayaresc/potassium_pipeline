#!/usr/bin/env python3
"""
Test script for MinimalConcentrationFeatures transformer.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from src.features.concentration_features import MinimalConcentrationFeatures

def test_minimal_concentration_features():
    """Test the MinimalConcentrationFeatures transformer."""
    print("=== Testing MinimalConcentrationFeatures ===")
    
    # Create synthetic spectral data
    n_samples = 100
    n_wavelengths = 50
    
    # Generate realistic spectral data
    X = np.random.randn(n_samples, n_wavelengths) * 100 + 1000
    X = np.abs(X)  # Intensities should be positive
    
    # Create realistic target concentrations (magnesium percentage)
    y = np.concatenate([
        np.random.uniform(0.15, 0.25, 25),  # Low concentrations
        np.random.uniform(0.25, 0.35, 50),  # Mid concentrations
        np.random.uniform(0.35, 0.45, 25)   # High concentrations
    ])
    
    print(f"Input data: {X.shape}")
    print(f"Target range: {y.min():.3f} - {y.max():.3f}")
    print(f"Target distribution: P25={np.percentile(y, 25):.3f}, P75={np.percentile(y, 75):.3f}")
    
    # Test the transformer
    transformer = MinimalConcentrationFeatures()
    
    try:
        # Fit with target values
        transformer.fit(X, y)
        print("✅ Fit completed successfully")
        
        # Transform the data
        X_enhanced = transformer.transform(X)
        print(f"✅ Transform completed: {X.shape} -> {X_enhanced.shape}")
        
        # Check feature names
        feature_names = transformer.get_feature_names_out()
        expected_additional_features = 5  # The 5 concentration features we added
        
        print(f"Additional features: {expected_additional_features}")
        print(f"Feature names: {feature_names}")
        
        # Check that we added the right number of features
        expected_total = n_wavelengths + expected_additional_features
        if X_enhanced.shape[1] == expected_total:
            print(f"✅ Feature count correct: {expected_total}")
        else:
            print(f"❌ Feature count mismatch: expected {expected_total}, got {X_enhanced.shape[1]}")
            return False
            
        # Check that concentration features are present
        X_enhanced_df = pd.DataFrame(X_enhanced)
        concentration_cols = X_enhanced_df.columns[-expected_additional_features:]
        
        print("Concentration feature statistics:")
        for i, col in enumerate(concentration_cols):
            values = X_enhanced_df[col]
            print(f"  {feature_names[i]}: mean={values.mean():.3f}, std={values.std():.3f}, range=[{values.min():.3f}, {values.max():.3f}]")
        
        # Test transform without fit (should work after fitting)
        X_test = np.random.randn(10, n_wavelengths) * 100 + 1000
        X_test = np.abs(X_test)
        X_test_enhanced = transformer.transform(X_test)
        
        if X_test_enhanced.shape[1] == expected_total:
            print("✅ Transform on new data works correctly")
        else:
            print(f"❌ Transform on new data failed: expected {expected_total} features, got {X_test_enhanced.shape[1]}")
            return False
            
        print("\n🎉 MinimalConcentrationFeatures test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal_concentration_features()
    if success:
        print("\n✅ All tests passed! MinimalConcentrationFeatures is ready for production.")
        print("\nUsage in raw spectral mode:")
        print("  python main.py autogluon --raw-spectral")
        print("  python main.py train --raw-spectral")
        print("  python main.py tune --raw-spectral")
    else:
        print("\n❌ Tests failed. Please check the implementation.")
        sys.exit(1)