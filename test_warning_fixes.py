#!/usr/bin/env python3
"""
Quick test script to verify that warnings have been fixed.
"""
import warnings
import numpy as np
import pandas as pd
from src.features.concentration_features import MinimalConcentrationFeatures
from src.features.feature_engineering import PandasStandardScaler

def test_no_divide_warnings():
    """Test that concentration features don't produce divide by zero warnings."""
    print("Testing concentration features for divide by zero warnings...")
    
    # Create test data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 10), columns=[f"feature_{i}" for i in range(10)])
    y = np.random.uniform(0.1, 0.5, 50)  # Magnesium concentration range
    
    # Test MinimalConcentrationFeatures
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        transformer = MinimalConcentrationFeatures()
        transformer.fit(X, y)
        X_transformed = transformer.transform(X)
        
        # Check for sklearn warnings
        sklearn_warnings = [warning for warning in w if 'sklearn' in str(warning.filename)]
        divide_warnings = [warning for warning in sklearn_warnings if 'divide' in str(warning.message).lower()]
        
        if divide_warnings:
            print(f"❌ Found {len(divide_warnings)} divide warnings:")
            for warning in divide_warnings:
                print(f"  - {warning.message}")
        else:
            print("✅ No divide by zero warnings found in concentration features")
    
    return len(divide_warnings) == 0

def test_pandas_preservation():
    """Test that PandasStandardScaler preserves DataFrame format."""
    print("Testing PandasStandardScaler for DataFrame preservation...")
    
    # Create test DataFrame
    X = pd.DataFrame(np.random.randn(20, 5), columns=[f"col_{i}" for i in range(5)])
    
    scaler = PandasStandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if isinstance(X_scaled, pd.DataFrame):
        if list(X_scaled.columns) == list(X.columns):
            print("✅ PandasStandardScaler preserves DataFrame format and column names")
            return True
        else:
            print("❌ PandasStandardScaler changed column names")
            return False
    else:
        print("❌ PandasStandardScaler returned numpy array instead of DataFrame")
        return False

def test_concentration_features_return_type():
    """Test that MinimalConcentrationFeatures returns DataFrame."""
    print("Testing MinimalConcentrationFeatures return type...")
    
    X = pd.DataFrame(np.random.randn(20, 5), columns=[f"col_{i}" for i in range(5)])
    y = np.random.uniform(0.1, 0.5, 20)
    
    transformer = MinimalConcentrationFeatures()
    transformer.fit(X, y)
    X_transformed = transformer.transform(X)
    
    if isinstance(X_transformed, pd.DataFrame):
        print("✅ MinimalConcentrationFeatures returns DataFrame")
        return True
    else:
        print("❌ MinimalConcentrationFeatures returns numpy array")
        return False

if __name__ == "__main__":
    print("Testing warning fixes...\n")
    
    test1 = test_no_divide_warnings()
    test2 = test_pandas_preservation()
    test3 = test_concentration_features_return_type()
    
    print(f"\nResults:")
    print(f"- Divide warnings fixed: {'✅' if test1 else '❌'}")
    print(f"- DataFrame preservation: {'✅' if test2 else '❌'}")
    print(f"- Feature names preservation: {'✅' if test3 else '❌'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 All warning fixes appear to be working!")
    else:
        print("\n⚠️  Some issues still need attention")