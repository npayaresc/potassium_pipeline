#!/usr/bin/env python3
"""
Test the fixed RawSpectralTransformer with actual data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.pipeline_config import config
from features.feature_engineering import RawSpectralTransformer

# Import the helper function from main
sys.path.insert(0, str(Path(__file__).parent))
from main import load_and_clean_data, run_data_preparation

def test_fixed_transformer():
    """Test the fixed RawSpectralTransformer."""
    print("=== Testing Fixed RawSpectralTransformer ===")
    
    test_config = config.model_copy(deep=True)
    test_config.use_raw_spectral_data = True
    test_config.run_timestamp = "20250813_test_fixed"
    
    try:
        # Load small dataset
        print("Loading data...")
        run_data_preparation(test_config)
        full_dataset, data_manager = load_and_clean_data(test_config)
        
        # Take first 3 samples
        test_data = full_dataset.head(3)
        print(f"Test data shape: {test_data.shape}")
        
        # Create transformer
        transformer = RawSpectralTransformer(test_config)
        
        # Fit and transform
        print("Fitting transformer...")
        transformer.fit(test_data.drop(columns=['Sample ID', 'Magnesium dm %']))
        
        print("Transforming data...")
        X_features = test_data.drop(columns=['Sample ID', 'Magnesium dm %'])
        result = transformer.transform(X_features)
        
        print(f"Result shape: {result.shape}")
        print(f"Non-NaN values: {result.notna().sum().sum()}")
        print(f"Total features: {result.shape[1]}")
        
        if result.notna().sum().sum() > 0:
            print("✅ SUCCESS: Raw spectral features extracted!")
            
            # Show sample of actual values
            non_nan_cols = result.columns[result.notna().any()][:10]
            print(f"\nSample values from {len(non_nan_cols)} features:")
            for col in non_nan_cols:
                values = result[col].dropna()
                if len(values) > 0:
                    print(f"  {col}: {values.iloc[0]:.2f}")
                    
            # Check feature ranges
            print(f"\nFeature statistics:")
            print(f"  Min value: {result.min().min():.2f}")
            print(f"  Max value: {result.max().max():.2f}")
            print(f"  Mean absolute value: {result.abs().mean().mean():.2f}")
            
            return True
        else:
            print("❌ STILL FAILING: No non-NaN values found")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_transformer()
    
    if success:
        print("\n" + "="*50)
        print("✅ Raw Spectral Transformer is now WORKING!")
        print("Ready to use with:")
        print("  python main.py train --raw-spectral")
        print("  python main.py autogluon --raw-spectral")
    else:
        print("\n" + "="*50)
        print("❌ Still needs debugging")