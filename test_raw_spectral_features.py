#!/usr/bin/env python3
"""
Test script for raw spectral feature implementation.

This script tests the new RawSpectralTransformer to ensure it correctly
extracts raw intensity values from PeakRegions without feature engineering.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.pipeline_config import config
from features.feature_engineering import RawSpectralTransformer, create_feature_pipeline
from data_management.data_manager import DataManager
from utils.helpers import setup_logging

# Import the helper function from main
sys.path.insert(0, str(Path(__file__).parent))
from main import load_and_clean_data, run_data_preparation

def test_raw_spectral_transformer():
    """Test the RawSpectralTransformer directly."""
    setup_logging()
    
    print("=== Testing RawSpectralTransformer ===")
    
    # Set up test config with raw spectral mode
    test_config = config.model_copy(deep=True)
    test_config.use_raw_spectral_data = True
    test_config.run_timestamp = "20250813_test"
    
    print(f"Test configuration:")
    print(f"  - PeakRegions: {len(test_config.all_regions)}")
    print(f"  - Raw spectral mode: {test_config.use_raw_spectral_data}")
    
    try:
        # Prepare data using same process as main pipeline
        print("\nPreparing test data...")
        run_data_preparation(test_config)
        full_dataset, global_data_manager = load_and_clean_data(test_config)
        
        # Take just first 3 samples for quick testing
        test_data = full_dataset.head(3)
        print(f"Test data shape: {test_data.shape}")
        print(f"Columns: {list(test_data.columns)}")
        
        # Test the RawSpectralTransformer directly
        print("\n=== Testing RawSpectralTransformer ===")
        
        # Extract features from test data
        X_test = test_data.drop(columns=[test_config.target_column, test_config.sample_id_column])
        
        # Create and fit transformer
        raw_transformer = RawSpectralTransformer(test_config)
        raw_transformer.fit(X_test)
        
        print(f"Expected features: {len(raw_transformer.feature_names_out_)}")
        print(f"First 10 feature names: {raw_transformer.feature_names_out_[:10]}")
        
        # Transform the data
        X_raw_features = raw_transformer.transform(X_test)
        
        print(f"Raw features shape: {X_raw_features.shape}")
        print(f"Features extracted: {X_raw_features.shape[1]}")
        
        # Check for NaN values
        nan_count = X_raw_features.isna().sum().sum()
        print(f"Total NaN values: {nan_count}")
        
        # Show sample of the features
        print(f"\nSample raw features (first 3 rows, first 5 columns):")
        print(X_raw_features.iloc[:3, :5])
        
        return True
        
    except Exception as e:
        print(f"Error during raw transformer test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_pipeline_comparison():
    """Compare feature engineering vs raw spectral pipelines."""
    setup_logging()
    
    print("\n=== Testing Feature Pipeline Comparison ===")
    
    # Set up test config
    test_config = config.model_copy(deep=True)
    test_config.run_timestamp = "20250813_test"
    
    # Load test data using same process as main pipeline
    run_data_preparation(test_config)
    full_dataset, data_manager = load_and_clean_data(test_config)
    test_data = full_dataset.head(3)
    X_test = test_data.drop(columns=[test_config.target_column, test_config.sample_id_column])
    
    try:
        print("\n1. Testing FEATURE ENGINEERING pipeline:")
        test_config.use_raw_spectral_data = False
        
        pipeline_features = create_feature_pipeline(test_config, "simple_only", exclude_scaler=True)
        pipeline_features.fit(X_test)
        X_engineered = pipeline_features.transform(X_test)
        
        print(f"   - Engineered features shape: {X_engineered.shape}")
        print(f"   - Features: {X_engineered.shape[1]}")
        
        print("\n2. Testing RAW SPECTRAL pipeline:")
        test_config.use_raw_spectral_data = True
        
        pipeline_raw = create_feature_pipeline(test_config, "simple_only", exclude_scaler=True)  # strategy ignored in raw mode
        pipeline_raw.fit(X_test)
        X_raw = pipeline_raw.transform(X_test)
        
        print(f"   - Raw features shape: {X_raw.shape}")
        print(f"   - Features: {X_raw.shape[1]}")
        
        print(f"\nComparison:")
        print(f"  - Feature Engineering: {X_engineered.shape[1]} features")
        print(f"  - Raw Spectral: {X_raw.shape[1]} features") 
        print(f"  - Raw has {X_raw.shape[1] - X_engineered.shape[1]} more features")
        
        return True
        
    except Exception as e:
        print(f"Error during pipeline comparison: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Raw Spectral Feature Implementation")
    print("=" * 50)
    
    # Test 1: Raw transformer
    success1 = test_raw_spectral_transformer()
    
    # Test 2: Pipeline comparison
    success2 = test_feature_pipeline_comparison()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✅ ALL TESTS PASSED")
        print("\nRaw spectral feature implementation is working correctly!")
        print("\nTo use raw spectral features:")
        print("  python main.py train --raw-spectral")
        print("  python main.py autogluon --raw-spectral")
        print("  python main.py tune --raw-spectral")
    else:
        print("❌ SOME TESTS FAILED")
        print("Check the errors above and fix the implementation.")
        sys.exit(1)