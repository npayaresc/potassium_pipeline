#!/usr/bin/env python3
"""Test script to verify training still works after prediction fixes."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from src.config.pipeline_config import Config
from src.features.feature_engineering import create_feature_pipeline
# from src.models.trainer import ModelTrainer  # Not needed for this test
from sklearn.model_selection import train_test_split
import ast

def test_feature_extraction():
    """Test that feature extraction works with both 1D and 2D arrays."""
    print("Testing feature extraction...")
    
    # Create config
    config = Config(
        run_timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
        data_dir=Path('data'),
        raw_data_dir=Path('data/raw/data_5278_Phase3'),
        processed_data_dir=Path('data/processed'),
        model_dir=Path('models'),
        reports_dir=Path('reports'),
        log_dir=Path('logs'),
        bad_files_dir=Path('bad_files'),
        averaged_files_dir=Path('data/averaged_files_per_sample'),
        cleansed_files_dir=Path('data/cleansed_files_per_sample'),
        bad_prediction_files_dir=Path('bad_prediction_files'),
        reference_data_path=Path('data/reference_data/Final_Lab_Data_Nico_New.xlsx')
    )
    
    # Load existing processed data
    train_data = pd.read_csv('data/processed/train_20250722_143651.csv')
    print(f"Loaded {len(train_data)} training samples")
    
    # Convert string representations back to arrays
    sample_data = []
    for _, row in train_data.head(5).iterrows():
        # Parse numpy array strings
        wavelengths_str = row['wavelengths'].strip('[]').replace('...', '')
        intensities_str = row['intensities'].strip('[]').replace('...', '')
        
        # For this test, create dummy arrays with correct shape
        # In real pipeline, these are loaded from files
        wavelengths = np.linspace(203, 1025, 2048)  # Standard wavelength range
        intensities = np.random.randn(2048)  # Random intensities for testing
        
        sample_data.append({
            'wavelengths': wavelengths,
            'intensities': intensities
        })
    
    test_df = pd.DataFrame(sample_data)
    y = train_data.head(5)['Nitrogen %']
    
    # Test feature pipeline
    pipeline = create_feature_pipeline(config, 'simple_only')
    
    try:
        # Test fit_transform (training scenario)
        features = pipeline.fit_transform(test_df)
        print(f"✓ Feature extraction successful: {features.shape}")
        print(f"  Generated {features.shape[1]} features")
        
        # Test transform (prediction scenario with single sample)
        single_sample = pd.DataFrame([sample_data[0]])
        single_features = pipeline.transform(single_sample)
        print(f"✓ Single sample transform successful: {single_features.shape}")
        
        return True, features, y
        
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_autogluon_compatibility(test_df, y):
    """Test that AutoGluon feature pipeline works."""
    print("\nTesting AutoGluon compatibility...")
    
    # Create config
    config = Config(
        run_timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
        data_dir=Path('data'),
        raw_data_dir=Path('data/raw/data_5278_Phase3'),
        processed_data_dir=Path('data/processed'),
        model_dir=Path('models'),
        reports_dir=Path('reports'),
        log_dir=Path('logs'),
        bad_files_dir=Path('bad_files'),
        averaged_files_dir=Path('data/averaged_files_per_sample'),
        cleansed_files_dir=Path('data/cleansed_files_per_sample'),
        bad_prediction_files_dir=Path('bad_prediction_files'),
        reference_data_path=Path('data/reference_data/Final_Lab_Data_Nico_New.xlsx')
    )
    
    # Test AutoGluon feature pipeline (without StandardScaler)
    pipeline = create_feature_pipeline(config, 'simple_only', exclude_scaler=True)
    
    try:
        features = pipeline.fit_transform(test_df)
        print(f"✓ AutoGluon feature extraction successful: {features.shape}")
        print(f"  Generated {features.shape[1]} features (no scaler)")
        return True
    except Exception as e:
        print(f"✗ AutoGluon feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_training(X, y):
    """Test that model training works with the transformed features."""
    print("\nTesting model training...")
    
    # Create config
    config = Config(
        run_timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
        data_dir=Path('data'),
        raw_data_dir=Path('data/raw/data_5278_Phase3'),
        processed_data_dir=Path('data/processed'),
        model_dir=Path('models'),
        reports_dir=Path('reports'),
        log_dir=Path('logs'),
        bad_files_dir=Path('bad_files'),
        averaged_files_dir=Path('data/averaged_files_per_sample'),
        cleansed_files_dir=Path('data/cleansed_files_per_sample'),
        bad_prediction_files_dir=Path('bad_prediction_files'),
        reference_data_path=Path('data/reference_data/Final_Lab_Data_Nico_New.xlsx')
    )
    
    # Test with a simple model
    # trainer = ModelTrainer(config)  # Not needed
    
    try:
        # Train a Ridge model (fast and simple)
        from sklearn.linear_model import Ridge
        model = Ridge()
        model.fit(X, y)
        
        # Test prediction
        predictions = model.predict(X)
        print(f"✓ Model training successful")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Sample predictions: {predictions[:3]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing training compatibility after prediction fixes")
    print("=" * 60)
    
    # Test 1: Feature extraction
    success, features, y = test_feature_extraction()
    
    if success and features is not None:
        # Test 2: Model training
        test_model_training(features, y)
        
        # Test 3: AutoGluon compatibility
        # Get the original test dataframe for AutoGluon test
        train_data = pd.read_csv('data/processed/train_20250722_143651.csv')
        sample_data = []
        for _, row in train_data.head(5).iterrows():
            wavelengths = np.linspace(203, 1025, 2048)
            intensities = np.random.randn(2048)
            sample_data.append({
                'wavelengths': wavelengths,
                'intensities': intensities
            })
        test_df = pd.DataFrame(sample_data)
        test_autogluon_compatibility(test_df, y)
    
    print("\n" + "=" * 60)
    print("Testing complete")
    print("=" * 60)