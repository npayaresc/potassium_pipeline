"""Test the NC_ratio fix"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.config.pipeline_config import config, Config
from src.features.feature_engineering import create_feature_pipeline

# Setup config
config.run_timestamp = "test_fix"
project_root = Path(__file__).resolve().parent
config.data_dir = project_root / "data"

print("=== TESTING NC_RATIO FIX ===")

# Create a fresh feature pipeline with the fix
feature_pipeline = create_feature_pipeline(config, "full_context")

# Test with a sample that had extreme values before
cleansed_dir = project_root / "data" / "cleansed_files_per_sample"
test_file = cleansed_dir / "1053789KENP4S009_G_2025_01_11.csv.txt"  # This one had NC_ratio = 2394.35

if test_file.exists():
    print(f"Testing with: {test_file.name}")
    
    df = pd.read_csv(test_file)
    wavelengths = df['Wavelength'].values
    intensity_cols = [col for col in df.columns if col != 'Wavelength']
    intensities = df[intensity_cols].values
    
    # Create input data
    input_data = pd.DataFrame([{
        'wavelengths': wavelengths,
        'intensities': intensities
    }])
    
    # Extract features with the fixed pipeline
    spectral_gen = feature_pipeline.named_steps['spectral_features']
    features = spectral_gen.transform(input_data)
    
    # Get feature names and check NC_ratio related features
    feature_names = spectral_gen.get_feature_names_out()
    
    nc_ratio_idx = list(feature_names).index('N_C_ratio')
    nc_squared_idx = list(feature_names).index('NC_ratio_squared')
    nc_cubic_idx = list(feature_names).index('NC_ratio_cubic')
    
    nc_ratio = features.iloc[0, nc_ratio_idx]
    nc_squared = features.iloc[0, nc_squared_idx]
    nc_cubic = features.iloc[0, nc_cubic_idx]
    
    print(f"\nFixed NC_ratio features:")
    print(f"  N_C_ratio: {nc_ratio:.6f}")
    print(f"  NC_ratio_squared: {nc_squared:.6f}")
    print(f"  NC_ratio_cubic: {nc_cubic:.6f}")
    
    # Check if values are within reasonable bounds
    print(f"\nBounds check:")
    print(f"  NC_ratio within [-50, 50]: {-50 <= nc_ratio <= 50}")
    print(f"  NC_squared <= 2500: {nc_squared <= 2500}")
    print(f"  NC_cubic <= 125000: {nc_cubic <= 125000}")
    
    # Now test the full pipeline (with imputer, clipper, scaler)
    print(f"\nTesting full pipeline transformation...")
    try:
        full_features = feature_pipeline.transform(input_data)
        print(f"Full pipeline successful: shape={full_features.shape}")
        print(f"Feature values: min={full_features.min():.6f}, max={full_features.max():.6f}")
        
        # Check if there are any extreme values that would corrupt a scaler
        extreme_mask = (np.abs(full_features) > 1e6)
        if extreme_mask.any():
            print(f"WARNING: Still found {extreme_mask.sum()} extreme values > 1e6")
        else:
            print("SUCCESS: No extreme values found - safe for scaler training")
            
    except Exception as e:
        print(f"Error in full pipeline: {e}")

else:
    print(f"Test file not found: {test_file}")

print("\nFix testing complete!")