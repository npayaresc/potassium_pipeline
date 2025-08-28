"""Quick test to verify NC_ratio fix produces reasonable values"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.config.pipeline_config import config, Config
from src.features.feature_engineering import create_feature_pipeline

# Setup config
config.run_timestamp = "quick_test"
project_root = Path(__file__).resolve().parent
config.data_dir = project_root / "data"

print("=== QUICK NC_RATIO FIX VERIFICATION ===")

# Create feature pipeline with the fix
feature_pipeline = create_feature_pipeline(config, "full_context")

# Test with 3 samples that had extreme NC_ratio values before
test_files = [
    "data/cleansed_files_per_sample/1053789KENP4S009_G_2025_01_11.csv.txt",  # Had NC_ratio = 2394.35
    "data/cleansed_files_per_sample/1053789KENP3S013_G_2025_01_18.csv.txt",  # Had NC_ratio = 1038.30
    "data/cleansed_files_per_sample/1053789KENP1S006_G_2025_01_11.csv.txt"   # Had NC_ratio = 761.45
]

all_data = []
for test_file in test_files:
    file_path = Path(test_file)
    if file_path.exists():
        print(f"Processing: {file_path.name}")
        
        df = pd.read_csv(file_path)
        wavelengths = df['Wavelength'].values
        intensity_cols = [col for col in df.columns if col != 'Wavelength']
        intensities = df[intensity_cols].values
        
        input_data = pd.DataFrame([{
            'wavelengths': wavelengths,
            'intensities': intensities
        }])
        
        all_data.append(input_data)

if all_data:
    # Combine all test data
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"\\nTesting with {len(combined_data)} samples...")
    
    # Extract features step by step
    spectral_gen = feature_pipeline.named_steps['spectral_features']
    imputer = feature_pipeline.named_steps['imputer']
    clipper = feature_pipeline.named_steps['clipper']
    scaler = feature_pipeline.named_steps['scaler']
    
    # Step 1: Spectral features (this is where NC_ratio is calculated)
    X1 = spectral_gen.transform(combined_data)
    print(f"After spectral features: shape={X1.shape}")
    print(f"  Feature value range: {X1.min():.2f} to {X1.max():.2f}")
    
    # Check for extreme values that would corrupt the scaler
    extreme_count = (np.abs(X1.values) > 1e6).sum()
    print(f"  Values > 1e6: {extreme_count}")
    
    if extreme_count == 0:
        print("  ✓ SUCCESS: No extreme values - NC_ratio fix is working!")
        
        # Continue with full pipeline
        X2 = imputer.transform(X1)
        X3 = clipper.transform(X2)
        
        # Fit scaler on this clean data
        scaler.fit(X3)
        print(f"  Scaler max mean: {scaler.mean_.max():.2e}")
        print(f"  Scaler max scale: {scaler.scale_.max():.2e}")
        
        if scaler.scale_.max() < 1e6:
            print("  ✓ SUCCESS: Scaler parameters are reasonable!")
        else:
            print("  ✗ WARNING: Scaler still has extreme values")
            
    else:
        print(f"  ✗ PROBLEM: Found {extreme_count} extreme values - fix needs more work")

else:
    print("No test files found!")

print("\\nQuick test complete!")