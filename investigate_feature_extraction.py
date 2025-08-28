"""Investigate if feature extraction is working correctly"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from src.config.pipeline_config import config, Config
from src.features.feature_engineering import create_feature_pipeline

# Setup config
config.run_timestamp = "debug"
project_root = Path(__file__).resolve().parent
config.data_dir = project_root / "data"
config.reference_data_path = project_root / "data" / "reference_data" / "Final_Lab_Data_Nico_New.xlsx"

print("=== INVESTIGATING FEATURE EXTRACTION ===")

# Load the model
model_path = project_root / "models" / "full_context_extratrees_20250723_220001.pkl"
pipeline = joblib.load(model_path)
feature_pipeline = pipeline.named_steps['features']

# Check if this is the same as creating a fresh feature pipeline
fresh_pipeline = create_feature_pipeline(config, "full_context")

print("=== COMPARING SAVED VS FRESH FEATURE PIPELINE ===")
print(f"Saved pipeline steps: {list(feature_pipeline.named_steps.keys())}")
print(f"Fresh pipeline steps: {list(fresh_pipeline.named_steps.keys())}")

# Test with training data sample
cleansed_files = list((project_root / "data" / "cleansed_files_per_sample").glob("*.csv.txt"))
if cleansed_files:
    # Load a sample
    sample_file = cleansed_files[0]
    sample_id = sample_file.name.replace('.csv.txt', '')
    print(f"\nTesting with sample: {sample_id}")
    
    df = pd.read_csv(sample_file)
    wavelengths = df['Wavelength'].values
    intensity_cols = [col for col in df.columns if col != 'Wavelength']
    intensities = df[intensity_cols].values
    
    # Create input data
    input_data = pd.DataFrame([{
        'wavelengths': wavelengths,
        'intensities': intensities
    }])
    
    print(f"Input data shape: {input_data.shape}")
    print(f"Wavelengths range: {wavelengths.min():.1f} - {wavelengths.max():.1f}")
    print(f"Intensities shape: {intensities.shape}")
    print(f"Intensities range: {intensities.min():.1f} - {intensities.max():.1f}")
    
    # Check nitrogen region specifically
    n_mask = (wavelengths >= 741) & (wavelengths <= 743)
    n_region_intensities = intensities[n_mask]
    print(f"\nNitrogen region (741-743nm):")
    print(f"Wavelength points: {n_mask.sum()}")
    print(f"Intensities in N region: {n_region_intensities.min():.1f} - {n_region_intensities.max():.1f}")
    print(f"Mean intensity in N region: {n_region_intensities.mean():.1f}")
    
    # Test spectral feature extraction step by step
    spectral_gen = feature_pipeline.named_steps['spectral_features']
    
    print(f"\n=== SPECTRAL FEATURE EXTRACTION DEBUG ===")
    print(f"Strategy: {spectral_gen.strategy}")
    print(f"Total expected features: {len(spectral_gen.get_feature_names_out())}")
    
    # Extract just spectral features (before imputation/scaling)
    raw_features = spectral_gen.transform(input_data)
    
    feature_names = spectral_gen.get_feature_names_out()
    print(f"Raw features shape: {raw_features.shape}")
    
    # Check nitrogen-related features
    nitrogen_features = {}
    for i, name in enumerate(feature_names):
        if 'N_I' in name and 'help' not in name:
            nitrogen_features[name] = raw_features.iloc[0, i] if hasattr(raw_features, 'iloc') else raw_features[0, i]
    
    print(f"\nNitrogen-related features (before scaling):")
    for name, value in nitrogen_features.items():
        print(f"  {name}: {value:.6f}")
    
    # Check if any features are unusually small or large
    if hasattr(raw_features, 'iloc'):
        feature_values = raw_features.iloc[0].values
    else:
        feature_values = raw_features[0]
    
    print(f"\nFeature value statistics:")
    print(f"  Min: {feature_values.min():.6f}")
    print(f"  Max: {feature_values.max():.6f}")
    print(f"  Mean: {feature_values.mean():.6f}")
    print(f"  Std: {feature_values.std():.6f}")
    print(f"  Number near zero (<1e-6): {(np.abs(feature_values) < 1e-6).sum()}")
    print(f"  Number very large (>1e6): {(np.abs(feature_values) > 1e6).sum()}")
    
    # Apply full pipeline
    full_features = feature_pipeline.transform(input_data)
    print(f"\nAfter full pipeline (impute + clip + scale):")
    if hasattr(full_features, 'shape'):
        print(f"  Shape: {full_features.shape}")
        print(f"  Min: {full_features.min():.6f}")
        print(f"  Max: {full_features.max():.6f}")
        print(f"  Mean: {full_features.mean():.6f}")
        print(f"  Std: {full_features.std():.6f}")
    
    # Final prediction
    prediction = pipeline.predict(input_data)[0]
    print(f"\nFinal prediction: {prediction:.4f}")
    
    # Check actual value if available
    ref_df = pd.read_excel(config.reference_data_path)
    if sample_id in ref_df['Sample ID'].values:
        actual = ref_df[ref_df['Sample ID'] == sample_id]['Nitrogen %'].iloc[0]
        print(f"Actual value: {actual:.2f}")
        print(f"Error: {abs(prediction - actual):.4f}")
        relative_error = abs(prediction - actual) / actual * 100
        print(f"Relative error: {relative_error:.1f}%")
        print(f"Within 20.5%: {relative_error <= 20.5}")
    
print("\n" + "="*80)
print("INVESTIGATION COMPLETE")
print("="*80)