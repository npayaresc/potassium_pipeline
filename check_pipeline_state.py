"""Check if the saved pipeline state matches expectations"""
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

print("=== CHECKING PIPELINE STATE ===")

# Load the saved model
model_path = project_root / "models" / "full_context_extratrees_20250723_220001.pkl"
pipeline = joblib.load(model_path)
saved_feature_pipeline = pipeline.named_steps['features']

# Create fresh feature pipeline
fresh_feature_pipeline = create_feature_pipeline(config, "full_context")

print("=== COMPARING PIPELINE COMPONENTS ===")

# Check scaler state
saved_scaler = saved_feature_pipeline.named_steps['scaler']
fresh_scaler = fresh_feature_pipeline.named_steps['scaler']

print(f"Saved scaler fitted: {hasattr(saved_scaler, 'mean_')}")
print(f"Fresh scaler fitted: {hasattr(fresh_scaler, 'mean_')}")

if hasattr(saved_scaler, 'mean_'):
    print(f"Saved scaler mean shape: {saved_scaler.mean_.shape}")
    print(f"Saved scaler mean range: {saved_scaler.mean_.min():.6f} to {saved_scaler.mean_.max():.6f}")
    print(f"Saved scaler scale range: {saved_scaler.scale_.min():.6f} to {saved_scaler.scale_.max():.6f}")

# Check imputer state
saved_imputer = saved_feature_pipeline.named_steps['imputer']
fresh_imputer = fresh_feature_pipeline.named_steps['imputer']

print(f"\nSaved imputer fitted: {hasattr(saved_imputer, 'statistics_')}")
print(f"Fresh imputer fitted: {hasattr(fresh_imputer, 'statistics_')}")

if hasattr(saved_imputer, 'statistics_'):
    print(f"Saved imputer statistics shape: {saved_imputer.statistics_.shape}")
    print(f"Saved imputer statistics range: {np.nanmin(saved_imputer.statistics_):.6f} to {np.nanmax(saved_imputer.statistics_):.6f}")
    print(f"Number of NaN in imputer stats: {np.isnan(saved_imputer.statistics_).sum()}")

# Check clipper state
saved_clipper = saved_feature_pipeline.named_steps['clipper']
print(f"\nSaved clipper fitted: {hasattr(saved_clipper, 'lower_bounds_')}")

if hasattr(saved_clipper, 'lower_bounds_'):
    print(f"Saved clipper bounds shape: {saved_clipper.lower_bounds_.shape}")
    print(f"Lower bounds range: {saved_clipper.lower_bounds_.min():.6f} to {saved_clipper.lower_bounds_.max():.6f}")
    print(f"Upper bounds range: {saved_clipper.upper_bounds_.min():.6f} to {saved_clipper.upper_bounds_.max():.6f}")

# Check spectral feature generator state
saved_spectral = saved_feature_pipeline.named_steps['spectral_features']
fresh_spectral = fresh_feature_pipeline.named_steps['spectral_features']

print(f"\nSpectral generator strategy: {saved_spectral.strategy}")
print(f"Feature names match: {saved_spectral.get_feature_names_out() == fresh_spectral.get_feature_names_out()}")

# Test with actual data to see where the issue occurs
cleansed_files = list((project_root / "data" / "cleansed_files_per_sample").glob("*.csv.txt"))
if cleansed_files:
    sample_file = cleansed_files[0]
    df = pd.read_csv(sample_file)
    wavelengths = df['Wavelength'].values
    intensity_cols = [col for col in df.columns if col != 'Wavelength']
    intensities = df[intensity_cols].values
    
    input_data = pd.DataFrame([{
        'wavelengths': wavelengths,
        'intensities': intensities
    }])
    
    print(f"\n=== STEP-BY-STEP FEATURE TRANSFORMATION ===")
    
    # Step 1: Spectral features
    X1 = saved_spectral.transform(input_data)
    print(f"After spectral features: shape={X1.shape}")
    print(f"  Has NaN: {pd.isna(X1).any().any() if hasattr(X1, 'any') else np.isnan(X1).any()}")
    print(f"  Min: {np.nanmin(X1):.6f}, Max: {np.nanmax(X1):.6f}")
    
    # Step 2: Imputer
    X2 = saved_imputer.transform(X1)
    print(f"After imputer: shape={X2.shape}")
    print(f"  Has NaN: {np.isnan(X2).any()}")
    print(f"  Min: {X2.min():.6f}, Max: {X2.max():.6f}")
    
    # Step 3: Clipper
    X3 = saved_clipper.transform(X2)
    print(f"After clipper: shape={X3.shape}")
    print(f"  Min: {X3.min():.6f}, Max: {X3.max():.6f}")
    
    # Step 4: Scaler
    X4 = saved_scaler.transform(X3)
    print(f"After scaler: shape={X4.shape}")
    print(f"  Min: {X4.min():.6f}, Max: {X4.max():.6f}")
    print(f"  Mean: {X4.mean():.6f}, Std: {X4.std():.6f}")
    
    # Compare with full pipeline
    X_full = saved_feature_pipeline.transform(input_data)
    print(f"\nFull pipeline result matches step-by-step: {np.allclose(X4, X_full)}")

print("\nDONE")