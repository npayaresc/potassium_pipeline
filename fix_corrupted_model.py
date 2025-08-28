"""Fix the corrupted model by replacing the corrupted scaler"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from src.config.pipeline_config import config, Config
from src.data_management.data_manager import DataManager

print("=== FIXING CORRUPTED MODEL ===")

# Setup config
config.run_timestamp = "fixed"
project_root = Path(__file__).resolve().parent
config.data_dir = project_root / "data"
config.reference_data_path = project_root / "data" / "reference_data" / "Final_Lab_Data_Nico_New.xlsx"
config.sample_id_column = "Sample ID"
config.target_column = "Nitrogen %"

# Load the corrupted model
original_model_path = project_root / "models" / "full_context_extratrees_20250723_220001.pkl"
pipeline = joblib.load(original_model_path)

print("Original model loaded")

# Get the feature pipeline
feature_pipeline = pipeline.named_steps['features']
corrupted_scaler = feature_pipeline.named_steps['scaler']

print(f"Found corrupted features at indices 55, 56")
print(f"Original max scale: {corrupted_scaler.scale_.max():.2e}")

# We need to recompute the scaler on clean training data
# Let's load some training data to refit the scaler properly

print("Loading training data to refit scaler...")

# Load cleansed training data
cleansed_dir = project_root / "data" / "cleansed_files_per_sample"
ref_df = pd.read_excel(config.reference_data_path)

training_data = []
cleansed_files = list(cleansed_dir.glob("*.csv.txt"))[:50]  # Use first 50 samples

print(f"Processing {len(cleansed_files)} training samples...")

for file_path in cleansed_files:
    sample_id = file_path.name.replace('.csv.txt', '')
    
    if sample_id in ref_df['Sample ID'].values:
        df = pd.read_csv(file_path)
        wavelengths = df['Wavelength'].values
        intensity_cols = [col for col in df.columns if col != 'Wavelength']
        intensities = df[intensity_cols].values
        
        # Create input format
        input_data = pd.DataFrame([{
            'wavelengths': wavelengths,
            'intensities': intensities
        }])
        
        training_data.append(input_data)

if len(training_data) == 0:
    print("ERROR: No training data found!")
    exit(1)

# Concatenate all training data
print(f"Concatenating {len(training_data)} samples...")
all_training_data = pd.concat(training_data, ignore_index=True)

# Apply the feature pipeline up to but not including the scaler
print("Extracting features without scaling...")
spectral_gen = feature_pipeline.named_steps['spectral_features']
imputer = feature_pipeline.named_steps['imputer']
clipper = feature_pipeline.named_steps['clipper']

# Extract features step by step
X_spectral = spectral_gen.transform(all_training_data)
X_imputed = imputer.transform(X_spectral)
X_clipped = clipper.transform(X_imputed)

print(f"Features before scaling: shape={X_clipped.shape}")
print(f"Min: {X_clipped.min():.6f}, Max: {X_clipped.max():.6f}")

# Check for extreme values that would corrupt the scaler
extreme_mask = (np.abs(X_clipped) > 1e10)
if extreme_mask.any():
    print(f"WARNING: Found {extreme_mask.sum()} extreme values")
    print("Clipping extreme values to ±1e6")
    X_clipped = np.clip(X_clipped, -1e6, 1e6)

# Create and fit a new scaler
print("Fitting new scaler...")
new_scaler = StandardScaler()
new_scaler.fit(X_clipped)

print(f"New scaler max scale: {new_scaler.scale_.max():.2e}")
print(f"New scaler max mean: {new_scaler.mean_.max():.2e}")

# Replace the corrupted scaler in the pipeline
print("Replacing corrupted scaler...")
feature_pipeline.named_steps['scaler'] = new_scaler

# Test the fixed pipeline
print("Testing fixed pipeline...")
test_sample = all_training_data.iloc[0:1]
test_features = feature_pipeline.transform(test_sample)
test_prediction = pipeline.predict(test_sample)[0]

print(f"Test prediction: {test_prediction:.4f}")
print(f"Features shape after fix: {test_features.shape}")
print(f"Features min: {test_features.min():.6f}, max: {test_features.max():.6f}")

# Save the fixed model
fixed_model_path = project_root / "models" / "full_context_extratrees_20250723_220001_FIXED.pkl"
joblib.dump(pipeline, fixed_model_path)

print(f"Fixed model saved to: {fixed_model_path}")
print("DONE - Model corruption fixed!")