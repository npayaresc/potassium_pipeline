"""Verify that the dimension fix is working correctly"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.config.pipeline_config import config, Config
from src.data_management.data_manager import DataManager
from src.cleansing.data_cleanser import DataCleanser
from src.models.predictor import Predictor
import joblib

# Setup config
config.run_timestamp = "debug"
project_root = Path(__file__).resolve().parent
config.data_dir = project_root / "data"
config.bad_prediction_files_dir = project_root / "bad_prediction_files"
config.reference_data_path = project_root / "data" / "reference_data" / "Final_Lab_Data_Nico_New.xlsx"
config.sample_id_column = "Sample ID"

print("=== VERIFYING DIMENSION FIX ===")

# Test the predictor's batch prediction method directly
predictor = Predictor(config)
input_dir = project_root / "data" / "raw" / "combo_6_8"

# Get sample files for one sample
sample_files = list(input_dir.glob("1087316SANP2S027_G_2025_02_07_*.csv.txt"))
print(f"Testing with {len(sample_files)} files for sample 1087316SANP2S027_G_2025_02_07")

# Manually replicate what the predictor does
data_manager = DataManager(config)
data_cleanser = DataCleanser(config)

# Step 1: Average files
wavelengths, averaged_intensities = data_manager.average_files_in_memory(sample_files)
print(f"After averaging: wavelengths={wavelengths.shape}, intensities={averaged_intensities.shape}")

# Step 2: Clean data
clean_intensities = data_cleanser.clean_spectra("1087316SANP2S027_G_2025_02_07", averaged_intensities)
print(f"After cleaning: intensities={clean_intensities.shape}")
print(f"Intensities remain 2D: {clean_intensities.ndim == 2}")

# Step 3: Create input dataframe as predictor does
input_data = pd.DataFrame([{
    "wavelengths": wavelengths,
    "intensities": clean_intensities  # This should be 2D now
}])

print(f"Input data created with intensities shape: {input_data.iloc[0]['intensities'].shape}")

# Step 4: Load model and test feature extraction
model_path = project_root / "models" / "full_context_extratrees_20250723_220001.pkl"
pipeline = joblib.load(model_path)

# Extract features step by step
feature_pipeline = pipeline.named_steps['features']
spectral_gen = feature_pipeline.named_steps['spectral_features']

# Test the spectral feature generation with 2D intensities
print(f"\n=== FEATURE EXTRACTION TEST ===")
X_spectral = spectral_gen.transform(input_data)
print(f"Features extracted: {X_spectral.shape}")

# Check specific nitrogen features
feature_names = spectral_gen.get_feature_names_out()
n_peak_idx = feature_names.index('N_I_peak_0')
c_peak_idx = feature_names.index('C_I_peak_0')

print(f"\nKey nitrogen features:")
print(f"N_I_peak_0: {X_spectral.iloc[0, n_peak_idx]:.6f}")
print(f"C_I_peak_0: {X_spectral.iloc[0, c_peak_idx]:.6f}")

# Compare with what we'd get if we used 1D (old way)
print(f"\n=== COMPARING 1D vs 2D INPUT ===")

# Create 1D version (old way)
clean_intensities_1d = np.mean(clean_intensities, axis=1)
input_data_1d = pd.DataFrame([{
    "wavelengths": wavelengths,
    "intensities": clean_intensities_1d
}])

X_spectral_1d = spectral_gen.transform(input_data_1d)
print(f"With 1D intensities - N_I_peak_0: {X_spectral_1d.iloc[0, n_peak_idx]:.6f}")
print(f"With 2D intensities - N_I_peak_0: {X_spectral.iloc[0, n_peak_idx]:.6f}")
print(f"Difference: {X_spectral.iloc[0, n_peak_idx] - X_spectral_1d.iloc[0, n_peak_idx]:.6f}")

# Make final predictions
prediction_2d = pipeline.predict(input_data)[0]
prediction_1d = pipeline.predict(input_data_1d)[0]

print(f"\n=== PREDICTION COMPARISON ===")
print(f"Prediction with 2D intensities: {prediction_2d:.4f}")
print(f"Prediction with 1D intensities: {prediction_1d:.4f}")
print(f"Difference: {prediction_2d - prediction_1d:.4f}")

# Get actual value
ref_df = pd.read_excel(config.reference_data_path)
actual_value = ref_df[ref_df['Sample ID'] == '1087316SANP2S027_G_2025_02_07']['Nitrogen %'].values[0]
print(f"Actual value: {actual_value:.2f}")
print(f"Error with 2D: {abs(prediction_2d - actual_value):.4f}")
print(f"Error with 1D: {abs(prediction_1d - actual_value):.4f}")