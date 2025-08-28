"""Debug script to understand prediction pipeline issues"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.config.pipeline_config import config, Config
from src.data_management.data_manager import DataManager
from src.cleansing.data_cleanser import DataCleanser

# Setup config
config.run_timestamp = "debug"
project_root = Path(__file__).resolve().parents[0]
config.data_dir = project_root / "data"
config.raw_data_dir = config.data_dir / "raw" / "data_5278_Phase3"
config.averaged_files_dir = config.data_dir / "averaged_files_per_sample"
config.cleansed_files_dir = config.data_dir / "cleansed_files_per_sample"
config.bad_files_dir = project_root / "bad_files"
config.bad_prediction_files_dir = project_root / "bad_prediction_files"
config.reference_data_path = project_root / "data" / "reference_data" / "Final_Lab_Data_Nico_New.xlsx"
config.sample_id_column = "Sample ID"

# Load model
model_path = project_root / "models" / "full_context_extratrees_20250723_220001.pkl"
pipeline = joblib.load(model_path)

print("=== MODEL PIPELINE STRUCTURE ===")
for name, step in pipeline.named_steps.items():
    print(f"{name}: {type(step).__name__}")

# Get a sample file from combo_6_8
sample_files = list((project_root / "data" / "raw" / "combo_6_8").glob("1087316SANP2S027_G_2025_02_07_*.csv.txt"))
print(f"\nUsing {len(sample_files)} files for sample 1087316SANP2S027_G_2025_02_07")

# Load and average the data (mimic prediction process)
data_manager = DataManager(config)
data_cleanser = DataCleanser(config)

wavelengths, averaged_intensities = data_manager.average_files_in_memory(sample_files)
print(f"\nWavelengths shape: {wavelengths.shape}")
print(f"Averaged intensities shape: {averaged_intensities.shape}")

# Clean the data
clean_intensities = data_cleanser.clean_spectra("1087316SANP2S027_G_2025_02_07", averaged_intensities)
if clean_intensities.ndim == 2:
    clean_intensities = np.mean(clean_intensities, axis=1)

print(f"Clean intensities shape: {clean_intensities.shape}")

# Create input dataframe as done in predictor
input_data = pd.DataFrame([{
    "wavelengths": wavelengths,
    "intensities": clean_intensities
}])

print(f"\nInput data shape: {input_data.shape}")
print(f"Input data columns: {input_data.columns.tolist()}")

# Transform through feature pipeline
feature_pipeline = pipeline.named_steps['features']
print("\n=== FEATURE PIPELINE STEPS ===")
for step_name, step in feature_pipeline.named_steps.items():
    print(f"{step_name}: {type(step).__name__}")

# Transform step by step
X = input_data
print(f"\nInitial X shape: {X.shape}")

# Step 1: Spectral features
spectral_gen = feature_pipeline.named_steps['spectral_features']
print(f"\nSpectral generator strategy: {spectral_gen.strategy}")
X_spectral = spectral_gen.transform(X)
print(f"After spectral features shape: {X_spectral.shape}")
print(f"First 5 features: {X_spectral.columns[:5].tolist() if hasattr(X_spectral, 'columns') else 'No columns'}")
print(f"Sample values: {X_spectral.iloc[0, :5].values if hasattr(X_spectral, 'iloc') else X_spectral[0, :5]}")

# Step 2: Imputer
imputer = feature_pipeline.named_steps['imputer']
X_imputed = imputer.transform(X_spectral)
print(f"\nAfter imputation shape: {X_imputed.shape}")
print(f"Any NaN remaining: {np.isnan(X_imputed).any()}")

# Step 3: Clipper
clipper = feature_pipeline.named_steps['clipper']
X_clipped = clipper.transform(X_imputed)
print(f"\nAfter clipping shape: {X_clipped.shape}")

# Step 4: Scaler
scaler = feature_pipeline.named_steps['scaler']
X_scaled = scaler.transform(X_clipped)
print(f"\nAfter scaling shape: {X_scaled.shape}")
print(f"Scaled mean: {X_scaled.mean():.4f}, std: {X_scaled.std():.4f}")
print(f"Scaled min: {X_scaled.min():.4f}, max: {X_scaled.max():.4f}")

# Now predict
model_pipeline = pipeline.named_steps['model']
print("\n=== MODEL PIPELINE ===")
for step_name, step in model_pipeline.named_steps.items():
    print(f"{step_name}: {type(step).__name__}")

# Get prediction
prediction = pipeline.predict(input_data)
print(f"\nPrediction: {prediction[0]:.4f}")

# Compare with ground truth
ref_df = pd.read_excel(config.reference_data_path)
actual_value = ref_df[ref_df['Sample ID'] == '1087316SANP2S027_G_2025_02_07']['Nitrogen %'].values[0]
print(f"Actual value: {actual_value:.2f}")
print(f"Difference: {prediction[0] - actual_value:.4f}")

# Check feature values for nitrogen-specific features
print("\n=== NITROGEN FEATURE VALUES ===")
feature_names = spectral_gen.get_feature_names_out()
for i, name in enumerate(feature_names):
    if 'N_I' in name and 'help' not in name:
        print(f"{name}: raw={X_spectral.iloc[0, i]:.4f}, scaled={X_scaled[0, i]:.4f}")