"""Debug training vs prediction data pipeline differences"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from src.config.pipeline_config import config, Config
from src.data_management.data_manager import DataManager
from src.cleansing.data_cleanser import DataCleanser
from src.features.feature_engineering import create_feature_pipeline

# Setup config
config.run_timestamp = "debug"
project_root = Path(__file__).resolve().parent
config.data_dir = project_root / "data"
config.raw_data_dir = config.data_dir / "raw" / "data_5278_Phase3"
config.averaged_files_dir = config.data_dir / "averaged_files_per_sample"
config.cleansed_files_dir = config.data_dir / "cleansed_files_per_sample"
config.bad_files_dir = project_root / "bad_files"
config.bad_prediction_files_dir = project_root / "bad_prediction_files"
config.reference_data_path = project_root / "data" / "reference_data" / "Final_Lab_Data_Nico_New.xlsx"
config.sample_id_column = "Sample ID"
config.target_column = "Nitrogen %"

print("=== TRAINING VS PREDICTION DATA FORMAT COMPARISON ===")

# Get a sample that was in training data for direct comparison
ref_df = pd.read_excel(config.reference_data_path)

# Load the model to see what training data looked like
model_path = project_root / "models" / "full_context_extratrees_20250723_220001.pkl"
pipeline = joblib.load(model_path)
feature_pipeline = pipeline.named_steps['features']

print(f"Model was trained with strategy: {feature_pipeline.named_steps['spectral_features'].strategy}")

# === SIMULATE TRAINING DATA FORMAT ===
print("\n=== TRAINING DATA FORMAT ===")

# Load a cleansed file (this is what training used)
cleansed_files = list((project_root / "data" / "cleansed_files_per_sample").glob("*.csv.txt"))
if cleansed_files:
    # Find a sample that's in reference data
    training_sample = None
    for file_path in cleansed_files[:10]:  # Check first 10
        sample_id = file_path.name.replace('.csv.txt', '')
        if sample_id in ref_df['Sample ID'].values:
            training_sample = sample_id
            break
    
    if training_sample:
        print(f"Using training sample: {training_sample}")
        
        # Load as training would
        cleansed_file = project_root / "data" / "cleansed_files_per_sample" / f"{training_sample}.csv.txt"
        df = pd.read_csv(cleansed_file)
        wavelengths = df['Wavelength'].values
        intensity_cols = [col for col in df.columns if col != 'Wavelength']
        intensities = df[intensity_cols].values
        
        print(f"Training format - wavelengths: {wavelengths.shape}, intensities: {intensities.shape}")
        
        # Get target value
        target = ref_df[ref_df['Sample ID'] == training_sample]['Nitrogen %'].iloc[0]
        
        # Create training format dataframe
        training_data = pd.DataFrame([{
            'Sample ID': training_sample,
            'wavelengths': wavelengths,
            'intensities': intensities,
            'Nitrogen %': target
        }])
        
        # Extract features as training does (drop sample_id and target first)
        X_train_raw = training_data.drop(columns=['Nitrogen %'])
        X_train_for_features = X_train_raw.drop(columns=['Sample ID'])
        
        print(f"Training input to feature pipeline: {X_train_for_features.shape}")
        print(f"Training input columns: {X_train_for_features.columns.tolist()}")
        
        # Transform features
        X_train_features = feature_pipeline.transform(X_train_for_features)
        print(f"Training features shape: {X_train_features.shape}")
        
        # Get some key feature values
        feature_names = feature_pipeline.named_steps['spectral_features'].get_feature_names_out()
        n_peak_idx = list(feature_names).index('N_I_peak_0')
        c_peak_idx = list(feature_names).index('C_I_peak_0')
        
        print(f"Training N_I_peak_0: {X_train_features[0, n_peak_idx]:.6f}")
        print(f"Training C_I_peak_0: {X_train_features[0, c_peak_idx]:.6f}")
        
        # Make prediction using full pipeline
        training_prediction = pipeline.predict(X_train_for_features)[0]
        print(f"Training-style prediction: {training_prediction:.4f}")
        print(f"Actual value: {target:.2f}")
        print(f"Training-style error: {abs(training_prediction - target):.4f}")

print("\n" + "="*80)

# === NOW TEST PREDICTION FORMAT ===
print("\n=== PREDICTION DATA FORMAT ===")

# Use combo_6_8 sample
combo_dir = project_root / "data" / "raw" / "combo_6_8"
sample_files = list(combo_dir.glob("1087316SANP2S027_G_2025_02_07_*.csv.txt"))

if sample_files:
    print(f"Using prediction sample: 1087316SANP2S027_G_2025_02_07")
    
    # Process as predictor does
    data_manager = DataManager(config)
    data_cleanser = DataCleanser(config)
    
    # Average files
    wavelengths, averaged_intensities = data_manager.average_files_in_memory(sample_files)
    print(f"After averaging: wavelengths={wavelengths.shape}, intensities={averaged_intensities.shape}")
    
    # Clean data  
    clean_intensities = data_cleanser.clean_spectra("1087316SANP2S027_G_2025_02_07", averaged_intensities)
    print(f"After cleaning: intensities={clean_intensities.shape}")
    
    # Create prediction format dataframe (NO sample_id)
    prediction_data = pd.DataFrame([{
        'wavelengths': wavelengths,
        'intensities': clean_intensities
    }])
    
    print(f"Prediction input to feature pipeline: {prediction_data.shape}")
    print(f"Prediction input columns: {prediction_data.columns.tolist()}")
    
    # Transform features
    X_pred_features = feature_pipeline.transform(prediction_data)
    print(f"Prediction features shape: {X_pred_features.shape}")
    
    print(f"Prediction N_I_peak_0: {X_pred_features[0, n_peak_idx]:.6f}")
    print(f"Prediction C_I_peak_0: {X_pred_features[0, c_peak_idx]:.6f}")
    
    # Make prediction
    prediction_prediction = pipeline.predict(prediction_data)[0]
    actual_value = ref_df[ref_df['Sample ID'] == '1087316SANP2S027_G_2025_02_07']['Nitrogen %'].iloc[0]
    print(f"Prediction-style prediction: {prediction_prediction:.4f}")
    print(f"Actual value: {actual_value:.2f}")
    print(f"Prediction-style error: {abs(prediction_prediction - actual_value):.4f}")

print("\n" + "="*80)
print("\n=== KEY DIFFERENCES TO INVESTIGATE ===")
print("1. Does the absence of 'Sample ID' column during prediction affect feature extraction?")
print("2. Are there differences in the data processing pipeline?")
print("3. Are there differences in wavelength alignment or intensity scaling?")
print("4. Could there be differences in the cleansing process?")