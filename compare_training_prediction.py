"""Compare data format between training and prediction"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load training data to see its format
# First, let's look at an averaged file
averaged_dir = Path("data/averaged_files_per_sample")
files = list(averaged_dir.glob("*.csv.txt")) if averaged_dir.exists() else []
sample_file = files[0] if files else None

if sample_file:
    print("=== CLEANSED FILE FORMAT ===")
    df = pd.read_csv(sample_file)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")
    
    # Extract wavelengths and intensities
    wavelengths = df['Wavelength'].values
    intensity_cols = [col for col in df.columns if col != 'Wavelength']
    intensities = df[intensity_cols].values
    
    print(f"\nWavelengths shape: {wavelengths.shape}")
    print(f"Intensities shape: {intensities.shape}")
    print(f"Intensity range: min={intensities.min():.2f}, max={intensities.max():.2f}")

# Check how training data is prepared in main.py
print("\n=== TRAINING DATA FORMAT (from load_and_clean_data) ===")
print("Training creates data with columns: [sample_id, wavelengths, intensities, target]")
print("Where:")
print("  - wavelengths: 1D array of wavelength values")
print("  - intensities: 2D array (n_wavelengths x n_shots)")
print("  - Each row is one sample")

# Check what the predictor does
print("\n=== PREDICTION DATA FORMAT (from predictor.py) ===")
print("Prediction creates data with columns: [wavelengths, intensities]")
print("Where:")
print("  - wavelengths: 1D array of wavelength values")  
print("  - intensities: 1D array (averaged across shots)")
print("  - NO sample_id column during prediction")

# The key difference
print("\n=== KEY DIFFERENCES ===")
print("1. Training: intensities are 2D (n_wavelengths x n_shots)")
print("2. Prediction: intensities are 1D (averaged)")
print("3. Training: includes sample_id column (dropped before feature transform)")
print("4. Prediction: no sample_id column")

# Load a training sample to verify
train_data_example = {
    "Sample ID": "1087316SANP2S027_G_2025_02_07",
    "wavelengths": wavelengths if 'wavelengths' in locals() else np.zeros(2500),
    "intensities": intensities if 'intensities' in locals() else np.zeros((2500, 20)),
    "Nitrogen %": 3.66
}

print(f"\n=== VERIFYING INTENSITY DIMENSIONS ===")
if 'intensities' in locals():
    print(f"Training intensities shape: {intensities.shape}")
    print(f"Prediction would average to: {np.mean(intensities, axis=1).shape}")
    
    # Check specific wavelength region for nitrogen (741-743 nm)
    n_mask = (wavelengths >= 741) & (wavelengths <= 743)
    print(f"\nNitrogen region (741-743 nm):")
    print(f"Number of wavelength points: {n_mask.sum()}")
    print(f"Intensity values in N region: min={intensities[n_mask].min():.2f}, max={intensities[n_mask].max():.2f}")
    print(f"Average intensity in N region: {intensities[n_mask].mean():.2f}")