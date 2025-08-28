"""Check for scaler corruption and investigate the issue"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

print("=== SCALER CORRUPTION INVESTIGATION ===")

# Load the corrupted model
model_path = Path("models/full_context_extratrees_20250723_220001.pkl")
pipeline = joblib.load(model_path)
feature_pipeline = pipeline.named_steps['features']
corrupted_scaler = feature_pipeline.named_steps['scaler']

print("=== CORRUPTED SCALER ANALYSIS ===")
print(f"Mean shape: {corrupted_scaler.mean_.shape}")
print(f"Scale shape: {corrupted_scaler.scale_.shape}")

# Find problematic features
problematic_indices = []
for i in range(len(corrupted_scaler.scale_)):
    if corrupted_scaler.scale_[i] > 1e10 or corrupted_scaler.mean_[i] > 1e10:
        problematic_indices.append(i)

print(f"\nProblematic feature indices: {problematic_indices}")
print(f"Number of corrupted features: {len(problematic_indices)}")

# Get feature names to identify which features are corrupted
spectral_gen = feature_pipeline.named_steps['spectral_features']
feature_names = spectral_gen.get_feature_names_out()

print(f"\nCorrupted features:")
for idx in problematic_indices:
    if idx < len(feature_names):
        print(f"  Index {idx}: {feature_names[idx]}")
        print(f"    Mean: {corrupted_scaler.mean_[idx]:.2e}")
        print(f"    Scale: {corrupted_scaler.scale_[idx]:.2e}")

# Check if there are other models that might be better
print(f"\n=== CHECKING OTHER MODELS ===")
model_dir = Path("models")
extratrees_models = list(model_dir.glob("*extratrees*.pkl"))
print(f"Found {len(extratrees_models)} ExtraTrees models:")

for model_file in sorted(extratrees_models)[-5:]:  # Check last 5
    try:
        test_pipeline = joblib.load(model_file)
        test_feature_pipeline = test_pipeline.named_steps['features']
        test_scaler = test_feature_pipeline.named_steps['scaler']
        
        max_scale = test_scaler.scale_.max()
        max_mean = test_scaler.mean_.max()
        
        print(f"  {model_file.name}:")
        print(f"    Max scale: {max_scale:.2e}")
        print(f"    Max mean: {max_mean:.2e}")
        print(f"    Corrupted: {'YES' if max_scale > 1e10 or max_mean > 1e10 else 'NO'}")
        
    except Exception as e:
        print(f"  {model_file.name}: ERROR - {e}")

# Let's check training summaries to see when this corruption happened
print(f"\n=== CHECKING TRAINING SUMMARIES ===")
reports_dir = Path("reports")
summary_files = list(reports_dir.glob("training_summary_*.csv"))

if summary_files:
    latest_summary = sorted(summary_files)[-1]
    print(f"Latest training summary: {latest_summary.name}")
    
    try:
        df = pd.read_csv(latest_summary)
        extratrees_rows = df[df['model_name'].str.contains('extratrees', case=False, na=False)]
        
        print(f"ExtraTrees performance in latest training:")
        if not extratrees_rows.empty:
            for _, row in extratrees_rows.iterrows():
                strategy = row.get('strategy', 'unknown')
                model = row.get('model_name', 'unknown')
                within_20_5 = row.get('within_20.5%', 'N/A')
                r2 = row.get('r2', 'N/A')
                print(f"  {strategy}_{model}: R2={r2}, Within20.5%={within_20_5}")
        else:
            print("  No ExtraTrees models found in summary")
            
    except Exception as e:
        print(f"Error reading summary: {e}")

print("\n=== RECOMMENDATION ===")
print("The scaler in the current model is severely corrupted with extreme values.")
print("This is likely caused by:")
print("1. Extreme outliers in the training data that weren't properly clipped")  
print("2. NaN or inf values that got converted to very large numbers")
print("3. A bug in the feature extraction that produced extreme values")
print("\nSolutions:")
print("1. Use a different model file that's not corrupted")
print("2. Retrain with proper outlier handling")
print("3. Fix the saved model by replacing the corrupted scaler with a proper one")