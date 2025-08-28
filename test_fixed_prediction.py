"""Test the fixed prediction pipeline"""
from pathlib import Path
from src.config.pipeline_config import config, Config
from src.models.predictor import Predictor

# Setup config
config.run_timestamp = "debug"
project_root = Path(__file__).resolve().parent
config.data_dir = project_root / "data"
config.bad_prediction_files_dir = project_root / "bad_prediction_files"
config.reference_data_path = project_root / "data" / "reference_data" / "Final_Lab_Data_Nico_New.xlsx"
config.sample_id_column = "Sample ID"

# Create predictor
predictor = Predictor(config)

# Test with batch predictions on a single sample
input_dir = project_root / "data" / "raw" / "combo_6_8"
model_path = project_root / "models" / "full_context_extratrees_20250723_220001.pkl"

# Test the same sample
sample_files = list(input_dir.glob("1087316SANP2S027_G_2025_02_07_*.csv.txt"))
print(f"Found {len(sample_files)} files for sample 1087316SANP2S027_G_2025_02_07")

# Use the single prediction method
if sample_files:
    prediction = predictor.make_prediction(sample_files[0], model_path)
    print(f"Prediction from fixed predictor: {prediction:.4f}")
    
    # Compare with expected value
    import pandas as pd
    ref_df = pd.read_excel(config.reference_data_path)
    actual_value = ref_df[ref_df['Sample ID'] == '1087316SANP2S027_G_2025_02_07']['Nitrogen %'].values[0]
    print(f"Actual value: {actual_value:.2f}")
    print(f"Difference: {prediction - actual_value:.4f}")