#!/usr/bin/env python3
"""Test script to verify PCA fix in predictor.py"""
from pathlib import Path
import sys
sys.path.append('/home/payanico/magnesium_pipeline')

from src.config.pipeline_config import Config
from src.models.predictor import Predictor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_pca_loading():
    """Test if PCA transformer is properly loaded."""
    # Create config
    config = Config(
        run_timestamp="test",
        data_dir=Path("/home/payanico/magnesium_pipeline/data"),
        raw_data_dir=Path("/home/payanico/magnesium_pipeline/data/raw/data_5278_Phase3"),
        processed_data_dir=Path("/home/payanico/magnesium_pipeline/data/processed"),
        model_dir=Path("/home/payanico/magnesium_pipeline/models"),
        reports_dir=Path("/home/payanico/magnesium_pipeline/reports"),
        log_dir=Path("/home/payanico/magnesium_pipeline/logs"),
        reference_data_path=Path("/home/payanico/magnesium_pipeline/data/reference_data/Final_Lab_Data_Nico_New.xlsx"),
        bad_files_dir=Path("/home/payanico/magnesium_pipeline/bad_files"),
        averaged_files_dir=Path("/home/payanico/magnesium_pipeline/data/averaged_files_per_sample"),
        cleansed_files_dir=Path("/home/payanico/magnesium_pipeline/data/cleansed_files_per_sample"),
        bad_prediction_files_dir=Path("/home/payanico/magnesium_pipeline/bad_prediction_files")
    )
    
    # Create predictor
    predictor = Predictor(config)
    
    # Test loading the AutoGluon model with PCA
    model_path = Path("/home/payanico/magnesium_pipeline/models/autogluon/full_context_20250803_143937")
    
    print(f"Testing model loading from: {model_path}")
    
    try:
        model, needs_manual_features = predictor._load_model(model_path)
        
        print(f"Model loaded successfully!")
        print(f"Needs manual features: {needs_manual_features}")
        print(f"Has PCA transformer: {hasattr(predictor, 'pca_transformer') and predictor.pca_transformer is not None}")
        
        if hasattr(predictor, 'pca_transformer') and predictor.pca_transformer is not None:
            print(f"PCA components: {predictor.pca_transformer.n_components_}")
            print(f"Variance explained: {predictor.pca_transformer.explained_variance_ratio_.sum():.2%}")
            print(f"Expected feature names: {model.feature_names_in_[:5]}...")  # Show first 5
        else:
            print("No PCA transformer found")
            
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("Testing PCA Fix in Predictor")
    print("="*60)
    
    success = test_pca_loading()
    
    if success:
        print("\n✅ PCA fix appears to be working!")
        print("The AutoGluon model should now receive the correct PCA-transformed features.")
    else:
        print("\n❌ PCA fix test failed!")