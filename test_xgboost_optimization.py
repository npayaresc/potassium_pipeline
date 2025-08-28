#!/usr/bin/env python3
"""
Quick test script for XGBoost optimization
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))

from src.models.optimize_xgboost import setup_pipeline_config, run_data_preparation, load_and_clean_data, XGBoostOptimizer
from src.data_management.data_manager import DataManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_optimization():
    # Setup
    cfg = setup_pipeline_config(use_gpu=True)
    run_data_preparation(cfg)
    full_dataset, data_manager = load_and_clean_data(cfg)
    train_df, test_df = data_manager.create_reproducible_splits(full_dataset)
    
    # Extract features and targets
    X_train = train_df.drop(columns=[cfg.target_column])
    y_train = train_df[cfg.target_column].values
    
    X_test = test_df.drop(columns=[cfg.target_column])
    y_test = test_df[cfg.target_column].values
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    
    # Initialize optimizer
    optimizer = XGBoostOptimizer(cfg, 'full_context')
    
    # Run quick optimization (5 trials, 300 seconds)
    logger.info("Starting quick optimization test...")
    study = optimizer.optimize(X_train, y_train, n_trials=5, timeout=300)
    
    # Train final model
    logger.info("Training final model...")
    final_pipeline, train_metrics, test_metrics = optimizer.train_final_model(
        X_train, y_train, X_test, y_test
    )
    
    # Results
    print(f"\n--- OPTIMIZATION TEST RESULTS ---")
    print(f"Best Score: {optimizer.best_score:.4f}")
    print(f"Train R²: {train_metrics['r2']:.4f}")
    print(f"Test R²: {test_metrics['r2']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print("----------------------------------")

if __name__ == "__main__":
    test_optimization()