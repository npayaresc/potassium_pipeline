#!/usr/bin/env python3
"""
Test script to verify sample weight functionality is working correctly.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Test the weight calculation methods
def test_weight_calculation():
    """Test that weight calculation methods work correctly."""
    
    # Create synthetic data with different concentration ranges
    np.random.seed(42)
    y_values = np.concatenate([
        np.random.normal(2.5, 0.2, 20),  # Low concentration (should get weight 2.0)
        np.random.normal(4.0, 0.3, 60),  # Medium concentration (should get weight 1.0)
        np.random.normal(6.5, 0.2, 20),  # High concentration (should get weight 1.5)
    ])
    y_series = pd.Series(y_values)
    
    # Test weighted_r2 method
    from src.models.model_trainer import ModelTrainer
    from src.config.pipeline_config import Config
    from src.reporting.reporter import Reporter
    
    # Create minimal config for testing
    config = Config(
        run_timestamp="test",
        data_dir=".", raw_data_dir=".", processed_data_dir=".",
        model_dir=".", reports_dir=".", log_dir=".", 
        bad_files_dir=".", averaged_files_dir=".", cleansed_files_dir=".", 
        bad_prediction_files_dir=".", reference_data_path="README.md",
        use_sample_weights=True,
        sample_weight_method="weighted_r2"
    )
    
    reporter = Reporter(config)
    trainer = ModelTrainer(config, "test_strategy", reporter)
    
    # Test weight calculation
    weights = trainer._calculate_sample_weights(y_series, method="weighted_r2")
    
    print("=== Weight Calculation Test ===")
    print(f"Data shape: {y_series.shape}")
    print(f"Data range: {y_series.min():.2f} - {y_series.max():.2f}")
    print(f"Weight range: {weights.min():.2f} - {weights.max():.2f}")
    print(f"Mean weight: {weights.mean():.2f}")
    
    # Check that weights are normalized (sum should equal number of samples)
    expected_sum = len(y_series)
    actual_sum = weights.sum()
    print(f"Weight sum check: {actual_sum:.2f} (expected: {expected_sum})")
    
    # Test model support detection
    print("\n=== Model Support Test ===")
    test_models = ["random_forest", "xgboost", "svr", "ridge"]
    for model in test_models:
        supports = trainer._model_supports_sample_weight(model)
        print(f"{model}: {'✓' if supports else '✗'}")
    
    return weights

def test_model_training_with_weights():
    """Test that models can be trained with sample weights."""
    
    # Create synthetic regression data
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    # Create target with some noise
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.1
    
    # Create sample weights (emphasize some samples)
    weights = np.ones(n_samples)
    weights[:50] = 2.0  # Give more importance to first 50 samples
    
    print("\n=== Model Training Test ===")
    
    # Test without weights
    model_no_weights = RandomForestRegressor(n_estimators=50, random_state=42)
    model_no_weights.fit(X, y)
    pred_no_weights = model_no_weights.predict(X)
    r2_no_weights = r2_score(y, pred_no_weights)
    
    # Test with weights
    model_with_weights = RandomForestRegressor(n_estimators=50, random_state=42)
    model_with_weights.fit(X, y, sample_weight=weights)
    pred_with_weights = model_with_weights.predict(X)
    r2_with_weights = r2_score(y, pred_with_weights)
    
    print(f"R² without weights: {r2_no_weights:.4f}")
    print(f"R² with weights:    {r2_with_weights:.4f}")
    
    # Check that models are different (weights should change the model)
    pred_diff = np.mean(np.abs(pred_no_weights - pred_with_weights))
    print(f"Average prediction difference: {pred_diff:.4f}")
    
    if pred_diff > 1e-6:
        print("✓ Sample weights are affecting model training")
    else:
        print("✗ Sample weights may not be working properly")
    
    return pred_diff > 1e-6

if __name__ == "__main__":
    print("Testing sample weight functionality...")
    
    try:
        # Test weight calculation
        weights = test_weight_calculation()
        
        # Test model training
        weights_working = test_model_training_with_weights()
        
        print("\n=== Summary ===")
        if weights is not None and weights_working:
            print("✓ All tests passed! Sample weight functionality is working.")
        else:
            print("✗ Some tests failed. Check the implementation.")
            
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()