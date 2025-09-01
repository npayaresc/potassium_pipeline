#!/usr/bin/env python3
"""
Test script to validate LightGBM GPU fallback functionality.
"""

import logging
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_lightgbm_gpu_fallback():
    """Test LightGBM GPU fallback by attempting GPU then falling back to CPU."""
    
    # Create sample data
    X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
    
    logger.info("Testing LightGBM GPU fallback functionality")
    
    # Test 1: Try GPU mode (expected to fail)
    try:
        logger.info("Attempting LightGBM with GPU mode...")
        gpu_model = LGBMRegressor(
            device='cuda',
            gpu_device_id=0,
            verbose=-1,
            random_state=42
        )
        gpu_model.fit(X, y)
        logger.info("✅ GPU mode succeeded (unexpected)")
        return "GPU"
    except Exception as gpu_error:
        logger.warning(f"❌ GPU mode failed as expected: {gpu_error}")
        
        # Test 2: Fallback to CPU mode
        try:
            logger.info("Falling back to CPU mode...")
            cpu_model = LGBMRegressor(
                device='cpu',
                verbose=-1,
                random_state=42
            )
            cpu_model.fit(X, y)
            logger.info("✅ CPU fallback succeeded!")
            
            # Test prediction
            predictions = cpu_model.predict(X[:10])
            logger.info(f"✅ Predictions working: {predictions[:3]}")
            
            return "CPU_FALLBACK"
        except Exception as cpu_error:
            logger.error(f"❌ CPU fallback also failed: {cpu_error}")
            return "FAILED"

if __name__ == "__main__":
    result = test_lightgbm_gpu_fallback()
    print(f"\nResult: {result}")