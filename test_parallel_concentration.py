#!/usr/bin/env python3
"""
Test script to verify that parallel processing works with concentration features.
"""
import sys
sys.path.append('/home/payanico/magnesium_pipeline')

from src.config.pipeline_config import config
from src.features.concentration_features import create_enhanced_feature_pipeline_with_concentration
import numpy as np
import pandas as pd

def test_parallel_concentration_pipeline():
    """Test that the concentration pipeline can be created with parallel support."""
    
    # Create test config with concentration features enabled
    test_config = config.model_copy(update={
        'use_concentration_features': True,
        'run_timestamp': '20250828_test'
    })
    
    print(f"Testing parallel concentration pipeline...")
    print(f"Configuration: use_concentration_features={test_config.use_concentration_features}")
    
    # Test without parallel processing
    pipeline_sequential = create_enhanced_feature_pipeline_with_concentration(
        test_config, 'simple_only', use_parallel=False, n_jobs=1
    )
    print(f"✓ Sequential pipeline created successfully")
    print(f"  Steps: {[step[0] for step in pipeline_sequential.steps]}")
    
    # Test with parallel processing
    pipeline_parallel = create_enhanced_feature_pipeline_with_concentration(
        test_config, 'simple_only', use_parallel=True, n_jobs=2
    )
    print(f"✓ Parallel pipeline created successfully")
    print(f"  Steps: {[step[0] for step in pipeline_parallel.steps]}")
    
    # Verify that the parallel version uses ParallelSpectralFeatureGenerator
    spectral_step = pipeline_parallel.named_steps['spectral_features']
    print(f"✓ Spectral feature generator type: {type(spectral_step).__name__}")
    
    if "Parallel" in type(spectral_step).__name__:
        print(f"✓ SUCCESS: Parallel processing is enabled in concentration pipeline!")
        print(f"  Worker count: {spectral_step.n_jobs}")
    else:
        print(f"✗ FAILED: Still using sequential processing")
        return False
        
    return True

if __name__ == "__main__":
    success = test_parallel_concentration_pipeline()
    if success:
        print(f"\n🎉 All tests passed! Parallel concentration features are working.")
    else:
        print(f"\n❌ Tests failed!")
        sys.exit(1)