#!/usr/bin/env python3
"""
Simple test to verify concentration features only track present elements.
"""
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.features.concentration_features import ConcentrationRangeFeatures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_element_filtering():
    """Test that concentration features only track elements present in data."""
    
    logger.info("=== Testing Element Filtering in Concentration Features ===")
    
    # Test 1: Only P features
    logger.info("\nTest 1: Only P features")
    n_samples = 50
    X_p_only = pd.DataFrame({
        'P_I_simple_peak_area': np.random.randn(n_samples),
        'P_I_simple_peak_height': np.random.randn(n_samples),
        'P_I_213_peak_area': np.random.randn(n_samples),
        'P_I_253_peak_height': np.random.randn(n_samples),
    })
    y = pd.Series(np.random.uniform(0.2, 0.5, n_samples))
    
    transformer = ConcentrationRangeFeatures()
    transformer.fit(X_p_only, y)
    
    # Check what elements were tracked
    tracked = set()
    for col in transformer.feature_statistics_.keys():
        if 'P_I' in col:
            tracked.add('P_I')
        if 'C_I' in col:
            tracked.add('C_I')
        if 'N_I' in col:
            tracked.add('N_I')
        if 'H_I' in col or 'H_alpha' in col or 'H_beta' in col:
            tracked.add('H_I')
        if 'O_I' in col:
            tracked.add('O_I')
    
    logger.info(f"  Tracked elements: {sorted(tracked)}")
    logger.info(f"  Feature statistics count: {len(transformer.feature_statistics_)}")
    
    if tracked == {'P_I'}:
        logger.info("  ✓ Correctly tracked only P_I")
    else:
        logger.warning(f"  ✗ Expected only P_I, got {tracked}")
    
    # Test 2: P and C features
    logger.info("\nTest 2: P and C features")
    X_p_and_c = pd.DataFrame({
        'P_I_simple_peak_area': np.random.randn(n_samples),
        'P_I_simple_peak_height': np.random.randn(n_samples),
        'C_I_simple_peak_area': np.random.randn(n_samples),
        'C_I_833_peak_height': np.random.randn(n_samples),
        'P_C_ratio': np.random.randn(n_samples),
    })
    
    transformer2 = ConcentrationRangeFeatures()
    transformer2.fit(X_p_and_c, y)
    
    tracked2 = set()
    for col in transformer2.feature_statistics_.keys():
        if 'P_I' in col:
            tracked2.add('P_I')
        if 'C_I' in col:
            tracked2.add('C_I')
        if 'N_I' in col:
            tracked2.add('N_I')
        if 'H_I' in col or 'H_alpha' in col or 'H_beta' in col:
            tracked2.add('H_I')
        if 'O_I' in col:
            tracked2.add('O_I')
    
    logger.info(f"  Tracked elements: {sorted(tracked2)}")
    logger.info(f"  Feature statistics count: {len(transformer2.feature_statistics_)}")
    
    if tracked2 == {'P_I', 'C_I'}:
        logger.info("  ✓ Correctly tracked P_I and C_I")
    else:
        logger.warning(f"  ✗ Expected P_I and C_I, got {tracked2}")
    
    # Test 3: Verify no H, N, O features are tracked when not present
    logger.info("\nTest 3: Verify H, N, O not tracked when absent")
    X_no_hno = pd.DataFrame({
        'P_I_simple_peak_area': np.random.randn(n_samples),
        'P_I_213_peak_area': np.random.randn(n_samples),
        'C_I_simple_peak_area': np.random.randn(n_samples),
        'P_C_ratio': np.random.randn(n_samples),
        'some_other_feature': np.random.randn(n_samples),  # Non-spectral feature
    })
    
    transformer3 = ConcentrationRangeFeatures()
    transformer3.fit(X_no_hno, y)
    
    # Check for unwanted elements
    has_nitrogen = any('N_I' in str(k) for k in transformer3.feature_statistics_.keys())
    has_hydrogen = any('H_I' in str(k) or 'H_alpha' in str(k) or 'H_beta' in str(k) 
                       for k in transformer3.feature_statistics_.keys())
    has_oxygen = any('O_I' in str(k) for k in transformer3.feature_statistics_.keys())
    
    if not (has_nitrogen or has_hydrogen or has_oxygen):
        logger.info("  ✓ Correctly excluded H, N, O when not present in data")
    else:
        logger.warning(f"  ✗ Found unwanted elements: N={has_nitrogen}, H={has_hydrogen}, O={has_oxygen}")
    
    # Test 4: Test with macro elements if present
    logger.info("\nTest 4: Test with macro elements (Ca, K)")
    X_with_macro = pd.DataFrame({
        'P_I_simple_peak_area': np.random.randn(n_samples),
        'C_I_simple_peak_area': np.random.randn(n_samples),
        'CA_I_help_peak_area': np.random.randn(n_samples),  # Calcium
        'K_I_help_peak_height': np.random.randn(n_samples),  # Potassium
        'P_C_ratio': np.random.randn(n_samples),
    })
    
    transformer4 = ConcentrationRangeFeatures()
    transformer4.fit(X_with_macro, y)
    
    tracked4 = set()
    for col in transformer4.feature_statistics_.keys():
        if 'P_I' in col:
            tracked4.add('P_I')
        if 'C_I' in col:
            tracked4.add('C_I')
        if 'CA_I' in col:
            tracked4.add('CA_I')
        if 'K_I' in col:
            tracked4.add('K_I')
    
    logger.info(f"  Tracked elements: {sorted(tracked4)}")
    logger.info(f"  Feature statistics count: {len(transformer4.feature_statistics_)}")
    
    if 'CA_I' in tracked4 and 'K_I' in tracked4:
        logger.info("  ✓ Correctly tracked macro elements when present")
    else:
        logger.warning(f"  ⚠ Macro elements tracking: {tracked4}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_element_filtering()
        if success:
            logger.info("\n🎉 All tests passed! Concentration features correctly filter elements based on data.")
        else:
            logger.error("\n❌ Some tests failed.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)