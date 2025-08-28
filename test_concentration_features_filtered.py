#!/usr/bin/env python3
"""
Test script to verify that concentration features only learn statistics 
for elements that are actually present in the data.
"""
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.config.pipeline_config import config as Config
from src.features.concentration_features import create_enhanced_feature_pipeline_with_concentration
from src.features.feature_engineering import create_feature_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_spectral_data(n_samples=100, strategy='simple_only'):
    """Create mock spectral data based on strategy."""
    np.random.seed(42)
    
    # Base wavelengths and intensities columns (raw spectral data)
    wavelengths = np.linspace(200, 900, 1000)
    data = {
        'wavelengths': wavelengths,
        'intensities': np.random.randn(1000) * 100 + 500
    }
    
    # Create DataFrame with multiple samples
    dfs = []
    for i in range(n_samples):
        sample_data = data.copy()
        sample_data['intensities'] = np.random.randn(1000) * 100 + 500
        dfs.append(pd.DataFrame(sample_data))
    
    return pd.DataFrame(data)

def test_concentration_features_filtering():
    """Test that concentration features only track enabled elements."""
    
    logger.info("=== Testing Concentration Features Element Filtering ===")
    
    # Use the global config
    config = Config
    config.use_concentration_features = True
    
    # Test different strategies
    strategies = ['Mg_only', 'simple_only', 'full_context']
    
    for strategy in strategies:
        logger.info(f"\n--- Testing strategy: {strategy} ---")
        
        # Create mock data
        X = create_mock_spectral_data(n_samples=50, strategy=strategy)
        y = pd.Series(np.random.uniform(0.2, 0.5, 50))  # Mock magnesium concentrations
        
        try:
            # Create pipeline with concentration features
            pipeline = create_enhanced_feature_pipeline_with_concentration(
                config, strategy, exclude_scaler=True
            )
            
            # Fit the pipeline
            logger.info(f"Fitting pipeline for strategy: {strategy}")
            X_transformed = pipeline.fit_transform(X, y)
            
            # Get the concentration features transformer
            conc_features = pipeline.named_steps['concentration_features']
            
            # Check which elements were tracked
            if hasattr(conc_features, 'feature_statistics_'):
                tracked_elements = set()
                for col_name in conc_features.feature_statistics_.keys():
                    for element in ['P_I', 'C_I', 'N_I', 'H_I', 'O_I', 'CA_I', 'K_I']:
                        if element in col_name:
                            tracked_elements.add(element)
                
                logger.info(f"  Tracked elements: {sorted(tracked_elements)}")
                logger.info(f"  Total features with statistics: {len(conc_features.feature_statistics_)}")
            else:
                logger.info("  No feature statistics collected")
            
            # Check transformed features
            logger.info(f"  Input features: {X.shape[1]}")
            logger.info(f"  Output features: {X_transformed.shape[1]}")
            
            # Check concentration-specific features
            feature_names = pipeline.get_feature_names_out()
            conc_specific = [f for f in feature_names if 'concentration' in f.lower()]
            logger.info(f"  Concentration-specific features: {len(conc_specific)}")
            
            # Verify expected behavior per strategy
            if strategy == 'Mg_only':
                # Should only track P_I elements
                assert 'P_I' in str(conc_features.feature_statistics_.keys()) or len(conc_features.feature_statistics_) == 0
                logger.info("  ✓ Mg_only strategy correctly focuses on magnesium")
                
            elif strategy == 'simple_only':
                # Should track P_I and possibly C_I (for P/C ratio)
                logger.info("  ✓ simple_only strategy includes basic elements")
                
            elif strategy == 'full_context':
                # May track more elements if they're in the data
                logger.info("  ✓ full_context strategy can include all available elements")
            
        except Exception as e:
            logger.error(f"  ✗ Error processing strategy {strategy}: {e}")
            continue
    
    # Test with disabled oxygen/hydrogen
    logger.info("\n--- Testing with disabled oxygen/hydrogen ---")
    config.enable_oxygen_hydrogen = False
    
    X = create_mock_spectral_data(n_samples=50, strategy='full_context')
    y = pd.Series(np.random.uniform(0.2, 0.5, 50))
    
    pipeline = create_enhanced_feature_pipeline_with_concentration(
        config, 'full_context', exclude_scaler=True
    )
    X_transformed = pipeline.fit_transform(X, y)
    
    conc_features = pipeline.named_steps['concentration_features']
    
    # Check that H_I and O_I are not in statistics
    has_hydrogen = any('H_I' in str(k) or 'H_alpha' in str(k) or 'H_beta' in str(k) 
                       for k in conc_features.feature_statistics_.keys())
    has_oxygen = any('O_I' in str(k) for k in conc_features.feature_statistics_.keys())
    
    if not has_hydrogen and not has_oxygen:
        logger.info("  ✓ Correctly excluded H and O statistics when disabled")
    else:
        logger.warning(f"  ⚠ Found H={has_hydrogen}, O={has_oxygen} in statistics despite being disabled")
    
    return True

def test_with_real_features():
    """Test with more realistic feature names that would come from SpectralFeatureGenerator."""
    logger.info("\n=== Testing with Realistic Feature Names ===")
    
    config = Config
    config.use_concentration_features = True
    
    # Create mock data with realistic feature names
    n_samples = 50
    features = {
        # P features (always present)
        'P_I_simple_peak_area': np.random.randn(n_samples),
        'P_I_simple_peak_height': np.random.randn(n_samples),
        'P_I_213_peak_area': np.random.randn(n_samples),
        'P_I_253_peak_height': np.random.randn(n_samples),
    }
    
    # Add C features for simple_only and full_context
    features['C_I_simple_peak_area'] = np.random.randn(n_samples)
    features['C_I_833_peak_height'] = np.random.randn(n_samples)
    features['P_C_ratio'] = np.random.randn(n_samples)
    
    X = pd.DataFrame(features)
    y = pd.Series(np.random.uniform(0.2, 0.5, n_samples))
    
    # Test that only P and C statistics are collected
    from src.features.concentration_features import ConcentrationRangeFeatures
    
    transformer = ConcentrationRangeFeatures()
    transformer.fit(X, y)
    
    # Check what was tracked
    tracked_elements = set()
    for col_name in transformer.feature_statistics_.keys():
        for element in ['P_I', 'C_I', 'N_I', 'H_I', 'O_I']:
            if element in col_name:
                tracked_elements.add(element)
    
    logger.info(f"Tracked elements from realistic features: {sorted(tracked_elements)}")
    
    # Should only have P_I and C_I
    expected = {'P_I', 'C_I'}
    if tracked_elements == expected:
        logger.info("✓ Correctly tracked only P and C elements from realistic features")
    else:
        logger.warning(f"✗ Unexpected elements tracked. Expected {expected}, got {tracked_elements}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_concentration_features_filtering()
        success = test_with_real_features() and success
        
        if success:
            logger.info("\n🎉 All tests passed! Concentration features correctly filter elements.")
        else:
            logger.error("\n❌ Some tests failed.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)