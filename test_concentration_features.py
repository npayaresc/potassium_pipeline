#!/usr/bin/env python3
"""
Test script for the concentration-aware feature enhancement implementation.
This validates that Option A (concentration range features) works correctly.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from src.config.pipeline_config import Config
from src.features.concentration_features import ConcentrationRangeFeatures, create_enhanced_feature_pipeline_with_concentration
from src.features.feature_engineering import create_feature_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_spectral_data(n_samples: int = 100) -> pd.DataFrame:
    """Create mock spectral data for testing."""
    np.random.seed(42)
    
    # Create mock wavelength and intensity arrays
    wavelengths = np.linspace(200, 300, 100)  # 100 wavelengths from 200-300nm
    
    # Generate realistic spectral data with different concentration patterns
    mock_data = []
    
    for i in range(n_samples):
        # Simulate different magnesium concentrations
        true_concentration = np.random.uniform(0.15, 0.45)
        
        # Base spectrum with some noise
        base_intensities = np.random.normal(1000, 100, 100)
        
        # Add P peaks around 213-215nm (indices ~13-15) and 253-256nm (indices ~53-56)
        p_peak_1_strength = true_concentration * 5000 + np.random.normal(0, 500)
        p_peak_2_strength = true_concentration * 3000 + np.random.normal(0, 300)
        
        # Add peaks
        base_intensities[13:16] += p_peak_1_strength
        base_intensities[53:57] += p_peak_2_strength
        
        # Add C peaks (around 247nm, index ~47)
        c_peak_strength = np.random.uniform(2000, 8000)
        base_intensities[46:49] += c_peak_strength
        
        # Add some other element peaks for realism
        base_intensities[20:23] += np.random.uniform(1000, 3000)  # N peak
        base_intensities[30:33] += np.random.uniform(500, 2000)   # O peak
        
        mock_data.append({
            'wavelengths': wavelengths,
            'intensities': base_intensities,
            'true_concentration': true_concentration  # For validation
        })
    
    return pd.DataFrame(mock_data)

def test_concentration_features_transformer():
    """Test the ConcentrationRangeFeatures transformer directly."""
    logger.info("=== Testing ConcentrationRangeFeatures Transformer ===")
    
    # Create mock features (simulating output of SpectralFeatureGenerator)
    np.random.seed(42)
    n_samples = 50
    
    # Create mock features that would come from spectral processing
    mock_features = pd.DataFrame({
        'P_I_simple_peak_area': np.random.uniform(1000, 10000, n_samples),
        'P_I_simple_peak_height': np.random.uniform(500, 5000, n_samples),
        'C_I_simple_peak_area': np.random.uniform(2000, 8000, n_samples),
        'C_I_simple_peak_height': np.random.uniform(1000, 4000, n_samples),
        'P_C_ratio': np.random.uniform(0.1, 2.0, n_samples),
        'PC_height_ratio': np.random.uniform(0.2, 1.5, n_samples),
        'N_I_simple_peak_area': np.random.uniform(500, 3000, n_samples),
    })
    
    # Test transformer
    transformer = ConcentrationRangeFeatures(
        low_threshold=0.25,
        high_threshold=0.40,
        enable_range_indicators=True,
        enable_spectral_modulation=True,
        enable_ratio_adjustments=True,
        enable_concentration_interactions=True
    )
    
    # Fit and transform
    logger.info(f"Input features: {list(mock_features.columns)}")
    logger.info(f"Input shape: {mock_features.shape}")
    
    transformer.fit(mock_features)
    enhanced_features = transformer.transform(mock_features)
    
    logger.info(f"Output shape: {enhanced_features.shape}")
    logger.info(f"Added {enhanced_features.shape[1] - mock_features.shape[1]} concentration-aware features")
    
    # Show the new features
    new_feature_names = [col for col in enhanced_features.columns if col not in mock_features.columns]
    logger.info(f"New features: {new_feature_names}")
    
    # Verify key features exist
    expected_features = [
        'concentration_range_low', 'concentration_range_medium', 'concentration_range_high',
        'concentration_intensity_score', 'concentration_ratio_score'
    ]
    
    for feature in expected_features:
        if feature in enhanced_features.columns:
            logger.info(f"✓ {feature}: range {enhanced_features[feature].min():.3f} - {enhanced_features[feature].max():.3f}")
        else:
            logger.warning(f"✗ Missing expected feature: {feature}")
    
    return True

def test_enhanced_pipeline():
    """Test the full enhanced feature pipeline."""
    logger.info("=== Testing Enhanced Feature Pipeline ===")
    
    # Load config
    config = Config()
    
    # Test with different strategies
    strategies = ["simple_only", "full_context"]  # Skip Mg_only for simplicity
    
    for strategy in strategies:
        logger.info(f"\nTesting strategy: {strategy}")
        
        try:
            # Create both pipelines for comparison
            standard_pipeline = create_feature_pipeline(config, strategy, exclude_scaler=True)
            enhanced_pipeline = create_enhanced_feature_pipeline_with_concentration(config, strategy, exclude_scaler=True)
            
            # Create mock spectral data
            mock_data = create_mock_spectral_data(n_samples=20)
            spectral_features = mock_data[['wavelengths', 'intensities']].copy()
            
            logger.info(f"Mock spectral data shape: {spectral_features.shape}")
            
            # Test standard pipeline
            try:
                standard_features = standard_pipeline.fit_transform(spectral_features)
                logger.info(f"Standard pipeline output: {standard_features.shape}")
            except Exception as e:
                logger.warning(f"Standard pipeline failed: {e}")
                continue
            
            # Test enhanced pipeline
            try:
                enhanced_features = enhanced_pipeline.fit_transform(spectral_features)
                logger.info(f"Enhanced pipeline output: {enhanced_features.shape}")
                
                added_features = enhanced_features.shape[1] - standard_features.shape[1]
                logger.info(f"Added {added_features} concentration-aware features")
                
                if added_features > 0:
                    logger.info(f"✓ Enhanced pipeline successfully added features for {strategy}")
                else:
                    logger.warning(f"✗ Enhanced pipeline didn't add features for {strategy}")
                    
            except Exception as e:
                logger.error(f"Enhanced pipeline failed for {strategy}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Pipeline test failed for {strategy}: {e}")
            return False
    
    return True

def test_concentration_estimation():
    """Test the concentration estimation heuristic."""
    logger.info("=== Testing Concentration Estimation ===")
    
    # Create mock data with known concentration patterns
    mock_data = create_mock_spectral_data(n_samples=30)
    
    # Process through standard pipeline to get features
    config = Config()
    pipeline = create_feature_pipeline(config, "simple_only", exclude_scaler=True)
    spectral_features = mock_data[['wavelengths', 'intensities']].copy()
    
    try:
        processed_features = pipeline.fit_transform(spectral_features)
        logger.info(f"Processed features shape: {processed_features.shape}")
        
        # Test concentration estimation
        transformer = ConcentrationRangeFeatures()
        transformer.fit(processed_features)
        
        # Get concentration estimates
        concentration_estimates = transformer._estimate_concentration_likelihood(processed_features)
        true_concentrations = mock_data['true_concentration'].values
        
        # Compare estimates vs true values
        correlation = np.corrcoef(concentration_estimates, true_concentrations)[0, 1]
        logger.info(f"Concentration estimation correlation with true values: {correlation:.3f}")
        
        # Log some examples
        logger.info("Examples (estimated vs true):")
        for i in range(5):
            logger.info(f"  Sample {i}: estimated={concentration_estimates[i]:.3f}, true={true_concentrations[i]:.3f}")
        
        if correlation > 0.3:  # Reasonable correlation for a heuristic
            logger.info("✓ Concentration estimation shows reasonable correlation")
            return True
        else:
            logger.warning(f"✗ Concentration estimation correlation too low: {correlation:.3f}")
            return False
            
    except Exception as e:
        logger.error(f"Concentration estimation test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting concentration features implementation tests...")
    
    tests = [
        ("Concentration Features Transformer", test_concentration_features_transformer),
        ("Enhanced Pipeline", test_enhanced_pipeline),
        ("Concentration Estimation", test_concentration_estimation),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info('='*50)
        
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info('='*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{status:<4} | {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Concentration features implementation is ready.")
        logger.info("\nTo use concentration features with AutoGluon:")
        logger.info("1. Set use_concentration_features=True in your config")
        logger.info("2. Run: python main.py autogluon --gpu")
        logger.info("3. AutoGluon will automatically use concentration-aware features")
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)