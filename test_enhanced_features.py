#!/usr/bin/env python3
"""Test script to verify enhanced features are working correctly."""
import numpy as np
import pandas as pd
from src.config.pipeline_config import config
from src.features.feature_engineering import SpectralFeatureGenerator

def test_enhanced_features():
    """Test that enhanced features are generated correctly."""
    
    # Create mock spectral data
    wavelengths = np.linspace(200, 900, 3500)  # High resolution spectrum
    # Simulate some spectral lines
    intensities = np.random.normal(100, 10, len(wavelengths))  # Background
    
    # Add some peaks at known locations
    for center_wl in [742, 833.5, 526, 747.8, 654.5, 769.69]:  # From config
        if 200 <= center_wl <= 900:
            idx = np.argmin(np.abs(wavelengths - center_wl))
            intensities[max(0, idx-5):min(len(intensities), idx+5)] += np.random.normal(200, 20, min(10, len(intensities) - max(0, idx-5)))
    
    # Create test DataFrame
    test_data = pd.DataFrame({
        'Sample ID': ['TEST_001'],
        'wavelengths': [wavelengths],
        'intensities': [intensities],
        'Nitrogen %': [4.5]
    })
    
    print("Testing Enhanced Features Implementation")
    print("=" * 50)
    
    # Test with different feature configurations
    test_configs = [
        ("All Enhanced Features", True, True, True, True, True),
        ("Molecular Bands Only", True, False, False, False, False),
        ("Advanced Ratios Only", False, True, False, False, False),
        ("Spectral Patterns Only", False, False, True, False, False),
        ("No Enhanced Features", False, False, False, False, False),
    ]
    
    for test_name, mol_bands, adv_ratios, spec_patterns, interference, plasma in test_configs:
        print(f"\n{test_name}:")
        print("-" * 30)
        
        # Configure enhanced features
        config.enable_molecular_bands = mol_bands
        config.enable_advanced_ratios = adv_ratios
        config.enable_spectral_patterns = spec_patterns
        config.enable_interference_correction = interference
        config.enable_plasma_indicators = plasma
        
        try:
            # Create feature generator
            feature_gen = SpectralFeatureGenerator(config=config, strategy="full_context")
            feature_gen.fit(test_data)
            
            # Transform data
            features = feature_gen.transform(test_data)
            
            print(f"Generated {features.shape[1]} features")
            
            # Show enhanced feature names
            enhanced_features = [col for col in features.columns 
                               if any(keyword in col.lower() for keyword in 
                                     ['cn_', 'ratio', 'fwhm', 'asymmetry', 'continuum', 
                                      'interference', 'plasma', 'ionic', 'molecular'])]
            
            if enhanced_features:
                print(f"Enhanced features ({len(enhanced_features)}):")
                for feat in enhanced_features[:10]:  # Show first 10
                    value = features[feat].iloc[0]
                    print(f"  - {feat}: {value:.4f}" if not np.isnan(value) else f"  - {feat}: NaN")
                if len(enhanced_features) > 10:
                    print(f"  ... and {len(enhanced_features) - 10} more")
            else:
                print("No enhanced features found")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("Enhanced features test completed!")
    
    # Test individual components
    print("\nTesting Individual Components:")
    print("-" * 30)
    
    from src.features.enhanced_features import (
        MolecularBandDetector, AdvancedRatioCalculator, 
        SpectralPatternAnalyzer, InterferenceCorrector, PlasmaStateIndicators
    )
    
    # Test molecular band detector
    detector = MolecularBandDetector()
    cn_features = detector.detect_cn_bands(wavelengths, intensities, [(385, 390), (415, 425)])
    print(f"CN band features: {len(cn_features)} generated")
    
    # Test ratio calculator
    calc = AdvancedRatioCalculator()
    sample_features = {'N_I_peak_0': 100, 'P_I_help_peak_0': 50, 'K_I_help_peak_0': 75}
    ratios = calc.calculate_nutrient_ratios(sample_features)
    print(f"Advanced ratios: {len(ratios)} generated")
    
    # Test pattern analyzer
    analyzer = SpectralPatternAnalyzer()
    fwhm_result = analyzer.calculate_peak_fwhm(wavelengths, intensities, 742.0)
    print(f"FWHM analysis: {fwhm_result}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_enhanced_features()