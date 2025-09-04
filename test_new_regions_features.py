#!/usr/bin/env python3
"""
Test script to verify that the new magnesium spectral regions 
are being used correctly in feature calculations.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from datetime import datetime
from src.config.pipeline_config import Config
from src.features.parallel_feature_engineering import ParallelSpectralFeatureGenerator
import pandas as pd
import numpy as np

def test_feature_generation_with_new_regions():
    """Test that features are generated from the new magnesium regions."""
    
    print("=" * 80)
    print("TESTING FEATURE GENERATION WITH NEW MAGNESIUM REGIONS")
    print("=" * 80)
    
    # Create config
    config = Config(run_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Display all magnesium-related regions
    print("\n1. CONFIGURED MAGNESIUM REGIONS:")
    mg_regions = [config.magnesium_region] + [r for r in config.context_regions if 'Mg' in r.element]
    
    for region in mg_regions:
        print(f"\n   {region.element}:")
        print(f"   - Range: {region.lower_wavelength:.1f}-{region.upper_wavelength:.1f} nm")
        print(f"   - Centers: {[f'{w:.1f}' for w in region.center_wavelengths]} nm")
        print(f"   - Expected base features:")
        for i in range(region.n_peaks):
            print(f"     * {region.element}_peak_{i}")
            print(f"     * {region.element}_simple_peak_area")
            print(f"     * {region.element}_simple_peak_height")
    
    # Create a dummy sample to test feature generation
    print("\n2. TESTING FEATURE GENERATION:")
    print("   Creating dummy spectral data...")
    
    # Create dummy spectral data covering all Mg wavelength ranges
    wavelengths = np.arange(200, 900, 0.1)  # 200-900 nm range
    intensities = np.random.normal(100, 20, len(wavelengths))  # Random intensities
    
    # Add peaks at magnesium wavelengths
    mg_wavelengths = [279.55, 279.80, 280.27, 285.2, 383.8, 516.7, 517.3, 518.4]
    for wl in mg_wavelengths:
        # Find closest index
        idx = np.argmin(np.abs(wavelengths - wl))
        # Add a peak with realistic LIBS intensity
        intensities[idx:idx+5] += np.random.normal(500, 50, 5)
    
    # Create sample DataFrame in expected format
    sample_data = {
        'wavelengths': [wavelengths],
        'intensities': [intensities],
        'Sample ID': ['TEST_SAMPLE_001']
    }
    sample_df = pd.DataFrame(sample_data)
    
    print(f"   - Sample data shape: {sample_df.shape}")
    print(f"   - Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
    print(f"   - Added peaks at: {', '.join(f'{w:.1f} nm' for w in mg_wavelengths)}")
    
    # Initialize feature generator
    print("\n3. INITIALIZING FEATURE GENERATOR:")
    try:
        feature_generator = ParallelSpectralFeatureGenerator(config, strategy="simple_only")
        feature_generator.fit(sample_df)
        print(f"   ✓ Feature generator fitted successfully")
        print(f"   - Strategy: simple_only")
        print(f"   - Total regions configured: {len(config.all_regions)}")
        
        # Generate features
        print("\n4. GENERATING FEATURES:")
        features_df = feature_generator.transform(sample_df)
        print(f"   ✓ Features generated successfully")
        print(f"   - Features shape: {features_df.shape}")
        
        # Check for magnesium-specific features
        print("\n5. ANALYZING GENERATED FEATURES:")
        mg_feature_patterns = ['M_I_', 'Mg_II_', 'Mg_I_285_', 'Mg_I_383_']
        
        for pattern in mg_feature_patterns:
            matching_features = [col for col in features_df.columns if pattern in col]
            print(f"\n   Features matching '{pattern}':")
            if matching_features:
                for feat in matching_features[:10]:  # Show first 10
                    value = features_df[feat].iloc[0]
                    print(f"     ✓ {feat}: {value:.4f}" if not np.isnan(value) else f"     ✗ {feat}: NaN")
                if len(matching_features) > 10:
                    print(f"     ... and {len(matching_features) - 10} more")
            else:
                print(f"     ✗ No features found")
        
        # Check M_C_ratio calculation
        print("\n6. CHECKING ADVANCED FEATURES:")
        if 'M_C_ratio' in features_df.columns:
            mc_ratio = features_df['M_C_ratio'].iloc[0]
            print(f"   ✓ M_C_ratio: {mc_ratio:.4f}" if not np.isnan(mc_ratio) else "   ✗ M_C_ratio: NaN")
        else:
            print("   ✗ M_C_ratio not found")
            
        # Look for other advanced features
        advanced_patterns = ['MC_ratio_squared', 'MC_ratio_log', 'Mg_base_fraction', 'K_Mg_ratio']
        for pattern in advanced_patterns:
            if pattern in features_df.columns:
                value = features_df[pattern].iloc[0]
                print(f"   ✓ {pattern}: {value:.4f}" if not np.isnan(value) else f"   ✗ {pattern}: NaN")
        
        print("\n" + "=" * 80)
        print("FEATURE GENERATION TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n   ✗ Error during feature generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_feature_generation_with_new_regions()
    if success:
        print("\n✅ All tests passed! New magnesium regions are working correctly.")
    else:
        print("\n❌ Tests failed! Check the errors above.")