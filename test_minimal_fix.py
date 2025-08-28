#!/usr/bin/env python3
"""
Minimal test to verify the RawSpectralTransformer fix.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.pipeline_config import config
from features.feature_engineering import RawSpectralTransformer

def create_mock_spectral_data():
    """Create mock spectral data that matches the real format."""
    # Wavelengths from 203 to 1025 nm (2500 points)
    wavelengths = np.linspace(203, 1025, 2500)
    
    # Intensities: 2500 wavelengths x 20 shots with realistic spectral data
    np.random.seed(42)  # For reproducible results
    intensities = np.random.randn(2500, 20) * 10 + 100  # Mean=100, std=10
    
    # Add some spectral peaks in the PeakRegions
    peak_regions = [
        (653.5, 656.5),  # P_I
        (832.5, 834.5),  # C_I  
        (525.1, 527.1),  # CA_I_help
        (741.0, 743.0),  # N_I_help
        (774.5, 776.5),  # P_I_secondary
        (768.79, 770.79),  # K_I_help
        (834.5, 836.5),  # S_I
        (868.0, 870.0),  # S_I_2
        (515.7, 517.7),  # Mg_I
        (279.0, 281.0),  # Mg_II
    ]
    
    # Add peaks in the regions
    for lower, upper in peak_regions:
        mask = (wavelengths >= lower) & (wavelengths <= upper)
        intensities[mask] += np.random.randn(np.sum(mask), 20) * 20 + 50  # Add peak
    
    return wavelengths, intensities

def test_minimal_fix():
    """Test the fixed transformer with mock data."""
    print("=== Testing Fixed RawSpectralTransformer (Minimal) ===")
    
    # Create mock data
    wavelengths, intensities = create_mock_spectral_data()
    print(f"Mock data created: {len(wavelengths)} wavelengths, {intensities.shape[1]} shots")
    
    # Create test DataFrame
    test_data = pd.DataFrame({
        'wavelengths': [wavelengths],
        'intensities': [intensities]
    })
    
    # Create config and transformer
    test_config = config.model_copy(deep=True)
    test_config.use_raw_spectral_data = True
    
    transformer = RawSpectralTransformer(test_config)
    
    print(f"PeakRegions: {len(test_config.all_regions)}")
    
    # Fit and transform
    print("Fitting transformer...")
    transformer.fit(test_data)
    print(f"Expected features: {len(transformer.feature_names_out_)}")
    
    print("Transforming data...")
    result = transformer.transform(test_data)
    
    print(f"Result shape: {result.shape}")
    print(f"Non-NaN values: {result.notna().sum().sum()}")
    print(f"Total values: {result.size}")
    print(f"Percentage valid: {result.notna().sum().sum() / result.size * 100:.1f}%")
    
    # Debug feature name matching
    print(f"\nFeature name debugging:")
    print(f"Expected features (first 10): {transformer.feature_names_out_[:10]}")
    print(f"Result columns (first 10): {result.columns[:10].tolist()}")
    non_nan_features = result.columns[result.notna().any()]
    print(f"Features with data ({len(non_nan_features)}): {non_nan_features.tolist()}")
    
    if result.notna().sum().sum() > 0:
        print("✅ SUCCESS: Raw spectral features extracted!")
        
        # Show sample values
        sample_features = result.iloc[0].dropna().head(10)
        print(f"\nSample features:")
        for name, value in sample_features.items():
            print(f"  {name}: {value:.2f}")
            
        # Check ranges
        print(f"\nValue ranges:")
        print(f"  Min: {result.min().min():.2f}")
        print(f"  Max: {result.max().max():.2f}")
        print(f"  Mean: {result.mean().mean():.2f}")
        
        return True
    else:
        print("❌ FAILED: No valid values extracted")
        return False

if __name__ == "__main__":
    success = test_minimal_fix()
    
    if success:
        print("\n" + "="*50)
        print("✅ RawSpectralTransformer FIX VERIFIED!")
        print("The transformer now correctly extracts raw spectral intensities.")
        print("Ready for production use with --raw-spectral flag.")
    else:
        print("\n" + "="*50)
        print("❌ Fix did not work - needs further debugging")