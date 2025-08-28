#!/usr/bin/env python3
"""
Debug script to understand why RawSpectralTransformer produces NaN values.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.pipeline_config import config
from features.feature_engineering import RawSpectralTransformer

# Import the helper function from main
sys.path.insert(0, str(Path(__file__).parent))
from main import load_and_clean_data, run_data_preparation

def debug_spectral_data_format():
    """Debug the actual format of spectral data."""
    print("=== Debugging Spectral Data Format ===")
    
    test_config = config.model_copy(deep=True)
    test_config.use_raw_spectral_data = True
    test_config.run_timestamp = "20250813_debug"
    
    try:
        # Load a small amount of data
        run_data_preparation(test_config)
        full_dataset, data_manager = load_and_clean_data(test_config)
        
        # Take just first sample
        sample = full_dataset.iloc[0]
        print(f"Sample columns: {list(full_dataset.columns)}")
        print(f"Sample shape: {full_dataset.shape}")
        
        # Check what's in wavelengths and intensities
        wavelengths = sample['wavelengths']
        intensities = sample['intensities']
        
        print(f"\nWavelengths type: {type(wavelengths)}")
        print(f"Wavelengths shape: {getattr(wavelengths, 'shape', 'No shape attribute')}")
        print(f"Wavelengths sample: {wavelengths[:10] if hasattr(wavelengths, '__getitem__') else wavelengths}")
        
        print(f"\nIntensities type: {type(intensities)}")
        print(f"Intensities shape: {getattr(intensities, 'shape', 'No shape attribute')}")
        print(f"Intensities sample: {intensities[:10] if hasattr(intensities, '__getitem__') else intensities}")
        
        # Check if they're arrays
        if hasattr(wavelengths, '__len__') and hasattr(intensities, '__len__'):
            print(f"\nWavelengths length: {len(wavelengths)}")
            print(f"Intensities length: {len(intensities)}")
            print(f"Lengths match: {len(wavelengths) == len(intensities)}")
            
            # Check ranges
            if len(wavelengths) > 0:
                wl_array = np.asarray(wavelengths)
                int_array = np.asarray(intensities)
                print(f"Wavelength range: {wl_array.min():.2f} - {wl_array.max():.2f} nm")
                print(f"Intensity range: {int_array.min():.2f} - {int_array.max():.2f}")
        
        return sample
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_raw_transformer_step_by_step(sample):
    """Debug the RawSpectralTransformer step by step."""
    if sample is None:
        return
        
    print("\n=== Debugging RawSpectralTransformer ===")
    
    test_config = config.model_copy(deep=True)
    test_config.use_raw_spectral_data = True
    
    # Create transformer
    transformer = RawSpectralTransformer(test_config)
    
    # Create single-row DataFrame
    test_data = pd.DataFrame([sample])
    
    print(f"PeakRegions configured: {len(test_config.all_regions)}")
    for region in test_config.all_regions:
        print(f"  - {region.element}: {region.lower_wavelength}-{region.upper_wavelength}nm")
    
    # Fit transformer
    transformer.fit(test_data)
    print(f"Expected features: {len(transformer.feature_names_out_)}")
    
    # Transform step by step
    print("\nProcessing sample...")
    wavelengths = np.asarray(sample['wavelengths'])
    intensities = np.asarray(sample['intensities'])
    
    print(f"Input wavelengths shape: {wavelengths.shape}")
    print(f"Input intensities shape: {intensities.shape}")
    
    # Check if reshaping is the issue
    if intensities.ndim == 1:
        intensities_2d = intensities.reshape(-1, 1)
        print(f"Reshaped intensities to: {intensities_2d.shape}")
    else:
        intensities_2d = intensities
        print(f"Intensities already 2D: {intensities_2d.shape}")
    
    # Try isolate_peaks
    try:
        region_results = transformer.extractor.isolate_peaks(
            wavelengths, 
            intensities_2d, 
            test_config.all_regions,
            baseline_correction=test_config.baseline_correction,
            area_normalization=False
        )
        
        print(f"Successfully isolated {len(region_results)} regions")
        
        for i, region_result in enumerate(region_results):
            element = region_result.region.element
            region_wl = region_result.wavelengths
            region_int = region_result.isolated_spectra
            print(f"  Region {i+1} ({element}): {len(region_wl)} wavelengths, intensities shape: {region_int.shape}")
            if len(region_wl) > 0:
                print(f"    Wavelength range: {region_wl.min():.2f}-{region_wl.max():.2f}nm")
                print(f"    Intensity range: {region_int.min():.2f}-{region_int.max():.2f}")
            
    except Exception as e:
        print(f"Error in isolate_peaks: {e}")
        import traceback
        traceback.print_exc()
    
    # Try full transform
    print("\nTrying full transform...")
    try:
        result = transformer.transform(test_data)
        print(f"Transform result shape: {result.shape}")
        print(f"Non-NaN values: {result.notna().sum().sum()}")
        print(f"Sample values: {result.iloc[0, :5].values}")
    except Exception as e:
        print(f"Error in transform: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Debugging Raw Spectral Transformer Issues")
    print("=" * 50)
    
    # Debug data format
    sample = debug_spectral_data_format()
    
    # Debug transformer
    debug_raw_transformer_step_by_step(sample)