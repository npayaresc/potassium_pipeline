#!/usr/bin/env python3
"""Debug script to understand feature count mismatch."""

import pandas as pd
import numpy as np
from src.config.pipeline_config import config
from src.features.concentration_features import create_enhanced_feature_pipeline_with_concentration

# Create a dummy sample data that covers the expected spectral regions for magnesium
# Check the config to see what regions are expected
print("Expected spectral regions:")
for region in config.all_regions:
    print(f"  {region.element}: {region.lower_wavelength}-{region.upper_wavelength} nm")

# Create wavelength range that covers all regions (need to span 279-870 nm)
wavelengths = np.linspace(270, 880, 5000)  # Broad range covering all spectral regions
dummy_data = pd.DataFrame({
    'wavelengths': [wavelengths] * 5,
    'intensities': [np.random.rand(len(wavelengths)) + 1000] * 5  # Add baseline to make it more realistic
})

print("Creating enhanced feature pipeline...")
pipeline = create_enhanced_feature_pipeline_with_concentration(config, 'simple_only', exclude_scaler=True)

print("Fitting pipeline...")
X_transformed = pipeline.fit_transform(dummy_data)
print(f"Transformed data shape: {X_transformed.shape}")

print("\nChecking individual steps:")
for name, step in pipeline.named_steps.items():
    print(f"Step: {name}")
    if hasattr(step, 'get_feature_names_out'):
        try:
            if name == 'spectral_features':
                feature_names = step.get_feature_names_out()
                print(f"  Feature count: {len(feature_names)}")
                print(f"  Sample features: {feature_names[:5]}...")
            elif name == 'concentration_features':
                # This step adds features to existing ones
                spectral_names = pipeline.named_steps['spectral_features'].get_feature_names_out()
                feature_names = step.get_feature_names_out(spectral_names)
                print(f"  Feature count: {len(feature_names)}")
                print(f"  Added features: {[n for n in feature_names if n not in spectral_names]}")
            else:
                print(f"  {name} doesn't change feature names")
        except Exception as e:
            print(f"  Error getting feature names: {e}")
    else:
        print(f"  No get_feature_names_out method")

print("\nTrying pipeline-level get_feature_names_out...")
try:
    pipeline_features = pipeline.get_feature_names_out(['wavelengths', 'intensities'])
    print(f"Pipeline feature count: {len(pipeline_features)}")
except Exception as e:
    print(f"Pipeline get_feature_names_out failed: {e}")

print("\nTesting manual feature name building...")
# Test the same logic as in the AutoGluon trainer
def build_feature_names_manually(pipeline, input_features):
    """Replicate the _build_feature_names_from_pipeline logic"""
    current_names = list(input_features)
    
    # Step 1: SpectralFeatureGenerator
    spectral_step = pipeline.named_steps.get('spectral_features')
    if spectral_step and hasattr(spectral_step, 'get_feature_names_out'):
        current_names = list(spectral_step.get_feature_names_out())
        print(f"After spectral_features: {len(current_names)} features")
    
    # Step 2: ConcentrationRangeFeatures
    concentration_step = pipeline.named_steps.get('concentration_features')
    if concentration_step and hasattr(concentration_step, 'get_feature_names_out'):
        concentration_names = concentration_step.get_feature_names_out(current_names)
        current_names = list(concentration_names)
        print(f"After concentration_features: {len(current_names)} features")
    
    return current_names

manual_features = build_feature_names_manually(pipeline, ['wavelengths', 'intensities'])
print(f"Manual feature building result: {len(manual_features)} features")
print(f"Expected: 100, Got: {len(manual_features)}, Match: {len(manual_features) == 100}")