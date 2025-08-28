#!/usr/bin/env python3
"""
Quick integration test to verify the full pipeline works with warning fixes.
"""
import warnings
import numpy as np
import pandas as pd
from src.config.pipeline_config import config
from src.features.feature_engineering import create_feature_pipeline

def test_pipeline_integration():
    """Test that the full feature pipeline works without warnings."""
    print("Testing full feature pipeline integration...")
    
    # Create mock spectral data
    np.random.seed(42)
    n_samples = 20
    n_wavelengths = 1000
    
    # Create wavelengths array (typical LIBS range)
    wavelengths = np.linspace(200, 900, n_wavelengths)
    
    # Create sample data
    data = []
    for i in range(n_samples):
        # Create mock spectrum with some peaks
        intensities = np.random.exponential(1000, n_wavelengths)
        # Add some magnesium peaks around the regions we care about
        for peak_wavelength in [214.9, 653.5, 775.5]:
            if 200 <= peak_wavelength <= 900:
                idx = int((peak_wavelength - 200) * n_wavelengths / 700)
                intensities[max(0, idx-5):min(n_wavelengths, idx+5)] += np.random.uniform(500, 2000)
        
        data.append({
            'Sample ID': f'sample_{i:03d}',
            'wavelengths': wavelengths,
            'intensities': intensities,
            'Magnesium dm %': np.random.uniform(0.1, 0.5)
        })
    
    df = pd.DataFrame(data)
    
    # Test pipeline creation and fitting
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Create feature pipeline
        pipeline = create_feature_pipeline(config, 'simple_only')
        
        # Split features and target
        X = df.drop(columns=['Magnesium dm %'])
        y = df['Magnesium dm %']
        
        # Fit and transform
        X_transformed = pipeline.fit_transform(X, y)
        
        # Check for warnings
        sklearn_warnings = [warning for warning in w if 'sklearn' in str(warning.filename)]
        divide_warnings = [warning for warning in sklearn_warnings if 'divide' in str(warning.message).lower()]
        
        print(f"Pipeline output type: {type(X_transformed)}")
        if isinstance(X_transformed, pd.DataFrame):
            print(f"Output shape: {X_transformed.shape}")
            print(f"Feature names preserved: {len(X_transformed.columns)} columns")
            print("✅ Pipeline returns DataFrame with feature names")
        else:
            print(f"Output shape: {X_transformed.shape}")
            print("❌ Pipeline returns numpy array - feature names lost")
        
        if divide_warnings:
            print(f"❌ Found {len(divide_warnings)} divide warnings in pipeline:")
            for warning in divide_warnings:
                print(f"  - {warning.message}")
        else:
            print("✅ No divide warnings in full pipeline")
        
        return len(divide_warnings) == 0 and isinstance(X_transformed, pd.DataFrame)

if __name__ == "__main__":
    success = test_pipeline_integration()
    if success:
        print("\n🎉 Pipeline integration test passed!")
    else:
        print("\n⚠️  Pipeline integration test failed")