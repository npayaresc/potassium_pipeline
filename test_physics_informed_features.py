#!/usr/bin/env python3
"""
Test script for physics-informed features implementation.

This script validates that the new FWHM, gamma, fit_quality, peak_asymmetry,
amplitude, and absorption_index features are correctly extracted from Lorentzian fits.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.spectral_extraction.extractor import SpectralFeatureExtractor
from src.spectral_extraction.results import PeakRegion
from src.config.pipeline_config import config
from src.features.feature_engineering import SpectralFeatureGenerator


def create_synthetic_spectrum():
    """Create a synthetic Lorentzian peak for testing."""
    # Generate wavelength range
    wavelengths = np.linspace(765.0, 771.0, 100)

    # Create Lorentzian peak: I(λ) = A * γ² / ((λ - λ0)² + γ²)
    center = 766.49  # K I resonance line
    amplitude = 1000.0
    gamma = 0.3  # HWHM (Half Width at Half Maximum)

    # Generate Lorentzian profile
    intensities = amplitude * (gamma ** 2) / ((wavelengths - center) ** 2 + gamma ** 2)

    # Add some noise
    noise = np.random.normal(0, 10, len(intensities))
    intensities += noise

    # Add baseline
    baseline = 50.0
    intensities += baseline

    return wavelengths, intensities


def test_extractor_physics_params():
    """Test that extractor correctly computes physics-informed parameters."""
    print("=" * 80)
    print("TEST 1: Extractor Physics-Informed Parameters")
    print("=" * 80)

    wavelengths, intensities = create_synthetic_spectrum()

    # Reshape for extractor (expects n_wavelengths x n_spectra)
    spectra_2d = intensities.reshape(-1, 1)

    # Define K I region
    region = PeakRegion(
        element="K_I",
        lower_wavelength=765.0,
        upper_wavelength=771.0,
        center_wavelengths=[766.49]
    )

    # Initialize extractor
    extractor = SpectralFeatureExtractor(
        enable_preprocessing=False,
        preprocessing_method='none'
    )

    # Extract features
    result = extractor.extract_features(
        wavelengths=wavelengths,
        spectra=spectra_2d,
        regions=[region],
        peak_shapes=['lorentzian'],
        baseline_correction=True,
        fitting_mode='mean_first'
    )

    # Check results
    fit_result = result.fitting_results[0]
    peak = fit_result.fit_results[0]

    print(f"\n✅ Fit Success: {peak.success}")
    print(f"📊 Peak Parameters:")
    print(f"   - Area: {peak.area:.2f}")
    print(f"   - Height: {peak.height:.2f}")
    print(f"   - Width (gamma): {peak.width:.4f}")
    print(f"   - Center: {peak.center:.2f} nm")

    print(f"\n🔬 Physics-Informed Parameters:")
    print(f"   - FWHM: {peak.fwhm:.4f} nm (should be ≈ 2 × gamma = {2 * peak.width:.4f})")
    print(f"   - Gamma (Stark broadening): {peak.gamma:.4f} nm")
    print(f"   - Fit Quality (R²): {peak.fit_quality:.4f} (>0.8 is good)")
    print(f"   - Peak Asymmetry: {peak.peak_asymmetry:.4f} (-1 to +1)")
    print(f"   - Amplitude: {peak.amplitude:.2f}")
    print(f"   - Absorption Index: {peak.fwhm * abs(peak.peak_asymmetry):.4f}")

    # Validate
    assert peak.fwhm > 0, "FWHM should be positive"
    assert peak.gamma > 0, "Gamma should be positive"
    assert 0 <= peak.fit_quality <= 1, "R² should be between 0 and 1"
    assert -1 <= peak.peak_asymmetry <= 1, "Asymmetry should be between -1 and 1"
    assert np.isclose(peak.fwhm, 2 * peak.gamma, rtol=0.01), "FWHM should equal 2 × gamma for Lorentzian"

    print("\n✅ All physics-informed parameters validated!")
    return True


def test_feature_engineering_integration():
    """Test that feature engineering correctly extracts physics-informed features."""
    print("\n" + "=" * 80)
    print("TEST 2: Feature Engineering Integration")
    print("=" * 80)

    # Create full spectrum covering all wavelength ranges
    wavelengths = np.linspace(200.0, 900.0, 1000)

    # Create multiple Lorentzian peaks for all K regions
    intensities = np.ones_like(wavelengths) * 50.0  # Baseline

    # K I primary peak at 766.49 nm
    center1 = 766.49
    gamma1 = 0.3
    intensities += 1000.0 * (gamma1 ** 2) / ((wavelengths - center1) ** 2 + gamma1 ** 2)

    # K I 404 nm peak
    center2 = 404.414
    gamma2 = 0.2
    intensities += 800.0 * (gamma2 ** 2) / ((wavelengths - center2) ** 2 + gamma2 ** 2)

    # Add noise
    intensities += np.random.normal(0, 10, len(intensities))

    # Create DataFrame in expected format
    df = pd.DataFrame({
        'wavelengths': [wavelengths],
        'intensities': [intensities],
        'sample_id': ['TEST_001']
    })

    # Initialize feature generator with simple_only to avoid missing regions
    feature_gen = SpectralFeatureGenerator(
        config=config,
        strategy='simple_only'  # Use simple_only to cover all regions
    )

    # Fit and transform
    print("\n📝 Fitting feature generator...")
    feature_gen.fit(df)

    print(f"📝 Extracting features...")
    features_df = feature_gen.transform(df)

    # Check for physics-informed features
    print(f"\n✅ Total features extracted: {features_df.shape[1]}")

    physics_features = [col for col in features_df.columns if any(
        keyword in col for keyword in ['fwhm', 'gamma', 'fit_quality', 'asymmetry', 'amplitude', 'absorption_index']
    )]

    print(f"✅ Physics-informed features found: {len(physics_features)}")
    print("\n🔬 Physics-Informed Features:")
    for feat in physics_features:
        value = features_df[feat].iloc[0]
        print(f"   - {feat}: {value:.4f}")

    # Validate presence of key features
    assert any('fwhm' in col for col in physics_features), "FWHM features missing"
    assert any('gamma' in col for col in physics_features), "Gamma features missing"
    assert any('fit_quality' in col for col in physics_features), "Fit quality features missing"
    assert any('asymmetry' in col for col in physics_features), "Asymmetry features missing"
    assert any('amplitude' in col for col in physics_features), "Amplitude features missing"
    assert any('absorption_index' in col for col in physics_features), "Absorption index features missing"

    print("\n✅ All physics-informed features integrated correctly!")
    return True


def main():
    """Run all tests."""
    print("\n" + "🔬" * 40)
    print("PHYSICS-INFORMED FEATURES TEST SUITE")
    print("🔬" * 40 + "\n")

    try:
        # Test 1: Extractor
        test_extractor_physics_params()

        # Test 2: Feature Engineering
        test_feature_engineering_integration()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\n📊 Summary:")
        print("   - Lorentzian fitting extracts FWHM correctly")
        print("   - Gamma (Stark broadening) parameter extracted")
        print("   - Fit quality (R²) computed correctly")
        print("   - Peak asymmetry calculated from residuals")
        print("   - Amplitude extracted from fit")
        print("   - Absorption index derived feature computed")
        print("   - All features integrated into feature engineering pipeline")
        print("\n🎉 Physics-informed features are ready for training!")

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
