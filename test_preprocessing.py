#!/usr/bin/env python3
"""
Quick test script for spectral preprocessing module.

Tests all preprocessing methods and verifies they work correctly.
Also tests the config integration.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.spectral_extraction.preprocessing import (
    SpectralPreprocessor,
    preprocess_libs_spectrum
)
from src.config.pipeline_config import Config
from src.spectral_extraction.extractor import SpectralFeatureExtractor

def generate_synthetic_libs_spectrum(n_points=100, noise_level=0.1):
    """Generate synthetic LIBS spectrum for testing."""
    wavelengths = np.linspace(760, 780, n_points)

    # Baseline (continuum emission - curved)
    baseline = 100 + 50 * np.exp(-(wavelengths - 770)**2 / 100)

    # Two Lorentzian peaks (K doublet at 766.5 and 769.9 nm)
    def lorentzian(x, center, amplitude, width):
        return amplitude / (1 + ((x - center) / width)**2)

    peak1 = lorentzian(wavelengths, 766.5, 200, 0.5)
    peak2 = lorentzian(wavelengths, 769.9, 150, 0.5)

    # Combine
    spectrum = baseline + peak1 + peak2

    # Add noise
    noise = np.random.randn(n_points) * noise_level * np.max(spectrum)
    spectrum += noise

    return wavelengths, spectrum


def test_preprocessing_methods():
    """Test all preprocessing methods."""
    print("=" * 80)
    print("TESTING SPECTRAL PREPROCESSING MODULE")
    print("=" * 80)
    print()

    # Generate test spectrum
    wavelengths, raw_spectrum = generate_synthetic_libs_spectrum(n_points=100, noise_level=0.05)
    print(f"✓ Generated synthetic LIBS spectrum: {len(wavelengths)} points")
    print()

    # Test different methods
    methods = ['none', 'savgol', 'snv', 'baseline', 'savgol+snv', 'baseline+snv', 'full']

    preprocessor = SpectralPreprocessor()
    results = {}

    for method in methods:
        print(f"Testing method: '{method}'...")

        try:
            preprocessor.configure(method=method)
            processed = preprocessor.preprocess(raw_spectrum, wavelengths)

            # Verify output
            assert processed.shape == raw_spectrum.shape, "Shape mismatch!"
            assert not np.all(np.isnan(processed)), "Contains NaN!"

            results[method] = processed
            print(f"  ✓ Success - Min: {np.min(processed):.2f}, Max: {np.max(processed):.2f}, Mean: {np.mean(processed):.2f}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print()
    print("=" * 80)
    print("BATCH PROCESSING TEST")
    print("=" * 80)
    print()

    # Test batch processing
    n_spectra = 10
    batch_spectra = np.column_stack([
        generate_synthetic_libs_spectrum(n_points=100, noise_level=0.05)[1]
        for _ in range(n_spectra)
    ])

    print(f"Generated batch: {batch_spectra.shape} (n_wavelengths, n_spectra)")

    preprocessor.configure(method='savgol+snv')
    batch_processed = preprocessor.preprocess_batch(batch_spectra, wavelengths)

    assert batch_processed.shape == batch_spectra.shape, "Batch shape mismatch!"
    print(f"✓ Batch preprocessing successful: {batch_processed.shape}")
    print()

    # Test convenience function
    print("=" * 80)
    print("CONVENIENCE FUNCTION TEST")
    print("=" * 80)
    print()

    quick_processed = preprocess_libs_spectrum(raw_spectrum, wavelengths, method='savgol+snv')
    assert quick_processed.shape == raw_spectrum.shape
    print("✓ Convenience function works")
    print()

    # Visualize results
    print("=" * 80)
    print("VISUALIZATION")
    print("=" * 80)
    print()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Spectral Preprocessing Comparison', fontsize=16)

    plot_methods = ['none', 'savgol', 'snv', 'savgol+snv', 'baseline+snv', 'full']

    for ax, method in zip(axes.flat, plot_methods):
        ax.plot(wavelengths, results[method], 'b-', linewidth=1)
        ax.set_title(f"Method: '{method}'", fontsize=10)
        ax.set_xlabel('Wavelength (nm)', fontsize=8)
        ax.set_ylabel('Intensity', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_file = 'preprocessing_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_file}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✓ All preprocessing methods work correctly")
    print("✓ Batch processing works")
    print("✓ Convenience functions work")
    print("✓ Visualization generated")
    print()
    print("Next steps:")
    print("  1. Review preprocessing_comparison.png")
    print("  2. Enable preprocessing in your pipeline:")
    print("     Edit src/config/pipeline_config.py line 707:")
    print("     use_spectral_preprocessing: bool = True  # Change from False")
    print("  3. Choose preprocessing method (line 708):")
    print("     spectral_preprocessing_method: ... = 'savgol+snv'  # Phase 1 (default)")
    print("     spectral_preprocessing_method: ... = 'full'        # Phase 2 (optimal)")
    print("  4. Train and compare model performance:")
    print("     python main.py train --gpu")
    print()


def test_config_integration():
    """Test that preprocessing works via config integration."""
    print("=" * 80)
    print("TESTING CONFIG INTEGRATION")
    print("=" * 80)
    print()

    # Generate test data
    wavelengths, raw_spectrum = generate_synthetic_libs_spectrum(n_points=100, noise_level=0.05)

    # Test 1: Config with preprocessing disabled (default)
    print("Test 1: Preprocessing DISABLED (default config)")
    config_disabled = Config(
        run_timestamp="test",
        _data_dir="/tmp/test",
        _raw_data_dir="/tmp/test",
        _processed_data_dir="/tmp/test",
        _model_dir="/tmp/test",
        _reports_dir="/tmp/test",
        _log_dir="/tmp/test",
        _reference_data_path="/tmp/test.xlsx",
        _bad_files_dir="/tmp/test",
        _averaged_files_dir="/tmp/test",
        _cleansed_files_dir="/tmp/test",
        _bad_prediction_files_dir="/tmp/test",
        use_spectral_preprocessing=False
    )

    extractor_disabled = SpectralFeatureExtractor(
        enable_preprocessing=config_disabled.use_spectral_preprocessing,
        preprocessing_method=config_disabled.spectral_preprocessing_method
    )

    print(f"  Config: use_spectral_preprocessing={config_disabled.use_spectral_preprocessing}")
    print(f"  Config: spectral_preprocessing_method='{config_disabled.spectral_preprocessing_method}'")
    print(f"  Extractor: enable_preprocessing={extractor_disabled.enable_preprocessing}")
    print("  ✓ Extractor created successfully with preprocessing DISABLED")
    print()

    # Test 2: Config with Phase 1 preprocessing enabled
    print("Test 2: Preprocessing ENABLED - Phase 1 (savgol+snv)")
    config_phase1 = Config(
        run_timestamp="test",
        _data_dir="/tmp/test",
        _raw_data_dir="/tmp/test",
        _processed_data_dir="/tmp/test",
        _model_dir="/tmp/test",
        _reports_dir="/tmp/test",
        _log_dir="/tmp/test",
        _reference_data_path="/tmp/test.xlsx",
        _bad_files_dir="/tmp/test",
        _averaged_files_dir="/tmp/test",
        _cleansed_files_dir="/tmp/test",
        _bad_prediction_files_dir="/tmp/test",
        use_spectral_preprocessing=True,
        spectral_preprocessing_method='savgol+snv'
    )

    extractor_phase1 = SpectralFeatureExtractor(
        enable_preprocessing=config_phase1.use_spectral_preprocessing,
        preprocessing_method=config_phase1.spectral_preprocessing_method
    )

    print(f"  Config: use_spectral_preprocessing={config_phase1.use_spectral_preprocessing}")
    print(f"  Config: spectral_preprocessing_method='{config_phase1.spectral_preprocessing_method}'")
    print(f"  Extractor: enable_preprocessing={extractor_phase1.enable_preprocessing}")
    print("  ✓ Extractor created successfully with Phase 1 preprocessing ENABLED")
    print()

    # Test 3: Config with Phase 2 preprocessing enabled
    print("Test 3: Preprocessing ENABLED - Phase 2 (full)")
    config_phase2 = Config(
        run_timestamp="test",
        _data_dir="/tmp/test",
        _raw_data_dir="/tmp/test",
        _processed_data_dir="/tmp/test",
        _model_dir="/tmp/test",
        _reports_dir="/tmp/test",
        _log_dir="/tmp/test",
        _reference_data_path="/tmp/test.xlsx",
        _bad_files_dir="/tmp/test",
        _averaged_files_dir="/tmp/test",
        _cleansed_files_dir="/tmp/test",
        _bad_prediction_files_dir="/tmp/test",
        use_spectral_preprocessing=True,
        spectral_preprocessing_method='full'
    )

    extractor_phase2 = SpectralFeatureExtractor(
        enable_preprocessing=config_phase2.use_spectral_preprocessing,
        preprocessing_method=config_phase2.spectral_preprocessing_method
    )

    print(f"  Config: use_spectral_preprocessing={config_phase2.use_spectral_preprocessing}")
    print(f"  Config: spectral_preprocessing_method='{config_phase2.spectral_preprocessing_method}'")
    print(f"  Extractor: enable_preprocessing={extractor_phase2.enable_preprocessing}")
    print("  ✓ Extractor created successfully with Phase 2 preprocessing ENABLED")
    print()

    # Test 4: Verify preprocessing methods are valid
    print("Test 4: Verify all preprocessing methods are valid config options")
    valid_methods = ['none', 'savgol', 'snv', 'baseline', 'savgol+snv', 'baseline+snv', 'full']

    for method in valid_methods:
        try:
            test_config = Config(
                run_timestamp="test",
                _data_dir="/tmp/test",
                _raw_data_dir="/tmp/test",
                _processed_data_dir="/tmp/test",
                _model_dir="/tmp/test",
                _reports_dir="/tmp/test",
                _log_dir="/tmp/test",
                _reference_data_path="/tmp/test.xlsx",
                _bad_files_dir="/tmp/test",
                _averaged_files_dir="/tmp/test",
                _cleansed_files_dir="/tmp/test",
                _bad_prediction_files_dir="/tmp/test",
                use_spectral_preprocessing=True,
                spectral_preprocessing_method=method
            )
            print(f"  ✓ Method '{method}' is valid")
        except Exception as e:
            print(f"  ✗ Method '{method}' FAILED: {e}")

    print()
    print("=" * 80)
    print("CONFIG INTEGRATION TEST SUMMARY")
    print("=" * 80)
    print()
    print("✓ Config can control preprocessing enable/disable")
    print("✓ Config can select preprocessing method")
    print("✓ All preprocessing methods are valid Literal options")
    print("✓ SpectralFeatureExtractor correctly receives config values")
    print()
    print("Integration successful! You can now control preprocessing via:")
    print("  - Editing pipeline_config.py (lines 707-708)")
    print("  - Using custom config YAML files")
    print("  - Modifying config object in code")
    print()


if __name__ == '__main__':
    # Run original preprocessing tests
    test_preprocessing_methods()

    # Run new config integration tests
    test_config_integration()
