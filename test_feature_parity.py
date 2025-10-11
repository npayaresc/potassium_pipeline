#!/usr/bin/env python3
"""
Test script to verify that sequential and parallel feature generation
produce the same number of features after the physics-informed features fix.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.pipeline_config import Config
from src.features.feature_engineering import SpectralFeatureGenerator
from src.features.parallel_feature_engineering import ParallelSpectralFeatureGenerator

def test_feature_count_parity():
    """Test that sequential and parallel generate the same number of features."""

    # Load config
    from datetime import datetime
    config = Config(run_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Create dummy spectral data (1 sample)
    dummy_wavelengths = np.linspace(200, 900, 2000)
    dummy_intensities = np.random.rand(2000) * 1000

    X_test = pd.DataFrame({
        'wavelengths': [dummy_wavelengths],
        'intensities': [dummy_intensities]
    })

    print("=" * 80)
    print("FEATURE COUNT PARITY TEST")
    print("=" * 80)

    # Test K_only strategy
    print("\nTesting K_only strategy:")
    print("-" * 40)

    # Sequential
    seq_gen = SpectralFeatureGenerator(config, strategy="K_only")
    seq_gen.fit(X_test)
    seq_features = seq_gen.get_feature_names_out()

    # Parallel
    par_gen = ParallelSpectralFeatureGenerator(config, strategy="K_only", n_jobs=1)
    par_gen.fit(X_test)
    par_features = par_gen.get_feature_names_out()

    print(f"Sequential features: {len(seq_features)}")
    print(f"Parallel features:   {len(par_features)}")
    print(f"Match: {'✓ YES' if len(seq_features) == len(par_features) else '✗ NO'}")

    if len(seq_features) != len(par_features):
        print("\n⚠️  MISMATCH DETECTED!")
        print(f"Difference: {abs(len(seq_features) - len(par_features))} features")

        # Find missing features
        seq_set = set(seq_features)
        par_set = set(par_features)

        only_in_seq = seq_set - par_set
        only_in_par = par_set - seq_set

        if only_in_seq:
            print(f"\nOnly in SEQUENTIAL ({len(only_in_seq)} features):")
            for feat in sorted(only_in_seq)[:10]:
                print(f"  - {feat}")
            if len(only_in_seq) > 10:
                print(f"  ... and {len(only_in_seq) - 10} more")

        if only_in_par:
            print(f"\nOnly in PARALLEL ({len(only_in_par)} features):")
            for feat in sorted(only_in_par)[:10]:
                print(f"  - {feat}")
            if len(only_in_par) > 10:
                print(f"  ... and {len(only_in_par) - 10} more")

        return False
    else:
        print("\n✓ Feature counts match perfectly!")

        # Check if feature names also match
        if set(seq_features) == set(par_features):
            print("✓ Feature names also match!")
        else:
            print("⚠️  Feature names differ (but count is the same)")

        # Show breakdown
        k_i_physics = [f for f in par_features if any(x in f for x in ['fwhm', 'gamma', 'fit_quality', 'asymmetry', 'amplitude', 'kurtosis', 'absorption_index'])]
        print(f"\nPhysics-informed features: {len(k_i_physics)}")
        print(f"  Examples: {k_i_physics[:5]}")

        return True

if __name__ == "__main__":
    success = test_feature_count_parity()
    sys.exit(0 if success else 1)
