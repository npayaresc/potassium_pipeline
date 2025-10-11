#!/usr/bin/env python3
"""
Test script to verify if parallel feature engineering produces deterministic results.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.pipeline_config import Config
from src.features.parallel_feature_engineering import ParallelSpectralFeatureGenerator
from src.features.feature_engineering import SpectralFeatureGenerator

def create_test_data(n_samples=5):
    """Create reproducible test data that covers all spectral regions."""
    np.random.seed(42)
    data = []

    # Create wavelength range that includes all regions needed
    wavelengths = np.linspace(200, 900, 700)  # Extended range

    for i in range(n_samples):
        # Create synthetic intensities with peaks in different regions
        intensities = np.ones_like(wavelengths) * 1000  # baseline

        # Add peaks for K regions (766-770nm and around 404nm)
        k_peak_1 = 500 * np.exp(-((wavelengths - 768) / 2)**2)  # K I peak
        k_peak_2 = 300 * np.exp(-((wavelengths - 404) / 2)**2)  # K II peak

        # Add peaks for other elements
        c_peak = 400 * np.exp(-((wavelengths - 658) / 3)**2)    # C I peak
        h_peak = 200 * np.exp(-((wavelengths - 656) / 2)**2)    # H alpha
        o_peak = 300 * np.exp(-((wavelengths - 777) / 2)**2)    # O I peak
        n_peak = 250 * np.exp(-((wavelengths - 746) / 2)**2)    # N I peak
        p_peak = 350 * np.exp(-((wavelengths - 255) / 2)**2)    # P I peak
        mg_peak = 400 * np.exp(-((wavelengths - 285) / 2)**2)   # Mg I peak

        # Combine all peaks
        intensities += k_peak_1 + k_peak_2 + c_peak + h_peak + o_peak + n_peak + p_peak + mg_peak

        # Add some noise
        intensities += np.random.normal(0, 30, len(wavelengths))

        # Ensure positive values
        intensities = np.maximum(intensities, 50)

        data.append({
            'wavelengths': wavelengths,
            'intensities': intensities
        })

    return pd.DataFrame(data)

def test_parallel_determinism():
    """Test if parallel feature engineering produces consistent results."""

    # Create config with required fields
    from datetime import datetime
    config = Config(run_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Create test data
    logger.info("Creating test data...")
    test_data = create_test_data(10)

    # Test 1: Run parallel generator multiple times
    logger.info("\n=== Testing Parallel Feature Generator Consistency ===")
    parallel_gen = ParallelSpectralFeatureGenerator(config, strategy="simple_only", n_jobs=2)

    # Fit once
    parallel_gen.fit(test_data)

    # Transform multiple times
    results_parallel = []
    for run in range(3):
        result = parallel_gen.transform(test_data)
        results_parallel.append(result)
        logger.info(f"Run {run+1}: Shape={result.shape}, Sum={result.values.sum():.6f}")

    # Check if all runs produce identical results
    all_equal_parallel = True
    for i in range(1, len(results_parallel)):
        if not np.allclose(results_parallel[0].values, results_parallel[i].values, rtol=1e-10, atol=1e-10):
            all_equal_parallel = False
            # Find differences
            diff = np.abs(results_parallel[0].values - results_parallel[i].values)
            max_diff = np.max(diff)
            logger.error(f"Parallel run {i+1} differs from run 1! Max difference: {max_diff}")
            # Find where differences occur
            diff_indices = np.where(diff > 1e-10)
            if len(diff_indices[0]) > 0:
                sample_idx = diff_indices[0][0]
                feature_idx = diff_indices[1][0]
                logger.error(f"First difference at sample {sample_idx}, feature {feature_idx}: "
                           f"{results_parallel[0].iloc[sample_idx, feature_idx]} vs "
                           f"{results_parallel[i].iloc[sample_idx, feature_idx]}")

    if all_equal_parallel:
        logger.info("✓ Parallel feature generation is deterministic across multiple runs")
    else:
        logger.error("✗ Parallel feature generation is NOT deterministic!")

    # Test 2: Compare parallel vs sequential
    logger.info("\n=== Testing Parallel vs Sequential Consistency ===")
    sequential_gen = SpectralFeatureGenerator(config, strategy="simple_only")
    sequential_gen.fit(test_data)
    result_sequential = sequential_gen.transform(test_data)

    logger.info(f"Sequential: Shape={result_sequential.shape}, Sum={result_sequential.values.sum():.6f}")
    logger.info(f"Parallel:   Shape={results_parallel[0].shape}, Sum={results_parallel[0].values.sum():.6f}")

    # Compare feature names
    seq_features = set(result_sequential.columns)
    par_features = set(results_parallel[0].columns)

    if seq_features != par_features:
        logger.error("Feature names differ!")
        logger.error(f"Only in sequential: {seq_features - par_features}")
        logger.error(f"Only in parallel: {par_features - seq_features}")
    else:
        logger.info("✓ Feature names match")

    # Compare values (allowing for small numerical differences)
    if result_sequential.shape == results_parallel[0].shape:
        # Reorder columns to match
        result_parallel_reordered = results_parallel[0][result_sequential.columns]
        if np.allclose(result_sequential.values, result_parallel_reordered.values, rtol=1e-5, atol=1e-8):
            logger.info("✓ Parallel and sequential produce similar results (within tolerance)")
        else:
            diff = np.abs(result_sequential.values - result_parallel_reordered.values)
            max_diff = np.max(diff)
            logger.warning(f"⚠ Results differ between parallel and sequential. Max difference: {max_diff}")
            if max_diff > 0.01:  # Significant difference
                logger.error("✗ Significant difference detected!")
    else:
        logger.error(f"✗ Shape mismatch: Sequential {result_sequential.shape} vs Parallel {results_parallel[0].shape}")

    # Test 3: Test with different n_jobs
    logger.info("\n=== Testing Different n_jobs Settings ===")
    for n_jobs in [1, 2, 4, -1]:
        parallel_gen_njobs = ParallelSpectralFeatureGenerator(config, strategy="simple_only", n_jobs=n_jobs)
        parallel_gen_njobs.fit(test_data)
        result_njobs = parallel_gen_njobs.transform(test_data)
        logger.info(f"n_jobs={n_jobs}: Shape={result_njobs.shape}, Sum={result_njobs.values.sum():.6f}")

        # Compare with first parallel result
        if not np.allclose(results_parallel[0].values, result_njobs[results_parallel[0].columns].values,
                          rtol=1e-5, atol=1e-8):
            logger.warning(f"⚠ Results with n_jobs={n_jobs} differ from n_jobs=2")

    return all_equal_parallel

if __name__ == "__main__":
    success = test_parallel_determinism()
    sys.exit(0 if success else 1)