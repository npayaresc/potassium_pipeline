#!/usr/bin/env python3
"""
Check preprocessing performance in the actual pipeline.
This script measures:
1. Whether preprocessing is actually being used
2. How much time preprocessing takes
3. Whether parallelization would help
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
import logging

from src.config.pipeline_config import Config
from src.data_management.data_manager import DataManager
from src.spectral_extraction.extractor import SpectralFeatureExtractor
from src.spectral_extraction.preprocessing import SpectralPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def measure_preprocessing_on_real_data():
    """Measure preprocessing performance on actual pipeline data."""

    print("=" * 80)
    print("PREPROCESSING PERFORMANCE CHECK")
    print("=" * 80)
    print()

    # 1. Check configuration
    print("1. Configuration Status")
    print("-" * 80)
    config = Config()
    print(f"   use_spectral_preprocessing: {config.use_spectral_preprocessing}")
    print(f"   spectral_preprocessing_method: '{config.spectral_preprocessing_method}'")
    print()

    if not config.use_spectral_preprocessing:
        print("   ⚠️  WARNING: Preprocessing is DISABLED in config!")
        print("   To enable, set use_spectral_preprocessing=True in pipeline_config.py")
        print()
        return

    # 2. Load some real data
    print("2. Loading Real Data")
    print("-" * 80)
    data_manager = DataManager(config)

    # Try to load averaged data
    averaged_dir = Path(config._averaged_files_dir)
    if not averaged_dir.exists():
        print(f"   ❌ Averaged files directory not found: {averaged_dir}")
        print("   Run the pipeline first to generate data.")
        return

    # Get first few files
    csv_files = list(averaged_dir.glob("*.csv"))[:10]
    if not csv_files:
        print(f"   ❌ No CSV files found in {averaged_dir}")
        return

    print(f"   ✓ Found {len(csv_files)} files to test")
    print()

    # 3. Load a sample file to check data shape
    print("3. Analyzing Data Shape")
    print("-" * 80)

    sample_file = csv_files[0]
    df = pd.read_csv(sample_file)
    wavelengths = df['wavelength'].values

    # Count spectrum columns (usually multiple shots per sample)
    spectrum_cols = [c for c in df.columns if c.startswith('spectrum_')]
    n_spectra = len(spectrum_cols)
    n_wavelengths = len(wavelengths)

    print(f"   Sample file: {sample_file.name}")
    print(f"   Wavelengths: {n_wavelengths}")
    print(f"   Spectra per sample: {n_spectra} (shots/replicates)")
    print()

    if n_spectra == 1:
        print("   ℹ️  Only 1 spectrum per sample - parallel preprocessing not needed")
        print("   (Already parallelized at sample level)")
    else:
        print(f"   ✓ Multiple spectra ({n_spectra}) - could benefit from parallel preprocessing")
    print()

    # 4. Measure preprocessing time per region
    print("4. Preprocessing Performance Test")
    print("-" * 80)

    # Simulate processing one region with multiple spectra
    spectra_array = df[[f'spectrum_{i}' for i in range(n_spectra)]].values.T  # Shape: (n_wavelengths, n_spectra)

    preprocessor = SpectralPreprocessor()
    preprocessor.configure(method=config.spectral_preprocessing_method)

    # Warmup
    _ = preprocessor.preprocess_batch(spectra_array, wavelengths)

    # Time it
    n_runs = 10
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = preprocessor.preprocess_batch(spectra_array, wavelengths)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"   Method: '{config.spectral_preprocessing_method}'")
    print(f"   Data shape: {spectra_array.shape} (n_wavelengths, n_spectra)")
    print(f"   Average time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"   Time per spectrum: {avg_time*1000/n_spectra:.2f} ms")
    print()

    # 5. Estimate total preprocessing time in pipeline
    print("5. Pipeline Impact Estimate")
    print("-" * 80)

    # Typical pipeline processes many samples with multiple regions
    typical_samples = len(csv_files)  # Or use actual count from data
    typical_regions = len(config.spectral_regions)  # Number of spectral regions

    total_preprocessing_time = avg_time * typical_samples * typical_regions

    print(f"   Estimated samples: ~{typical_samples}")
    print(f"   Spectral regions: {typical_regions}")
    print(f"   Estimated total preprocessing time: {total_preprocessing_time:.2f} seconds")
    print(f"   = {total_preprocessing_time/60:.2f} minutes")
    print()

    # 6. Parallel potential
    print("6. Parallelization Analysis")
    print("-" * 80)

    if n_spectra <= 2:
        print("   ✅ CURRENT APPROACH IS OPTIMAL")
        print("   - Few spectra per sample (≤2)")
        print("   - Already parallelized at sample level")
        print("   - No need for batch-level parallelization")
    else:
        print("   ⚡ POTENTIAL FOR OPTIMIZATION")
        print(f"   - {n_spectra} spectra per sample processed sequentially")
        print(f"   - Could achieve {n_spectra}x speedup with parallel preprocess_batch()")
        print(f"   - Estimated time savings: {total_preprocessing_time * (1 - 1/n_spectra):.2f} seconds")
        print()
        print("   Recommendation:")
        print("   - Add joblib parallelization to preprocess_batch() method")
        print("   - Use threading (not multiprocessing) to avoid pickling overhead")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Preprocessing is ENABLED: {config.spectral_preprocessing_method}")
    print(f"✓ Processing time per region: {avg_time*1000:.2f} ms")
    print(f"✓ Current parallelization: Per-sample (via --feature-parallel)")

    if n_spectra > 2:
        print(f"⚡ Optimization opportunity: Parallelize {n_spectra} spectra within each sample")
    else:
        print("✅ Current parallelization strategy is optimal")
    print()


if __name__ == "__main__":
    measure_preprocessing_on_real_data()
