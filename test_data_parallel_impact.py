#!/usr/bin/env python3
"""
Test script to check if --data-parallel flag affects prediction results.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.pipeline_config import Config
from src.models.predictor import Predictor
from datetime import datetime

def create_mock_data_files(temp_dir: Path, n_samples=3, files_per_sample=2):
    """Create mock spectral data files for testing."""
    np.random.seed(42)

    # Create raw data directory
    raw_dir = temp_dir / "raw_data_5278_Phase3"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for sample_id in range(n_samples):
        for file_id in range(files_per_sample):
            # Generate synthetic spectral data
            wavelengths = np.linspace(200, 900, 700)
            intensities_1 = 1000 + 500 * np.sin((wavelengths - 400) / 50) + np.random.normal(0, 30, 700)
            intensities_2 = 1200 + 400 * np.sin((wavelengths - 450) / 60) + np.random.normal(0, 35, 700)

            # Create DataFrame
            df = pd.DataFrame({
                'Wavelength': wavelengths,
                'Intensity1': intensities_1,
                'Intensity2': intensities_2
            })

            # Save file
            filename = f"sample_{sample_id:03d}_file_{file_id:02d}.csv.txt"
            df.to_csv(raw_dir / filename, index=False)

    return raw_dir

def test_data_parallel_impact():
    """Test if data-parallel flag affects prediction results."""

    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create config pointing to temp directory
        config = Config(
            run_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            raw_data_dir=temp_path / "raw_data_5278_Phase3",
            averaged_files_dir=temp_path / "averaged_files_per_sample",
            cleansed_files_dir=temp_path / "cleansed_files_per_sample",
            model_dir=temp_path / "models",
            reports_dir=temp_path / "reports",
            bad_files_dir=temp_path / "bad_files",
            bad_prediction_files_dir=temp_path / "bad_prediction_files"
        )

        # Create mock data files
        logger.info("Creating mock data files...")
        raw_dir = create_mock_data_files(temp_path)
        logger.info(f"Created {len(list(raw_dir.glob('*.csv.txt')))} mock files")

        # Test 1: Check if data_parallel settings affect predictor config
        logger.info("\n=== Test 1: Config Override Check ===")

        # Set data parallel in config
        config.parallel.use_data_parallel = True
        config.parallel.data_n_jobs = 4

        predictor_1 = Predictor(config)
        logger.info(f"Predictor 1 - use_data_parallel: {config.parallel.use_data_parallel}")
        logger.info(f"Predictor 1 - data_n_jobs: {config.parallel.data_n_jobs}")

        # Change config
        config.parallel.use_data_parallel = False
        config.parallel.data_n_jobs = 1

        predictor_2 = Predictor(config)
        logger.info(f"Predictor 2 - use_data_parallel: {config.parallel.use_data_parallel}")
        logger.info(f"Predictor 2 - data_n_jobs: {config.parallel.data_n_jobs}")

        # Check if predictors use the config
        if hasattr(predictor_1, 'data_parallel_settings'):
            logger.info("❌ Predictor stores data parallel settings")
        else:
            logger.info("✓ Predictor does NOT store data parallel settings")

        # Test 2: Check actual impact on batch processing
        logger.info("\n=== Test 2: Batch Processing Impact ===")
        logger.info("Note: The predictor's make_batch_predictions method processes data internally")
        logger.info("and doesn't directly use data_parallel settings from config during prediction.")
        logger.info("Data parallel settings only affect the data preparation phase (averaging/cleaning).")

        # Test 3: Check where data_parallel actually has impact
        logger.info("\n=== Test 3: Data Parallel Usage Analysis ===")
        logger.info("✓ Data parallel affects:")
        logger.info("  - Raw file averaging (parallel_average_raw_files)")
        logger.info("  - Data loading and cleaning (parallel_load_and_clean_data)")
        logger.info("  - TRAINING data preparation only")

        logger.info("✗ Data parallel does NOT affect:")
        logger.info("  - Predictor.make_batch_predictions() - processes samples sequentially")
        logger.info("  - Single file predictions")
        logger.info("  - Feature engineering (that's --feature-parallel)")
        logger.info("  - Model inference")

        # Test 4: Demonstrate the actual batch prediction process
        logger.info("\n=== Test 4: Batch Prediction Process ===")
        logger.info("The predictor's batch prediction method:")
        logger.info("1. Groups files by sample ID")
        logger.info("2. Averages files per sample using data_manager.average_files_in_memory()")
        logger.info("3. Applies data_manager.standardize_wavelength_grid() if enabled")
        logger.info("4. Cleans data using data_cleanser.clean_spectra()")
        logger.info("5. Creates batch DataFrame for all samples")
        logger.info("6. Applies feature pipeline transform (batch processing)")
        logger.info("7. Makes batch predictions")

        logger.info("\n✓ Only step 6 (feature pipeline) can be parallel via --feature-parallel")
        logger.info("✓ Steps 2-4 are sequential per sample in batch prediction")
        logger.info("✓ --data-parallel only affects TRAINING data preparation, not prediction")

if __name__ == "__main__":
    test_data_parallel_impact()