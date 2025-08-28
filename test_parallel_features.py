#!/usr/bin/env python
"""
Test script for parallel feature engineering.
Compares performance and results between sequential and parallel processing.
"""

import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from src.config.pipeline_config import config
from src.features.feature_engineering import create_feature_pipeline
from src.data_management.data_manager import DataManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_parallel_features():
    """Test parallel vs sequential feature generation."""
    
    # Load some sample data
    data_manager = DataManager(config)
    
    # Load and prepare the dataset
    logger.info("Loading dataset...")
    
    # Check if cleansed data exists
    cleansed_dir = Path("/home/payanico/magnesium_pipeline/data/cleansed_files_per_sample")
    cleansed_files = list(cleansed_dir.glob("*.csv.txt"))
    if not cleansed_files:
        logger.error("No cleansed data available. Please run data preparation first.")
        return
    
    # Load a few files for testing
    test_samples = []
    for file_path in cleansed_files[:20]:  # Load up to 20 files for testing
        wavelengths, intensities = data_manager.load_spectral_file(file_path)
        if wavelengths is not None and intensities is not None:
            sample_id = file_path.stem
            # Average intensities across shots if multiple columns
            if intensities.ndim > 1:
                avg_intensities = np.mean(intensities, axis=1)
            else:
                avg_intensities = intensities
            test_samples.append({
                'SampleID': sample_id,
                'wavelengths': wavelengths,
                'intensities': avg_intensities,
                'Mg_concentration_percent': 0.3  # Dummy value for testing
            })
    
    if not test_samples:
        logger.error("Could not load any spectral files for testing.")
        return
    
    full_dataset = pd.DataFrame(test_samples)
    
    # Take a subset for testing (to see the difference clearly)
    test_size = min(100, len(full_dataset))
    test_data = full_dataset.head(test_size).copy()
    
    logger.info(f"Testing with {test_size} samples")
    
    # Test sequential processing
    logger.info("\n" + "="*60)
    logger.info("Testing SEQUENTIAL feature generation...")
    logger.info("="*60)
    
    sequential_pipeline = create_feature_pipeline(
        config, 
        strategy="simple_only", 
        use_parallel=False
    )
    
    start_time = time.time()
    sequential_features = sequential_pipeline.fit_transform(test_data)
    sequential_time = time.time() - start_time
    
    logger.info(f"Sequential processing completed in {sequential_time:.2f} seconds")
    logger.info(f"Output shape: {sequential_features.shape}")
    
    # Test parallel processing with different worker counts
    for n_jobs in [-1, 4, 2]:
        logger.info("\n" + "="*60)
        if n_jobs == -1:
            logger.info(f"Testing PARALLEL feature generation (all cores)...")
        else:
            logger.info(f"Testing PARALLEL feature generation ({n_jobs} cores)...")
        logger.info("="*60)
        
        parallel_pipeline = create_feature_pipeline(
            config, 
            strategy="simple_only", 
            use_parallel=True,
            n_jobs=n_jobs
        )
        
        start_time = time.time()
        parallel_features = parallel_pipeline.fit_transform(test_data)
        parallel_time = time.time() - start_time
        
        logger.info(f"Parallel processing completed in {parallel_time:.2f} seconds")
        logger.info(f"Output shape: {parallel_features.shape}")
        logger.info(f"Speedup: {sequential_time/parallel_time:.2f}x")
        
        # Verify results are identical (allowing for small floating point differences)
        if isinstance(sequential_features, pd.DataFrame) and isinstance(parallel_features, pd.DataFrame):
            # Sort columns to ensure same order
            sequential_sorted = sequential_features.reindex(sorted(sequential_features.columns), axis=1)
            parallel_sorted = parallel_features.reindex(sorted(parallel_features.columns), axis=1)
            
            # Check if values are close (accounting for floating point precision)
            are_close = np.allclose(
                sequential_sorted.fillna(0).values, 
                parallel_sorted.fillna(0).values, 
                rtol=1e-5, 
                atol=1e-8,
                equal_nan=True
            )
            
            if are_close:
                logger.info("✓ Results match between sequential and parallel processing")
            else:
                logger.warning("⚠ Results differ between sequential and parallel processing")
                
                # Find differences
                diff_mask = ~np.isclose(
                    sequential_sorted.fillna(0).values,
                    parallel_sorted.fillna(0).values,
                    rtol=1e-5,
                    atol=1e-8
                )
                
                if diff_mask.any():
                    diff_locations = np.where(diff_mask)
                    num_diffs = len(diff_locations[0])
                    logger.warning(f"Found {num_diffs} differences")
                    
                    # Show first few differences
                    for i in range(min(5, num_diffs)):
                        row_idx = diff_locations[0][i]
                        col_idx = diff_locations[1][i]
                        col_name = sequential_sorted.columns[col_idx]
                        seq_val = sequential_sorted.iloc[row_idx, col_idx]
                        par_val = parallel_sorted.iloc[row_idx, col_idx]
                        logger.warning(f"  Row {row_idx}, Col '{col_name}': {seq_val} vs {par_val}")
    
    logger.info("\n" + "="*60)
    logger.info("Parallel feature testing completed!")
    logger.info("="*60)

if __name__ == "__main__":
    test_parallel_features()