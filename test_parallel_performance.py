#!/usr/bin/env python3
"""
Test script to compare sequential vs parallel prediction performance
with a real CatBoost model on 60 sample IDs.
"""

import logging
import time
import pandas as pd
from pathlib import Path
import sys
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_performance_comparison():
    """Test sequential vs parallel performance with real CatBoost model."""

    # Configuration
    model_path = "/home/payanico/potassium_pipeline/models/simple_only_catboost_20250909_194354.pkl"
    raw_data_dir = "/home/payanico/potassium_pipeline/data/raw/data_5278_Phase3"
    max_samples = 60

    logger.info("=== PARALLEL DATA PROCESSING PERFORMANCE TEST ===")
    logger.info(f"Model: {Path(model_path).name}")
    logger.info(f"Data directory: {raw_data_dir}")
    logger.info(f"Max samples: {max_samples}")

    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return False

    # Check if data directory exists
    if not Path(raw_data_dir).exists():
        logger.error(f"Data directory not found: {raw_data_dir}")
        return False

    # Test 1: Sequential processing
    logger.info(f"\n🔄 TEST 1: SEQUENTIAL PROCESSING")
    start_time = time.time()

    try:
        from src.models.predictor import Predictor
        from src.config.pipeline_config import Config
        from datetime import datetime

        # Create config for sequential processing
        config_seq = Config(run_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
        config_seq.parallel.use_data_parallel = False
        config_seq.parallel.data_n_jobs = 1

        predictor_seq = Predictor(config_seq)

        # Run sequential prediction
        results_seq = predictor_seq.make_batch_predictions(
            input_dir=Path(raw_data_dir),
            model_path=Path(model_path),
            max_samples=max_samples
        )

        sequential_time = time.time() - start_time
        logger.info(f"✅ Sequential processing completed in {sequential_time:.2f} seconds")
        logger.info(f"   Processed samples: {len(results_seq)}")
        logger.info(f"   Successful predictions: {results_seq['PredictedValue'].notna().sum()}")
        logger.info(f"   Failed predictions: {results_seq['PredictedValue'].isna().sum()}")

        # Show sample results
        if len(results_seq) > 0:
            successful_results = results_seq[results_seq['PredictedValue'].notna()]
            if len(successful_results) > 0:
                logger.info(f"   Sample predictions: {successful_results['PredictedValue'].head(3).tolist()}")
                logger.info(f"   Prediction range: {successful_results['PredictedValue'].min():.4f} - {successful_results['PredictedValue'].max():.4f}")

    except Exception as e:
        logger.error(f"❌ Sequential processing failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

    # Test 2: Parallel processing (2 workers)
    logger.info(f"\n⚡ TEST 2: PARALLEL PROCESSING (2 workers)")
    start_time = time.time()

    try:
        # Create config for parallel processing
        config_par = Config(run_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
        config_par.parallel.use_data_parallel = True
        config_par.parallel.data_n_jobs = 2

        predictor_par = Predictor(config_par)

        # Run parallel prediction
        results_par = predictor_par.make_batch_predictions(
            input_dir=Path(raw_data_dir),
            model_path=Path(model_path),
            max_samples=max_samples
        )

        parallel_time_2 = time.time() - start_time
        logger.info(f"✅ Parallel processing (2 workers) completed in {parallel_time_2:.2f} seconds")
        logger.info(f"   Processed samples: {len(results_par)}")
        logger.info(f"   Successful predictions: {results_par['PredictedValue'].notna().sum()}")
        logger.info(f"   Failed predictions: {results_par['PredictedValue'].isna().sum()}")

        # Show sample results
        if len(results_par) > 0:
            successful_results = results_par[results_par['PredictedValue'].notna()]
            if len(successful_results) > 0:
                logger.info(f"   Sample predictions: {successful_results['PredictedValue'].head(3).tolist()}")
                logger.info(f"   Prediction range: {successful_results['PredictedValue'].min():.4f} - {successful_results['PredictedValue'].max():.4f}")

    except Exception as e:
        logger.error(f"❌ Parallel processing (2 workers) failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

    # Test 3: Parallel processing (4 workers)
    logger.info(f"\n⚡ TEST 3: PARALLEL PROCESSING (4 workers)")
    start_time = time.time()

    try:
        # Create config for parallel processing with more workers
        config_par4 = Config(run_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
        config_par4.parallel.use_data_parallel = True
        config_par4.parallel.data_n_jobs = 4

        predictor_par4 = Predictor(config_par4)

        # Run parallel prediction
        results_par4 = predictor_par4.make_batch_predictions(
            input_dir=Path(raw_data_dir),
            model_path=Path(model_path),
            max_samples=max_samples
        )

        parallel_time_4 = time.time() - start_time
        logger.info(f"✅ Parallel processing (4 workers) completed in {parallel_time_4:.2f} seconds")
        logger.info(f"   Processed samples: {len(results_par4)}")
        logger.info(f"   Successful predictions: {results_par4['PredictedValue'].notna().sum()}")
        logger.info(f"   Failed predictions: {results_par4['PredictedValue'].isna().sum()}")

        # Show sample results
        if len(results_par4) > 0:
            successful_results = results_par4[results_par4['PredictedValue'].notna()]
            if len(successful_results) > 0:
                logger.info(f"   Sample predictions: {successful_results['PredictedValue'].head(3).tolist()}")
                logger.info(f"   Prediction range: {successful_results['PredictedValue'].min():.4f} - {successful_results['PredictedValue'].max():.4f}")

    except Exception as e:
        logger.error(f"❌ Parallel processing (4 workers) failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

    # Performance comparison
    logger.info(f"\n📊 PERFORMANCE COMPARISON:")
    logger.info(f"   Sequential (1 worker):  {sequential_time:.2f}s")
    logger.info(f"   Parallel (2 workers):   {parallel_time_2:.2f}s")
    logger.info(f"   Parallel (4 workers):   {parallel_time_4:.2f}s")

    speedup_2 = sequential_time / parallel_time_2
    speedup_4 = sequential_time / parallel_time_4

    logger.info(f"\n🚀 SPEEDUP ANALYSIS:")
    logger.info(f"   2 workers speedup: {speedup_2:.2f}x")
    logger.info(f"   4 workers speedup: {speedup_4:.2f}x")

    if speedup_2 > 1.1:
        logger.info(f"   ✅ Parallel processing provides significant speedup!")
    else:
        logger.info(f"   ⚠️  Limited speedup - may be I/O bound or small dataset")

    # Results comparison
    logger.info(f"\n🔍 RESULTS CONSISTENCY CHECK:")

    try:
        # Compare successful predictions
        seq_successful = results_seq[results_seq['PredictedValue'].notna()].copy()
        par_successful = results_par[results_par['PredictedValue'].notna()].copy()

        if len(seq_successful) == len(par_successful):
            logger.info(f"   ✅ Same number of successful predictions: {len(seq_successful)}")

            # Sort both by sampleId for comparison
            seq_successful = seq_successful.sort_values('sampleId').reset_index(drop=True)
            par_successful = par_successful.sort_values('sampleId').reset_index(drop=True)

            # Compare sample IDs
            if seq_successful['sampleId'].equals(par_successful['sampleId']):
                logger.info(f"   ✅ Sample IDs match perfectly")

                # Compare predictions
                pred_diff = np.abs(seq_successful['PredictedValue'] - par_successful['PredictedValue'])
                max_diff = pred_diff.max()
                mean_diff = pred_diff.mean()

                logger.info(f"   Prediction differences - Max: {max_diff:.8f}, Mean: {mean_diff:.8f}")

                if max_diff < 1e-10:
                    logger.info(f"   ✅ Predictions are IDENTICAL (deterministic)")
                elif max_diff < 1e-6:
                    logger.info(f"   ✅ Predictions are NEARLY IDENTICAL (numerical precision)")
                else:
                    logger.warning(f"   ⚠️  Predictions differ beyond expected numerical precision")

            else:
                logger.warning(f"   ⚠️  Sample IDs don't match - ordering issue")
        else:
            logger.warning(f"   ⚠️  Different number of successful predictions")
            logger.warning(f"      Sequential: {len(seq_successful)}, Parallel: {len(par_successful)}")

    except Exception as e:
        logger.error(f"   ❌ Results comparison failed: {e}")

    logger.info(f"\n✅ PARALLEL DATA PROCESSING TEST COMPLETED")
    return True

if __name__ == "__main__":
    success = test_performance_comparison()
    sys.exit(0 if success else 1)