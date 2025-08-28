#!/usr/bin/env python3
"""
Quick test to verify parallel data operations work.
"""
import logging
from src.config.pipeline_config import Config
from src.data_management.data_manager import DataManager
from src.data_management.parallel_data_manager import parallel_average_raw_files
from pathlib import Path

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_parallel_data_operations():
    """Test that parallel data operations don't crash."""
    try:
        # Create config with timestamp
        from datetime import datetime
        cfg = Config(run_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
        logger.info(f"Config loaded")
        
        # Check if we have any raw data files
        raw_files = list(cfg.raw_data_dir.glob('*.csv.txt'))
        logger.info(f"Found {len(raw_files)} raw files in {cfg.raw_data_dir}")
        
        if len(raw_files) == 0:
            logger.warning("No raw data files found - skipping test")
            return True
            
        # Test parallel averaging (with small number of workers for testing)
        data_manager = DataManager(cfg)
        logger.info("Testing parallel averaging with 2 workers...")
        parallel_average_raw_files(data_manager, n_jobs=2)
        
        logger.info("✅ Parallel data operations test passed!")
        return True
        
    except ImportError as e:
        logger.error(f"Import error - parallel functionality not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing parallel data operations...")
    success = test_parallel_data_operations()
    if success:
        logger.info("🎉 Test completed successfully!")
    else:
        logger.error("❌ Test failed!")
        exit(1)