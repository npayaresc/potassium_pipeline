#!/usr/bin/env python3
"""
Test script to verify config saving functionality.
"""
import sys
from pathlib import Path
from datetime import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.pipeline_config import Config
from reporting.reporter import Reporter

def test_config_saving():
    """Test the config saving functionality."""
    print("Testing config saving functionality...")
    
    # Create a test config with minimal valid paths
    test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config = Config(
        run_timestamp=test_timestamp,
        data_dir=Path.cwd() / "data",
        raw_data_dir=Path.cwd() / "data" / "raw" / "data_5278_Phase3",
        processed_data_dir=Path.cwd() / "data" / "processed",
        model_dir=Path.cwd() / "models",
        reports_dir=Path.cwd() / "reports",
        log_dir=Path.cwd() / "logs",
        bad_files_dir=Path.cwd() / "bad_files",
        averaged_files_dir=Path.cwd() / "data" / "averaged_files_per_sample",
        cleansed_files_dir=Path.cwd() / "data" / "cleansed_files_per_sample",
        bad_prediction_files_dir=Path.cwd() / "bad_prediction_files",
        reference_data_path=Path.cwd() / "data" / "Reference_data_5278_Phase3.xlsx"
    )
    
    # Create a reporter and save config
    reporter = Reporter(config)
    
    try:
        saved_path = reporter.save_config()
        print(f"✓ Config successfully saved to: {saved_path}")
        
        # Verify the file exists and contains expected content
        if saved_path.exists():
            print(f"✓ Config file exists: {saved_path}")
            with open(saved_path, 'r') as f:
                import json
                config_data = json.load(f)
                print(f"✓ Config file contains {len(config_data)} configuration keys")
                print(f"✓ Timestamp in config: {config_data.get('run_timestamp')}")
        else:
            print("✗ Config file does not exist!")
            return False
            
    except Exception as e:
        print(f"✗ Error saving config: {e}")
        return False
    
    print("Config saving test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_config_saving()
    sys.exit(0 if success else 1)