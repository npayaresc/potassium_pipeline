#!/usr/bin/env python3
"""
Second test file for verifying outline works across multiple files
"""

import os
from typing import Dict, List, Optional

class DataProcessor:
    """Main data processing class"""
    
    def __init__(self, config: Dict[str, str]):
        """Initialize the data processor"""
        self.config = config
        self.data_cache: List[Dict] = []
        
    def load_data(self, file_path: str) -> Optional[Dict]:
        """Load data from file"""
        if not os.path.exists(file_path):
            return None
        return {"data": "loaded"}
    
    def process_data(self, raw_data: Dict) -> Dict:
        """Process the raw data"""
        processed = {
            "original": raw_data,
            "processed_at": "2025-08-16"
        }
        return processed
    
    @classmethod
    def from_config_file(cls, config_path: str):
        """Create instance from config file"""
        config = {"path": config_path}
        return cls(config)
    
    @staticmethod
    def validate_data(data: Dict) -> bool:
        """Validate data structure"""
        return isinstance(data, dict) and len(data) > 0

def utility_function(param1: str, param2: int = 10) -> str:
    """A utility function with default parameters"""
    return f"{param1} - {param2}"

def another_utility(items: List[str]) -> List[str]:
    """Process a list of items"""
    return [item.upper() for item in items]

# Constants
MAX_ITEMS = 100
DEFAULT_CONFIG = {"timeout": 30, "retries": 3}

# Global variable
global_processor: Optional[DataProcessor] = None

if __name__ == "__main__":
    processor = DataProcessor(DEFAULT_CONFIG)
    print(utility_function("test", 42))
