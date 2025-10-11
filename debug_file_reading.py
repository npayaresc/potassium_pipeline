#!/usr/bin/env python3
"""Debug script to test file reading."""

import pandas as pd
from pathlib import Path

def test_file_reading():
    """Test reading one file."""
    file_path = Path('/home/payanico/potassium_pipeline/data/raw/OneDrive_1_9-22-2025/16-09-2025/MPN_0000_002025_CNGS7330000001_POT_0000_S00531_P_Y_27062025_1000_01.csv.txt')

    print(f"Testing file: {file_path.name}")

    # Find where the data starts (after metadata)
    data_start_line = 0
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            print(f"Line {i}: {line.strip()[:100]}")
            if 'Wavelength' in line and 'Intensity' in line:
                data_start_line = i
                print(f"Found data start at line {i}")
                break

    # Read the actual data
    try:
        df = pd.read_csv(file_path, skiprows=data_start_line)
        print(f"Successfully read {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head())

        # Clean column names (remove spaces)
        df.columns = [col.strip() for col in df.columns]
        print(f"Cleaned columns: {list(df.columns)}")

        # Test intensity column selection
        intensity_cols_10 = [f'Intensity{i}' for i in range(1, 11)]
        intensity_cols_20 = [f'Intensity{i}' for i in range(1, 21)]

        available_10 = [col for col in intensity_cols_10 if col in df.columns]
        available_20 = [col for col in intensity_cols_20 if col in df.columns]

        print(f"Available 10-shot columns: {len(available_10)}")
        print(f"Available 20-shot columns: {len(available_20)}")

        if len(available_20) == 20:
            print("SUCCESS: Can extract 20 shots")
            intensity_data = df[available_20].values
            print(f"Intensity data shape: {intensity_data.shape}")
        else:
            print("ERROR: Cannot extract 20 shots")

    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_file_reading()