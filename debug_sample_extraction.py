#!/usr/bin/env python3
"""Debug sample ID extraction."""

import re
from pathlib import Path
from collections import defaultdict

def extract_sample_id(filename: str) -> str:
    """Extract sample ID from filename (e.g., S00531 from the filename)."""
    # Pattern: Look for S followed by exactly 5 digits, but ensure it's the right one
    # In the filename: MPN_0000_002025_CNGS7330000001_POT_0000_S00531_P_Y_27062025_1000_01.csv.txt
    # We want S00531, not S73300 (which comes from CNGS7330000001)

    # Look for S followed by 5 digits that comes after "_POT_0000_"
    match = re.search(r'_POT_0000_(S\d{5})_', filename)
    if match:
        return match.group(1)

    # Alternative: look for S followed by 5 digits near the end of the filename
    matches = re.findall(r'S\d{5}', filename)
    if matches:
        # Take the last match (most likely the sample ID)
        return matches[-1]

    # Fallback: use everything before the last underscore minus the shot number
    name_without_ext = filename.replace('.csv.txt', '')
    parts = name_without_ext.split('_')
    # Remove the last part (shot number like 01, 02)
    return '_'.join(parts[:-1])

def test_sample_extraction():
    """Test sample ID extraction."""
    raw_data_dir = Path('/home/payanico/potassium_pipeline/data/raw/OneDrive_1_9-22-2025')

    sample_groups = defaultdict(list)
    all_files = list(raw_data_dir.rglob("*.csv.txt"))

    print(f"Found {len(all_files)} files")

    for file_path in all_files[:10]:  # Test first 10 files
        sample_id = extract_sample_id(file_path.name)
        sample_groups[sample_id].append(file_path)
        print(f"File: {file_path.name}")
        print(f"  -> Sample ID: {sample_id}")
        print()

    print(f"Total unique samples: {len(sample_groups)}")
    for sample_id, files in list(sample_groups.items())[:5]:
        print(f"Sample {sample_id}: {len(files)} files")

if __name__ == "__main__":
    test_sample_extraction()