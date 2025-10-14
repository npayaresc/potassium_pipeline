#!/usr/bin/env python3
"""
Fix Sample IDs in Excel reference file by removing trailing '_0' suffix.

This script:
1. Checks if Sample IDs have incorrect trailing '_0'
2. Compares Sample IDs with actual raw filenames
3. Creates corrected Excel file if transformation is needed
"""

import pandas as pd
from pathlib import Path
import re
from collections import Counter

def extract_sample_prefix_from_filename(filename: str) -> str:
    """
    Extract sample ID prefix from raw filename.
    Example: MPN_0000_002025_CNGS7330000001_POT_0000_S00531_P_Y_27062025_1000_01.csv.txt
    Returns: MPN_0000_002025_CNGS7330000001_POT_0000_S00531_P_Y_27062025_1000
    """
    # Remove .csv.txt extension
    name = filename.replace('.csv.txt', '')

    # Split by underscore and check if last part is a 2-digit shot number
    parts = name.split('_')
    if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) == 2:
        return '_'.join(parts[:-1])

    return name

def check_and_fix_sample_ids(excel_path: Path, raw_data_dir: Path, output_path: Path = None):
    """
    Check Sample IDs in Excel file and fix if needed.

    Args:
        excel_path: Path to Excel reference file
        raw_data_dir: Path to raw data directory with subdirectories
        output_path: Output path for corrected Excel (default: same name with _corrected suffix)
    """
    print("=" * 100)
    print("SAMPLE ID CHECKER AND CORRECTOR")
    print("=" * 100)

    # Load Excel file
    print(f"\n1. Loading Excel file: {excel_path}")
    df = pd.read_excel(excel_path)
    print(f"   ✓ Loaded {len(df)} rows")

    # Check required columns
    required_cols = ['Sample ID', 'Sample No.']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"   ✗ Missing required columns: {missing_cols}")
        return

    # Analyze Sample ID patterns in Excel
    print("\n2. Analyzing Sample ID patterns in Excel:")
    sample_ids = df['Sample ID'].dropna()
    print(f"   Total Sample IDs: {len(sample_ids)}")

    # Check how many end with '_0'
    trailing_zero_pattern = re.compile(r'_0$')
    ids_with_trailing_zero = [sid for sid in sample_ids if trailing_zero_pattern.search(str(sid))]

    print(f"   Sample IDs ending with '_0': {len(ids_with_trailing_zero)}")
    print(f"   Sample IDs (first 5): {list(sample_ids.head())}")

    # Collect raw filenames
    print(f"\n3. Scanning raw data directory: {raw_data_dir}")
    raw_files = list(raw_data_dir.glob("**/*.csv.txt"))
    print(f"   Found {len(raw_files)} raw files")

    if len(raw_files) == 0:
        print("   ⚠️  No raw files found - cannot validate Sample IDs")
        return

    # Extract unique sample prefixes from raw filenames
    raw_sample_prefixes = set()
    for file_path in raw_files:
        prefix = extract_sample_prefix_from_filename(file_path.name)
        raw_sample_prefixes.add(prefix)

    print(f"   Unique sample prefixes in raw files: {len(raw_sample_prefixes)}")
    print(f"   Sample prefixes (first 5): {sorted(list(raw_sample_prefixes))[:5]}")

    # Check matching between Excel and raw files
    print("\n4. Checking matching between Excel Sample IDs and raw filenames:")

    # Direct matching (current)
    direct_matches = set(sample_ids) & raw_sample_prefixes
    print(f"   Direct matches (current): {len(direct_matches)}")

    # Check if removing '_0' would improve matching
    # Use regex to specifically remove trailing '_0' (not other '0's)
    corrected_sample_ids = []
    for sid in sample_ids:
        sid_str = str(sid)
        # Only remove '_0' at the very end
        if sid_str.endswith('_0'):
            corrected_sample_ids.append(sid_str[:-2])  # Remove last 2 characters: '_0'
        else:
            corrected_sample_ids.append(sid_str)

    corrected_matches = set(corrected_sample_ids) & raw_sample_prefixes
    print(f"   Matches after removing trailing '_0': {len(corrected_matches)}")

    # Determine if correction is needed
    improvement = len(corrected_matches) - len(direct_matches)

    print("\n5. Analysis Results:")
    if improvement > 0:
        print(f"   ✓ Correction NEEDED: Would improve matching by {improvement} samples")
        print(f"   ✓ Match rate improvement: {len(direct_matches)/len(sample_ids)*100:.1f}% → {len(corrected_matches)/len(sample_ids)*100:.1f}%")

        # Apply correction
        print("\n6. Applying correction...")
        df_corrected = df.copy()

        # Remove trailing '_0' from Sample ID column (last 2 characters only if ends with '_0')
        df_corrected['Sample ID'] = df_corrected['Sample ID'].apply(
            lambda x: str(x)[:-2] if pd.notna(x) and str(x).endswith('_0') else x
        )

        # Also check and fix 'Raw files_Sample ID' column if it exists
        if 'Raw files_Sample ID' in df_corrected.columns:
            df_corrected['Raw files_Sample ID'] = df_corrected['Raw files_Sample ID'].apply(
                lambda x: str(x)[:-2] if pd.notna(x) and str(x).endswith('_0') else x
            )
            print("   ✓ Also corrected 'Raw files_Sample ID' column")

        # Determine output path
        if output_path is None:
            output_path = excel_path.parent / f"{excel_path.stem}_corrected{excel_path.suffix}"

        # Save corrected file
        df_corrected.to_excel(output_path, index=False)
        print(f"   ✓ Corrected Excel saved to: {output_path}")

        # Show examples of changes
        print("\n7. Example corrections:")
        changes = []
        for i in range(min(5, len(sample_ids))):
            old_id = sample_ids.iloc[i]
            new_id = df_corrected['Sample ID'].iloc[i]
            if old_id != new_id:
                changes.append((old_id, new_id))

        for old, new in changes:
            print(f"   {old} → {new}")

        if not changes:
            print("   (No changes in first 5 samples)")

        return output_path
    else:
        print(f"   ✓ No correction needed - Sample IDs already match raw filenames correctly")
        print(f"   ✓ Current match rate: {len(direct_matches)/len(sample_ids)*100:.1f}%")
        return None

if __name__ == "__main__":
    # Configuration
    excel_path = Path("data/reference_data/lab_Element_Rough_141025.xlsx")
    raw_data_dir = Path("data/raw/10-14-2025")

    print(f"\nInput Excel: {excel_path}")
    print(f"Raw data directory: {raw_data_dir}")
    print()

    # Check and fix
    corrected_path = check_and_fix_sample_ids(excel_path, raw_data_dir)

    print("\n" + "=" * 100)
    if corrected_path:
        print("CORRECTION COMPLETE")
        print(f"Corrected file saved to: {corrected_path}")
        print("\nNext steps:")
        print("1. Review the corrected file")
        print("2. Update pipeline_config.py to use the corrected file")
        print(f"   Change: _reference_data_path=str(BASE_PATH / 'data' / 'reference_data' / '{corrected_path.name}')")
    else:
        print("NO CORRECTION NEEDED")
    print("=" * 100)
