#!/usr/bin/env python3
"""Detailed analysis of NEW data for potassium pipeline with sample ID matching focus."""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Import pipeline configuration
from src.config.pipeline_config import config

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("=" * 100)
print("POTASSIUM PIPELINE - DETAILED NEW DATA ANALYSIS")
print(f"Configuration: {config.raw_data_dir} + {config.reference_data_path.name}")
print("=" * 100)

# ============================================================================
# PART 1: LOAD AND EXAMINE REFERENCE FILE IN DETAIL
# ============================================================================
print("\n" + "=" * 100)
print("PART 1: DETAILED REFERENCE FILE ANALYSIS")
print("=" * 100)

# Use reference data path from config
ref_file = config.reference_data_path
print(f"\nLoading: {ref_file}")

df_ref = pd.read_excel(ref_file)
print(f"✓ Shape: {df_ref.shape[0]} rows × {df_ref.shape[1]} columns")
print(f"\nAll columns:")
for i, col in enumerate(df_ref.columns, 1):
    print(f"  {i:2d}. {col}")

# Show first few rows of all ID columns
print("\n" + "-" * 100)
print("Sample of ID Columns (first 20 rows):")
print("-" * 100)
id_cols = ['Sample No.', 'Sample ID', 'Raw files_Sample ID', 'match']
df_sample = df_ref[id_cols].head(20)
print(df_sample.to_string())

# Analyze each ID column
print("\n" + "-" * 100)
print("ID Column Analysis:")
print("-" * 100)

for col in id_cols:
    if col in df_ref.columns:
        print(f"\n{col}:")
        print(f"  Total values: {df_ref[col].notna().sum()}/{len(df_ref)}")
        print(f"  Unique values: {df_ref[col].nunique()}")
        print(f"  Data type: {df_ref[col].dtype}")

        # Show sample values
        valid_values = df_ref[col].dropna().unique()[:10]
        print(f"  Sample values (first 10):")
        for val in valid_values:
            print(f"    • {val}")

# Analyze potassium column
print("\n" + "-" * 100)
print("Potassium Data Analysis:")
print("-" * 100)

# Try to find the potassium column - handle both formats (newline vs space)
k_col = config.target_column
if k_col not in df_ref.columns:
    # Try with newline instead of space
    k_col_newline = k_col.replace(' ', '\n', 1)
    if k_col_newline in df_ref.columns:
        k_col = k_col_newline
    else:
        # Try to find any column containing 'K 766' or 'K\n766'
        k_cols = [c for c in df_ref.columns if 'K' in c and '766' in c]
        if k_cols:
            k_col = k_cols[0]
            print(f"⚠️ Using detected K column: '{k_col}'")
        else:
            print(f"❌ Could not find potassium column. Available columns: {list(df_ref.columns)}")
            k_col = None

if k_col and k_col in df_ref.columns:
    k_data = df_ref[k_col].dropna()
    print(f"\n{k_col}:")
    print(f"  Valid samples: {len(k_data)}/{len(df_ref)} ({len(k_data)/len(df_ref)*100:.1f}%)")
    print(f"  Missing samples: {df_ref[k_col].isna().sum()}")
    print(f"  Range: [{k_data.min():.4f}, {k_data.max():.4f}]")
    print(f"  Mean ± Std: {k_data.mean():.4f} ± {k_data.std():.4f}")
    print(f"  Median: {k_data.median():.4f}")
    print(f"  Q1 / Q3: {k_data.quantile(0.25):.4f} / {k_data.quantile(0.75):.4f}")

    # Show rows with K values and their IDs
    print("\n  Sample rows with K values (first 10):")
    df_with_k = df_ref[df_ref[k_col].notna()][id_cols + [k_col]].head(10)
    print(df_with_k.to_string(index=False))

# ============================================================================
# PART 2: ANALYZE RAW DATA FILES
# ============================================================================
print("\n" + "=" * 100)
print("PART 2: RAW DATA FILES ANALYSIS")
print("=" * 100)

# Use raw data directory from config
raw_dir = config.raw_data_dir
print(f"\nDirectory: {raw_dir}")

# Check if directory exists
if not raw_dir.exists():
    print(f"⚠️ Directory does not exist: {raw_dir}")
    print(f"⚠️ Skipping raw data file analysis")
    all_files = []
    subdirs = []
else:
    # Collect all files
    all_files = []
    subdirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    print(f"✓ Found {len(subdirs)} subdirectories")

    for subdir in subdirs:
        files = list(subdir.glob("*.csv.txt"))
        all_files.extend(files)
        print(f"  • {subdir.name}: {len(files)} files")

    print(f"\nTotal files: {len(all_files)}")

# Analyze filename patterns (only if we have files)
if all_files:
    print("\n" + "-" * 100)
    print("Filename Pattern Analysis:")
    print("-" * 100)

    print("\nSample filenames (first 10):")
    for f in all_files[:10]:
        print(f"  {f.name}")

    # Try to parse filenames to extract components
    print("\n" + "-" * 100)
    print("Extracting Sample Identifiers from Filenames:")
    print("-" * 100)

    # Pattern: MPN_0000_002025_CNGS7330000001_POT_0000_S00531_P_Y_27062025_1000_0_01.csv.txt
    # Let's extract different parts
    sample_ids_extracted = []
    sample_numbers = []
    cngs_numbers = []

    pattern = r'MPN_\d+_\d+_(CNGS\d+)_POT_\d+_(S\d+)_'

    for f in all_files:
        filename = f.name

        # Try to extract CNGS number and S number
        match = re.search(pattern, filename)
        if match:
            cngs = match.group(1)
            sample_s = match.group(2)
            cngs_numbers.append(cngs)
            sample_numbers.append(sample_s)
            sample_ids_extracted.append(f"{cngs}_{sample_s}")

    print(f"Successfully parsed: {len(sample_ids_extracted)}/{len(all_files)} files")

    if sample_numbers:
        print(f"\nUnique CNGS numbers: {len(set(cngs_numbers))}")
        print(f"Sample CNGS numbers (first 10): {sorted(set(cngs_numbers))[:10]}")

        print(f"\nUnique S numbers: {len(set(sample_numbers))}")
        print(f"Sample S numbers (first 10): {sorted(set(sample_numbers))[:10]}")

        print(f"\nUnique combined IDs: {len(set(sample_ids_extracted))}")
        print(f"Sample combined IDs (first 10): {sorted(set(sample_ids_extracted))[:10]}")
else:
    # No files found, set empty defaults
    sample_ids_extracted = []
    sample_numbers = []
    cngs_numbers = []

# ============================================================================
# PART 3: ATTEMPT TO MATCH SAMPLES
# ============================================================================
print("\n" + "=" * 100)
print("PART 3: SAMPLE MATCHING ANALYSIS")
print("=" * 100)

# Check if any of the ID columns in reference match the extracted IDs
print("\nAttempting to match reference IDs with raw data filenames...")

# Create a mapping of all possible IDs from reference
ref_ids = {
    'Sample No.': set(df_ref['Sample No.'].dropna().astype(str)),
    'Sample ID': set(df_ref['Sample ID'].dropna().astype(str)),
    'Raw files_Sample ID': set(df_ref['Raw files_Sample ID'].dropna().astype(str)) if 'Raw files_Sample ID' in df_ref.columns else set()
}

# Create sets of extracted IDs from filenames
raw_ids = {
    'Full filename': set(f.name for f in all_files),
    'Filename stem': set(f.stem for f in all_files),
    'CNGS numbers': set(cngs_numbers),
    'S numbers': set(sample_numbers),
    'Combined (CNGS_S)': set(sample_ids_extracted)
}

print(f"\nReference ID sets:")
for name, id_set in ref_ids.items():
    print(f"  • {name}: {len(id_set)} unique values")

print(f"\nRaw data ID sets:")
for name, id_set in raw_ids.items():
    print(f"  • {name}: {len(id_set)} unique values")

# Try all combinations of matching
print("\n" + "-" * 100)
print("Matching Attempts:")
print("-" * 100)

best_match_count = 0
best_match_info = None

for ref_name, ref_set in ref_ids.items():
    for raw_name, raw_set in raw_ids.items():
        matches = ref_set & raw_set
        if len(matches) > best_match_count:
            best_match_count = len(matches)
            best_match_info = (ref_name, raw_name, matches)

        print(f"\n  {ref_name} ↔ {raw_name}:")
        print(f"    Matches: {len(matches)}")
        if matches:
            print(f"    Sample matches: {sorted(list(matches))[:5]}")

if best_match_info and best_match_count > 0:
    ref_name, raw_name, matches = best_match_info
    print("\n" + "=" * 100)
    print(f"✓ BEST MATCH FOUND!")
    print("=" * 100)
    print(f"Reference column: {ref_name}")
    print(f"Raw data field: {raw_name}")
    print(f"Matched samples: {len(matches)}")
    print(f"\nSample matched IDs (first 20):")
    for mid in sorted(list(matches))[:20]:
        print(f"  • {mid}")
else:
    print("\n" + "=" * 100)
    print("⚠️ NO DIRECT MATCHES FOUND")
    print("=" * 100)
    print("\nPossible reasons:")
    print("  1. Sample IDs are encoded differently")
    print("  2. Different naming conventions")
    print("  3. Need manual mapping file")
    print("  4. Check 'Raw files_Sample ID' column more carefully")

# ============================================================================
# PART 4: CHECK RAW FILES_SAMPLE ID COLUMN MORE CAREFULLY
# ============================================================================
print("\n" + "=" * 100)
print("PART 4: DETAILED 'Raw files_Sample ID' COLUMN ANALYSIS")
print("=" * 100)

if 'Raw files_Sample ID' in df_ref.columns:
    raw_file_ids = df_ref['Raw files_Sample ID'].dropna()
    print(f"\nTotal non-null values: {len(raw_file_ids)}")
    print(f"\nAll unique values (first 50):")
    for i, val in enumerate(raw_file_ids.unique()[:50], 1):
        print(f"  {i:2d}. {val}")

    # Check if these might be partial filenames
    print("\n" + "-" * 100)
    print("Checking if 'Raw files_Sample ID' matches any part of actual filenames:")
    print("-" * 100)

    partial_matches = []
    for ref_id in raw_file_ids.unique()[:20]:  # Check first 20
        ref_id_str = str(ref_id)
        matched_files = [f.name for f in all_files if ref_id_str in f.name]
        if matched_files:
            partial_matches.append((ref_id_str, matched_files))
            print(f"\n  '{ref_id_str}' found in {len(matched_files)} files:")
            for fname in matched_files[:3]:
                print(f"    • {fname}")

    if partial_matches:
        print(f"\n✓ Found {len(partial_matches)} partial matches!")
    else:
        print(f"\n✗ No partial matches found")

# ============================================================================
# PART 5: CHECK FILE CONTENTS
# ============================================================================
print("\n" + "=" * 100)
print("PART 5: SPECTRAL FILE CONTENT ANALYSIS")
print("=" * 100)

# Read a few sample files
sample_files = all_files[:5]
print(f"\nAnalyzing {len(sample_files)} sample files...")

for i, sample_file in enumerate(sample_files, 1):
    print(f"\n{'-' * 100}")
    print(f"File {i}: {sample_file.name}")
    print(f"Subdir: {sample_file.parent.name}")

    try:
        # Try different separators
        for sep in ['\t', ',', ' ', ';']:
            try:
                df_spec = pd.read_csv(sample_file, sep=sep, header=None)
                if df_spec.shape[1] == 2:
                    df_spec.columns = ['wavelength', 'intensity']
                    print(f"✓ Successfully read with separator: '{sep}'")
                    print(f"  Shape: {df_spec.shape[0]} wavelength points")
                    print(f"  Wavelength range: {df_spec['wavelength'].min():.2f} - {df_spec['wavelength'].max():.2f} nm")
                    print(f"  Intensity range: {df_spec['intensity'].min():.2f} - {df_spec['intensity'].max():.2f}")
                    print(f"  First 5 rows:")
                    print(df_spec.head().to_string(index=False))
                    break
            except:
                continue
    except Exception as e:
        print(f"✗ Error reading file: {e}")

# ============================================================================
# PART 6: DATA QUALITY SUMMARY
# ============================================================================
print("\n" + "=" * 100)
print("PART 6: DATA QUALITY SUMMARY")
print("=" * 100)

print("\n📊 REFERENCE DATA:")
print(f"  • Total samples: {len(df_ref)}")
if k_col:
    print(f"  • Samples with K values: {df_ref[k_col].notna().sum()} ({df_ref[k_col].notna().sum()/len(df_ref)*100:.1f}%)")
    print(f"  • Missing K values: {df_ref[k_col].isna().sum()}")
    print(f"  • K concentration range: {k_data.min():.3f} - {k_data.max():.3f}%")
else:
    print(f"  ⚠️ Potassium column not found in reference data")

print("\n📁 RAW SPECTRAL DATA:")
print(f"  • Total files: {len(all_files)}")
print(f"  • Subdirectories: {len(subdirs)}")
print(f"  • Unique CNGS numbers: {len(set(cngs_numbers))}")
print(f"  • Unique S numbers: {len(set(sample_numbers))}")

print("\n🔗 MATCHING STATUS:")
if best_match_count > 0:
    print(f"  ✓ Successfully matched: {best_match_count} samples")
    print(f"  ✓ Can proceed with training")
else:
    print(f"  ⚠️ No automatic matching found")
    print(f"  ⚠️ Manual intervention required")

# ============================================================================
# PART 7: CONCENTRATION DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("PART 7: CONCENTRATION DISTRIBUTION FOR MODELING")
print("=" * 100)

if k_col and k_col in df_ref.columns:
    k_data = df_ref[k_col].dropna()

    print(f"\n📈 Statistical Summary:")
    print(f"  Count:    {len(k_data)}")
    print(f"  Mean:     {k_data.mean():.4f}%")
    print(f"  Std Dev:  {k_data.std():.4f}%")
    print(f"  Min:      {k_data.min():.4f}%")
    print(f"  25%:      {k_data.quantile(0.25):.4f}%")
    print(f"  50%:      {k_data.median():.4f}%")
    print(f"  75%:      {k_data.quantile(0.75):.4f}%")
    print(f"  Max:      {k_data.max():.4f}%")

    print(f"\n📊 Concentration Ranges:")
    ranges = [
        ("Zero", 0, 0),
        ("Very Low", 0, 0.1),
        ("Low", 0.1, 0.5),
        ("Low-Medium", 0.5, 2.0),
        ("Medium", 2.0, 5.0),
        ("Medium-High", 5.0, 10.0),
        ("High", 10.0, 15.0),
        ("Very High", 15.0, float('inf'))
    ]

    for label, min_val, max_val in ranges:
        if max_val == 0:
            count = (k_data == 0).sum()
        elif max_val == float('inf'):
            count = (k_data >= min_val).sum()
        else:
            count = ((k_data >= min_val) & (k_data < max_val)).sum()
        pct = count / len(k_data) * 100
        print(f"  {label:15s} ({min_val:5.1f} - {max_val:5.1f}%): {count:4d} samples ({pct:5.1f}%)")

    # Create enhanced visualization
    print("\n" + "=" * 100)
    print("Creating enhanced visualization...")
    print("=" * 100)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main title
    fig.suptitle('NEW Data - Potassium Concentration Analysis\n' +
                 f'Lab_data_updated_potassium2.xlsx ({len(k_data)} samples)',
                 fontsize=16, fontweight='bold')

    # 1. Histogram with KDE
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(k_data, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(k_data)
    x_range = np.linspace(k_data.min(), k_data.max(), 200)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    ax1.axvline(k_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {k_data.mean():.2f}%')
    ax1.axvline(k_data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {k_data.median():.2f}%')
    ax1.set_xlabel('Potassium (%)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Distribution with KDE', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Box plot with stats
    ax2 = fig.add_subplot(gs[0, 2])
    bp = ax2.boxplot(k_data, vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)
    ax2.set_ylabel('Potassium (%)', fontsize=12)
    ax2.set_title('Box Plot', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add quartile labels
    q1, q2, q3 = k_data.quantile([0.25, 0.5, 0.75])
    ax2.text(1.15, q1, f'Q1: {q1:.2f}', fontsize=10)
    ax2.text(1.15, q2, f'Q2: {q2:.2f}', fontsize=10, color='red', fontweight='bold')
    ax2.text(1.15, q3, f'Q3: {q3:.2f}', fontsize=10)

    # 3. Cumulative distribution
    ax3 = fig.add_subplot(gs[1, 0])
    sorted_vals = np.sort(k_data)
    ax3.plot(sorted_vals, np.arange(1, len(sorted_vals) + 1) / len(sorted_vals),
             linewidth=2, color='darkblue')
    ax3.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50%')
    ax3.set_xlabel('Potassium (%)', fontsize=11)
    ax3.set_ylabel('Cumulative Probability', fontsize=11)
    ax3.set_title('CDF', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Q-Q plot
    ax4 = fig.add_subplot(gs[1, 1])
    from scipy import stats
    stats.probplot(k_data, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normal)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Violin plot
    ax5 = fig.add_subplot(gs[1, 2])
    parts = ax5.violinplot([k_data], vert=True, showmeans=True, showmedians=True, showextrema=True)
    ax5.set_ylabel('Potassium (%)', fontsize=11)
    ax5.set_title('Violin Plot', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Concentration ranges bar chart
    ax6 = fig.add_subplot(gs[2, :])
    range_counts = []
    range_labels = []
    for label, min_val, max_val in ranges:
        if max_val == 0:
            count = (k_data == 0).sum()
        elif max_val == float('inf'):
            count = (k_data >= min_val).sum()
        else:
            count = ((k_data >= min_val) & (k_data < max_val)).sum()
        range_counts.append(count)
        range_labels.append(f"{label}\n({min_val:.1f}-{max_val:.1f}%)")

    colors = ['gray', 'lightcoral', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen', 'purple']
    bars = ax6.bar(range(len(range_counts)), range_counts, color=colors[:len(range_counts)],
                   edgecolor='black', alpha=0.8)
    ax6.set_xticks(range(len(range_labels)))
    ax6.set_xticklabels(range_labels, rotation=0, fontsize=9)
    ax6.set_ylabel('Count', fontsize=12)
    ax6.set_title('Concentration Range Distribution (Model Training Perspective)', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, range_counts)):
        if count > 0:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({count/len(k_data)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_file = config.reports_dir / "NEW_data_detailed_analysis.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

print("\n" + "=" * 100)
print("DETAILED ANALYSIS COMPLETE")
print("=" * 100)
