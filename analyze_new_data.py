#!/usr/bin/env python3
"""Comprehensive analysis of new data and reference files for potassium pipeline."""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("=" * 100)
print("POTASSIUM PIPELINE - NEW DATA ANALYSIS")
print("=" * 100)

# ============================================================================
# PART 1: ANALYZE REFERENCE DATA FILES
# ============================================================================
print("\n" + "=" * 100)
print("PART 1: REFERENCE DATA ANALYSIS")
print("=" * 100)

ref_files = {
    "OLD": Path("data/reference_data/Final_Lab_Data_Nico_New.xlsx"),
    "NEW": Path("data/reference_data/Lab_data_updated_potassium2.xlsx"),
    "NEW_v1": Path("data/reference_data/Lab_data_updated_potassium.xlsx")
}

ref_data = {}
for name, filepath in ref_files.items():
    if filepath.exists():
        print(f"\n{'-' * 100}")
        print(f"Loading: {name}")
        print(f"File: {filepath}")
        try:
            df = pd.read_excel(filepath)
            ref_data[name] = df
            print(f"✓ Loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"  Columns: {list(df.columns)}")

            # Find potassium-related columns
            k_cols = [col for col in df.columns if 'k' in col.lower() or 'potassium' in col.lower() or 'potasio' in col.lower()]
            id_cols = [col for col in df.columns if any(x in col.lower() for x in ['id', 'sample', 'name', 'muestra'])]

            if k_cols:
                print(f"\n  Potassium columns: {k_cols}")
                for col in k_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        valid_data = df[col].dropna()
                        if len(valid_data) > 0:
                            print(f"    • {col}:")
                            print(f"      - Valid samples: {len(valid_data)}/{len(df)}")
                            print(f"      - Range: [{valid_data.min():.4f}, {valid_data.max():.4f}]")
                            print(f"      - Mean±Std: {valid_data.mean():.4f}±{valid_data.std():.4f}")
                            print(f"      - Median: {valid_data.median():.4f}")

                            # Distribution analysis
                            q1, q3 = valid_data.quantile([0.25, 0.75])
                            print(f"      - Q1/Q3: {q1:.4f}/{q3:.4f}")
                            print(f"      - Zero values: {(valid_data == 0).sum()}")
                            print(f"      - Very low (<0.1): {((valid_data > 0) & (valid_data < 0.1)).sum()}")
                            print(f"      - Low (0.1-0.5): {((valid_data >= 0.1) & (valid_data < 0.5)).sum()}")
                            print(f"      - Medium (0.5-2.0): {((valid_data >= 0.5) & (valid_data < 2.0)).sum()}")
                            print(f"      - High (2.0+): {(valid_data >= 2.0).sum()}")

            if id_cols:
                print(f"\n  ID columns: {id_cols}")
                for col in id_cols:
                    print(f"    • {col}: {df[col].nunique()} unique values")

        except Exception as e:
            print(f"✗ Error loading {name}: {e}")
    else:
        print(f"\n{'-' * 100}")
        print(f"File not found: {name} - {filepath}")

# Compare reference files if we have multiple
if len(ref_data) > 1:
    print("\n" + "=" * 100)
    print("REFERENCE FILE COMPARISON")
    print("=" * 100)

    for name, df in ref_data.items():
        print(f"  {name}: {df.shape[0]} samples, {df.shape[1]} columns")

    # If both OLD and NEW exist, compare them
    if "OLD" in ref_data and "NEW" in ref_data:
        old_df = ref_data["OLD"]
        new_df = ref_data["NEW"]

        print(f"\n  Sample count difference: {new_df.shape[0] - old_df.shape[0]:+d}")
        print(f"  Column count difference: {new_df.shape[1] - old_df.shape[1]:+d}")

# ============================================================================
# PART 2: ANALYZE RAW SPECTRAL DATA FILES
# ============================================================================
print("\n" + "=" * 100)
print("PART 2: RAW SPECTRAL DATA ANALYSIS")
print("=" * 100)

data_dirs = {
    "OLD (Phase3)": Path("data/raw/data_5278_Phase3"),
    "NEW (newdata)": Path("data/raw/newdata"),
    "NEW (OneDrive)": Path("data/raw/OneDrive_1_9-22-2025")
}

data_info = {}
for name, dirpath in data_dirs.items():
    print(f"\n{'-' * 100}")
    print(f"Analyzing: {name}")
    print(f"Directory: {dirpath}")

    if not dirpath.exists():
        print(f"✗ Directory not found")
        continue

    # Count files
    if name.startswith("NEW"):
        # New data has subdirectories by date
        subdirs = sorted([d for d in dirpath.iterdir() if d.is_dir()])
        print(f"✓ Found {len(subdirs)} subdirectories:")

        total_files = 0
        sample_ids = set()

        for subdir in subdirs:
            files = list(subdir.glob("*.csv.txt"))
            total_files += len(files)
            print(f"  • {subdir.name}: {len(files)} files")

            # Extract sample IDs from filenames
            for f in files:
                # Filename format: 1053789KENP1S001_G_2025_01_04_1.csv.txt
                # Sample ID is typically the first part before _G_
                parts = f.stem.split('_G_')
                if parts:
                    sample_id = parts[0]
                    sample_ids.add(sample_id)

        print(f"\n  Total files: {total_files}")
        print(f"  Unique sample IDs: {len(sample_ids)}")

        data_info[name] = {
            'total_files': total_files,
            'sample_ids': sample_ids,
            'subdirs': [d.name for d in subdirs]
        }

        # Show first few sample IDs
        if sample_ids:
            print(f"  Sample IDs (first 10): {sorted(list(sample_ids))[:10]}")
    else:
        # Old data is flat
        files = list(dirpath.glob("*.csv.txt"))
        print(f"✓ Found {len(files)} files")

        sample_ids = set()
        for f in files:
            parts = f.stem.split('_G_')
            if parts:
                sample_id = parts[0]
                sample_ids.add(sample_id)

        print(f"  Unique sample IDs: {len(sample_ids)}")

        data_info[name] = {
            'total_files': len(files),
            'sample_ids': sample_ids
        }

        # Show first few sample IDs
        if sample_ids:
            print(f"  Sample IDs (first 10): {sorted(list(sample_ids))[:10]}")

# ============================================================================
# PART 3: SAMPLE ONE SPECTRAL FILE
# ============================================================================
print("\n" + "=" * 100)
print("PART 3: SPECTRAL FILE FORMAT ANALYSIS")
print("=" * 100)

# Try to read one file from new data
new_data_dir = Path("data/raw/newdata")
if new_data_dir.exists():
    # Get first subdirectory
    subdirs = sorted([d for d in new_data_dir.iterdir() if d.is_dir()])
    if subdirs:
        first_subdir = subdirs[0]
        sample_files = list(first_subdir.glob("*.csv.txt"))

        if sample_files:
            sample_file = sample_files[0]
            print(f"\nSample file: {sample_file.name}")
            print(f"From: {first_subdir.name}")

            try:
                # Try to read the file
                df_spectral = pd.read_csv(sample_file, sep='\t', header=None, names=['wavelength', 'intensity'])
                print(f"✓ Successfully read file")
                print(f"  Shape: {df_spectral.shape[0]} wavelength points")
                print(f"  Wavelength range: [{df_spectral['wavelength'].min():.2f}, {df_spectral['wavelength'].max():.2f}] nm")
                print(f"  Intensity range: [{df_spectral['intensity'].min():.2f}, {df_spectral['intensity'].max():.2f}]")
                print(f"\nFirst 10 rows:")
                print(df_spectral.head(10))
            except Exception as e:
                print(f"✗ Error reading file: {e}")

# ============================================================================
# PART 4: CROSS-REFERENCE ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("PART 4: CROSS-REFERENCE ANALYSIS (Reference Data vs Raw Data)")
print("=" * 100)

# Check if sample IDs in reference data match raw data files
if "NEW" in ref_data and "NEW (newdata)" in data_info:
    new_ref = ref_data["NEW"]
    new_raw_samples = data_info["NEW (newdata)"]['sample_ids']

    # Try to find ID column in reference data
    id_cols = [col for col in new_ref.columns if any(x in col.lower() for x in ['id', 'sample', 'name', 'muestra'])]

    if id_cols:
        id_col = id_cols[0]
        ref_sample_ids = set(new_ref[id_col].dropna().astype(str))

        print(f"\nReference file ({id_col}):")
        print(f"  Sample IDs in reference: {len(ref_sample_ids)}")
        print(f"  Sample IDs in raw data: {len(new_raw_samples)}")

        # Find matches
        matched = ref_sample_ids & new_raw_samples
        ref_only = ref_sample_ids - new_raw_samples
        raw_only = new_raw_samples - ref_sample_ids

        print(f"\n  Matched samples: {len(matched)}")
        print(f"  Reference only (no raw data): {len(ref_only)}")
        print(f"  Raw data only (no reference): {len(raw_only)}")

        if ref_only:
            print(f"\n  Samples in reference but missing raw data (first 10):")
            for sid in sorted(list(ref_only))[:10]:
                print(f"    • {sid}")

        if raw_only:
            print(f"\n  Samples with raw data but missing reference (first 10):")
            for sid in sorted(list(raw_only))[:10]:
                print(f"    • {sid}")

# ============================================================================
# PART 5: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 100)
print("PART 5: CREATING VISUALIZATIONS")
print("=" * 100)

# Create output directory
output_dir = Path("reports")
output_dir.mkdir(exist_ok=True)

# Plot potassium distributions for all reference files
for name, df in ref_data.items():
    k_cols = [col for col in df.columns if 'k' in col.lower() or 'potassium' in col.lower() or 'potasio' in col.lower()]

    for col in k_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            valid_data = df[col].dropna()

            if len(valid_data) > 0:
                print(f"\nCreating plots for {name} - {col}...")

                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle(f'{name} - {col} Distribution Analysis', fontsize=16, fontweight='bold')

                # 1. Histogram
                axes[0, 0].hist(valid_data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
                axes[0, 0].set_xlabel(f'{col} (%)', fontsize=11)
                axes[0, 0].set_ylabel('Frequency', fontsize=11)
                axes[0, 0].set_title('Distribution', fontsize=12, fontweight='bold')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].axvline(valid_data.mean(), color='red', linestyle='--', label=f'Mean: {valid_data.mean():.3f}')
                axes[0, 0].axvline(valid_data.median(), color='green', linestyle='--', label=f'Median: {valid_data.median():.3f}')
                axes[0, 0].legend()

                # 2. Box plot
                bp = axes[0, 1].boxplot(valid_data, vert=True, patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                axes[0, 1].set_ylabel(f'{col} (%)', fontsize=11)
                axes[0, 1].set_title('Box Plot', fontsize=12, fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3, axis='y')

                # 3. Violin plot
                parts = axes[0, 2].violinplot([valid_data], vert=True, showmeans=True, showmedians=True)
                axes[0, 2].set_ylabel(f'{col} (%)', fontsize=11)
                axes[0, 2].set_title('Violin Plot', fontsize=12, fontweight='bold')
                axes[0, 2].grid(True, alpha=0.3, axis='y')

                # 4. Cumulative distribution
                sorted_vals = np.sort(valid_data)
                axes[1, 0].plot(sorted_vals, np.arange(1, len(sorted_vals) + 1) / len(sorted_vals), linewidth=2)
                axes[1, 0].set_xlabel(f'{col} (%)', fontsize=11)
                axes[1, 0].set_ylabel('Cumulative Probability', fontsize=11)
                axes[1, 0].set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3)

                # 5. Q-Q plot
                from scipy import stats
                stats.probplot(valid_data, dist="norm", plot=axes[1, 1])
                axes[1, 1].set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3)

                # 6. Concentration ranges bar chart
                ranges = {
                    'Zero': (valid_data == 0).sum(),
                    'Very Low\n(0-0.1%)': ((valid_data > 0) & (valid_data < 0.1)).sum(),
                    'Low\n(0.1-0.5%)': ((valid_data >= 0.1) & (valid_data < 0.5)).sum(),
                    'Medium\n(0.5-2.0%)': ((valid_data >= 0.5) & (valid_data < 2.0)).sum(),
                    'High\n(2.0+%)': (valid_data >= 2.0).sum()
                }

                colors = ['gray', 'lightcoral', 'orange', 'lightgreen', 'darkgreen']
                axes[1, 2].bar(ranges.keys(), ranges.values(), color=colors, edgecolor='black', alpha=0.8)
                axes[1, 2].set_ylabel('Count', fontsize=11)
                axes[1, 2].set_title('Concentration Range Distribution', fontsize=12, fontweight='bold')
                axes[1, 2].grid(True, alpha=0.3, axis='y')
                axes[1, 2].tick_params(axis='x', rotation=0)

                # Add count labels on bars
                for i, (label, count) in enumerate(ranges.items()):
                    axes[1, 2].text(i, count, str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

                plt.tight_layout()

                safe_name = f"{name}_{col}".replace(" ", "_").replace("%", "pct").replace("/", "_")
                output_file = output_dir / f"reference_analysis_{safe_name}.png"
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"  ✓ Saved: {output_file}")
                plt.close()

# Compare OLD vs NEW if both exist
if "OLD" in ref_data and "NEW" in ref_data:
    print("\n" + "=" * 100)
    print("Creating comparison plots between OLD and NEW reference data...")
    print("=" * 100)

    old_df = ref_data["OLD"]
    new_df = ref_data["NEW"]

    # Find common potassium columns
    old_k_cols = [col for col in old_df.columns if 'k' in col.lower() or 'potassium' in col.lower() or 'potasio' in col.lower()]
    new_k_cols = [col for col in new_df.columns if 'k' in col.lower() or 'potassium' in col.lower() or 'potasio' in col.lower()]

    if old_k_cols and new_k_cols:
        old_col = old_k_cols[0]
        new_col = new_k_cols[0]

        old_data = old_df[old_col].dropna()
        new_data = new_df[new_col].dropna()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('OLD vs NEW Reference Data Comparison', fontsize=16, fontweight='bold')

        # 1. Histograms overlay
        axes[0, 0].hist(old_data, bins=50, alpha=0.6, label=f'OLD (n={len(old_data)})', color='blue', edgecolor='black')
        axes[0, 0].hist(new_data, bins=50, alpha=0.6, label=f'NEW (n={len(new_data)})', color='red', edgecolor='black')
        axes[0, 0].set_xlabel('Potassium (%)', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Distribution Comparison', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Box plots side by side
        bp = axes[0, 1].boxplot([old_data, new_data], labels=['OLD', 'NEW'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        axes[0, 1].set_ylabel('Potassium (%)', fontsize=11)
        axes[0, 1].set_title('Box Plot Comparison', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 3. CDFs overlay
        sorted_old = np.sort(old_data)
        sorted_new = np.sort(new_data)
        axes[1, 0].plot(sorted_old, np.arange(1, len(sorted_old) + 1) / len(sorted_old),
                       label='OLD', linewidth=2, color='blue')
        axes[1, 0].plot(sorted_new, np.arange(1, len(sorted_new) + 1) / len(sorted_new),
                       label='NEW', linewidth=2, color='red')
        axes[1, 0].set_xlabel('Potassium (%)', fontsize=11)
        axes[1, 0].set_ylabel('Cumulative Probability', fontsize=11)
        axes[1, 0].set_title('Cumulative Distribution Comparison', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Statistics table
        axes[1, 1].axis('off')
        stats_data = [
            ['Metric', 'OLD', 'NEW', 'Difference'],
            ['Count', f'{len(old_data)}', f'{len(new_data)}', f'{len(new_data) - len(old_data):+d}'],
            ['Mean', f'{old_data.mean():.4f}', f'{new_data.mean():.4f}', f'{new_data.mean() - old_data.mean():+.4f}'],
            ['Median', f'{old_data.median():.4f}', f'{new_data.median():.4f}', f'{new_data.median() - old_data.median():+.4f}'],
            ['Std Dev', f'{old_data.std():.4f}', f'{new_data.std():.4f}', f'{new_data.std() - old_data.std():+.4f}'],
            ['Min', f'{old_data.min():.4f}', f'{new_data.min():.4f}', f'{new_data.min() - old_data.min():+.4f}'],
            ['Max', f'{old_data.max():.4f}', f'{new_data.max():.4f}', f'{new_data.max() - old_data.max():+.4f}'],
        ]

        table = axes[1, 1].table(cellText=stats_data, cellLoc='center', loc='center',
                                colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        axes[1, 1].set_title('Statistical Summary', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        output_file = output_dir / "reference_comparison_OLD_vs_NEW.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
print(f"\nReports saved to: {output_dir.absolute()}")
