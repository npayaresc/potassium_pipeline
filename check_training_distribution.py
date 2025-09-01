#!/usr/bin/env python3
"""Check the distribution of training data to understand high concentration issues."""
import pandas as pd
import numpy as np
from pathlib import Path

# Load reference data to see full distribution
reference_file = Path("/home/payanico/magnesium_pipeline/data/reference_data/Final_Lab_Data_Nico_New.xlsx")
df = pd.read_excel(reference_file)

# Find magnesium column
target_col = None
for col in df.columns:
    if 'magnesium' in col.lower() or 'mg %' in col.lower():
        target_col = col
        break

if target_col:
    mg_values = df[target_col].dropna()
    
    print(f"\n=== Full Dataset Distribution ===")
    print(f"Total samples: {len(mg_values)}")
    print(f"Range: {mg_values.min():.3f} - {mg_values.max():.3f}")
    print(f"Mean: {mg_values.mean():.3f}, Std: {mg_values.std():.3f}")
    
    # Count samples in ranges
    print("\n=== Sample Counts by Range ===")
    ranges = [(0.0, 0.2), (0.2, 0.3), (0.3, 0.35), (0.35, 0.39), 
              (0.39, 0.45), (0.45, 0.5), (0.5, 1.0)]
    
    for low, high in ranges:
        count = len(mg_values[(mg_values >= low) & (mg_values < high)])
        percentage = (count / len(mg_values)) * 100
        print(f"{low:.2f}-{high:.2f}: {count:4d} samples ({percentage:5.1f}%)")
    
    # Check effect of target filtering
    print(f"\n=== Effect of Target Filtering (0.2-0.5) ===")
    filtered = mg_values[(mg_values >= 0.2) & (mg_values <= 0.5)]
    excluded = mg_values[(mg_values < 0.2) | (mg_values > 0.5)]
    print(f"Samples kept: {len(filtered)} ({len(filtered)/len(mg_values)*100:.1f}%)")
    print(f"Samples excluded: {len(excluded)} ({len(excluded)/len(mg_values)*100:.1f}%)")
    
    if len(excluded) > 0:
        print(f"\nExcluded samples range: {excluded.min():.3f} - {excluded.max():.3f}")
        print(f"Excluded samples >0.5: {len(excluded[excluded > 0.5])}")
        print(f"Excluded samples <0.2: {len(excluded[excluded < 0.2])}")