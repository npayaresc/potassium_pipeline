#!/usr/bin/env python3
"""
Analyze the actual distribution of magnesium values in the reference data.
This will help verify if the sample weighting is appropriate.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
from pathlib import Path

def analyze_reference_distribution(reference_file_path):
    """Analyze the distribution of target values in reference data."""
    print(f"\nAnalyzing reference data from: {reference_file_path}")
    
    # Load reference data
    df = pd.read_excel(reference_file_path)
    df.columns = df.columns.str.strip()
    
    # Look for magnesium column (try different names)
    target_col = None
    for col in ['Magnesium %', 'Magnesium %', 'P %', 'P%', 'magnesium', 'magnesium']:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        print("Available columns:", df.columns.tolist())
        raise ValueError("Could not find magnesium concentration column")
    
    print(f"\nFound target column: {target_col}")
    
    # Get magnesium values
    p_values = df[target_col].dropna()
    print(f"Number of samples: {len(p_values)}")
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    print(f"Mean: {p_values.mean():.4f}%")
    print(f"Median: {p_values.median():.4f}%")
    print(f"Std Dev: {p_values.std():.4f}%")
    print(f"Min: {p_values.min():.4f}%")
    print(f"Max: {p_values.max():.4f}%")
    
    # Percentiles
    print("\n=== Percentiles ===")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for p in percentiles:
        value = np.percentile(p_values, p)
        print(f"P{p:2d}: {value:.4f}%")
    
    # Count samples in ranges
    print("\n=== Sample Distribution by Range ===")
    ranges = [(0.0, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.3), 
              (0.3, 0.4), (0.4, 0.5), (0.5, 1.0)]
    
    for low, high in ranges:
        count = len(p_values[(p_values >= low) & (p_values < high)])
        percentage = (count / len(p_values)) * 100
        print(f"{low:.2f}-{high:.2f}%: {count:4d} samples ({percentage:5.1f}%)")
    
    # Calculate what weights would be with different methods
    print("\n=== Sample Weights Analysis ===")
    
    # Legacy method (adapted for magnesium)
    legacy_weights = np.ones_like(p_values, dtype=float)
    for i, val in enumerate(p_values):
        if 0.1 <= val < 0.15: legacy_weights[i] = 2.5
        elif 0.15 <= val < 0.20: legacy_weights[i] = 2.0
        elif 0.20 <= val < 0.30: legacy_weights[i] = 1.2
        elif 0.30 <= val < 0.40: legacy_weights[i] = 1.5
        elif 0.40 <= val <= 0.50: legacy_weights[i] = 2.5
        else: legacy_weights[i] = 1.0
    
    print("\nLegacy method weights by range:")
    for low, high in ranges:
        mask = (p_values >= low) & (p_values < high)
        if mask.any():
            avg_weight = legacy_weights[mask].mean()
            print(f"{low:.2f}-{high:.2f}%: avg weight = {avg_weight:.2f}")
    
    # Distribution-based method
    try:
        kde = gaussian_kde(p_values.values)
        densities = kde(p_values.values)
        dist_weights = 1.0 / (densities + 1e-8)
        dist_weights = np.clip(dist_weights, 0.2, 5.0)
        dist_weights = dist_weights * len(p_values) / np.sum(dist_weights)
        
        print("\nDistribution-based method weights by range:")
        for low, high in ranges:
            mask = (p_values >= low) & (p_values < high)
            if mask.any():
                avg_weight = dist_weights[mask].mean()
                print(f"{low:.2f}-{high:.2f}%: avg weight = {avg_weight:.2f}")
    except Exception as e:
        print(f"Could not calculate KDE weights: {e}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Histogram
    ax = axes[0, 0]
    ax.hist(p_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Magnesium %')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Magnesium Concentrations')
    ax.axvline(p_values.mean(), color='red', linestyle='--', label=f'Mean: {p_values.mean():.3f}%')
    ax.axvline(p_values.median(), color='green', linestyle='--', label=f'Median: {p_values.median():.3f}%')
    ax.legend()
    
    # 2. KDE plot
    ax = axes[0, 1]
    sns.kdeplot(data=p_values, ax=ax, fill=True)
    ax.set_xlabel('Magnesium %')
    ax.set_ylabel('Density')
    ax.set_title('Kernel Density Estimate')
    
    # 3. Box plot with percentiles
    ax = axes[1, 0]
    box_data = ax.boxplot([p_values], vert=False, patch_artist=True)
    box_data['boxes'][0].set_facecolor('lightblue')
    ax.set_xlabel('Magnesium %')
    ax.set_title('Box Plot with Quartiles')
    
    # 4. Cumulative distribution
    ax = axes[1, 1]
    sorted_values = np.sort(p_values)
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax.plot(sorted_values, cumulative, marker='.', markersize=2)
    ax.set_xlabel('Magnesium %')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('magnesium_distribution_analysis.png', dpi=150)
    print("\nSaved distribution plot to: magnesium_distribution_analysis.png")
    
    return p_values

if __name__ == "__main__":
    # Update this path to your reference data file
    reference_file = Path("/home/payanico/magnesium_pipeline/data/reference_data/Final_Lab_Data_Nico_New.xlsx")
    
    if not reference_file.exists():
        print(f"Reference file not found: {reference_file}")
        print("Please update the path to your magnesium reference data file.")
    else:
        try:
            p_values = analyze_reference_distribution(reference_file)
        except Exception as e:
            print(f"Error analyzing distribution: {e}")