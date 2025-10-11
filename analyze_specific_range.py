#!/usr/bin/env python3
"""
Analyze specific range [0.2, 2.0] % Mg as requested
"""

import pandas as pd
import numpy as np
from typing import Dict

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate all the metrics we typically report."""
    # Basic regression metrics
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Relative metrics
    mean_true = np.mean(y_true)
    rrmse = (rmse / mean_true * 100) if mean_true != 0 else 0
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Percentage accuracy metrics
    within_20_5 = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.205) * 100
    within_15 = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.15) * 100
    within_10 = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.10) * 100
    
    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'rrmse': rrmse,
        'mape': mape,
        'within_20.5%': within_20_5,
        'within_15%': within_15,
        'within_10%': within_10,
        'n_samples': len(y_true),
        'mean_true': mean_true,
        'min_true': np.min(y_true),
        'max_true': np.max(y_true)
    }

def analyze_range_0_2_to_2_0():
    """Analyze the specific range [0.2, 2.0] % Mg."""
    
    file_path = "/home/payanico/magnesium_pipeline/reports/predictions_simple_only_autogluon_20250904_202610.csv"
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"Loaded predictions from: {file_path}")
    print(f"Total samples: {len(df)}")
    print()
    
    # Calculate overall metrics first
    overall_metrics = calculate_metrics(df['ElementValue'].values, df['PredictedValue'].values)
    
    # Filter for the requested range [0.2, 2.0]
    range_start = 0.2
    range_end = 2.0
    
    mask = (df['ElementValue'] >= range_start) & (df['ElementValue'] <= range_end)
    range_df = df[mask]
    
    if len(range_df) == 0:
        print(f"No samples found in range [{range_start}, {range_end}]")
        return
    
    # Calculate metrics for the specific range
    range_metrics = calculate_metrics(range_df['ElementValue'].values, range_df['PredictedValue'].values)
    
    print("="*70)
    print(f"ANALYSIS FOR RANGE [0.2, 2.0] % Mg")
    print("="*70)
    print()
    
    print("RANGE CHARACTERISTICS:")
    print(f"Range: [{range_start:.1f}, {range_end:.1f}] % Mg")
    print(f"Range Width: {range_end - range_start:.1f} % Mg")
    print(f"Number of Samples: {range_metrics['n_samples']} ({range_metrics['n_samples']/len(df)*100:.1f}% of total)")
    print(f"Mean True Value: {range_metrics['mean_true']:.4f} % Mg")
    print(f"Concentration Range: {range_metrics['min_true']:.3f} - {range_metrics['max_true']:.3f} % Mg")
    print()
    
    print("PERFORMANCE METRICS:")
    print("="*50)
    print(f"R²: {range_metrics['r2']:.6f}")
    print(f"MAE: {range_metrics['mae']:.6f}")
    print(f"RMSE: {range_metrics['rmse']:.6f}")
    print(f"RRMSE: {range_metrics['rrmse']:.2f}%")
    print(f"MAPE: {range_metrics['mape']:.2f}%")
    print()
    print("ACCURACY METRICS:")
    print(f"Within 20.5%: {range_metrics['within_20.5%']:.2f}%")
    print(f"Within 15%: {range_metrics['within_15%']:.2f}%")
    print(f"Within 10%: {range_metrics['within_10%']:.2f}%")
    print()
    
    print("COMPARISON TO OVERALL DATASET:")
    print("="*50)
    print(f"{'Metric':<15} {'Range [0.2,2.0]':<15} {'Overall':<15} {'Difference':<15}")
    print("-" * 60)
    print(f"{'Within 20.5%':<15} {range_metrics['within_20.5%']:<15.2f} {overall_metrics['within_20.5%']:<15.2f} {range_metrics['within_20.5%'] - overall_metrics['within_20.5%']:+.2f}")
    print(f"{'Within 15%':<15} {range_metrics['within_15%']:<15.2f} {overall_metrics['within_15%']:<15.2f} {range_metrics['within_15%'] - overall_metrics['within_15%']:+.2f}")
    print(f"{'Within 10%':<15} {range_metrics['within_10%']:<15.2f} {overall_metrics['within_10%']:<15.2f} {range_metrics['within_10%'] - overall_metrics['within_10%']:+.2f}")
    print(f"{'R²':<15} {range_metrics['r2']:<15.6f} {overall_metrics['r2']:<15.6f} {range_metrics['r2'] - overall_metrics['r2']:+.6f}")
    print(f"{'RMSE':<15} {range_metrics['rmse']:<15.6f} {overall_metrics['rmse']:<15.6f} {range_metrics['rmse'] - overall_metrics['rmse']:+.6f}")
    print(f"{'MAE':<15} {range_metrics['mae']:<15.6f} {overall_metrics['mae']:<15.6f} {range_metrics['mae'] - overall_metrics['mae']:+.6f}")
    print(f"{'RRMSE':<15} {range_metrics['rrmse']:<15.2f} {overall_metrics['rrmse']:<15.2f} {range_metrics['rrmse'] - overall_metrics['rrmse']:+.2f}")
    print(f"{'MAPE':<15} {range_metrics['mape']:<15.2f} {overall_metrics['mape']:<15.2f} {range_metrics['mape'] - overall_metrics['mape']:+.2f}")
    print()
    
    # Show detailed breakdown by concentration sub-ranges
    print("BREAKDOWN BY SUB-RANGES:")
    print("="*60)
    sub_ranges = [
        (0.2, 0.5, "Low-Medium"),
        (0.5, 1.0, "Medium"), 
        (1.0, 1.5, "Medium-High"),
        (1.5, 2.0, "High")
    ]
    
    print(f"{'Sub-range':<15} {'Description':<15} {'Samples':<8} {'Within 20.5%':<12} {'R²':<10} {'RMSE':<10}")
    print("-" * 80)
    
    for sub_start, sub_end, description in sub_ranges:
        sub_mask = (range_df['ElementValue'] >= sub_start) & (range_df['ElementValue'] <= sub_end)
        sub_df = range_df[sub_mask]
        
        if len(sub_df) >= 5:  # Only analyze if we have enough samples
            sub_metrics = calculate_metrics(sub_df['ElementValue'].values, sub_df['PredictedValue'].values)
            print(f"[{sub_start:.1f}, {sub_end:.1f}]    {description:<15} {sub_metrics['n_samples']:<8} {sub_metrics['within_20.5%']:<12.1f} {sub_metrics['r2']:<10.4f} {sub_metrics['rmse']:<10.4f}")
        else:
            print(f"[{sub_start:.1f}, {sub_end:.1f}]    {description:<15} {len(sub_df):<8} {'Too few':<12} {'samples':<10} {'N/A':<10}")
    
    print()
    
    # Show some example predictions in this range
    print("SAMPLE PREDICTIONS IN RANGE [0.2, 2.0]:")
    print("="*70)
    
    # Sort by true value and show representative samples
    range_df_sorted = range_df.sort_values('ElementValue')
    
    print(f"{'Sample ID':<25} {'True':<8} {'Predicted':<10} {'Error%':<8} {'Within 20.5%'}")
    print("-" * 70)
    
    # Show first 15, middle 10, and last 15 samples
    n_samples = len(range_df_sorted)
    
    # First 15 samples (lowest concentrations)
    print("Lowest concentrations:")
    for idx, (_, row) in enumerate(range_df_sorted.head(15).iterrows()):
        error_pct = abs((row['ElementValue'] - row['PredictedValue']) / row['ElementValue']) * 100
        within_threshold = "✓" if error_pct <= 20.5 else "✗"
        print(f"{row['sampleId'][:25]:<25} {row['ElementValue']:<8.3f} {row['PredictedValue']:<10.3f} {error_pct:<8.1f} {within_threshold}")
    
    # Middle 10 samples
    if n_samples > 40:
        print("\nMiddle concentrations:")
        middle_start = n_samples // 2 - 5
        middle_end = n_samples // 2 + 5
        for idx, (_, row) in enumerate(range_df_sorted.iloc[middle_start:middle_end].iterrows()):
            error_pct = abs((row['ElementValue'] - row['PredictedValue']) / row['ElementValue']) * 100
            within_threshold = "✓" if error_pct <= 20.5 else "✗"
            print(f"{row['sampleId'][:25]:<25} {row['ElementValue']:<8.3f} {row['PredictedValue']:<10.3f} {error_pct:<8.1f} {within_threshold}")
    
    # Last 15 samples (highest concentrations)
    print("\nHighest concentrations:")
    for idx, (_, row) in enumerate(range_df_sorted.tail(15).iterrows()):
        error_pct = abs((row['ElementValue'] - row['PredictedValue']) / row['ElementValue']) * 100
        within_threshold = "✓" if error_pct <= 20.5 else "✗"
        print(f"{row['sampleId'][:25]:<25} {row['ElementValue']:<8.3f} {row['PredictedValue']:<10.3f} {error_pct:<8.1f} {within_threshold}")

if __name__ == "__main__":
    analyze_range_0_2_to_2_0()