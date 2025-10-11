#!/usr/bin/env python3
"""
Custom Range Analysis Tool for Magnesium Predictions

This script accepts low_limit and high_limit parameters to analyze any concentration range
and calculates all relevant metrics for that range.

Usage:
    python analyze_custom_range.py --low-limit 0.2 --high-limit 1.5
    python analyze_custom_range.py --low-limit 0.15 --high-limit 2.5 --predictions-file path/to/predictions.csv
"""

import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Tuple
from pathlib import Path

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate all the metrics we typically report."""
    if len(y_true) == 0:
        return {
            'r2': np.nan, 'mae': np.nan, 'rmse': np.nan, 'rrmse': np.nan, 'mape': np.nan,
            'within_20.5%': np.nan, 'within_15%': np.nan, 'within_10%': np.nan,
            'n_samples': 0, 'mean_true': np.nan, 'min_true': np.nan, 'max_true': np.nan
        }
    
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
    
    # MAPE with zero-division protection
    relative_errors = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(relative_errors[y_true != 0]) * 100
    
    # Percentage accuracy metrics
    within_20_5 = np.mean(relative_errors <= 0.205) * 100
    within_15 = np.mean(relative_errors <= 0.15) * 100
    within_10 = np.mean(relative_errors <= 0.10) * 100
    
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

def create_sub_ranges(low_limit: float, high_limit: float, num_sub_ranges: int = 5) -> List[Tuple[float, float, str]]:
    """Create evenly spaced sub-ranges within the specified limits."""
    range_width = high_limit - low_limit
    sub_range_width = range_width / num_sub_ranges
    
    sub_ranges = []
    for i in range(num_sub_ranges):
        sub_start = low_limit + i * sub_range_width
        sub_end = low_limit + (i + 1) * sub_range_width
        
        # Create descriptive names
        if num_sub_ranges == 5:
            descriptions = ["Very Low", "Low", "Medium", "High", "Very High"]
        elif num_sub_ranges == 4:
            descriptions = ["Low", "Low-Medium", "Medium-High", "High"]
        elif num_sub_ranges == 3:
            descriptions = ["Low", "Medium", "High"]
        else:
            descriptions = [f"Range {i+1}" for i in range(num_sub_ranges)]
        
        sub_ranges.append((sub_start, sub_end, descriptions[i]))
    
    return sub_ranges

def analyze_custom_range(predictions_file: str, low_limit: float, high_limit: float, 
                        show_samples: bool = True, max_samples_display: int = 20):
    """Analyze a custom concentration range and calculate all metrics."""
    
    # Validate inputs
    if low_limit >= high_limit:
        raise ValueError(f"low_limit ({low_limit}) must be less than high_limit ({high_limit})")
    
    if not Path(predictions_file).exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    
    # Load data
    try:
        df = pd.read_csv(predictions_file)
        print(f"Loaded predictions from: {predictions_file}")
    except Exception as e:
        raise ValueError(f"Error loading predictions file: {e}")
    
    # Validate required columns
    required_cols = ['sampleId', 'ElementValue', 'PredictedValue']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Total samples in file: {len(df)}")
    print(f"Concentration range in file: {df['ElementValue'].min():.3f} - {df['ElementValue'].max():.3f} % Mg")
    print()
    
    # Calculate overall metrics first
    overall_metrics = calculate_metrics(df['ElementValue'].values, df['PredictedValue'].values)
    
    # Filter for the requested range
    mask = (df['ElementValue'] >= low_limit) & (df['ElementValue'] <= high_limit)
    range_df = df[mask]
    
    if len(range_df) == 0:
        print(f"❌ No samples found in range [{low_limit}, {high_limit}] % Mg")
        print(f"Available range: {df['ElementValue'].min():.3f} - {df['ElementValue'].max():.3f} % Mg")
        return
    
    # Calculate metrics for the specific range
    range_metrics = calculate_metrics(range_df['ElementValue'].values, range_df['PredictedValue'].values)
    
    print("="*80)
    print(f"ANALYSIS FOR CUSTOM RANGE [{low_limit}, {high_limit}] % Mg")
    print("="*80)
    print()
    
    print("RANGE CHARACTERISTICS:")
    print("="*50)
    print(f"Range: [{low_limit}, {high_limit}] % Mg")
    print(f"Range Width: {high_limit - low_limit:.3f} % Mg")
    print(f"Number of Samples: {range_metrics['n_samples']} ({range_metrics['n_samples']/len(df)*100:.1f}% of total)")
    print(f"Mean True Value: {range_metrics['mean_true']:.4f} % Mg")
    print(f"Actual Range: {range_metrics['min_true']:.3f} - {range_metrics['max_true']:.3f} % Mg")
    
    # Count excluded samples
    excluded_low = len(df[df['ElementValue'] < low_limit])
    excluded_high = len(df[df['ElementValue'] > high_limit])
    print(f"Samples excluded (< {low_limit}): {excluded_low}")
    print(f"Samples excluded (> {high_limit}): {excluded_high}")
    print()
    
    print("PERFORMANCE METRICS:")
    print("="*50)
    print(f"R²: {range_metrics['r2']:.6f}")
    print(f"MAE: {range_metrics['mae']:.6f} % Mg")
    print(f"RMSE: {range_metrics['rmse']:.6f} % Mg")
    print(f"RRMSE: {range_metrics['rrmse']:.2f}%")
    print(f"MAPE: {range_metrics['mape']:.2f}%")
    print()
    
    print("ACCURACY METRICS:")
    print("="*50)
    print(f"Within 20.5%: {range_metrics['within_20.5%']:.2f}%")
    print(f"Within 15%: {range_metrics['within_15%']:.2f}%")
    print(f"Within 10%: {range_metrics['within_10%']:.2f}%")
    print()
    
    print("COMPARISON TO OVERALL DATASET:")
    print("="*70)
    print(f"{'Metric':<15} {'Custom Range':<15} {'Overall':<15} {'Difference':<15} {'Status':<10}")
    print("-" * 75)
    
    metrics_comparison = [
        ('Within 20.5%', 'within_20.5%', '%'),
        ('Within 15%', 'within_15%', '%'),
        ('Within 10%', 'within_10%', '%'),
        ('R²', 'r2', ''),
        ('RMSE', 'rmse', ''),
        ('MAE', 'mae', ''),
        ('RRMSE', 'rrmse', '%'),
        ('MAPE', 'mape', '%')
    ]
    
    for metric_name, metric_key, unit in metrics_comparison:
        range_val = range_metrics[metric_key]
        overall_val = overall_metrics[metric_key]
        diff = range_val - overall_val
        
        # Determine status
        if metric_key in ['within_20.5%', 'within_15%', 'within_10%', 'r2']:
            status = "Better" if diff > 0 else "Worse" if diff < 0 else "Same"
        else:  # Lower is better for error metrics
            status = "Better" if diff < 0 else "Worse" if diff > 0 else "Same"
        
        print(f"{metric_name:<15} {range_val:<15.2f} {overall_val:<15.2f} {diff:+15.2f} {status:<10}")
    
    print()
    
    # Sub-range analysis
    print("BREAKDOWN BY SUB-RANGES:")
    print("="*80)
    
    # Determine appropriate number of sub-ranges based on range width and sample count
    range_width = high_limit - low_limit
    if range_width <= 0.5 or range_metrics['n_samples'] < 50:
        num_sub_ranges = 3
    elif range_width <= 1.5 or range_metrics['n_samples'] < 100:
        num_sub_ranges = 4
    else:
        num_sub_ranges = 5
    
    sub_ranges = create_sub_ranges(low_limit, high_limit, num_sub_ranges)
    
    print(f"{'Sub-range':<20} {'Description':<12} {'Samples':<8} {'Within 20.5%':<12} {'R²':<10} {'RMSE':<10}")
    print("-" * 90)
    
    for sub_start, sub_end, description in sub_ranges:
        sub_mask = (range_df['ElementValue'] >= sub_start) & (range_df['ElementValue'] <= sub_end)
        sub_df = range_df[sub_mask]
        
        if len(sub_df) >= 3:  # Only analyze if we have enough samples
            sub_metrics = calculate_metrics(sub_df['ElementValue'].values, sub_df['PredictedValue'].values)
            print(f"[{sub_start:.3f}, {sub_end:.3f}]  {description:<12} {sub_metrics['n_samples']:<8} {sub_metrics['within_20.5%']:<12.1f} {sub_metrics['r2']:<10.4f} {sub_metrics['rmse']:<10.4f}")
        else:
            print(f"[{sub_start:.3f}, {sub_end:.3f}]  {description:<12} {len(sub_df):<8} {'Too few samples':<32}")
    
    print()
    
    # Sample predictions display
    if show_samples and range_metrics['n_samples'] > 0:
        print("SAMPLE PREDICTIONS IN CUSTOM RANGE:")
        print("="*80)
        
        # Sort by true value
        range_df_sorted = range_df.sort_values('ElementValue')
        
        print(f"{'Sample ID':<30} {'True':<8} {'Predicted':<10} {'Error%':<8} {'Within 20.5%'}")
        print("-" * 75)
        
        # Display strategy based on sample count
        n_samples = len(range_df_sorted)
        
        if n_samples <= max_samples_display:
            # Show all samples
            for _, row in range_df_sorted.iterrows():
                error_pct = abs((row['ElementValue'] - row['PredictedValue']) / row['ElementValue']) * 100
                within_threshold = "✓" if error_pct <= 20.5 else "✗"
                print(f"{row['sampleId'][:30]:<30} {row['ElementValue']:<8.3f} {row['PredictedValue']:<10.3f} {error_pct:<8.1f} {within_threshold}")
        else:
            # Show representative samples: first, middle, last
            show_each = max_samples_display // 3
            
            print(f"Lowest concentrations (showing {show_each}):")
            for _, row in range_df_sorted.head(show_each).iterrows():
                error_pct = abs((row['ElementValue'] - row['PredictedValue']) / row['ElementValue']) * 100
                within_threshold = "✓" if error_pct <= 20.5 else "✗"
                print(f"{row['sampleId'][:30]:<30} {row['ElementValue']:<8.3f} {row['PredictedValue']:<10.3f} {error_pct:<8.1f} {within_threshold}")
            
            print(f"\nMiddle concentrations (showing {show_each}):")
            middle_start = n_samples // 2 - show_each // 2
            middle_end = middle_start + show_each
            for _, row in range_df_sorted.iloc[middle_start:middle_end].iterrows():
                error_pct = abs((row['ElementValue'] - row['PredictedValue']) / row['ElementValue']) * 100
                within_threshold = "✓" if error_pct <= 20.5 else "✗"
                print(f"{row['sampleId'][:30]:<30} {row['ElementValue']:<8.3f} {row['PredictedValue']:<10.3f} {error_pct:<8.1f} {within_threshold}")
            
            print(f"\nHighest concentrations (showing {show_each}):")
            for _, row in range_df_sorted.tail(show_each).iterrows():
                error_pct = abs((row['ElementValue'] - row['PredictedValue']) / row['ElementValue']) * 100
                within_threshold = "✓" if error_pct <= 20.5 else "✗"
                print(f"{row['sampleId'][:30]:<30} {row['ElementValue']:<8.3f} {row['PredictedValue']:<10.3f} {error_pct:<8.1f} {within_threshold}")
            
            remaining = n_samples - (3 * show_each)
            if remaining > 0:
                print(f"\n... and {remaining} more samples in between")
    
    # Summary and recommendations
    print(f"\nSUMMARY:")
    print("="*50)
    
    accuracy_improvement = range_metrics['within_20.5%'] - overall_metrics['within_20.5%']
    r2_change = range_metrics['r2'] - overall_metrics['r2']
    
    print(f"📊 Range Coverage: {range_metrics['n_samples']}/{len(df)} samples ({range_metrics['n_samples']/len(df)*100:.1f}%)")
    print(f"🎯 Accuracy: {range_metrics['within_20.5%']:.2f}% within 20.5% error")
    print(f"📈 vs Overall: {accuracy_improvement:+.2f} percentage points")
    print(f"📉 R² Change: {r2_change:+.4f}")
    
    if accuracy_improvement > 5:
        print("✅ This range shows SIGNIFICANTLY better accuracy than the overall dataset")
    elif accuracy_improvement > 0:
        print("✅ This range shows improved accuracy compared to the overall dataset")
    else:
        print("⚠️  This range shows similar or worse accuracy than the overall dataset")
    
    if range_metrics['n_samples'] / len(df) > 0.8:
        print("✅ Excellent coverage - includes most samples")
    elif range_metrics['n_samples'] / len(df) > 0.6:
        print("✅ Good coverage - includes majority of samples")
    else:
        print("⚠️  Limited coverage - excludes many samples")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze custom concentration range for magnesium predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_custom_range.py --low-limit 0.2 --high-limit 1.5
  python analyze_custom_range.py --low-limit 0.15 --high-limit 2.5 --predictions-file path/to/predictions.csv
  python analyze_custom_range.py --low-limit 0.3 --high-limit 1.0 --no-samples
        """
    )
    
    parser.add_argument("--low-limit", type=float, required=True,
                       help="Lower limit of concentration range (percent Mg)")
    parser.add_argument("--high-limit", type=float, required=True,
                       help="Upper limit of concentration range (percent Mg)")
    parser.add_argument("--predictions-file", type=str, 
                       default="/home/payanico/magnesium_pipeline/reports/predictions_simple_only_autogluon_20250904_202610.csv",
                       help="Path to predictions CSV file (default: AutoGluon predictions file)")
    parser.add_argument("--no-samples", action="store_true",
                       help="Skip displaying sample predictions")
    parser.add_argument("--max-samples", type=int, default=20,
                       help="Maximum number of sample predictions to display (default: 20)")
    
    args = parser.parse_args()
    
    try:
        analyze_custom_range(
            predictions_file=args.predictions_file,
            low_limit=args.low_limit,
            high_limit=args.high_limit,
            show_samples=not args.no_samples,
            max_samples_display=args.max_samples
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())