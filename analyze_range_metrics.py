#!/usr/bin/env python3
"""
Range Analysis Tool for Magnesium Predictions

This script finds the continuous concentration range with the highest within_20.5% accuracy
and calculates all relevant metrics for that range.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import sys
from pathlib import Path

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

def find_best_continuous_range(df: pd.DataFrame, 
                             min_samples: int = 10,
                             step_size: float = 0.01) -> Tuple[float, float, Dict[str, float]]:
    """
    Find the continuous concentration range with the highest within_20.5% accuracy.
    
    Args:
        df: DataFrame with ElementValue and PredictedValue columns
        min_samples: Minimum number of samples required in a range
        step_size: Step size for range boundaries
        
    Returns:
        Tuple of (range_start, range_end, best_metrics)
    """
    y_true = df['ElementValue'].values
    y_pred = df['PredictedValue'].values
    
    # Sort by true values for continuous range analysis
    sorted_indices = np.argsort(y_true)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    min_val = np.min(y_true)
    max_val = np.max(y_true)
    
    best_within_20_5 = -1
    best_range = None
    best_metrics = None
    
    print(f"Analyzing concentration ranges from {min_val:.3f} to {max_val:.3f}")
    print(f"Total samples: {len(df)}")
    print("="*60)
    
    # Try different range starting points
    range_start = min_val
    while range_start < max_val - 0.1:  # Ensure minimum range width
        
        # Try different range ending points
        range_end = range_start + 0.1  # Minimum range width
        while range_end <= max_val:
            
            # Get samples in this range
            mask = (y_true_sorted >= range_start) & (y_true_sorted <= range_end)
            y_true_range = y_true_sorted[mask]
            y_pred_range = y_pred_sorted[mask]
            
            if len(y_true_range) >= min_samples:
                # Calculate metrics for this range
                metrics = calculate_metrics(y_true_range, y_pred_range)
                
                # Check if this is the best range so far
                if metrics['within_20.5%'] > best_within_20_5:
                    best_within_20_5 = metrics['within_20.5%']
                    best_range = (range_start, range_end)
                    best_metrics = metrics
                    
                    print(f"New best range: [{range_start:.3f}, {range_end:.3f}]")
                    print(f"  Samples: {metrics['n_samples']}")
                    print(f"  Within 20.5%: {metrics['within_20.5%']:.1f}%")
                    print(f"  R²: {metrics['r2']:.4f}")
                    print()
            
            range_end += step_size
        
        range_start += step_size
    
    return best_range[0], best_range[1], best_metrics

def analyze_prediction_file(file_path: str):
    """Analyze the prediction file and find the best continuous range."""
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"Loaded predictions from: {file_path}")
    print(f"Total samples: {len(df)}")
    print(f"Concentration range: {df['ElementValue'].min():.3f} - {df['ElementValue'].max():.3f}")
    print()
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(df['ElementValue'].values, df['PredictedValue'].values)
    print("OVERALL METRICS:")
    print("="*50)
    print(f"R²: {overall_metrics['r2']:.6f}")
    print(f"MAE: {overall_metrics['mae']:.6f}")
    print(f"RMSE: {overall_metrics['rmse']:.6f}")
    print(f"RRMSE: {overall_metrics['rrmse']:.2f}%")
    print(f"MAPE: {overall_metrics['mape']:.2f}%")
    print(f"Within 20.5%: {overall_metrics['within_20.5%']:.2f}%")
    print(f"Within 15%: {overall_metrics['within_15%']:.2f}%")
    print(f"Within 10%: {overall_metrics['within_10%']:.2f}%")
    print()
    
    # Find best continuous range
    print("FINDING BEST CONTINUOUS RANGE:")
    print("="*50)
    range_start, range_end, best_metrics = find_best_continuous_range(df, min_samples=10, step_size=0.01)
    
    print("BEST CONTINUOUS RANGE RESULTS:")
    print("="*50)
    print(f"Optimal Range: [{range_start:.3f}, {range_end:.3f}] % Mg")
    print(f"Range Width: {range_end - range_start:.3f} % Mg")
    print()
    print("METRICS FOR BEST RANGE:")
    print(f"Number of Samples: {best_metrics['n_samples']}")
    print(f"Mean True Value: {best_metrics['mean_true']:.6f}")
    print(f"R²: {best_metrics['r2']:.6f}")
    print(f"MAE: {best_metrics['mae']:.6f}")
    print(f"RMSE: {best_metrics['rmse']:.6f}")
    print(f"RRMSE: {best_metrics['rrmse']:.2f}%")
    print(f"MAPE: {best_metrics['mape']:.2f}%")
    print(f"Within 20.5%: {best_metrics['within_20.5%']:.2f}%")
    print(f"Within 15%: {best_metrics['within_15%']:.2f}%")
    print(f"Within 10%: {best_metrics['within_10%']:.2f}%")
    print()
    
    # Show improvement over overall metrics
    print("IMPROVEMENT OVER OVERALL:")
    print("="*30)
    print(f"Within 20.5%: +{best_metrics['within_20.5%'] - overall_metrics['within_20.5%']:.1f} percentage points")
    print(f"R²: +{best_metrics['r2'] - overall_metrics['r2']:.4f}")
    print(f"RMSE: {best_metrics['rmse'] - overall_metrics['rmse']:.4f} (negative is better)")
    print()
    
    # Get the samples in the best range for detailed analysis
    df_sorted = df.sort_values('ElementValue')
    mask = (df_sorted['ElementValue'] >= range_start) & (df_sorted['ElementValue'] <= range_end)
    best_range_samples = df_sorted[mask]
    
    print("SAMPLE BREAKDOWN IN BEST RANGE:")
    print("="*40)
    print(f"Sample ID examples (first 10):")
    for idx, row in best_range_samples.head(10).iterrows():
        error_pct = abs((row['ElementValue'] - row['PredictedValue']) / row['ElementValue']) * 100
        print(f"  {row['sampleId'][:20]:20} | True: {row['ElementValue']:.3f} | Pred: {row['PredictedValue']:.3f} | Error: {error_pct:.1f}%")
    
    if len(best_range_samples) > 10:
        print(f"  ... and {len(best_range_samples) - 10} more samples")

if __name__ == "__main__":
    file_path = "/home/payanico/magnesium_pipeline/reports/predictions_simple_only_autogluon_20250904_202610.csv"
    analyze_prediction_file(file_path)