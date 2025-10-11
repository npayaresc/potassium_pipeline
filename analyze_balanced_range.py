#!/usr/bin/env python3
"""
Balanced Range Analysis Tool

Find the best continuous range considering both within_20.5% accuracy AND R² performance.
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

def score_range(metrics: Dict[str, float]) -> float:
    """
    Calculate a balanced score for a range considering both accuracy and R².
    Higher score is better.
    """
    # Only consider ranges with reasonable R² (> -1.0) and sufficient accuracy
    if metrics['r2'] < -1.0 or metrics['within_20.5%'] < 50:
        return -1000  # Heavily penalize poor ranges
    
    # Balanced scoring: 70% weight on within_20.5%, 30% weight on R²
    accuracy_score = metrics['within_20.5%']  # 0-100
    r2_score = max(0, metrics['r2']) * 100   # Convert R² to 0-100 scale (clip negative)
    
    balanced_score = 0.7 * accuracy_score + 0.3 * r2_score
    
    # Bonus for high sample count (stability)
    sample_bonus = min(10, metrics['n_samples'] / 10)  # Up to 10 point bonus
    
    return balanced_score + sample_bonus

def find_balanced_range(df: pd.DataFrame, min_samples: int = 15, step_size: float = 0.02) -> Tuple[float, float, Dict[str, float]]:
    """
    Find the continuous concentration range with the best balanced performance.
    """
    y_true = df['ElementValue'].values
    y_pred = df['PredictedValue'].values
    
    # Sort by true values
    sorted_indices = np.argsort(y_true)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    min_val = np.min(y_true)
    max_val = np.max(y_true)
    
    best_score = -1000
    best_range = None
    best_metrics = None
    
    print(f"Analyzing concentration ranges from {min_val:.3f} to {max_val:.3f}")
    print(f"Looking for balanced performance (accuracy + R²)")
    print("="*60)
    
    # Try different range starting points
    range_start = min_val
    while range_start < max_val - 0.2:  # Ensure minimum range width
        
        # Try different range ending points
        range_end = range_start + 0.2  # Minimum range width
        while range_end <= max_val:
            
            # Get samples in this range
            mask = (y_true_sorted >= range_start) & (y_true_sorted <= range_end)
            y_true_range = y_true_sorted[mask]
            y_pred_range = y_pred_sorted[mask]
            
            if len(y_true_range) >= min_samples:
                # Calculate metrics for this range
                metrics = calculate_metrics(y_true_range, y_pred_range)
                score = score_range(metrics)
                
                # Check if this is the best range so far
                if score > best_score:
                    best_score = score
                    best_range = (range_start, range_end)
                    best_metrics = metrics
                    
                    print(f"New best range: [{range_start:.3f}, {range_end:.3f}]")
                    print(f"  Score: {score:.1f}")
                    print(f"  Samples: {metrics['n_samples']}")
                    print(f"  Within 20.5%: {metrics['within_20.5%']:.1f}%")
                    print(f"  R²: {metrics['r2']:.4f}")
                    print()
            
            range_end += step_size
        
        range_start += step_size
    
    return best_range[0], best_range[1], best_metrics

def analyze_multiple_ranges(df: pd.DataFrame):
    """Analyze and compare multiple good ranges."""
    
    print("\n" + "="*70)
    print("ANALYZING MULTIPLE GOOD RANGES")
    print("="*70)
    
    # Define some promising ranges based on different criteria
    ranges_to_test = [
        # Low concentration range
        (0.1, 0.6, "Low Concentration"),
        # Mid concentration range  
        (0.4, 1.2, "Mid Concentration"),
        # High concentration range
        (1.0, 2.0, "High Concentration"),
        # Sweet spot around 0.5-1.0
        (0.5, 1.0, "Sweet Spot"),
        # Broader mid-range
        (0.3, 1.5, "Broad Mid-Range")
    ]
    
    results = []
    
    for range_start, range_end, description in ranges_to_test:
        # Get samples in this range
        mask = (df['ElementValue'] >= range_start) & (df['ElementValue'] <= range_end)
        subset_df = df[mask]
        
        if len(subset_df) >= 10:
            metrics = calculate_metrics(subset_df['ElementValue'].values, subset_df['PredictedValue'].values)
            score = score_range(metrics)
            
            results.append({
                'range': f"[{range_start:.1f}, {range_end:.1f}]",
                'description': description,
                'score': score,
                'samples': metrics['n_samples'],
                'r2': metrics['r2'],
                'within_20_5': metrics['within_20.5%'],
                'rmse': metrics['rmse'],
                'mape': metrics['mape']
            })
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"{'Range':<12} {'Description':<15} {'Score':<8} {'Samples':<8} {'R²':<8} {'Within20.5%':<12} {'RMSE':<8} {'MAPE':<8}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['range']:<12} {r['description']:<15} {r['score']:<8.1f} {r['samples']:<8} {r['r2']:<8.4f} {r['within_20_5']:<12.1f} {r['rmse']:<8.4f} {r['mape']:<8.1f}")
    
    return results

def main():
    file_path = "/home/payanico/magnesium_pipeline/reports/predictions_simple_only_autogluon_20250904_202610.csv"
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"Loaded predictions from: {file_path}")
    print(f"Total samples: {len(df)}")
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(df['ElementValue'].values, df['PredictedValue'].values)
    print("\nOVERALL METRICS:")
    print("="*50)
    print(f"R²: {overall_metrics['r2']:.6f}")
    print(f"MAE: {overall_metrics['mae']:.6f}")
    print(f"RMSE: {overall_metrics['rmse']:.6f}")
    print(f"RRMSE: {overall_metrics['rrmse']:.2f}%")
    print(f"Within 20.5%: {overall_metrics['within_20.5%']:.2f}%")
    print(f"Overall Balanced Score: {score_range(overall_metrics):.1f}")
    
    # Find best balanced range
    print("\n" + "="*60)
    print("FINDING BEST BALANCED RANGE:")
    print("="*60)
    range_start, range_end, best_metrics = find_balanced_range(df, min_samples=15, step_size=0.02)
    
    print("\nBEST BALANCED RANGE RESULTS:")
    print("="*50)
    print(f"Optimal Range: [{range_start:.3f}, {range_end:.3f}] % Mg")
    print(f"Range Width: {range_end - range_start:.3f} % Mg")
    print(f"Balanced Score: {score_range(best_metrics):.1f}")
    print()
    print("METRICS FOR BEST BALANCED RANGE:")
    print(f"Number of Samples: {best_metrics['n_samples']}")
    print(f"Mean True Value: {best_metrics['mean_true']:.6f} % Mg")
    print(f"R²: {best_metrics['r2']:.6f}")
    print(f"MAE: {best_metrics['mae']:.6f}")
    print(f"RMSE: {best_metrics['rmse']:.6f}")
    print(f"RRMSE: {best_metrics['rrmse']:.2f}%")
    print(f"MAPE: {best_metrics['mape']:.2f}%")
    print(f"Within 20.5%: {best_metrics['within_20.5%']:.2f}%")
    print(f"Within 15%: {best_metrics['within_15%']:.2f}%")
    print(f"Within 10%: {best_metrics['within_10%']:.2f}%")
    
    # Analyze predefined ranges
    analyze_multiple_ranges(df)
    
    # Show some sample predictions in the best range
    mask = (df['ElementValue'] >= range_start) & (df['ElementValue'] <= range_end)
    best_range_samples = df[mask].sort_values('ElementValue')
    
    print(f"\nSAMPLE PREDICTIONS IN BEST RANGE [{range_start:.3f}, {range_end:.3f}]:")
    print("="*80)
    print(f"{'Sample ID':<25} {'True':<8} {'Predicted':<10} {'Error%':<8} {'Within 20.5%'}")
    print("-" * 80)
    
    for idx, row in best_range_samples.iterrows():
        error_pct = abs((row['ElementValue'] - row['PredictedValue']) / row['ElementValue']) * 100
        within_threshold = "✓" if error_pct <= 20.5 else "✗"
        print(f"{row['sampleId'][:25]:<25} {row['ElementValue']:<8.3f} {row['PredictedValue']:<10.3f} {error_pct:<8.1f} {within_threshold}")

if __name__ == "__main__":
    main()