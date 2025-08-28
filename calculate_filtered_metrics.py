#!/usr/bin/env python3
"""
Calculate regression metrics for AutoGluon predictions filtered to ElementValue between 0.2 and 0.5.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def calculate_filtered_metrics(predictions_file, min_value=0.2, max_value=0.5):
    """
    Calculate regression metrics for predictions filtered by ElementValue range.
    
    Args:
        predictions_file: Path to predictions CSV file
        min_value: Minimum ElementValue to include
        max_value: Maximum ElementValue to include
    
    Returns:
        Dictionary of metrics
    """
    # Load predictions
    df = pd.read_csv(predictions_file)
    
    print(f"\n=== Original Data ===")
    print(f"Total samples: {len(df)}")
    print(f"ElementValue range: {df['ElementValue'].min():.3f} - {df['ElementValue'].max():.3f}")
    
    # Filter by ElementValue range
    filtered_df = df[(df['ElementValue'] >= min_value) & (df['ElementValue'] <= max_value)]
    
    print(f"\n=== Filtered Data ({min_value} ≤ ElementValue ≤ {max_value}) ===")
    print(f"Filtered samples: {len(filtered_df)} ({len(filtered_df)/len(df)*100:.1f}% of total)")
    print(f"Filtered range: {filtered_df['ElementValue'].min():.3f} - {filtered_df['ElementValue'].max():.3f}")
    
    if len(filtered_df) == 0:
        print("No samples in the specified range!")
        return None
    
    # Extract true and predicted values
    y_true = filtered_df['ElementValue'].values
    y_pred = filtered_df['PredictedValue'].values
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # RRMSE (Relative Root Mean Squared Error)
    rrmse = (rmse / np.mean(y_true)) * 100
    
    # Within 20.5% accuracy
    percent_errors = np.abs((y_true - y_pred) / y_true) * 100
    within_20_5 = np.sum(percent_errors <= 20.5) / len(percent_errors) * 100
    
    metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'rrmse': rrmse,
        'within_20.5%': within_20_5
    }
    
    # Print results in the requested format
    print(f"\n--- MODEL TRAINING SUMMARY ---")
    print(f"{'strategy':<12} {'model_name':<10} {'r2':>6} {'rmse':>7} {'mae':>7} {'mape':>8} {'rrmse':>7} {'within_20.5%':>13}")
    print(f"{'full_context':<12} {'autogluon':<10} {r2:6.4f} {rmse:7.4f} {mae:7.4f} {mape:8.4f} {rrmse:7.4f} {within_20_5:13.4f}")
    
    # Additional analysis by concentration ranges
    print(f"\n=== Performance by Concentration Range (Filtered Data) ===")
    bins = [0.2, 0.25, 0.3, 0.35, 0.39, 0.45, 0.5]
    filtered_df['conc_bin'] = pd.cut(filtered_df['ElementValue'], bins=bins, include_lowest=True)
    
    print(f"{'Range':<15} {'N':<5} {'MAE':<8} {'MAPE%':<8} {'R²':<8}")
    print("-" * 50)
    
    for bin_range in filtered_df['conc_bin'].cat.categories:
        mask = filtered_df['conc_bin'] == bin_range
        if mask.sum() > 1:
            subset = filtered_df[mask]
            bin_mae = np.abs(subset['PredictedValue'] - subset['ElementValue']).mean()
            bin_mape = (np.abs(subset['PredictedValue'] - subset['ElementValue']) / subset['ElementValue']).mean() * 100
            bin_r2 = r2_score(subset['ElementValue'], subset['PredictedValue'])
            print(f"{str(bin_range):<15} {len(subset):<5} {bin_mae:<8.4f} {bin_mape:<8.2f} {bin_r2:<8.4f}")
    
    return metrics

if __name__ == "__main__":
    predictions_file = "/home/payanico/magnesium_pipeline/reports/predictions_raw-spectral_autogluon_raw-spectral_20250814_184731.csv"
    
    print("="*70)
    print("AutoGluon Predictions Analysis - Filtered to 0.2-0.5 Range")
    print("="*70)
    
    metrics = calculate_filtered_metrics(predictions_file, min_value=0.1, max_value=0.5)
    
    if metrics:
        print(f"\n=== Summary of Filtered Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric:<15}: {value:>10.4f}")