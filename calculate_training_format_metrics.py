#!/usr/bin/env python3
"""
Calculate regression metrics for AutoGluon predictions using the same format as training.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to Python path to import the utils
sys.path.append('/home/payanico/nitrogen_pipeline')

from src.utils.helpers import calculate_regression_metrics

def main():
    # Load the predictions file
    predictions_file = '/home/payanico/nitrogen_pipeline/reports/predictions_simple_only_autogluon_20250722_214846.csv'
    
    print("Loading predictions file...")
    df = pd.read_csv(predictions_file)
    
    print(f"Original dataset: {len(df)} samples")
    print(f"ElementValue range: {df['ElementValue'].min():.3f} - {df['ElementValue'].max():.3f}")
    
    # Filter for ElementValue between 4.2 and 5.2
    filtered_df = df[(df['ElementValue'] >= 4.2) & (df['ElementValue'] <= 5.2)]
    
    print(f"Filtered dataset (ElementValue 4.2-5.2): {len(filtered_df)} samples")
    
    if len(filtered_df) == 0:
        print("No samples found in the specified range!")
        return
    
    # Extract actual and predicted values
    y_true = filtered_df['ElementValue'].values
    y_pred = filtered_df['PredictedValue'].values
    
    # Use the same metrics calculation function as training
    metrics = calculate_regression_metrics(y_true, y_pred)
    
    # Format output exactly like training summary
    print("\n--- MODEL EVALUATION SUMMARY ---")
    print("Strategy: simple_only")
    print("Model: autogluon")
    print("Filter: ElementValue 4.2-5.2")
    print(f"Samples: {len(filtered_df)}")
    
    # Display metrics in training format (4 decimal places)
    display_cols = ['r2', 'rmse', 'mae', 'mape', 'rrmse', 'within_20.5%']
    
    print("\nPerformance Metrics:")
    for col in display_cols:
        if col in metrics:
            print(f"{col}: {metrics[col]:.4f}")
    
    print("----------------------------\n")
    
    # Also create a summary DataFrame like in training
    results_data = {
        'strategy': 'simple_only',
        'model_name': 'autogluon',
        'filter': 'ElementValue_4.2-5.2',
        'n_samples': len(filtered_df),
        **metrics
    }
    
    results_df = pd.DataFrame([results_data])
    
    # Display in tabular format like training summary
    print("Detailed Results Table:")
    print(results_df[['strategy', 'model_name', 'filter', 'n_samples'] + display_cols].to_string(index=False, float_format="%.4f"))

if __name__ == "__main__":
    main()