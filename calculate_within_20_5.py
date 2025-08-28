"""Calculate Within20.5% metric exactly as implemented in the pipeline"""
import pandas as pd
import numpy as np
from src.utils.helpers import calculate_regression_metrics

# Load predictions and reference data
predictions_df = pd.read_csv('reports/combo_6_8_final_test.csv')
ref_df = pd.read_excel('data/reference_data/Final_Lab_Data_Nico_New.xlsx')

# Merge to compare successful predictions only
merged = predictions_df.merge(ref_df, left_on='sampleId', right_on='Sample ID', how='inner')
successful = merged[merged['Status'] == 'Success']
successful = successful.dropna(subset=['PredictedValue', 'Nitrogen %'])

print('=== WITHIN 20.5% CALCULATION ===')
print(f'Total successful predictions: {len(successful)}')

if len(successful) > 0:
    y_true = successful['Nitrogen %'].values
    y_pred = successful['PredictedValue'].values
    
    # Calculate exactly as in helpers.py
    # First handle division by zero by replacing zeros with small values
    y_true_safe = np.where(y_true == 0, 1e-8, y_true)
    
    # Calculate relative errors
    relative_errors = np.abs((y_true - y_pred) / y_true_safe)
    
    # Calculate within 20.5% exactly as in pipeline
    within_20_5_percent = np.mean(relative_errors <= 0.205) * 100
    
    print(f'Within 20.5% (pipeline method): {within_20_5_percent:.4f}%')
    
    # Show details for verification
    print('\n=== DETAILED BREAKDOWN ===')
    
    # Create detailed comparison
    comparison = pd.DataFrame({
        'sampleId': successful['sampleId'].values,
        'y_true': y_true,
        'y_pred': y_pred,
        'absolute_error': np.abs(y_true - y_pred),
        'relative_error': relative_errors,
        'within_20_5': relative_errors <= 0.205
    })
    
    print(f'Samples within 20.5%: {(relative_errors <= 0.205).sum()}/{len(relative_errors)}')
    print(f'Percentage: {within_20_5_percent:.1f}%')
    
    # Show the first 10 samples
    print('\n=== FIRST 10 SAMPLES ===')
    display_df = comparison.head(10).copy()
    display_df['relative_error_pct'] = display_df['relative_error'] * 100
    print(display_df[['sampleId', 'y_true', 'y_pred', 'absolute_error', 'relative_error_pct', 'within_20_5']].to_string(index=False))
    
    # Compare with the calculate_regression_metrics function
    print('\n=== VERIFICATION USING calculate_regression_metrics ===')
    metrics = calculate_regression_metrics(y_true, y_pred)
    print(f'Within 20.5% from calculate_regression_metrics: {metrics["within_20.5%"]:.4f}%')
    
    # Show distribution of relative errors
    print('\n=== RELATIVE ERROR DISTRIBUTION ===')
    print(f'Min relative error: {relative_errors.min()*100:.2f}%')
    print(f'Max relative error: {relative_errors.max()*100:.2f}%')
    print(f'Mean relative error: {relative_errors.mean()*100:.2f}%')
    print(f'Median relative error: {np.median(relative_errors)*100:.2f}%')
    
    # Show worst performers
    print('\n=== WORST RELATIVE ERRORS ===')
    worst_df = comparison.nlargest(5, 'relative_error').copy()
    worst_df['relative_error_pct'] = worst_df['relative_error'] * 100
    print(worst_df[['sampleId', 'y_true', 'y_pred', 'relative_error_pct']].to_string(index=False))
    
else:
    print('No successful predictions to analyze!')