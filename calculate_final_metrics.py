"""Calculate final metrics on combo_6_8 predictions"""
import pandas as pd
import numpy as np
from src.utils.helpers import calculate_regression_metrics

# Load predictions and reference data
predictions_df = pd.read_csv('reports/combo_6_8_final_test.csv')
ref_df = pd.read_excel('data/reference_data/Final_Lab_Data_Nico_New.xlsx')

print('=== PREDICTION RESULTS SUMMARY ===')
print(f'Total samples processed: {len(predictions_df)}')
success_count = (predictions_df['Status'] == 'Success').sum()
fail_count = len(predictions_df) - success_count
print(f'Successful predictions: {success_count}')
print(f'Failed predictions: {fail_count}')

# Merge to compare successful predictions only
merged = predictions_df.merge(ref_df, left_on='sampleId', right_on='Sample ID', how='inner')
successful = merged[merged['Status'] == 'Success']
successful = successful.dropna(subset=['PredictedValue', 'Nitrogen %'])

print(f'\nMatched with reference data: {len(successful)} samples')

if len(successful) > 0:
    # Calculate metrics
    y_true = successful['Nitrogen %'].values
    y_pred = successful['PredictedValue'].values
    
    metrics = calculate_regression_metrics(y_true, y_pred)
    
    print('\n=== PREDICTION METRICS ===')
    for name, value in metrics.items():
        print(f'{name.upper()}: {value:.4f}')
    
    print(f'\n=== VALUE RANGES ===')
    print(f'Predicted values: min={y_pred.min():.2f}, max={y_pred.max():.2f}, mean={y_pred.mean():.2f}')
    print(f'Actual values: min={y_true.min():.2f}, max={y_true.max():.2f}, mean={y_true.mean():.2f}')
    
    print(f'\n=== SAMPLE COMPARISONS (first 20) ===')
    comparison = successful[['sampleId', 'PredictedValue', 'Nitrogen %']].head(20)
    comparison['Error'] = comparison['PredictedValue'] - comparison['Nitrogen %']
    print(comparison.to_string(index=False))
    
    print(f'\n=== ERROR STATISTICS ===')
    errors = y_pred - y_true
    print(f'Mean error: {errors.mean():.4f}')
    print(f'Std error: {errors.std():.4f}')
    print(f'Min error: {errors.min():.4f}')
    print(f'Max error: {errors.max():.4f}')
    
    # Check distribution of errors
    print(f'\n=== ERROR DISTRIBUTION ===')
    print(f'Errors within ±0.5: {(np.abs(errors) <= 0.5).sum()}/{len(errors)} ({(np.abs(errors) <= 0.5).mean()*100:.1f}%)')
    print(f'Errors within ±1.0: {(np.abs(errors) <= 1.0).sum()}/{len(errors)} ({(np.abs(errors) <= 1.0).mean()*100:.1f}%)')
    print(f'Errors within ±1.5: {(np.abs(errors) <= 1.5).sum()}/{len(errors)} ({(np.abs(errors) <= 1.5).mean()*100:.1f}%)')
    
else:
    print('No successful predictions to analyze!')

# Save the successful predictions with errors for analysis
if len(successful) > 0:
    analysis_df = successful[['sampleId', 'PredictedValue', 'Nitrogen %']].copy()
    analysis_df['Error'] = analysis_df['PredictedValue'] - analysis_df['Nitrogen %']
    analysis_df['AbsError'] = analysis_df['Error'].abs()
    analysis_df = analysis_df.sort_values('AbsError', ascending=False)
    
    print(f'\n=== WORST 10 PREDICTIONS ===')
    print(analysis_df[['sampleId', 'PredictedValue', 'Nitrogen %', 'Error', 'AbsError']].head(10).to_string(index=False))
    
    print(f'\n=== BEST 10 PREDICTIONS ===')
    print(analysis_df[['sampleId', 'PredictedValue', 'Nitrogen %', 'Error', 'AbsError']].tail(10).to_string(index=False))