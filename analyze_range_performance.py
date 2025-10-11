#!/usr/bin/env python3
"""Analyze model performance within the configured 0.2-5.3% range."""

import pandas as pd
import numpy as np

# Read XGBoost predictions (best model)
df = pd.read_csv('reports/predictions_K_only_xgboost_20251005_210418.csv')

# Calculate errors
df['error'] = df['PredictedValue'] - df['ElementValue']
df['abs_error'] = np.abs(df['error'])
df['pct_error'] = 100 * df['abs_error'] / df['ElementValue']

# Define concentration ranges WITHIN TARGET 0.2-5.3%
ranges = [
    (0.2, 1.0, 'Low (Within Target)'),
    (1.0, 2.0, 'Medium-Low (Within Target)'),
    (2.0, 3.0, 'Medium (Within Target)'),
    (3.0, 4.0, 'High (Within Target)'),
    (4.0, 5.3, 'Very High (Within Target)')
]

print('=' * 90)
print('POTASSIUM ANALYSIS - CONFIGURED RANGE: 0.2% to 5.3%')
print('Model: XGBoost K_only (R² = 0.623)')
print('=' * 90)

for min_val, max_val, label in ranges:
    mask = (df['ElementValue'] >= min_val) & (df['ElementValue'] < max_val)
    subset = df[mask]

    if len(subset) == 0:
        continue

    bias = subset['error'].mean()
    bias_direction = "over" if bias > 0 else "under"

    print(f'\n{label}: {min_val:.1f}-{max_val:.1f}%')
    print(f'  Samples: {len(subset):3d} ({100*len(subset)/len(df):.1f}% of test set)')
    print(f'  Mean True: {subset["ElementValue"].mean():5.3f}%')
    print(f'  Mean Pred: {subset["PredictedValue"].mean():5.3f}%')
    print(f'  Bias: {bias:+6.3f} ({bias_direction}-predicting)')
    print(f'  RMSE: {np.sqrt(np.mean(subset["error"]**2)):5.3f}')
    print(f'  MAE: {subset["abs_error"].mean():5.3f}')
    print(f'  MAPE: {subset["pct_error"].mean():5.1f}%')
    within_20 = (subset['pct_error'] <= 20).sum()
    within_10 = (subset['pct_error'] <= 10).sum()
    print(f'  Within ±20%: {within_20:3d}/{len(subset):3d} ({within_20 / len(subset) * 100:4.1f}%)')
    print(f'  Within ±10%: {within_10:3d}/{len(subset):3d} ({within_10 / len(subset) * 100:4.1f}%)')

# Overall statistics
print('\n' + '='*90)
print('OVERALL STATISTICS (0.2% - 5.3% range)')
print('='*90)
print(f'Total samples: {len(df)}')
print(f'Concentration range: {df["ElementValue"].min():.3f}% - {df["ElementValue"].max():.3f}%')
print(f'Mean concentration: {df["ElementValue"].mean():.3f}%')
print(f'RMSE: {np.sqrt(np.mean(df["error"]**2)):.3f}')
print(f'MAE: {df["abs_error"].mean():.3f}')
print(f'MAPE: {df["pct_error"].mean():.1f}%')
print(f'Bias: {df["error"].mean():+.3f}')
within_20_total = (df['pct_error'] <= 20).sum()
within_10_total = (df['pct_error'] <= 10).sum()
print(f'Within ±20%: {within_20_total}/{len(df)} ({within_20_total / len(df) * 100:.1f}%)')
print(f'Within ±10%: {within_10_total}/{len(df)} ({within_10_total / len(df) * 100:.1f}%)')
