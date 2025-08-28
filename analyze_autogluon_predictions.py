#!/usr/bin/env python3
"""Analyze AutoGluon predictions to understand concentration-dependent performance."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns

# Load predictions
df = pd.read_csv('/home/payanico/magnesium_pipeline/reports/predictions_full_context_autogluon_20250731_210301.csv')

# Calculate errors
df['error'] = df['PredictedValue'] - df['ElementValue']
df['abs_error'] = np.abs(df['error'])
df['percent_error'] = (df['abs_error'] / df['ElementValue']) * 100

# Create concentration bins
bins = [0.0, 0.25, 0.30, 0.35, 0.39, 0.45, 0.50]
df['conc_bin'] = pd.cut(df['ElementValue'], bins=bins, include_lowest=True)

# Analyze by concentration range
print("\n=== Performance Analysis by Concentration Range ===")
print(f"{'Range':<15} {'N':<5} {'MAE':<8} {'MAPE%':<8} {'R²':<8} {'Avg True':<10} {'Avg Pred':<10}")
print("-" * 80)

for bin_range in df['conc_bin'].cat.categories:
    mask = df['conc_bin'] == bin_range
    if mask.sum() > 0:
        subset = df[mask]
        mae = subset['abs_error'].mean()
        mape = subset['percent_error'].mean()
        r2 = r2_score(subset['ElementValue'], subset['PredictedValue']) if len(subset) > 1 else np.nan
        avg_true = subset['ElementValue'].mean()
        avg_pred = subset['PredictedValue'].mean()
        
        print(f"{str(bin_range):<15} {len(subset):<5} {mae:<8.4f} {mape:<8.2f} {r2:<8.4f} {avg_true:<10.4f} {avg_pred:<10.4f}")

# Special analysis for high concentrations
high_conc = df[df['ElementValue'] >= 0.39]
print(f"\n=== Detailed Analysis for High Concentrations (≥ 0.39) ===")
print(f"Number of samples: {len(high_conc)}")
print(f"True value range: {high_conc['ElementValue'].min():.3f} - {high_conc['ElementValue'].max():.3f}")
print(f"Predicted range: {high_conc['PredictedValue'].min():.3f} - {high_conc['PredictedValue'].max():.3f}")
print(f"Average true value: {high_conc['ElementValue'].mean():.3f}")
print(f"Average predicted: {high_conc['PredictedValue'].mean():.3f}")
print(f"Systematic bias: {high_conc['error'].mean():.3f}")

# Identify worst predictions
worst_predictions = df.nlargest(10, 'abs_error')
print(f"\n=== 10 Worst Predictions ===")
print(worst_predictions[['sampleId', 'ElementValue', 'PredictedValue', 'error', 'percent_error']].to_string())

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Actual vs Predicted scatter
ax = axes[0, 0]
scatter = ax.scatter(df['ElementValue'], df['PredictedValue'], c=df['ElementValue'], 
                     cmap='viridis', alpha=0.6, s=50)
ax.plot([0.2, 0.5], [0.2, 0.5], 'r--', label='Perfect prediction')
ax.set_xlabel('True Magnesium %')
ax.set_ylabel('Predicted Magnesium %')
ax.set_title('Predictions vs True Values')
ax.legend()
plt.colorbar(scatter, ax=ax, label='True Value')

# Highlight high concentration region
ax.axvline(x=0.39, color='red', linestyle=':', alpha=0.5, label='High conc threshold')
ax.fill_betweenx([0.2, 0.5], 0.39, 0.5, alpha=0.1, color='red')

# 2. Error vs True Value
ax = axes[0, 1]
ax.scatter(df['ElementValue'], df['error'], alpha=0.6)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('True Magnesium %')
ax.set_ylabel('Prediction Error')
ax.set_title('Prediction Error vs True Value')
ax.axvline(x=0.39, color='red', linestyle=':', alpha=0.5)

# 3. Box plot by concentration bins
ax = axes[1, 0]
df.boxplot(column='percent_error', by='conc_bin', ax=ax)
ax.set_xlabel('Concentration Range')
ax.set_ylabel('Absolute Percent Error')
ax.set_title('Error Distribution by Concentration Range')
plt.sca(ax)
plt.xticks(rotation=45)

# 4. Histogram of predictions
ax = axes[1, 1]
ax.hist(df['ElementValue'], bins=20, alpha=0.5, label='True values', color='blue')
ax.hist(df['PredictedValue'], bins=20, alpha=0.5, label='Predictions', color='orange')
ax.axvline(x=0.39, color='red', linestyle=':', alpha=0.7, label='High conc threshold')
ax.set_xlabel('Magnesium %')
ax.set_ylabel('Count')
ax.set_title('Distribution of True vs Predicted Values')
ax.legend()

plt.tight_layout()
plt.savefig('autogluon_prediction_analysis.png', dpi=150)
print("\nPlot saved to: autogluon_prediction_analysis.png")

# Additional analysis for extreme values
print(f"\n=== Prediction Range Compression ===")
print(f"True values std: {df['ElementValue'].std():.4f}")
print(f"Predictions std: {df['PredictedValue'].std():.4f}")
print(f"Std ratio (pred/true): {df['PredictedValue'].std() / df['ElementValue'].std():.4f}")