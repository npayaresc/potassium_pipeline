#!/usr/bin/env python3
"""
Systematic Bias Analysis for Nitrogen Prediction Models

This script calculates comprehensive metrics to detect systematic bias patterns
in model predictions across different concentration ranges.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, List, Tuple, Optional
import glob

def calculate_systematic_bias_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive systematic bias metrics.
    
    Args:
        y_true: True concentration values
        y_pred: Predicted concentration values
        
    Returns:
        Dictionary with bias metrics
    """
    residuals = y_pred - y_true
    
    metrics = {}
    
    # 1. Overall bias metrics
    metrics['mean_bias'] = np.mean(residuals)  # Overall systematic bias
    metrics['median_bias'] = np.median(residuals)  # Robust bias measure
    metrics['bias_std'] = np.std(residuals)  # Variability of bias
    
    # 2. Range-specific bias (quintiles)
    quintiles = np.percentile(y_true, [20, 40, 60, 80])
    
    for i, (lower, upper) in enumerate(zip([0] + list(quintiles), list(quintiles) + [100])):
        if i == 0:
            mask = y_true <= quintiles[0]
            range_name = f'Q1_low_{quintiles[0]:.1f}'
        elif i == 4:
            mask = y_true > quintiles[3]
            range_name = f'Q5_high_{quintiles[3]:.1f}'
        else:
            mask = (y_true > quintiles[i-1]) & (y_true <= quintiles[i])
            range_name = f'Q{i+1}_{quintiles[i-1]:.1f}_{quintiles[i]:.1f}'
        
        if np.sum(mask) > 0:
            metrics[f'bias_{range_name}'] = np.mean(residuals[mask])
            metrics[f'mae_{range_name}'] = np.mean(np.abs(residuals[mask]))
            metrics[f'count_{range_name}'] = np.sum(mask)
    
    # 3. Concentration-dependent bias trend
    # Fit linear regression: residuals = a * y_true + b
    if len(y_true) > 5:
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, residuals)
        metrics['bias_slope'] = slope  # How bias changes with concentration
        metrics['bias_intercept'] = intercept
        metrics['bias_trend_r2'] = r_value**2
        metrics['bias_trend_p_value'] = p_value
    
    # 4. Extreme value bias
    p5, p95 = np.percentile(y_true, [5, 95])
    
    # Low extreme bias
    low_mask = y_true <= p5
    if np.sum(low_mask) > 0:
        metrics['bias_low_extreme'] = np.mean(residuals[low_mask])
        metrics['mae_low_extreme'] = np.mean(np.abs(residuals[low_mask]))
    
    # High extreme bias
    high_mask = y_true >= p95
    if np.sum(high_mask) > 0:
        metrics['bias_high_extreme'] = np.mean(residuals[high_mask])
        metrics['mae_high_extreme'] = np.mean(np.abs(residuals[high_mask]))
    
    # 5. Calibration metrics
    # Perfect calibration would have slope=1, intercept=0 for y_pred vs y_true
    if len(y_true) > 5:
        cal_slope, cal_intercept, cal_r, cal_p, cal_se = stats.linregress(y_true, y_pred)
        metrics['calibration_slope'] = cal_slope  # Should be ~1.0
        metrics['calibration_intercept'] = cal_intercept  # Should be ~0.0
        metrics['calibration_r2'] = cal_r**2
    
    # 6. Proportional bias (relative to concentration level)
    # Bias as percentage of true value
    relative_bias = residuals / y_true * 100
    metrics['mean_relative_bias_pct'] = np.mean(relative_bias)
    metrics['median_relative_bias_pct'] = np.median(relative_bias)
    
    # 7. Heteroscedasticity (bias variance changes with concentration)
    # Split into low/high concentration groups
    median_conc = np.median(y_true)
    low_conc_mask = y_true <= median_conc
    high_conc_mask = y_true > median_conc
    
    if np.sum(low_conc_mask) > 0 and np.sum(high_conc_mask) > 0:
        low_var = np.var(residuals[low_conc_mask])
        high_var = np.var(residuals[high_conc_mask])
        # F-test for equal variances
        f_stat = max(low_var, high_var) / min(low_var, high_var)
        metrics['variance_ratio_low_high'] = f_stat
        metrics['low_conc_std'] = np.sqrt(low_var)
        metrics['high_conc_std'] = np.sqrt(high_var)
    
    # 8. Standard regression metrics for context
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mape'] = np.mean(np.abs(relative_bias))
    
    return metrics

def create_bias_visualization(y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str, save_path: Optional[Path] = None) -> None:
    """
    Create comprehensive bias visualization plots.
    """
    residuals = y_pred - y_true
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Systematic Bias Analysis: {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Calibration plot (predicted vs actual)
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=30)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect calibration')
    
    # Add calibration line
    slope, intercept, r_val, _, _ = stats.linregress(y_true, y_pred)
    line_pred = slope * y_true + intercept
    axes[0, 0].plot(y_true, line_pred, 'g-', lw=2, 
                   label=f'Calibration line (slope={slope:.3f})')
    
    axes[0, 0].set_xlabel('True Concentration (%)')
    axes[0, 0].set_ylabel('Predicted Concentration (%)')
    axes[0, 0].set_title(f'Calibration Plot (R²={r_val**2:.3f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals vs predicted (heteroscedasticity check)
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=30)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    
    # Add trend line for residuals
    res_slope, res_intercept, _, _, _ = stats.linregress(y_pred, residuals)
    trend_line = res_slope * y_pred + res_intercept
    axes[0, 1].plot(y_pred, trend_line, 'g-', lw=2, 
                   label=f'Trend (slope={res_slope:.4f})')
    
    axes[0, 1].set_xlabel('Predicted Concentration (%)')
    axes[0, 1].set_ylabel('Residuals (Pred - True)')
    axes[0, 1].set_title('Residuals vs Predicted')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals vs true values (concentration-dependent bias)
    axes[0, 2].scatter(y_true, residuals, alpha=0.6, s=30)
    axes[0, 2].axhline(y=0, color='r', linestyle='--', lw=2)
    
    # Add trend line
    true_slope, true_intercept, _, _, _ = stats.linregress(y_true, residuals)
    true_trend = true_slope * y_true + true_intercept
    axes[0, 2].plot(y_true, true_trend, 'g-', lw=2, 
                   label=f'Bias trend (slope={true_slope:.4f})')
    
    axes[0, 2].set_xlabel('True Concentration (%)')
    axes[0, 2].set_ylabel('Residuals (Pred - True)')
    axes[0, 2].set_title('Concentration-Dependent Bias')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Residual distribution
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero bias')
    axes[1, 0].axvline(x=np.mean(residuals), color='g', linestyle='-', lw=2, 
                      label=f'Mean bias ({np.mean(residuals):.3f})')
    axes[1, 0].set_xlabel('Residuals (Pred - True)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Q-Q plot (normality check)
    from scipy.stats import probplot
    probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Bias by concentration ranges
    quintiles = np.percentile(y_true, [20, 40, 60, 80])
    range_labels = ['Q1 (Low)', 'Q2', 'Q3 (Med)', 'Q4', 'Q5 (High)']
    range_biases = []
    range_counts = []
    
    for i in range(5):
        if i == 0:
            mask = y_true <= quintiles[0]
        elif i == 4:
            mask = y_true > quintiles[3]
        else:
            mask = (y_true > quintiles[i-1]) & (y_true <= quintiles[i])
        
        if np.sum(mask) > 0:
            range_biases.append(np.mean(residuals[mask]))
            range_counts.append(np.sum(mask))
        else:
            range_biases.append(0)
            range_counts.append(0)
    
    bars = axes[1, 2].bar(range_labels, range_biases, alpha=0.7, 
                         color=['red' if b < -0.1 else 'blue' if b > 0.1 else 'green' 
                               for b in range_biases])
    axes[1, 2].axhline(y=0, color='black', linestyle='-', lw=1)
    axes[1, 2].set_xlabel('Concentration Range')
    axes[1, 2].set_ylabel('Mean Bias')
    axes[1, 2].set_title('Bias by Concentration Quintiles')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, range_counts):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01 * np.sign(height),
                       f'n={count}', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bias analysis plot saved to: {save_path}")
    
    plt.show()

def analyze_model_predictions(predictions_file: str, model_name: str = None) -> Dict[str, float]:
    """
    Analyze systematic bias from a predictions CSV file.
    
    Args:
        predictions_file: Path to CSV file with columns ['ActualValue', 'PredictedValue']
        model_name: Name of the model for visualization
        
    Returns:
        Dictionary with bias metrics
    """
    # Load predictions
    df = pd.read_csv(predictions_file)
    
    # Try different column name possibilities
    actual_col = None
    pred_col = None
    
    for col in df.columns:
        if 'element' in col.lower() or 'true' in col.lower() or 'nitrogen' in col.lower():
            actual_col = col
        elif 'predict' in col.lower() or 'pred' in col.lower():
            pred_col = col
    
    if actual_col is None or pred_col is None:
        print("Available columns:", df.columns.tolist())
        raise ValueError("Could not find actual and predicted value columns. "
                        "Expected columns containing 'actual'/'true' and 'predicted'/'pred'")
    
    y_true = df[actual_col].values
    y_pred = df[pred_col].values
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    print(f"Analyzing {len(y_true)} predictions from {predictions_file}")
    print(f"Actual range: {y_true.min():.2f} - {y_true.max():.2f}")
    print(f"Predicted range: {y_pred.min():.2f} - {y_pred.max():.2f}")
    
    # Calculate bias metrics
    metrics = calculate_systematic_bias_metrics(y_true, y_pred)
    
    # Create visualization
    if model_name is None:
        model_name = Path(predictions_file).stem
    
    save_path = Path(predictions_file).parent / f"bias_analysis_{model_name}.png"
    create_bias_visualization(y_true, y_pred, model_name, save_path)
    
    return metrics

def print_bias_interpretation(metrics: Dict[str, float]) -> None:
    """
    Print interpretation of bias metrics.
    """
    print("\n" + "="*60)
    print("SYSTEMATIC BIAS ANALYSIS RESULTS")
    print("="*60)
    
    # Overall bias assessment
    mean_bias = metrics.get('mean_bias', 0)
    bias_slope = metrics.get('bias_slope', 0)
    cal_slope = metrics.get('calibration_slope', 1)
    
    print(f"\n📊 OVERALL BIAS ASSESSMENT:")
    print(f"   Mean Bias: {mean_bias:+.4f} ({'✓ Good' if abs(mean_bias) < 0.1 else '⚠ Moderate' if abs(mean_bias) < 0.2 else '✗ High'})")
    print(f"   Bias Trend Slope: {bias_slope:+.4f} ({'✓ No trend' if abs(bias_slope) < 0.02 else '⚠ Slight trend' if abs(bias_slope) < 0.05 else '✗ Strong trend'})")
    print(f"   Calibration Slope: {cal_slope:.4f} ({'✓ Well calibrated' if 0.95 <= cal_slope <= 1.05 else '⚠ Some miscalibration' if 0.9 <= cal_slope <= 1.1 else '✗ Poor calibration'})")
    
    # Range-specific bias
    print(f"\n📈 CONCENTRATION RANGE BIAS:")
    ranges = ['Q1_low', 'Q2_', 'Q3_', 'Q4_', 'Q5_high']
    for range_key in ranges:
        bias_key = [k for k in metrics.keys() if k.startswith(f'bias_{range_key}')]
        if bias_key:
            bias_val = metrics[bias_key[0]]
            range_name = bias_key[0].replace('bias_', '').replace('_', ' ').title()
            status = '✓ Good' if abs(bias_val) < 0.1 else '⚠ Moderate' if abs(bias_val) < 0.2 else '✗ High'
            print(f"   {range_name}: {bias_val:+.4f} ({status})")
    
    # Extreme values
    low_bias = metrics.get('bias_low_extreme')
    high_bias = metrics.get('bias_high_extreme')
    if low_bias is not None and high_bias is not None:
        print(f"\n🎯 EXTREME VALUE BIAS:")
        print(f"   Low Extreme (≤P5): {low_bias:+.4f} ({'✓ Good' if abs(low_bias) < 0.15 else '⚠ Moderate' if abs(low_bias) < 0.3 else '✗ High'})")
        print(f"   High Extreme (≥P95): {high_bias:+.4f} ({'✓ Good' if abs(high_bias) < 0.15 else '⚠ Moderate' if abs(high_bias) < 0.3 else '✗ High'})")
    
    # Recommendations
    print(f"\n💡 CALIBRATION RECOMMENDATIONS:")
    
    needs_calibration = False
    
    if abs(mean_bias) > 0.1:
        print(f"   • Apply global bias correction: subtract {mean_bias:.4f}")
        needs_calibration = True
    
    if abs(bias_slope) > 0.02:
        if bias_slope > 0:
            print(f"   • Model under-predicts at low concentrations, over-predicts at high")
            print(f"   • Consider: pred_calibrated = pred * {1-abs(bias_slope):.3f}")
        else:
            print(f"   • Model over-predicts at low concentrations, under-predicts at high")
            print(f"   • Consider: pred_calibrated = pred * {1+abs(bias_slope):.3f}")
        needs_calibration = True
    
    if not (0.95 <= cal_slope <= 1.05):
        print(f"   • Poor overall calibration: pred_calibrated = pred / {cal_slope:.3f}")
        needs_calibration = True
    
    # Check range-specific issues
    range_issues = []
    for range_key in ranges:
        bias_key = [k for k in metrics.keys() if k.startswith(f'bias_{range_key}')]
        if bias_key and abs(metrics[bias_key[0]]) > 0.15:
            range_issues.append((range_key, metrics[bias_key[0]]))
    
    if range_issues:
        print(f"   • Range-specific calibration needed:")
        for range_name, bias in range_issues:
            print(f"     - {range_name}: adjust by {-bias:+.3f}")
        needs_calibration = True
    
    if not needs_calibration:
        print(f"   ✓ Model is well calibrated - minimal post-calibration needed!")
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"   R²: {metrics.get('r2', 0):.4f}")
    print(f"   RMSE: {metrics.get('rmse', 0):.4f}")
    print(f"   MAE: {metrics.get('mae', 0):.4f}")
    print(f"   MAPE: {metrics.get('mape', 0):.2f}%")
    
    print("="*60)

def main():
    """
    Main function to analyze all prediction files in the reports directory.
    """
    import argparse
    # parser = argparse.ArgumentParser(description='Analyze systematic bias in model predictions')
    # parser.add_argument('--file', '-f', type=str, help='Specific prediction file to analyze')
    # parser.add_argument('--dir', '-d', type=str, default='reports', help='Directory containing prediction files')
    # parser.add_argument('--pattern', '-p', type=str, default='predictions_*.csv', help='File pattern to match')
    
    # args = parser.parse_args()
    args = argparse.Namespace()
    args.file = "/home/payanico/magnesium_pipeline/reports/predictions_simple_only_autogluon_20250803_175608.csv"
    
    if args.file:
        # Analyze single file
        metrics = analyze_model_predictions(args.file)
        print_bias_interpretation(metrics)
    else:
        # Analyze all files matching pattern
        reports_dir = Path(args.dir)
        pattern = args.pattern
        
        prediction_files = list(reports_dir.glob(pattern))
        
        if not prediction_files:
            print(f"No files found matching pattern '{pattern}' in directory '{reports_dir}'")
            print("Available files:")
            for f in reports_dir.glob("*.csv"):
                print(f"  {f.name}")
            return
        
        print(f"Found {len(prediction_files)} prediction files to analyze...\n")
        
        all_metrics = {}
        
        for pred_file in prediction_files:
            print(f"\n{'='*60}")
            print(f"ANALYZING: {pred_file.name}")
            print(f"{'='*60}")
            
            try:
                metrics = analyze_model_predictions(str(pred_file))
                all_metrics[pred_file.stem] = metrics
                print_bias_interpretation(metrics)
                
            except Exception as e:
                print(f"Error analyzing {pred_file.name}: {e}")
                continue
        
        # Summary comparison
        if len(all_metrics) > 1:
            print(f"\n{'='*60}")
            print("MODEL COMPARISON SUMMARY")
            print(f"{'='*60}")
            
            comparison_df = pd.DataFrame(all_metrics).T
            key_metrics = ['mean_bias', 'bias_slope', 'calibration_slope', 'r2', 'rmse']
            
            for metric in key_metrics:
                if metric in comparison_df.columns:
                    print(f"\n{metric.upper().replace('_', ' ')}:")
                    sorted_models = comparison_df[metric].sort_values()
                    for model, value in sorted_models.items():
                        print(f"  {model}: {value:.4f}")

if __name__ == "__main__":
    main()