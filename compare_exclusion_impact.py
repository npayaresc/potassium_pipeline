#!/usr/bin/env python3
"""Compare model performance with and without suspicious sample exclusion."""

import subprocess
import re
import json
from datetime import datetime

def run_training(exclude_suspects=False):
    """Run training and capture metrics."""
    cmd = ["python", "main.py", "train", "--models", "ridge", "--strategy", "simple_only"]

    if exclude_suspects:
        cmd.extend(["--exclude-suspects", "reports/mislabel_analysis/suspicious_samples_min_confidence_2.csv"])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse output for metrics
    output = result.stdout + result.stderr

    # Extract sample counts
    samples_pattern = r"Train data \((\d+) samples\).*Test data \((\d+) samples\)"
    samples_match = re.search(samples_pattern, output)
    train_samples = int(samples_match.group(1)) if samples_match else 0
    test_samples = int(samples_match.group(2)) if samples_match else 0

    # Extract exclusion info
    excluded_count = 0
    exclusion_percentage = 0.0
    if exclude_suspects:
        excluded_pattern = r"Actually excluded: (\d+) samples"
        excluded_match = re.search(excluded_pattern, output)
        if excluded_match:
            excluded_count = int(excluded_match.group(1))

        percentage_pattern = r"Exclusion percentage: ([\d.]+)%"
        percentage_match = re.search(percentage_pattern, output)
        if percentage_match:
            exclusion_percentage = float(percentage_match.group(1))

    # Extract performance metrics
    r2_pattern = r"'r2': ([\d.]+)"
    rmse_pattern = r"'rmse': .*?\(([\d.]+)\)"
    mae_pattern = r"'mae': ([\d.]+)"
    mape_pattern = r"'mape': .*?\(([\d.]+)\)"

    r2 = float(re.search(r2_pattern, output).group(1)) if re.search(r2_pattern, output) else 0
    rmse = float(re.search(rmse_pattern, output).group(1)) if re.search(rmse_pattern, output) else 0
    mae = float(re.search(mae_pattern, output).group(1)) if re.search(mae_pattern, output) else 0
    mape = float(re.search(mape_pattern, output).group(1)) if re.search(mape_pattern, output) else 0

    return {
        'train_samples': train_samples,
        'test_samples': test_samples,
        'total_samples': train_samples + test_samples,
        'excluded_count': excluded_count,
        'exclusion_percentage': exclusion_percentage,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

def main():
    print("=" * 80)
    print("COMPARING MODEL PERFORMANCE WITH AND WITHOUT SUSPICIOUS SAMPLE EXCLUSION")
    print("=" * 80)
    print()

    # Run without exclusion
    print("1. Training WITHOUT exclusion...")
    print("-" * 40)
    without_exclusion = run_training(exclude_suspects=False)
    print()

    # Run with exclusion
    print("2. Training WITH exclusion...")
    print("-" * 40)
    with_exclusion = run_training(exclude_suspects=True)
    print()

    # Display comparison
    print("=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    print("\nSample Counts:")
    print(f"  Without exclusion: {without_exclusion['total_samples']} total ({without_exclusion['train_samples']} train, {without_exclusion['test_samples']} test)")
    print(f"  With exclusion:    {with_exclusion['total_samples']} total ({with_exclusion['train_samples']} train, {with_exclusion['test_samples']} test)")
    print(f"  Samples excluded:  {with_exclusion['excluded_count']} ({with_exclusion['exclusion_percentage']:.1f}%)")

    print("\nPerformance Metrics:")
    print(f"                    Without Exclusion    With Exclusion    Change")
    print(f"  R² Score:         {without_exclusion['r2']:.4f}             {with_exclusion['r2']:.4f}          {(with_exclusion['r2'] - without_exclusion['r2']):.4f} ({((with_exclusion['r2'] - without_exclusion['r2'])/without_exclusion['r2']*100):+.1f}%)")
    print(f"  RMSE:             {without_exclusion['rmse']:.4f}             {with_exclusion['rmse']:.4f}          {(with_exclusion['rmse'] - without_exclusion['rmse']):.4f} ({((with_exclusion['rmse'] - without_exclusion['rmse'])/without_exclusion['rmse']*100):+.1f}%)")
    print(f"  MAE:              {without_exclusion['mae']:.4f}             {with_exclusion['mae']:.4f}          {(with_exclusion['mae'] - without_exclusion['mae']):.4f} ({((with_exclusion['mae'] - without_exclusion['mae'])/without_exclusion['mae']*100):+.1f}%)")
    print(f"  MAPE:             {without_exclusion['mape']:.2f}            {with_exclusion['mape']:.2f}           {(with_exclusion['mape'] - without_exclusion['mape']):.2f} ({((with_exclusion['mape'] - without_exclusion['mape'])/without_exclusion['mape']*100):+.1f}%)")

    print("\nSummary:")
    if with_exclusion['r2'] > without_exclusion['r2']:
        print("✓ Excluding suspicious samples IMPROVED model performance!")
        print(f"  - R² improved by {((with_exclusion['r2'] - without_exclusion['r2'])/without_exclusion['r2']*100):.1f}%")
        print(f"  - RMSE reduced by {((without_exclusion['rmse'] - with_exclusion['rmse'])/without_exclusion['rmse']*100):.1f}%")
    else:
        print("⚠ Excluding suspicious samples did not improve performance.")
        print("  Consider reviewing the mislabel detection parameters.")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'without_exclusion': without_exclusion,
        'with_exclusion': with_exclusion,
        'improvement': {
            'r2_change': with_exclusion['r2'] - without_exclusion['r2'],
            'rmse_change': with_exclusion['rmse'] - without_exclusion['rmse'],
            'mae_change': with_exclusion['mae'] - without_exclusion['mae'],
            'mape_change': with_exclusion['mape'] - without_exclusion['mape']
        }
    }

    output_file = f"reports/exclusion_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()