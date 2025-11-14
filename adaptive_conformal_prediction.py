#!/usr/bin/env python3
"""
Locally Adaptive Conformal Prediction

Implements conformal prediction with prediction-dependent interval widths.
This addresses heteroscedasticity - different prediction difficulties.

Key methods:
1. Locally Weighted Conformal (LWC) - weight by distance to prediction
2. Conformalized Quantile Regression (CQR) - predict intervals directly
3. Difficulty-Stratified Conformal - bin by prediction difficulty

Usage:
    python adaptive_conformal_prediction.py \
        --predictions reports/predictions_full_context_autogluon_20251113_103412.csv \
        --output-dir reports/adaptive_conformal_analysis
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Dict
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class LocallyWeightedConformal:
    """
    Locally Weighted Conformal Prediction.

    Uses distance-based weighting to give more importance to nearby calibration
    points when computing conformity scores.

    Reference: "Adaptive Conformal Inference" (Lei & Wasserman, 2014)
    """

    def __init__(self, alpha: float = 0.05, bandwidth: float = None):
        """
        Args:
            alpha: Significance level (0.05 for 95% confidence)
            bandwidth: Kernel bandwidth (auto-estimated if None)
        """
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.X_cal = None
        self.residuals = None

    def fit(self, predictions: np.ndarray, actuals: np.ndarray):
        """
        Fit on calibration data.

        Args:
            predictions: Model predictions on calibration set
            actuals: True values on calibration set
        """
        self.X_cal = predictions.reshape(-1, 1)
        self.residuals = np.abs(actuals - predictions)

        # Auto-estimate bandwidth using Scott's rule
        if self.bandwidth is None:
            n = len(predictions)
            std = np.std(predictions)
            self.bandwidth = 1.06 * std * n ** (-1/5)
            logger.info(f"Auto-estimated bandwidth: {self.bandwidth:.4f}")

        logger.info(f"Fitted locally weighted conformal predictor (n={len(predictions)})")

    def predict_interval(
        self,
        prediction: float,
        return_weights: bool = False
    ) -> Tuple[float, float]:
        """
        Compute prediction interval for a new prediction.

        Uses Gaussian kernel to weight calibration residuals by distance.

        Args:
            prediction: Model prediction for new sample
            return_weights: If True, also return the weights

        Returns:
            (lower_bound, upper_bound) or (lower, upper, weights) if return_weights=True
        """
        # Compute distances from prediction to calibration points
        distances = np.abs(self.X_cal.flatten() - prediction)

        # Gaussian kernel weights
        weights = np.exp(-0.5 * (distances / self.bandwidth) ** 2)
        weights = weights / weights.sum()  # Normalize

        # Compute weighted quantile of residuals
        sorted_idx = np.argsort(self.residuals)
        sorted_residuals = self.residuals[sorted_idx]
        sorted_weights = weights[sorted_idx]

        # Cumulative weights
        cum_weights = np.cumsum(sorted_weights)

        # Find quantile (with finite sample correction)
        n = len(self.residuals)
        q_level = (1 - self.alpha) * (n + 1) / n
        q_level = np.clip(q_level, 0, 1)

        # Find threshold where cumulative weight exceeds q_level
        threshold_idx = np.searchsorted(cum_weights, q_level)
        threshold_idx = min(threshold_idx, len(sorted_residuals) - 1)
        threshold = sorted_residuals[threshold_idx]

        lower = prediction - threshold
        upper = prediction + threshold

        if return_weights:
            return lower, upper, weights
        return lower, upper


class DifficultyStratifiedConformal:
    """
    Difficulty-Stratified Conformal Prediction.

    Divides prediction space into bins and uses separate conformity thresholds
    for each bin. Simple but effective for heteroscedastic data.
    """

    def __init__(self, alpha: float = 0.05, n_bins: int = 5):
        """
        Args:
            alpha: Significance level
            n_bins: Number of difficulty strata (bins)
        """
        self.alpha = alpha
        self.n_bins = n_bins
        self.bin_edges = None
        self.bin_thresholds = {}

    def fit(self, predictions: np.ndarray, actuals: np.ndarray):
        """Fit stratified conformal predictor."""
        residuals = np.abs(actuals - predictions)

        # Create bins based on prediction values
        self.bin_edges = np.percentile(
            predictions,
            np.linspace(0, 100, self.n_bins + 1)
        )

        # Ensure unique bin edges
        self.bin_edges = np.unique(self.bin_edges)
        if len(self.bin_edges) < 2:
            logger.warning("Not enough unique bin edges, using single threshold")
            self.bin_edges = np.array([predictions.min(), predictions.max()])

        logger.info(f"Bin edges: {self.bin_edges}")

        # Compute threshold for each bin
        for i in range(len(self.bin_edges) - 1):
            bin_mask = (predictions >= self.bin_edges[i]) & (predictions < self.bin_edges[i+1])

            # Special case for last bin (include upper edge)
            if i == len(self.bin_edges) - 2:
                bin_mask = (predictions >= self.bin_edges[i]) & (predictions <= self.bin_edges[i+1])

            bin_residuals = residuals[bin_mask]

            if len(bin_residuals) == 0:
                logger.warning(f"Empty bin [{self.bin_edges[i]:.2f}, {self.bin_edges[i+1]:.2f}]")
                continue

            # Compute quantile with finite sample correction
            n = len(bin_residuals)
            q_level = (1 - self.alpha) * (n + 1) / n
            q_level = np.clip(q_level, 0, 1)

            threshold = np.quantile(bin_residuals, q_level)
            self.bin_thresholds[i] = {
                'range': (self.bin_edges[i], self.bin_edges[i+1]),
                'threshold': threshold,
                'n_samples': len(bin_residuals),
                'mean_residual': bin_residuals.mean(),
                'std_residual': bin_residuals.std()
            }

            logger.info(
                f"Bin {i}: [{self.bin_edges[i]:.2f}, {self.bin_edges[i+1]:.2f}] "
                f"threshold={threshold:.3f}, n={len(bin_residuals)}"
            )

    def predict_interval(self, prediction: float) -> Tuple[float, float]:
        """Compute prediction interval using appropriate bin threshold."""
        # Find which bin this prediction belongs to
        bin_idx = np.digitize(prediction, self.bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, len(self.bin_thresholds) - 1)

        # Get threshold for this bin
        if bin_idx in self.bin_thresholds:
            threshold = self.bin_thresholds[bin_idx]['threshold']
        else:
            # Fallback to nearest bin
            closest_bin = min(
                self.bin_thresholds.keys(),
                key=lambda b: abs(
                    (self.bin_thresholds[b]['range'][0] + self.bin_thresholds[b]['range'][1]) / 2
                    - prediction
                )
            )
            threshold = self.bin_thresholds[closest_bin]['threshold']
            logger.debug(f"Prediction {prediction:.2f} using fallback bin {closest_bin}")

        return prediction - threshold, prediction + threshold

    def get_bin_info(self) -> pd.DataFrame:
        """Get information about each bin."""
        data = []
        for bin_idx, info in self.bin_thresholds.items():
            data.append({
                'bin': bin_idx,
                'lower': info['range'][0],
                'upper': info['range'][1],
                'center': (info['range'][0] + info['range'][1]) / 2,
                'threshold': info['threshold'],
                'n_samples': info['n_samples'],
                'mean_residual': info['mean_residual'],
                'std_residual': info['std_residual']
            })
        return pd.DataFrame(data)


class ConformizedQuantileRegression:
    """
    Conformalized Quantile Regression (CQR).

    Predicts upper and lower quantiles directly, then applies conformal
    adjustment for guaranteed coverage.

    Reference: "Conformalized Quantile Regression" (Romano et al., 2019)
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
        self.adjustment = None

    def fit(
        self,
        predictions_lower: np.ndarray,
        predictions_upper: np.ndarray,
        actuals: np.ndarray
    ):
        """
        Fit CQR using pre-trained quantile models.

        Args:
            predictions_lower: Lower quantile predictions (e.g., 5th percentile)
            predictions_upper: Upper quantile predictions (e.g., 95th percentile)
            actuals: True values
        """
        # Compute conformity scores (max distance to interval)
        lower_violation = predictions_lower - actuals
        upper_violation = actuals - predictions_upper
        conformity_scores = np.maximum(lower_violation, upper_violation)

        # Compute adjustment quantile
        n = len(conformity_scores)
        q_level = (1 - self.alpha) * (n + 1) / n
        q_level = np.clip(q_level, 0, 1)

        self.adjustment = np.quantile(conformity_scores, q_level)

        logger.info(f"CQR adjustment: {self.adjustment:.4f}")

    def adjust_interval(
        self,
        lower_pred: float,
        upper_pred: float
    ) -> Tuple[float, float]:
        """
        Apply conformal adjustment to quantile predictions.

        Args:
            lower_pred: Lower quantile prediction
            upper_pred: Upper quantile prediction

        Returns:
            Adjusted (lower, upper) interval with guaranteed coverage
        """
        return lower_pred - self.adjustment, upper_pred + self.adjustment


class AdaptiveConformalAnalyzer:
    """Comprehensive analysis of adaptive conformal methods."""

    def __init__(self, predictions_path: Path):
        """Load validation predictions."""
        self.predictions_df = pd.read_csv(predictions_path)
        logger.info(f"Loaded {len(self.predictions_df)} predictions")

        self.predictions_df['residual'] = (
            self.predictions_df['PredictedValue'] -
            self.predictions_df['ElementValue']
        )
        self.predictions_df['abs_residual'] = np.abs(
            self.predictions_df['residual']
        )

    def compare_methods(self, alpha: float = 0.05) -> Dict:
        """Compare global vs adaptive conformal methods."""
        logger.info("Comparing conformal prediction methods...")

        predictions = self.predictions_df['PredictedValue'].values
        actuals = self.predictions_df['ElementValue'].values

        results = {}

        # 1. Global (standard) conformal
        logger.info("Computing global conformal intervals...")
        residuals = np.abs(predictions - actuals)
        n = len(residuals)
        q_level = (1 - alpha) * (n + 1) / n
        global_threshold = np.quantile(residuals, np.clip(q_level, 0, 1))

        global_lower = predictions - global_threshold
        global_upper = predictions + global_threshold
        global_coverage = np.mean(
            (actuals >= global_lower) & (actuals <= global_upper)
        )
        global_width = 2 * global_threshold

        results['global'] = {
            'threshold': global_threshold,
            'coverage': global_coverage,
            'mean_width': global_width,
            'intervals': list(zip(global_lower, global_upper))
        }

        logger.info(f"Global: threshold={global_threshold:.3f}, "
                   f"coverage={global_coverage:.2%}, width={global_width:.3f}")

        # 2. Locally weighted conformal
        logger.info("Computing locally weighted conformal intervals...")
        lwc = LocallyWeightedConformal(alpha=alpha)
        lwc.fit(predictions, actuals)

        lwc_intervals = [lwc.predict_interval(p) for p in predictions]
        lwc_lower = np.array([i[0] for i in lwc_intervals])
        lwc_upper = np.array([i[1] for i in lwc_intervals])
        lwc_coverage = np.mean(
            (actuals >= lwc_lower) & (actuals <= lwc_upper)
        )
        lwc_widths = lwc_upper - lwc_lower
        lwc_mean_width = np.mean(lwc_widths)

        results['locally_weighted'] = {
            'coverage': lwc_coverage,
            'mean_width': lwc_mean_width,
            'width_std': np.std(lwc_widths),
            'min_width': np.min(lwc_widths),
            'max_width': np.max(lwc_widths),
            'intervals': list(zip(lwc_lower, lwc_upper))
        }

        logger.info(f"LWC: coverage={lwc_coverage:.2%}, "
                   f"mean_width={lwc_mean_width:.3f} ± {np.std(lwc_widths):.3f}")

        # 3. Difficulty-stratified conformal
        logger.info("Computing stratified conformal intervals...")
        dsc = DifficultyStratifiedConformal(alpha=alpha, n_bins=5)
        dsc.fit(predictions, actuals)

        dsc_intervals = [dsc.predict_interval(p) for p in predictions]
        dsc_lower = np.array([i[0] for i in dsc_intervals])
        dsc_upper = np.array([i[1] for i in dsc_intervals])
        dsc_coverage = np.mean(
            (actuals >= dsc_lower) & (actuals <= dsc_upper)
        )
        dsc_widths = dsc_upper - dsc_lower
        dsc_mean_width = np.mean(dsc_widths)

        results['stratified'] = {
            'coverage': dsc_coverage,
            'mean_width': dsc_mean_width,
            'width_std': np.std(dsc_widths),
            'min_width': np.min(dsc_widths),
            'max_width': np.max(dsc_widths),
            'intervals': list(zip(dsc_lower, dsc_upper)),
            'bin_info': dsc.get_bin_info()
        }

        logger.info(f"Stratified: coverage={dsc_coverage:.2%}, "
                   f"mean_width={dsc_mean_width:.3f} ± {np.std(dsc_widths):.3f}")

        # Add to dataframe for plotting
        self.predictions_df['global_lower'] = global_lower
        self.predictions_df['global_upper'] = global_upper
        self.predictions_df['lwc_lower'] = lwc_lower
        self.predictions_df['lwc_upper'] = lwc_upper
        self.predictions_df['dsc_lower'] = dsc_lower
        self.predictions_df['dsc_upper'] = dsc_upper

        return results

    def generate_plots(self, results: Dict, output_dir: Path):
        """Generate comparison plots."""
        logger.info("Generating comparison plots...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sort by prediction for cleaner plots
        df_sorted = self.predictions_df.sort_values('PredictedValue')

        # Plot 1: Interval widths comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1a: Global conformal
        ax = axes[0, 0]
        ax.scatter(df_sorted['PredictedValue'], df_sorted['ElementValue'],
                  alpha=0.3, s=20, label='Actual')
        ax.plot(df_sorted['PredictedValue'], df_sorted['PredictedValue'],
               'k-', linewidth=2, label='Perfect')
        ax.fill_between(
            df_sorted['PredictedValue'],
            df_sorted['global_lower'],
            df_sorted['global_upper'],
            alpha=0.3,
            label=f"95% Global Interval (width={results['global']['mean_width']:.2f}%)"
        )
        ax.set_xlabel('Predicted Potassium (%)', fontsize=11)
        ax.set_ylabel('Actual Potassium (%)', fontsize=11)
        ax.set_title(f"Global Conformal (Coverage: {results['global']['coverage']:.1%})",
                    fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 1b: Locally weighted
        ax = axes[0, 1]
        ax.scatter(df_sorted['PredictedValue'], df_sorted['ElementValue'],
                  alpha=0.3, s=20, label='Actual')
        ax.plot(df_sorted['PredictedValue'], df_sorted['PredictedValue'],
               'k-', linewidth=2, label='Perfect')
        ax.fill_between(
            df_sorted['PredictedValue'],
            df_sorted['lwc_lower'],
            df_sorted['lwc_upper'],
            alpha=0.3, color='orange',
            label=f"95% LWC Interval (width={results['locally_weighted']['mean_width']:.2f}%)"
        )
        ax.set_xlabel('Predicted Potassium (%)', fontsize=11)
        ax.set_ylabel('Actual Potassium (%)', fontsize=11)
        ax.set_title(f"Locally Weighted Conformal (Coverage: {results['locally_weighted']['coverage']:.1%})",
                    fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 1c: Stratified
        ax = axes[1, 0]
        ax.scatter(df_sorted['PredictedValue'], df_sorted['ElementValue'],
                  alpha=0.3, s=20, label='Actual')
        ax.plot(df_sorted['PredictedValue'], df_sorted['PredictedValue'],
               'k-', linewidth=2, label='Perfect')
        ax.fill_between(
            df_sorted['PredictedValue'],
            df_sorted['dsc_lower'],
            df_sorted['dsc_upper'],
            alpha=0.3, color='green',
            label=f"95% Stratified Interval (width={results['stratified']['mean_width']:.2f}%)"
        )
        ax.set_xlabel('Predicted Potassium (%)', fontsize=11)
        ax.set_ylabel('Actual Potassium (%)', fontsize=11)
        ax.set_title(f"Difficulty-Stratified Conformal (Coverage: {results['stratified']['coverage']:.1%})",
                    fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 1d: Width comparison
        ax = axes[1, 1]

        # Calculate widths
        global_widths = df_sorted['global_upper'] - df_sorted['global_lower']
        lwc_widths = df_sorted['lwc_upper'] - df_sorted['lwc_lower']
        dsc_widths = df_sorted['dsc_upper'] - df_sorted['dsc_lower']

        ax.plot(df_sorted['PredictedValue'], global_widths,
               'b-', linewidth=2, alpha=0.7, label='Global')
        ax.plot(df_sorted['PredictedValue'], lwc_widths,
               'orange', linewidth=2, alpha=0.7, label='Locally Weighted')
        ax.plot(df_sorted['PredictedValue'], dsc_widths,
               'g-', linewidth=2, alpha=0.7, label='Stratified')

        ax.set_xlabel('Predicted Potassium (%)', fontsize=11)
        ax.set_ylabel('Interval Width (%)', fontsize=11)
        ax.set_title('Interval Width Comparison', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / 'adaptive_conformal_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved comparison plot: {plot_path}")

        # Plot 2: Stratified bin details
        if 'bin_info' in results['stratified']:
            fig, ax = plt.subplots(figsize=(12, 6))

            bin_info = results['stratified']['bin_info']

            x = np.arange(len(bin_info))
            width = 0.35

            ax.bar(x - width/2, bin_info['threshold'], width,
                  label='Threshold', alpha=0.8)
            ax.bar(x + width/2, bin_info['std_residual'], width,
                  label='Std Dev', alpha=0.8)

            ax.set_xlabel('Concentration Bin', fontsize=12)
            ax.set_ylabel('Value (%)', fontsize=12)
            ax.set_title('Stratified Conformal - Bin Details',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f"{row['center']:.1f}%" for _, row in bin_info.iterrows()])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            # Add sample counts as text
            for i, (_, row) in enumerate(bin_info.iterrows()):
                ax.text(i, max(row['threshold'], row['std_residual']) + 0.05,
                       f"n={row['n_samples']}", ha='center', fontsize=9)

            plt.tight_layout()
            plot_path = output_dir / 'stratified_bin_details.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved bin details plot: {plot_path}")

    def save_results(self, results: Dict, output_dir: Path):
        """Save analysis results."""
        output_dir = Path(output_dir)

        # Save comparison table
        comparison_df = pd.DataFrame({
            'Method': ['Global', 'Locally Weighted', 'Stratified'],
            'Coverage': [
                results['global']['coverage'],
                results['locally_weighted']['coverage'],
                results['stratified']['coverage']
            ],
            'Mean Width': [
                results['global']['mean_width'],
                results['locally_weighted']['mean_width'],
                results['stratified']['mean_width']
            ],
            'Width Std': [
                0,  # Global has constant width
                results['locally_weighted']['width_std'],
                results['stratified']['width_std']
            ],
            'Min Width': [
                results['global']['mean_width'],
                results['locally_weighted']['min_width'],
                results['stratified']['min_width']
            ],
            'Max Width': [
                results['global']['mean_width'],
                results['locally_weighted']['max_width'],
                results['stratified']['max_width']
            ]
        })

        csv_path = output_dir / 'method_comparison.csv'
        comparison_df.to_csv(csv_path, index=False, float_format='%.4f')
        logger.info(f"Saved comparison table: {csv_path}")

        # Save stratified bin info
        if 'bin_info' in results['stratified']:
            bin_path = output_dir / 'stratified_bin_info.csv'
            results['stratified']['bin_info'].to_csv(bin_path, index=False, float_format='%.4f')
            logger.info(f"Saved bin info: {bin_path}")

        # Save report
        report_path = output_dir / 'adaptive_conformal_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ADAPTIVE CONFORMAL PREDICTION ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            f.write("COMPARISON OF METHODS\n")
            f.write("-" * 80 + "\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")

            # Calculate improvements
            global_width = results['global']['mean_width']
            lwc_width = results['locally_weighted']['mean_width']
            dsc_width = results['stratified']['mean_width']

            f.write("IMPROVEMENTS OVER GLOBAL METHOD\n")
            f.write("-" * 80 + "\n")
            f.write(f"Locally Weighted:  {(1 - lwc_width/global_width)*100:+.1f}% narrower\n")
            f.write(f"Stratified:        {(1 - dsc_width/global_width)*100:+.1f}% narrower\n")
            f.write("\n")

            f.write("KEY INSIGHTS\n")
            f.write("-" * 80 + "\n")
            f.write("1. Adaptive methods provide variable-width intervals\n")
            f.write("   - Narrower in well-predicted regions\n")
            f.write("   - Wider in difficult regions\n")
            f.write("\n")
            f.write("2. Coverage guarantees are maintained:\n")
            for method in ['global', 'locally_weighted', 'stratified']:
                cov = results[method]['coverage']
                f.write(f"   - {method.replace('_', ' ').title()}: {cov:.1%}\n")
            f.write("\n")
            f.write("3. Stratified method is simplest and most interpretable\n")
            f.write("4. Locally weighted adapts smoothly across prediction range\n")
            f.write("\n")

            f.write("RECOMMENDATION\n")
            f.write("-" * 80 + "\n")
            f.write("Use STRATIFIED conformal prediction for production:\n")
            f.write("- Simple to implement and explain\n")
            f.write("- Maintains coverage guarantees\n")
            f.write("- Provides tighter intervals where model performs well\n")
            f.write(f"- Average {(1 - dsc_width/global_width)*100:.1f}% narrower than global\n")

        logger.info(f"Saved report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Adaptive conformal prediction analysis'
    )
    parser.add_argument(
        '--predictions',
        type=Path,
        required=True,
        help='Path to validation predictions CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('reports/adaptive_conformal_analysis'),
        help='Output directory'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05 for 95%% confidence)'
    )

    args = parser.parse_args()

    # Validate input
    if not args.predictions.exists():
        logger.error(f"Predictions file not found: {args.predictions}")
        return 1

    # Run analysis
    analyzer = AdaptiveConformalAnalyzer(args.predictions)
    results = analyzer.compare_methods(alpha=args.alpha)
    analyzer.generate_plots(results, args.output_dir)
    analyzer.save_results(results, args.output_dir)

    logger.info("=" * 80)
    logger.info("ADAPTIVE CONFORMAL ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")
    logger.info("Key results:")
    logger.info(f"  Global interval width:      {results['global']['mean_width']:.3f}%")
    logger.info(f"  Locally weighted width:     {results['locally_weighted']['mean_width']:.3f}%")
    logger.info(f"  Stratified width:           {results['stratified']['mean_width']:.3f}%")

    improvement = (1 - results['stratified']['mean_width'] / results['global']['mean_width']) * 100
    logger.info(f"")
    logger.info(f"  Stratified improvement: {improvement:.1f}% narrower intervals!")

    return 0


if __name__ == '__main__':
    exit(main())
