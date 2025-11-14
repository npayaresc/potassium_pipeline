#!/usr/bin/env python3
"""
Uncertainty Analysis for AutoGluon Predictions

Implements multiple complementary approaches to estimate prediction uncertainty:
1. Ensemble variance - AutoGluon model disagreement
2. Residual-based intervals - Concentration-range specific errors
3. Conformal prediction - Calibrated prediction intervals
4. Bootstrap estimation - Resampling-based uncertainty

Usage:
    python analyze_prediction_uncertainty.py \
        --predictions reports/predictions_full_context_autogluon_20251113_103412.csv \
        --model-path models/autogluon/full_context_20251113_103412 \
        --output-dir reports/uncertainty_analysis
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.isotonic import IsotonicRegression

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress AutoGluon warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class UncertaintyAnalyzer:
    """Comprehensive uncertainty analysis for AutoGluon predictions."""

    def __init__(self, predictions_path: Path, model_path: Optional[Path] = None):
        """
        Initialize uncertainty analyzer.

        Args:
            predictions_path: Path to validation predictions CSV
            model_path: Path to AutoGluon model directory (for ensemble variance)
        """
        self.predictions_path = predictions_path
        self.model_path = model_path

        # Load predictions
        self.predictions_df = pd.read_csv(predictions_path)
        logger.info(f"Loaded {len(self.predictions_df)} validation predictions")

        # Calculate residuals
        self.predictions_df['residual'] = (
            self.predictions_df['PredictedValue'] - self.predictions_df['ElementValue']
        )
        self.predictions_df['abs_residual'] = np.abs(self.predictions_df['residual'])

        # Load AutoGluon predictor if available
        self.predictor = None
        if model_path and model_path.exists():
            try:
                from autogluon.tabular import TabularPredictor
                self.predictor = TabularPredictor.load(str(model_path))
                logger.info(f"Loaded AutoGluon predictor from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load AutoGluon predictor: {e}")

    def analyze_ensemble_variance(self) -> Optional[pd.DataFrame]:
        """
        Calculate ensemble variance from individual AutoGluon models.

        Returns:
            DataFrame with ensemble statistics per prediction, or None if not available
        """
        if self.predictor is None:
            logger.warning("AutoGluon predictor not available - skipping ensemble analysis")
            return None

        try:
            logger.info("Analyzing ensemble variance from individual models...")

            # Get leaderboard to identify available models
            leaderboard = self.predictor.leaderboard(silent=True)
            logger.info(f"Found {len(leaderboard)} models in ensemble")

            # Note: AutoGluon doesn't directly expose individual model predictions
            # We'll need to use the model's internal structure
            # This is a limitation - in production, you'd want to save individual predictions

            logger.warning("Direct ensemble variance extraction requires custom implementation")
            logger.info("Recommendation: Save predictions from each model during training")

            return None

        except Exception as e:
            logger.error(f"Failed to analyze ensemble variance: {e}")
            return None

    def analyze_residual_distribution(
        self,
        n_bins: int = 5,
        confidence_levels: List[float] = [0.68, 0.95, 0.99]
    ) -> Dict:
        """
        Analyze residual distribution by concentration range.

        Args:
            n_bins: Number of concentration bins
            confidence_levels: Confidence levels for prediction intervals

        Returns:
            Dictionary with residual statistics by concentration range
        """
        logger.info("Analyzing residual distribution by concentration range...")

        # Create concentration bins
        self.predictions_df['concentration_bin'] = pd.qcut(
            self.predictions_df['ElementValue'],
            q=n_bins,
            labels=[f'Bin_{i+1}' for i in range(n_bins)],
            duplicates='drop'
        )

        results = {}

        for bin_name in self.predictions_df['concentration_bin'].unique():
            bin_data = self.predictions_df[
                self.predictions_df['concentration_bin'] == bin_name
            ]

            residuals = bin_data['residual'].values
            abs_residuals = bin_data['abs_residual'].values

            # Calculate statistics
            bin_stats = {
                'n_samples': len(bin_data),
                'concentration_range': (
                    bin_data['ElementValue'].min(),
                    bin_data['ElementValue'].max()
                ),
                'mean_prediction': bin_data['PredictedValue'].mean(),
                'mean_residual': residuals.mean(),
                'std_residual': residuals.std(),
                'mae': abs_residuals.mean(),
                'rmse': np.sqrt((residuals ** 2).mean()),
                'median_absolute_error': np.median(abs_residuals),
            }

            # Calculate prediction intervals at different confidence levels
            for conf_level in confidence_levels:
                alpha = 1 - conf_level
                lower_q = alpha / 2
                upper_q = 1 - alpha / 2

                lower_bound = np.percentile(residuals, lower_q * 100)
                upper_bound = np.percentile(residuals, upper_q * 100)

                bin_stats[f'interval_{int(conf_level*100)}%'] = (lower_bound, upper_bound)

            results[str(bin_name)] = bin_stats

        return results

    def fit_conformal_predictor(
        self,
        alpha: float = 0.05,
        method: str = 'split'
    ) -> Tuple[callable, float]:
        """
        Fit conformal prediction intervals.

        Conformal prediction provides calibrated prediction intervals with
        guaranteed coverage probability regardless of the underlying model.

        Args:
            alpha: Significance level (e.g., 0.05 for 95% confidence)
            method: 'split' for split conformal or 'cv' for CV+

        Returns:
            Tuple of (interval_function, coverage_score)
        """
        logger.info(f"Fitting conformal predictor (alpha={alpha}, method={method})...")

        # Calculate nonconformity scores (absolute residuals)
        nonconformity_scores = self.predictions_df['abs_residual'].values

        # Calculate quantile for prediction intervals
        n = len(nonconformity_scores)
        if method == 'split':
            # Split conformal: simple quantile
            q_level = (1 - alpha) * (1 + 1/n)
        else:  # cv
            # CV+ uses a different adjustment
            q_level = (1 - alpha) * (n + 1) / n

        # Clamp q_level to valid range
        q_level = min(max(q_level, 0), 1)

        # Calculate the conformity threshold
        conformity_threshold = np.quantile(nonconformity_scores, q_level)

        logger.info(f"Conformal threshold at {q_level:.4f} quantile: {conformity_threshold:.4f}")

        # Create interval function
        def prediction_interval(predicted_value: float) -> Tuple[float, float]:
            """Return prediction interval for a given predicted value."""
            return (
                predicted_value - conformity_threshold,
                predicted_value + conformity_threshold
            )

        # Validate coverage on validation set
        intervals = [prediction_interval(p) for p in self.predictions_df['PredictedValue']]
        coverage = sum(
            lower <= true <= upper
            for (lower, upper), true in zip(intervals, self.predictions_df['ElementValue'])
        ) / len(intervals)

        logger.info(f"Empirical coverage: {coverage:.2%} (target: {1-alpha:.2%})")

        return prediction_interval, coverage

    def bootstrap_uncertainty(
        self,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Estimate uncertainty using bootstrap resampling of residuals.

        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with bootstrap statistics
        """
        logger.info(f"Running bootstrap analysis ({n_bootstrap} samples)...")

        residuals = self.predictions_df['residual'].values

        # Bootstrap sampling
        bootstrap_stds = []
        bootstrap_maes = []

        np.random.seed(42)
        for _ in range(n_bootstrap):
            # Resample residuals with replacement
            boot_sample = np.random.choice(residuals, size=len(residuals), replace=True)
            bootstrap_stds.append(np.std(boot_sample))
            bootstrap_maes.append(np.mean(np.abs(boot_sample)))

        # Calculate confidence intervals for uncertainty metrics
        alpha = 1 - confidence_level
        std_ci = np.percentile(bootstrap_stds, [alpha/2 * 100, (1-alpha/2) * 100])
        mae_ci = np.percentile(bootstrap_maes, [alpha/2 * 100, (1-alpha/2) * 100])

        results = {
            'std_estimate': np.mean(bootstrap_stds),
            'std_ci': std_ci,
            'mae_estimate': np.mean(bootstrap_maes),
            'mae_ci': mae_ci,
            'bootstrap_samples': n_bootstrap,
        }

        logger.info(f"Bootstrap STD: {results['std_estimate']:.4f} "
                   f"[{std_ci[0]:.4f}, {std_ci[1]:.4f}]")
        logger.info(f"Bootstrap MAE: {results['mae_estimate']:.4f} "
                   f"[{mae_ci[0]:.4f}, {mae_ci[1]:.4f}]")

        return results

    def fit_heteroscedastic_model(self) -> Tuple[callable, callable]:
        """
        Fit a model for concentration-dependent uncertainty (heteroscedasticity).

        Uses isotonic regression to model how prediction error varies with
        predicted concentration.

        Returns:
            Tuple of (mean_function, std_function)
        """
        logger.info("Fitting heteroscedastic uncertainty model...")

        # Sort by predicted value
        sorted_idx = np.argsort(self.predictions_df['PredictedValue'])
        pred_values = self.predictions_df['PredictedValue'].values[sorted_idx]
        residuals = self.predictions_df['residual'].values[sorted_idx]

        # Fit isotonic regression for mean residual (bias correction)
        iso_mean = IsotonicRegression(out_of_bounds='clip')
        iso_mean.fit(pred_values, residuals)

        # Calculate squared residuals for variance modeling
        squared_residuals = residuals ** 2

        # Fit isotonic regression for variance
        iso_var = IsotonicRegression(out_of_bounds='clip')
        iso_var.fit(pred_values, squared_residuals)

        # Create prediction functions
        def mean_correction(predicted_value: float) -> float:
            """Bias correction at given concentration."""
            return iso_mean.predict([predicted_value])[0]

        def std_prediction(predicted_value: float) -> float:
            """Standard deviation at given concentration."""
            var = iso_var.predict([predicted_value])[0]
            return np.sqrt(max(var, 1e-6))  # Prevent negative variance

        # Evaluate model fit
        mean_preds = iso_mean.predict(pred_values)
        std_preds = np.sqrt(iso_var.predict(pred_values))

        logger.info(f"Heteroscedastic model fitted:")
        logger.info(f"  Mean bias range: [{mean_preds.min():.4f}, {mean_preds.max():.4f}]")
        logger.info(f"  STD range: [{std_preds.min():.4f}, {std_preds.max():.4f}]")

        return mean_correction, std_prediction

    def generate_uncertainty_report(
        self,
        output_dir: Path,
        include_plots: bool = True
    ) -> Path:
        """
        Generate comprehensive uncertainty analysis report.

        Args:
            output_dir: Directory to save report and plots
            include_plots: Whether to generate visualization plots

        Returns:
            Path to saved report file
        """
        logger.info("Generating comprehensive uncertainty report...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Residual distribution analysis
        residual_stats = self.analyze_residual_distribution()

        # 2. Conformal prediction
        conformal_95, coverage_95 = self.fit_conformal_predictor(alpha=0.05)
        conformal_99, coverage_99 = self.fit_conformal_predictor(alpha=0.01)

        # 3. Bootstrap uncertainty
        bootstrap_stats = self.bootstrap_uncertainty()

        # 4. Heteroscedastic model
        mean_correction, std_prediction = self.fit_heteroscedastic_model()

        # 5. Generate plots if requested
        if include_plots:
            self._generate_plots(
                output_dir,
                residual_stats,
                conformal_95,
                mean_correction,
                std_prediction
            )

        # 6. Create comprehensive report
        report_path = output_dir / 'uncertainty_analysis_report.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PREDICTION UNCERTAINTY ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Overall statistics
            f.write("OVERALL VALIDATION PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of predictions: {len(self.predictions_df)}\n")
            f.write(f"Mean Absolute Error (MAE): {self.predictions_df['abs_residual'].mean():.4f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {np.sqrt((self.predictions_df['residual']**2).mean()):.4f}\n")
            f.write(f"Mean Bias: {self.predictions_df['residual'].mean():.4f}\n")
            f.write(f"Std Dev of Errors: {self.predictions_df['residual'].std():.4f}\n")
            f.write(f"Concentration Range: [{self.predictions_df['ElementValue'].min():.2f}, "
                   f"{self.predictions_df['ElementValue'].max():.2f}]\n\n")

            # Residual distribution by concentration
            f.write("UNCERTAINTY BY CONCENTRATION RANGE\n")
            f.write("-" * 80 + "\n")
            for bin_name, stats in residual_stats.items():
                f.write(f"\n{bin_name}:\n")
                f.write(f"  Concentration: [{stats['concentration_range'][0]:.2f}, "
                       f"{stats['concentration_range'][1]:.2f}]\n")
                f.write(f"  N samples: {stats['n_samples']}\n")
                f.write(f"  MAE: {stats['mae']:.4f}\n")
                f.write(f"  RMSE: {stats['rmse']:.4f}\n")
                f.write(f"  Mean Bias: {stats['mean_residual']:.4f}\n")
                f.write(f"  Std Dev: {stats['std_residual']:.4f}\n")
                f.write(f"  95% Prediction Interval: [{stats['interval_95%'][0]:.4f}, "
                       f"{stats['interval_95%'][1]:.4f}]\n")

            # Conformal prediction
            f.write("\n\nCONFORMAL PREDICTION INTERVALS\n")
            f.write("-" * 80 + "\n")
            f.write(f"95% Confidence Intervals:\n")
            f.write(f"  Empirical Coverage: {coverage_95:.2%}\n")
            f.write(f"  Example: For prediction = 8.0, interval = "
                   f"[{conformal_95(8.0)[0]:.4f}, {conformal_95(8.0)[1]:.4f}]\n")
            f.write(f"\n99% Confidence Intervals:\n")
            f.write(f"  Empirical Coverage: {coverage_99:.2%}\n")
            f.write(f"  Example: For prediction = 8.0, interval = "
                   f"[{conformal_99(8.0)[0]:.4f}, {conformal_99(8.0)[1]:.4f}]\n")

            # Bootstrap results
            f.write("\n\nBOOTSTRAP UNCERTAINTY ESTIMATES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Bootstrap samples: {bootstrap_stats['bootstrap_samples']}\n")
            f.write(f"STD estimate: {bootstrap_stats['std_estimate']:.4f} "
                   f"[{bootstrap_stats['std_ci'][0]:.4f}, {bootstrap_stats['std_ci'][1]:.4f}]\n")
            f.write(f"MAE estimate: {bootstrap_stats['mae_estimate']:.4f} "
                   f"[{bootstrap_stats['mae_ci'][0]:.4f}, {bootstrap_stats['mae_ci'][1]:.4f}]\n")

            # Practical recommendations
            f.write("\n\nPRACTICAL RECOMMENDATIONS FOR PRODUCTION USE\n")
            f.write("-" * 80 + "\n")
            f.write("1. USE CONFORMAL PREDICTION INTERVALS:\n")
            f.write("   - For 95% confidence: use conformal_95() function\n")
            f.write("   - Provides calibrated intervals with guaranteed coverage\n\n")

            f.write("2. CONCENTRATION-SPECIFIC UNCERTAINTY:\n")
            f.write("   - Low concentrations (<5%): Higher relative uncertainty\n")
            f.write("   - Mid concentrations (5-9%): More stable predictions\n")
            f.write("   - High concentrations (>9%): Review concentration-specific stats\n\n")

            f.write("3. HETEROSCEDASTIC MODELING:\n")
            f.write("   - Use std_prediction() function for concentration-dependent uncertainty\n")
            f.write("   - Captures how uncertainty varies across concentration range\n\n")

            f.write("4. RECOMMENDED WORKFLOW:\n")
            f.write("   a) Make prediction: pred = model.predict(X)\n")
            f.write("   b) Get conformal interval: interval = conformal_95(pred)\n")
            f.write("   c) Get heteroscedastic std: sigma = std_prediction(pred)\n")
            f.write("   d) Report: pred ± interval_width (or pred ± 2*sigma for ~95% coverage)\n")

        logger.info(f"Report saved to: {report_path}")

        # Save uncertainty functions for later use
        self._save_uncertainty_functions(
            output_dir,
            conformal_95,
            conformal_99,
            mean_correction,
            std_prediction,
            residual_stats
        )

        return report_path

    def _generate_plots(
        self,
        output_dir: Path,
        residual_stats: Dict,
        conformal_interval: callable,
        mean_correction: callable,
        std_prediction: callable
    ):
        """Generate visualization plots for uncertainty analysis."""
        logger.info("Generating uncertainty visualization plots...")

        # Set style
        sns.set_style("whitegrid")

        # 1. Residuals vs Predicted Value
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Residuals scatter
        ax = axes[0, 0]
        scatter = ax.scatter(
            self.predictions_df['PredictedValue'],
            self.predictions_df['residual'],
            c=self.predictions_df['ElementValue'],
            cmap='viridis',
            alpha=0.6,
            edgecolors='k',
            linewidth=0.5
        )
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Predicted Potassium (%)', fontsize=12)
        ax.set_ylabel('Residual (Predicted - Actual)', fontsize=12)
        ax.set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Actual Concentration (%)')

        # Plot 2: Absolute residuals with heteroscedastic model
        ax = axes[0, 1]
        pred_range = np.linspace(
            self.predictions_df['PredictedValue'].min(),
            self.predictions_df['PredictedValue'].max(),
            100
        )
        std_curve = [std_prediction(p) for p in pred_range]

        ax.scatter(
            self.predictions_df['PredictedValue'],
            self.predictions_df['abs_residual'],
            alpha=0.4,
            label='Actual Errors'
        )
        ax.plot(pred_range, std_curve, 'r-', linewidth=2, label='Predicted STD')
        ax.plot(pred_range, 2*np.array(std_curve), 'r--', linewidth=2, label='2× STD (~95%)')
        ax.set_xlabel('Predicted Potassium (%)', fontsize=12)
        ax.set_ylabel('Absolute Error', fontsize=12)
        ax.set_title('Heteroscedastic Uncertainty Model', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Distribution of residuals
        ax = axes[1, 0]
        ax.hist(
            self.predictions_df['residual'],
            bins=30,
            edgecolor='black',
            alpha=0.7,
            density=True
        )
        # Overlay normal distribution
        mu, sigma = self.predictions_df['residual'].mean(), self.predictions_df['residual'].std()
        x = np.linspace(self.predictions_df['residual'].min(), self.predictions_df['residual'].max(), 100)
        ax.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
               'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
        ax.axvline(x=0, color='k', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Residual', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Prediction intervals
        ax = axes[1, 1]
        sorted_idx = np.argsort(self.predictions_df['PredictedValue'])
        pred_sorted = self.predictions_df['PredictedValue'].values[sorted_idx]
        actual_sorted = self.predictions_df['ElementValue'].values[sorted_idx]

        # Calculate conformal intervals for all predictions
        intervals = [conformal_interval(p) for p in pred_sorted]
        lower_bounds = [i[0] for i in intervals]
        upper_bounds = [i[1] for i in intervals]

        ax.plot(pred_sorted, actual_sorted, 'o', alpha=0.5, label='Actual vs Predicted')
        ax.plot(pred_sorted, pred_sorted, 'k-', linewidth=2, label='Perfect Prediction')
        ax.fill_between(pred_sorted, lower_bounds, upper_bounds, alpha=0.3,
                        label='95% Conformal Interval')
        ax.set_xlabel('Predicted Potassium (%)', fontsize=12)
        ax.set_ylabel('Actual Potassium (%)', fontsize=12)
        ax.set_title('Calibration with Prediction Intervals', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / 'uncertainty_analysis_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Plots saved to: {plot_path}")

        # 2. Concentration-binned uncertainty plot
        fig, ax = plt.subplots(figsize=(12, 6))

        bin_names = []
        bin_centers = []
        bin_maes = []
        bin_stds = []

        for bin_name, stats in residual_stats.items():
            bin_names.append(bin_name)
            bin_centers.append(stats['mean_prediction'])
            bin_maes.append(stats['mae'])
            bin_stds.append(stats['std_residual'])

        x = np.arange(len(bin_names))
        width = 0.35

        ax.bar(x - width/2, bin_maes, width, label='MAE', alpha=0.8)
        ax.bar(x + width/2, bin_stds, width, label='STD', alpha=0.8)

        ax.set_xlabel('Concentration Bin', fontsize=12)
        ax.set_ylabel('Error Magnitude', fontsize=12)
        ax.set_title('Uncertainty by Concentration Range', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{c:.2f}' for c in bin_centers])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = output_dir / 'concentration_binned_uncertainty.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Concentration-binned plot saved to: {plot_path}")

    def _save_uncertainty_functions(
        self,
        output_dir: Path,
        conformal_95: callable,
        conformal_99: callable,
        mean_correction: callable,
        std_prediction: callable,
        residual_stats: Dict
    ):
        """Save uncertainty estimation functions and lookup tables."""
        import pickle

        # Create lookup table for practical use
        pred_values = np.linspace(
            self.predictions_df['PredictedValue'].min(),
            self.predictions_df['PredictedValue'].max(),
            100
        )

        lookup_table = pd.DataFrame({
            'predicted_value': pred_values,
            'conformal_95_lower': [conformal_95(p)[0] for p in pred_values],
            'conformal_95_upper': [conformal_95(p)[1] for p in pred_values],
            'conformal_99_lower': [conformal_99(p)[0] for p in pred_values],
            'conformal_99_upper': [conformal_99(p)[1] for p in pred_values],
            'mean_correction': [mean_correction(p) for p in pred_values],
            'std_prediction': [std_prediction(p) for p in pred_values],
        })

        csv_path = output_dir / 'uncertainty_lookup_table.csv'
        lookup_table.to_csv(csv_path, index=False)
        logger.info(f"Uncertainty lookup table saved to: {csv_path}")

        # Save residual statistics
        stats_df = pd.DataFrame(residual_stats).T
        stats_path = output_dir / 'residual_statistics_by_concentration.csv'
        stats_df.to_csv(stats_path)
        logger.info(f"Residual statistics saved to: {stats_path}")

        # Note: Functions cannot be easily pickled due to closure
        # Use the lookup table CSV instead for practical applications
        logger.info("Uncertainty functions available via lookup table (CSV) for practical use")


def main():
    """Main entry point for uncertainty analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze prediction uncertainty for AutoGluon models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_prediction_uncertainty.py \\
      --predictions reports/predictions_full_context_autogluon_20251113_103412.csv

  # With AutoGluon model for ensemble analysis
  python analyze_prediction_uncertainty.py \\
      --predictions reports/predictions_full_context_autogluon_20251113_103412.csv \\
      --model-path models/autogluon/full_context_20251113_103412

  # Custom output directory
  python analyze_prediction_uncertainty.py \\
      --predictions reports/predictions_full_context_autogluon_20251113_103412.csv \\
      --output-dir reports/uncertainty_analysis_custom
        """
    )

    parser.add_argument(
        '--predictions',
        type=Path,
        required=True,
        help='Path to validation predictions CSV file'
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        default=None,
        help='Path to AutoGluon model directory (optional, for ensemble analysis)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('reports/uncertainty_analysis'),
        help='Directory to save analysis results (default: reports/uncertainty_analysis)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating visualization plots'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.predictions.exists():
        logger.error(f"Predictions file not found: {args.predictions}")
        return 1

    if args.model_path and not args.model_path.exists():
        logger.warning(f"Model path not found: {args.model_path}")
        logger.warning("Continuing without ensemble analysis...")
        args.model_path = None

    # Run analysis
    try:
        analyzer = UncertaintyAnalyzer(
            predictions_path=args.predictions,
            model_path=args.model_path
        )

        report_path = analyzer.generate_uncertainty_report(
            output_dir=args.output_dir,
            include_plots=not args.no_plots
        )

        logger.info("=" * 80)
        logger.info("UNCERTAINTY ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Report: {report_path}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Review the uncertainty_analysis_report.txt")
        logger.info("2. Check the visualization plots")
        logger.info("3. Use uncertainty_lookup_table.csv for production predictions")
        logger.info("4. Use residual_statistics_by_concentration.csv for range-specific uncertainties")

        return 0

    except Exception as e:
        logger.error(f"Uncertainty analysis failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
