#!/usr/bin/env python3
"""
Compare LIBS analysis using 10 shots vs 20 shots per measurement.

This script processes raw LIBS data to compare the impact of using:
- First 10 shots only (Intensity1 to Intensity10)
- All 20 shots (Intensity1 to Intensity20)

Each sample has 2 measurement files (_01 and _02), and each measurement contains 20 shots.
We'll compare the averaging, noise characteristics, and spectral quality between the two approaches.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.pipeline_config import Config
from src.data_management.data_manager import DataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('shots_comparison_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

class ShotsComparison:
    """Compare 10 vs 20 shots analysis for LIBS measurements."""

    def __init__(self, raw_data_dir: Path):
        """Initialize with raw data directory."""
        self.raw_data_dir = raw_data_dir
        self.results_10_shots = {}
        self.results_20_shots = {}
        self.sample_groups = defaultdict(list)

    def extract_sample_id(self, filename: str) -> str:
        """Extract sample ID from filename (e.g., S00531 from the filename)."""
        # Pattern: Look for S followed by exactly 5 digits, but ensure it's the right one
        # In the filename: MPN_0000_002025_CNGS7330000001_POT_0000_S00531_P_Y_27062025_1000_01.csv.txt
        # We want S00531, not S73300 (which comes from CNGS7330000001)
        import re

        # Look for S followed by 5 digits that comes after "_POT_0000_"
        match = re.search(r'_POT_0000_(S\d{5})_', filename)
        if match:
            return match.group(1)

        # Alternative: look for S followed by 5 digits near the end of the filename
        matches = re.findall(r'S\d{5}', filename)
        if matches:
            # Take the last match (most likely the sample ID)
            return matches[-1]

        # Fallback: use everything before the last underscore minus the shot number
        name_without_ext = filename.replace('.csv.txt', '')
        parts = name_without_ext.split('_')
        # Remove the last part (shot number like 01, 02)
        return '_'.join(parts[:-1])

    def read_spectral_file(self, file_path: Path) -> pd.DataFrame:
        """Read a raw spectral file and return the data."""
        # Find where the data starts (after metadata)
        data_start_line = 0
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if 'Wavelength' in line and 'Intensity' in line:
                    data_start_line = i
                    break

        # Read the actual data
        df = pd.read_csv(file_path, skiprows=data_start_line)

        # Clean column names (remove spaces)
        df.columns = [col.strip() for col in df.columns]

        return df

    def group_files_by_sample(self):
        """Group files by sample ID."""
        logger.info(f"Scanning directory: {self.raw_data_dir}")

        # Recursively find all .csv.txt files
        all_files = list(self.raw_data_dir.rglob("*.csv.txt"))
        logger.info(f"Found {len(all_files)} total files")

        for file_path in all_files:
            sample_id = self.extract_sample_id(file_path.name)
            self.sample_groups[sample_id].append(file_path)

        logger.info(f"Grouped into {len(self.sample_groups)} unique samples")

        # Log sample distribution
        for sample_id, files in list(self.sample_groups.items())[:5]:
            logger.debug(f"Sample {sample_id}: {len(files)} files")

    def process_shots(self, file_paths: List[Path], num_shots: int) -> Dict[str, np.ndarray]:
        """
        Process files using either 10 or 20 shots.

        Args:
            file_paths: List of file paths for a sample
            num_shots: Number of shots to use (10 or 20)

        Returns:
            Dictionary with averaged spectra and statistics
        """
        all_spectra = []
        wavelengths = None

        for file_path in file_paths:
            try:
                df = self.read_spectral_file(file_path)

                if wavelengths is None:
                    wavelengths = df['Wavelength'].values

                # Select intensity columns based on num_shots
                intensity_cols = [f'Intensity{i}' for i in range(1, num_shots + 1)]

                # Check if columns exist
                available_cols = [col for col in intensity_cols if col in df.columns]
                if len(available_cols) < num_shots:
                    logger.warning(f"File {file_path.name} has only {len(available_cols)} intensity columns")
                    continue

                # Get the intensity data
                intensity_data = df[available_cols].values  # Shape: (n_wavelengths, n_shots)
                all_spectra.append(intensity_data)

            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                continue

        if not all_spectra:
            return None

        # Stack all measurements: (n_files, n_wavelengths, n_shots)
        stacked_spectra = np.stack(all_spectra, axis=0)

        # Calculate statistics across all measurements and shots
        # First average across shots for each measurement
        avg_per_measurement = np.mean(stacked_spectra, axis=2)  # (n_files, n_wavelengths)

        # Then calculate statistics across measurements
        mean_spectrum = np.mean(avg_per_measurement, axis=0)
        std_spectrum = np.std(avg_per_measurement, axis=0)

        # Calculate shot-to-shot variability (within measurement)
        shot_variability = np.mean([np.std(spectrum, axis=1) for spectrum in stacked_spectra], axis=0)

        # Calculate measurement-to-measurement variability
        measurement_variability = std_spectrum

        # Calculate signal-to-noise ratio
        snr = np.divide(mean_spectrum, std_spectrum,
                       out=np.zeros_like(mean_spectrum), where=std_spectrum!=0)

        return {
            'wavelengths': wavelengths,
            'mean_spectrum': mean_spectrum,
            'std_spectrum': std_spectrum,
            'shot_variability': shot_variability,
            'measurement_variability': measurement_variability,
            'snr': snr,
            'n_measurements': len(all_spectra),
            'n_shots': num_shots,
            'raw_data': stacked_spectra
        }

    def analyze_all_samples(self):
        """Process all samples with both 10 and 20 shots."""
        logger.info("Starting analysis for all samples...")

        for i, (sample_id, file_paths) in enumerate(self.sample_groups.items()):
            if i % 10 == 0:
                logger.info(f"Processing sample {i+1}/{len(self.sample_groups)}: {sample_id}")

            # Process with 10 shots
            result_10 = self.process_shots(file_paths, 10)
            if result_10:
                self.results_10_shots[sample_id] = result_10

            # Process with 20 shots
            result_20 = self.process_shots(file_paths, 20)
            if result_20:
                self.results_20_shots[sample_id] = result_20

        logger.info(f"Processed {len(self.results_10_shots)} samples with 10 shots")
        logger.info(f"Processed {len(self.results_20_shots)} samples with 20 shots")

    def calculate_statistics(self) -> pd.DataFrame:
        """Calculate comparative statistics for 10 vs 20 shots."""
        stats_data = []

        # Get samples that have both 10 and 20 shot results
        common_samples = set(self.results_10_shots.keys()) & set(self.results_20_shots.keys())
        logger.info(f"Comparing {len(common_samples)} samples with both configurations")

        if not common_samples:
            logger.warning("No samples found with both 10 and 20 shot results!")
            # Return empty dataframe with expected columns
            return pd.DataFrame(columns=['sample_id', 'snr_10_shots', 'snr_20_shots',
                                        'avg_shot_var_10', 'avg_shot_var_20',
                                        'avg_measurement_var_10', 'avg_measurement_var_20',
                                        'peak_intensity_10', 'peak_intensity_20',
                                        'cv_at_peak_10', 'cv_at_peak_20',
                                        'snr_improvement', 'shot_var_reduction', 'cv_improvement'])

        for sample_id in common_samples:
            res_10 = self.results_10_shots[sample_id]
            res_20 = self.results_20_shots[sample_id]

            # Calculate key metrics
            stats_data.append({
                'sample_id': sample_id,
                # Average SNR across spectrum
                'snr_10_shots': np.median(res_10['snr'][np.isfinite(res_10['snr'])]),
                'snr_20_shots': np.median(res_20['snr'][np.isfinite(res_20['snr'])]),
                # Average variability
                'avg_shot_var_10': np.mean(res_10['shot_variability']),
                'avg_shot_var_20': np.mean(res_20['shot_variability']),
                'avg_measurement_var_10': np.mean(res_10['measurement_variability']),
                'avg_measurement_var_20': np.mean(res_20['measurement_variability']),
                # Peak intensity (max signal)
                'peak_intensity_10': np.max(res_10['mean_spectrum']),
                'peak_intensity_20': np.max(res_20['mean_spectrum']),
                # Coefficient of variation at peak
                'cv_at_peak_10': res_10['std_spectrum'][np.argmax(res_10['mean_spectrum'])] /
                                 np.max(res_10['mean_spectrum']) if np.max(res_10['mean_spectrum']) > 0 else np.nan,
                'cv_at_peak_20': res_20['std_spectrum'][np.argmax(res_20['mean_spectrum'])] /
                                 np.max(res_20['mean_spectrum']) if np.max(res_20['mean_spectrum']) > 0 else np.nan,
            })

        df_stats = pd.DataFrame(stats_data)

        # Calculate improvement factors
        if not df_stats.empty:
            df_stats['snr_improvement'] = (df_stats['snr_20_shots'] - df_stats['snr_10_shots']) / df_stats['snr_10_shots'] * 100
            df_stats['shot_var_reduction'] = (df_stats['avg_shot_var_10'] - df_stats['avg_shot_var_20']) / df_stats['avg_shot_var_10'] * 100
            df_stats['cv_improvement'] = (df_stats['cv_at_peak_10'] - df_stats['cv_at_peak_20']) / df_stats['cv_at_peak_10'] * 100

        return df_stats

    def plot_comparisons(self, df_stats: pd.DataFrame, output_dir: Path):
        """Create visualization comparing 10 vs 20 shots."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. SNR Comparison
        ax = axes[0, 0]
        ax.scatter(df_stats['snr_10_shots'], df_stats['snr_20_shots'], alpha=0.6)
        max_snr = max(df_stats['snr_10_shots'].max(), df_stats['snr_20_shots'].max())
        ax.plot([0, max_snr], [0, max_snr], 'r--', label='Equal SNR')
        ax.set_xlabel('SNR (10 shots)')
        ax.set_ylabel('SNR (20 shots)')
        ax.set_title('Signal-to-Noise Ratio Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Shot Variability Comparison
        ax = axes[0, 1]
        ax.scatter(df_stats['avg_shot_var_10'], df_stats['avg_shot_var_20'], alpha=0.6)
        max_var = max(df_stats['avg_shot_var_10'].max(), df_stats['avg_shot_var_20'].max())
        ax.plot([0, max_var], [0, max_var], 'r--', label='Equal Variability')
        ax.set_xlabel('Shot Variability (10 shots)')
        ax.set_ylabel('Shot Variability (20 shots)')
        ax.set_title('Shot-to-Shot Variability Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. CV at Peak Comparison
        ax = axes[0, 2]
        ax.scatter(df_stats['cv_at_peak_10'], df_stats['cv_at_peak_20'], alpha=0.6)
        max_cv = max(df_stats['cv_at_peak_10'].max(), df_stats['cv_at_peak_20'].max())
        ax.plot([0, max_cv], [0, max_cv], 'r--', label='Equal CV')
        ax.set_xlabel('CV at Peak (10 shots)')
        ax.set_ylabel('CV at Peak (20 shots)')
        ax.set_title('Coefficient of Variation at Peak')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. SNR Improvement Distribution
        ax = axes[1, 0]
        ax.hist(df_stats['snr_improvement'].dropna(), bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', label='No improvement')
        ax.set_xlabel('SNR Improvement (%)')
        ax.set_ylabel('Count')
        ax.set_title(f'SNR Improvement Distribution\nMean: {df_stats["snr_improvement"].mean():.1f}%')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Variability Reduction Distribution
        ax = axes[1, 1]
        ax.hist(df_stats['shot_var_reduction'].dropna(), bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', label='No reduction')
        ax.set_xlabel('Variability Reduction (%)')
        ax.set_ylabel('Count')
        ax.set_title(f'Shot Variability Reduction\nMean: {df_stats["shot_var_reduction"].mean():.1f}%')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Summary Statistics Box Plot
        ax = axes[1, 2]
        improvement_data = pd.DataFrame({
            'SNR\nImprovement': df_stats['snr_improvement'],
            'Variability\nReduction': df_stats['shot_var_reduction'],
            'CV\nImprovement': df_stats['cv_improvement']
        })
        bp = ax.boxplot([improvement_data[col].dropna() for col in improvement_data.columns],
                        labels=improvement_data.columns, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Summary of Improvements (20 vs 10 shots)')
        ax.grid(True, alpha=0.3)

        plt.suptitle('LIBS Analysis: 10 vs 20 Shots Comparison', fontsize=16, y=1.02)
        plt.tight_layout()

        output_file = output_dir / 'shots_comparison_plots.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Plots saved to {output_file}")

    def plot_spectral_examples(self, n_examples: int = 3, output_dir: Path = None):
        """Plot example spectra comparing 10 vs 20 shots."""
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Get random samples
        common_samples = list(set(self.results_10_shots.keys()) & set(self.results_20_shots.keys()))
        if len(common_samples) > n_examples:
            example_samples = np.random.choice(common_samples, n_examples, replace=False)
        else:
            example_samples = common_samples

        fig, axes = plt.subplots(n_examples, 2, figsize=(16, 4*n_examples))
        if n_examples == 1:
            axes = axes.reshape(1, -1)

        for i, sample_id in enumerate(example_samples):
            res_10 = self.results_10_shots[sample_id]
            res_20 = self.results_20_shots[sample_id]

            # Plot mean spectra
            ax = axes[i, 0]
            ax.plot(res_10['wavelengths'], res_10['mean_spectrum'],
                   label='10 shots', alpha=0.8, linewidth=1)
            ax.plot(res_20['wavelengths'], res_20['mean_spectrum'],
                   label='20 shots', alpha=0.8, linewidth=1)
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Intensity')
            ax.set_title(f'Sample {sample_id}: Mean Spectra')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot SNR comparison
            ax = axes[i, 1]
            ax.plot(res_10['wavelengths'], res_10['snr'],
                   label='10 shots', alpha=0.8, linewidth=1)
            ax.plot(res_20['wavelengths'], res_20['snr'],
                   label='20 shots', alpha=0.8, linewidth=1)
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('SNR')
            ax.set_title(f'Sample {sample_id}: Signal-to-Noise Ratio')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, np.percentile(np.concatenate([res_10['snr'], res_20['snr']]), 95)])

        plt.suptitle('Example Spectral Comparisons: 10 vs 20 Shots', fontsize=14, y=1.02)
        plt.tight_layout()

        if output_dir:
            output_file = output_dir / 'example_spectra_comparison.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Example spectra saved to {output_file}")
        plt.show()

    def generate_summary_report(self, df_stats: pd.DataFrame, output_dir: Path):
        """Generate a text summary report of the analysis."""
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / 'shots_comparison_report.txt'

        with open(report_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("LIBS ANALYSIS: 10 vs 20 SHOTS COMPARISON REPORT\n")
            f.write("="*60 + "\n\n")

            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Directory: {self.raw_data_dir}\n")
            f.write(f"Total Samples Analyzed: {len(df_stats)}\n\n")

            f.write("-"*40 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-"*40 + "\n\n")

            # SNR Analysis
            f.write("Signal-to-Noise Ratio (SNR):\n")
            f.write(f"  10 shots - Mean: {df_stats['snr_10_shots'].mean():.2f}, "
                   f"Median: {df_stats['snr_10_shots'].median():.2f}\n")
            f.write(f"  20 shots - Mean: {df_stats['snr_20_shots'].mean():.2f}, "
                   f"Median: {df_stats['snr_20_shots'].median():.2f}\n")
            f.write(f"  Average Improvement: {df_stats['snr_improvement'].mean():.1f}%\n")
            f.write(f"  Samples with improved SNR: {(df_stats['snr_improvement'] > 0).sum()} "
                   f"({(df_stats['snr_improvement'] > 0).sum()/len(df_stats)*100:.1f}%)\n\n")

            # Variability Analysis
            f.write("Shot-to-Shot Variability:\n")
            f.write(f"  10 shots - Mean: {df_stats['avg_shot_var_10'].mean():.2f}\n")
            f.write(f"  20 shots - Mean: {df_stats['avg_shot_var_20'].mean():.2f}\n")
            f.write(f"  Average Reduction: {df_stats['shot_var_reduction'].mean():.1f}%\n")
            f.write(f"  Samples with reduced variability: {(df_stats['shot_var_reduction'] > 0).sum()} "
                   f"({(df_stats['shot_var_reduction'] > 0).sum()/len(df_stats)*100:.1f}%)\n\n")

            # CV Analysis
            f.write("Coefficient of Variation at Peak:\n")
            f.write(f"  10 shots - Mean: {df_stats['cv_at_peak_10'].mean():.4f}\n")
            f.write(f"  20 shots - Mean: {df_stats['cv_at_peak_20'].mean():.4f}\n")
            f.write(f"  Average Improvement: {df_stats['cv_improvement'].mean():.1f}%\n\n")

            f.write("-"*40 + "\n")
            f.write("STATISTICAL SIGNIFICANCE\n")
            f.write("-"*40 + "\n\n")

            # Perform paired t-tests
            t_stat_snr, p_val_snr = stats.ttest_rel(df_stats['snr_20_shots'],
                                                     df_stats['snr_10_shots'])
            f.write(f"Paired t-test for SNR: t={t_stat_snr:.3f}, p={p_val_snr:.4f}\n")

            t_stat_var, p_val_var = stats.ttest_rel(df_stats['avg_shot_var_20'],
                                                    df_stats['avg_shot_var_10'])
            f.write(f"Paired t-test for Variability: t={t_stat_var:.3f}, p={p_val_var:.4f}\n\n")

            f.write("-"*40 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("-"*40 + "\n\n")

            # Generate recommendations based on results
            snr_improvement = df_stats['snr_improvement'].mean()
            var_reduction = df_stats['shot_var_reduction'].mean()

            if snr_improvement > 10 and var_reduction > 10:
                f.write("STRONG RECOMMENDATION: Use 20 shots\n")
                f.write("Rationale: Significant improvements in both SNR and variability.\n")
            elif snr_improvement > 5 or var_reduction > 5:
                f.write("MODERATE RECOMMENDATION: Use 20 shots\n")
                f.write("Rationale: Noticeable improvements in signal quality.\n")
            else:
                f.write("WEAK RECOMMENDATION: 10 shots may be sufficient\n")
                f.write("Rationale: Minimal improvements observed with 20 shots.\n")

            f.write(f"\nEstimated time savings with 10 shots: ~50%\n")
            f.write(f"Quality trade-off: {abs(snr_improvement):.1f}% SNR difference\n")

            f.write("\n" + "="*60 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*60 + "\n")

        logger.info(f"Summary report saved to {report_file}")

        # Also save detailed statistics to CSV
        csv_file = output_dir / 'shots_comparison_statistics.csv'
        df_stats.to_csv(csv_file, index=False)
        logger.info(f"Detailed statistics saved to {csv_file}")

        return report_file


def main():
    """Main execution function."""
    # Set up paths
    raw_data_dir = Path('/home/payanico/potassium_pipeline/data/raw/OneDrive_1_9-22-2025')
    output_dir = Path('reports/shots_comparison')

    logger.info("Starting 10 vs 20 shots comparison analysis...")

    # Initialize analyzer
    analyzer = ShotsComparison(raw_data_dir)

    # Group files by sample
    analyzer.group_files_by_sample()

    # Analyze all samples
    analyzer.analyze_all_samples()

    # Calculate statistics
    df_stats = analyzer.calculate_statistics()

    if df_stats.empty:
        print("\n" + "="*60)
        print("NO DATA TO ANALYZE")
        print("="*60)
        print("No samples could be processed successfully.")
        print("Check the log file for detailed error information.")
        return df_stats

    # Generate visualizations
    analyzer.plot_comparisons(df_stats, output_dir)

    # Plot example spectra
    analyzer.plot_spectral_examples(n_examples=3, output_dir=output_dir)

    # Generate summary report
    report_file = analyzer.generate_summary_report(df_stats, output_dir)

    # Print summary to console
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nKey Findings:")
    print(f"  • Average SNR Improvement: {df_stats['snr_improvement'].mean():.1f}%")
    print(f"  • Average Variability Reduction: {df_stats['shot_var_reduction'].mean():.1f}%")
    print(f"  • Samples with better SNR (20 shots): {(df_stats['snr_improvement'] > 0).sum()}/{len(df_stats)}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Full report: {report_file}")

    return df_stats


if __name__ == "__main__":
    df_results = main()