"""
Reporting Module

Handles the aggregation, visualization, and saving of model performance
metrics and prediction results.
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import optuna
import json

logger = logging.getLogger(__name__)

class Reporter:
    """Aggregates and saves all results from a pipeline run."""

    def __init__(self, config):
        """
        Initializes the Reporter.

        Args:
            config: The pipeline configuration object.
        """
        self.config = config
        self.results = []
    
    def _get_display_strategy_name(self, strategy: str) -> str:
        """
        Returns the strategy name to use for display purposes.
        When raw_spectral_data is enabled, returns 'raw-spectral' instead of the original strategy.
        
        Args:
            strategy: The original strategy name
            
        Returns:
            The strategy name to use for display
        """
        if self.config.use_raw_spectral_data:
            return "raw-spectral"
        return strategy

    def add_run_results(self, strategy: str, model_name: str, metrics: dict, params: dict):
        """
        Adds the results of a single model training run to the aggregate list.

        Args:
            strategy: The feature strategy used (e.g., 'simple_only').
            model_name: The name of the model trained (e.g., 'random_forest').
            metrics: A dictionary of performance metrics (e.g., {'r2': 0.9, 'rmse': 0.1}).
            params: A dictionary of the model's parameters.
        """
        display_strategy = self._get_display_strategy_name(strategy)
        result_entry = {
            'strategy': display_strategy,
            'model_name': model_name,
            **metrics,
            'params': params,
        }
        self.results.append(result_entry)
        logger.debug(f"Added results for {display_strategy}/{model_name}.")

    def save_summary_report(self, strategy: str = None, model_name: str = None):
        """
        Saves the aggregated results of all runs to a CSV file and prints a summary.
        
        Args:
            strategy: Optional strategy name to include in filename
            model_name: Optional model name to include in filename
        """
        if not self.results:
            logger.warning("No results to report. Skipping summary report.")
            return

        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values(by='r2', ascending=False).reset_index(drop=True)

        # Build filename with strategy and model names if provided
        filename_parts = ["training_summary"]
        if strategy:
            display_strategy = self._get_display_strategy_name(strategy)
            filename_parts.append(display_strategy)
        if model_name:
            filename_parts.append(model_name)
        if self.config.use_raw_spectral_data:
            filename_parts.append("raw-spectral")
        filename_parts.append(self.config.run_timestamp)
        
        filename = "_".join(filename_parts) + ".csv"
        report_path = self.config.reports_dir / filename
        results_df.to_csv(report_path, index=False)
        logger.info(f"Full training summary saved to: {report_path}")

        # Print a formatted summary to the console
        print("\n--- MODEL TRAINING SUMMARY ---")
        display_cols = ['strategy', 'model_name', 'r2', 'rmse', 'mae', 'mape', 'rrmse', 'within_20.5%']
        # Ensure all display columns exist, filling with NaN if not
        for col in display_cols:
            if col not in results_df.columns:
                results_df[col] = np.nan
        
        print(results_df[display_cols].to_string(index=False, float_format="%.4f"))
        print("----------------------------\n")

    def save_prediction_results(
        self, y_true: pd.Series, y_pred: np.ndarray,
        sample_ids: pd.Series, strategy: str, model_name: str
    ):
        """
        Saves detailed prediction results (true vs. predicted) for a specific model.

        Args:
            y_true: Series of true target values.
            y_pred: Array of predicted values.
            sample_ids: Series of sample IDs corresponding to the test set.
            strategy: The feature strategy used.
            model_name: The name of the model.
        """
        predictions_df = pd.DataFrame({
            'sampleId': sample_ids,
            'ElementValue': y_true,
            'PredictedValue': y_pred
        })
        
        display_strategy = self._get_display_strategy_name(strategy)
        filename_parts = ["predictions", display_strategy, model_name]
        if self.config.use_raw_spectral_data:
            filename_parts.append("raw-spectral")
        filename_parts.append(self.config.run_timestamp)
        filename = "_".join(filename_parts) + ".csv"
        save_path = self.config.reports_dir / filename
        predictions_df.to_csv(save_path, index=False)
        logger.info(f"Predictions for {display_strategy}/{model_name} saved to: {save_path}")
        
        return predictions_df # Return for further analysis

    def generate_calibration_plot(self, predictions_df: pd.DataFrame, strategy: str, model_name: str):
        """
        Generates and saves a calibration plot (predicted vs. actual).

        Args:
            predictions_df: DataFrame with 'ElementValue' and 'PredictedValue'.
            strategy: The feature strategy used.
            model_name: The name of the model.
        """
        y_true = predictions_df['ElementValue']
        y_pred = predictions_df['PredictedValue']
        
        r2 = r2_score(y_true, y_pred)
        display_strategy = self._get_display_strategy_name(strategy)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', label='Predicted vs. Actual')
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        plt.plot(lims, lims, 'r--', alpha=0.75, lw=2, label='Ideal 1:1 Line')
        
        plt.title(f'Calibration Curve: {display_strategy} / {model_name}', fontsize=16)
        plt.xlabel('Actual Magnesium Concentration (%)', fontsize=12)
        plt.ylabel('Predicted Magnesium Concentration (%)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        
        textstr = f'$R^2 = {r2:.4f}$'
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        filename_parts = ["calibration_plot", display_strategy, model_name]
        if self.config.use_raw_spectral_data:
            filename_parts.append("raw-spectral")
        filename_parts.append(self.config.run_timestamp)
        plot_filename = "_".join(filename_parts) + ".png"
        save_path = self.config.reports_dir / plot_filename
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Calibration plot saved to: {save_path}")
        
    def save_tuning_report(self, study: optuna.Study, strategy: str, model_name: str = None):
        """
        Saves the results of a hyperparameter tuning study to a CSV file.

        Args:
            study: The completed Optuna study object.
            strategy: The feature strategy used for the tuning run.
            model_name: Optional model name to include in filename.
        """
        results_df = study.trials_dataframe()
        display_strategy = self._get_display_strategy_name(strategy)
        
        # Build filename with strategy and model name
        filename_parts = ["tuning_report", display_strategy]
        if model_name:
            filename_parts.append(model_name)
        if self.config.use_raw_spectral_data:
            filename_parts.append("raw-spectral")
        filename_parts.append(self.config.run_timestamp)
        
        filename = "_".join(filename_parts) + ".csv"
        report_path = self.config.reports_dir / filename
        results_df.to_csv(report_path, index=False)
        logger.info(f"Tuning study report for strategy '{display_strategy}'{f' and model {model_name}' if model_name else ''} saved to: {report_path}")

    def save_validation_metrics(self, strategy: str, model_name: str, metrics: dict, params: dict):
        """
        Saves validation metrics from hyperparameter tuning to a separate CSV file.
        
        Args:
            strategy: The feature strategy used.
            model_name: The name of the model.
            metrics: Dictionary of validation metrics.
            params: Dictionary of model parameters.
        """
        display_strategy = self._get_display_strategy_name(strategy)
        
        # Create a single row with all the information
        validation_data = {
            'strategy': display_strategy,
            'model_name': model_name,
            'timestamp': self.config.run_timestamp,
            **metrics,  # Unpack all metrics (r2, rmse, mae, etc.)
            'params': str(params)  # Convert params dict to string
        }
        
        # Convert to DataFrame
        validation_df = pd.DataFrame([validation_data])
        
        # Save to CSV file
        filename_parts = ["validation_metrics", display_strategy, model_name]
        if self.config.use_raw_spectral_data:
            filename_parts.append("raw-spectral")
        filename_parts.append(self.config.run_timestamp)
        filename = "_".join(filename_parts) + ".csv"
        save_path = self.config.reports_dir / filename
        validation_df.to_csv(save_path, index=False)
        
        logger.info(f"Validation metrics for {display_strategy}/{model_name} saved to: {save_path}")
        
        return save_path

    def save_config(self):
        """
        Saves the pipeline configuration to a JSON file with timestamp.
        
        Returns:
            Path: The path where the config was saved.
        """
        # Convert the config to a dictionary, handling Pydantic serialization
        config_dict = self.config.model_dump()
        
        # Remove non-serializable path objects and convert to strings
        for key, value in config_dict.items():
            if hasattr(value, '__fspath__'):  # Path objects
                config_dict[key] = str(value)
        
        # Create the filename with timestamp
        filename_parts = ["pipeline_config"]
        if self.config.use_raw_spectral_data:
            filename_parts.append("raw-spectral")
        filename_parts.append(self.config.run_timestamp)
        config_filename = "_".join(filename_parts) + ".json"
        save_path = self.config.reports_dir / config_filename
        
        # Save the config as JSON
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"Pipeline configuration saved to: {save_path}")
        return save_path