"""
Utility functions, custom classes, and logging configuration.
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config.pipeline_config import config

def setup_logging() -> None:
    """Configures the logging for the entire application."""
    log_file_path = Path(config.log_dir) / config.log_file
    # Ensure logs directory exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=config.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file_path)
        ]
    )

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates a dictionary of comprehensive regression metrics.

    Args:
        y_true: Array of true target values.
        y_pred: Array of predicted values.

    Returns:
        A dictionary containing R-squared, RMSE, MAE, and other metrics.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Avoid division by zero for percentage-based metrics
    y_true_safe = np.where(y_true == 0, 1e-6, y_true)
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    rrmse = rmse / np.mean(y_true_safe) * 100
    
    relative_errors = np.abs((y_true - y_pred) / y_true_safe)
    within_20_5_percent = np.mean(relative_errors <= 0.205) * 100
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'rrmse': rrmse,
        'within_20.5%': within_20_5_percent
    }

class OutlierClipper(BaseEstimator, TransformerMixin):
    """A scikit-learn compatible transformer to clip extreme values."""
    def __init__(self, percentile: float = 99.9, factor: float = 1.5):
        self.percentile = percentile
        self.factor = factor
        self.clipping_values_ = None
        self.feature_names_in_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Calculates the clipping thresholds."""
        percentile_values = np.nanpercentile(X, self.percentile, axis=0)
        self.clipping_values_ = percentile_values * self.factor

        # Store feature names if input is DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]

        logging.info(f"OutlierClipper fitted for {X.shape[1]} features.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies clipping to the data."""
        X_copy = X.copy()
        if self.clipping_values_ is not None:
            clip_bounds = np.nan_to_num(self.clipping_values_, nan=np.inf)
            X_copy = np.minimum(X_copy, clip_bounds)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Returns feature names for output features."""
        if input_features is None:
            # Use stored feature names from fit
            if self.feature_names_in_ is not None:
                return np.array(self.feature_names_in_)
            # Fallback to generic names
            n_features = len(self.clipping_values_) if self.clipping_values_ is not None else 0
            return np.array([f"x{i}" for i in range(n_features)])
        else:
            # Return the same feature names as input (clipping doesn't change feature names)
            return np.array(input_features)