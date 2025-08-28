"""
Data Cleansing Module: Performs outlier detection and removal on spectral data.
"""
import logging
import numpy as np
import pandas as pd

from src.config.pipeline_config import Config
from src.outlier_detection.detector import SpectralOutlierDetector

logger = logging.getLogger(__name__)

class DataCleanser:
    """A class to perform data cleansing operations on spectral data."""
    def __init__(self, config: Config):
        self.config = config
        self.outlier_detector = SpectralOutlierDetector(
            method=config.outlier_method, threshold=config.outlier_threshold
        )
        self.audit_log = []

    def clean_spectra(self, file_path: str, intensities: np.ndarray) -> np.ndarray:
        """Detects and removes outliers from a single spectral dataset."""
        # Handle 1D array (single spectrum) - no outlier detection needed
        if intensities.ndim == 1:
            self.audit_log.append({"file": file_path, "status": "single_spectrum", "removed": 0, "total": 1})
            return intensities
            
        n_total = intensities.shape[1]
        if n_total == 0:
            self.audit_log.append({"file": file_path, "status": "skipped", "reason": "no spectra"})
            return np.array([])
        try:
            clean, outliers = self.outlier_detector.detect_outliers_single(intensities)
            n_removed = len(outliers)
            outlier_pct = (n_removed / n_total) * 100

            if outlier_pct > self.config.max_outlier_percentage:
                logger.debug(f"Skipping {file_path}: Excessive outliers ({outlier_pct:.1f}%)")
                self.audit_log.append({"file": file_path, "status": "skipped", "reason": "excessive outliers"})
                return np.array([])
            
            self.audit_log.append({"file": file_path, "status": "cleaned", "removed": n_removed, "total": n_total})
            # If clean is 2D with single spectrum, flatten to 1D
            if clean.ndim == 2 and clean.shape[1] == 1:
                return clean.flatten()
            return clean
        except ValueError as e:
            logger.error(f"Outlier detection failed for {file_path}: {e}")
            self.audit_log.append({"file": file_path, "status": "failed", "error": str(e)})
            return np.array([])

    def get_audit_report(self) -> pd.DataFrame:
        """Returns the cleansing operations as a pandas DataFrame."""
        return pd.DataFrame(self.audit_log)