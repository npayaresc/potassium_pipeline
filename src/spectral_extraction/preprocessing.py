"""
Spectral Preprocessing Module for LIBS Data

This module provides preprocessing techniques specifically designed for
Laser-Induced Breakdown Spectroscopy (LIBS) data to reduce noise, normalize
intensities, and remove baseline effects.

Key features:
1. Savitzky-Golay smoothing - Reduces shot-to-shot noise
2. Standard Normal Variate (SNV) - Normalizes laser power variations
3. Asymmetric Least Squares (ALS) baseline - Advanced continuum removal
4. Configurable preprocessing pipeline

Author: Adapted for potassium_pipeline
Date: 2025-10-05
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Optional, Literal, Tuple
import logging

logger = logging.getLogger(__name__)


class SpectralPreprocessor:
    """
    Preprocessing utilities for LIBS spectroscopy data.

    This class implements standard spectral preprocessing techniques to:
    - Reduce measurement noise (Savitzky-Golay smoothing)
    - Normalize intensity variations (SNV)
    - Remove baseline/continuum effects (ALS)

    Typical usage:
        preprocessor = SpectralPreprocessor()
        preprocessor.configure(method='full')  # Enable all preprocessing
        clean_spectrum = preprocessor.preprocess(raw_spectrum)
    """

    def __init__(self):
        """Initialize preprocessor with default parameters."""
        self.method = 'none'  # Default: no preprocessing

        # Savitzky-Golay parameters
        self.savgol_window = 11  # Window length (odd number)
        self.savgol_polyorder = 2  # Polynomial order

        # SNV parameters
        self.snv_epsilon = 1e-10  # Minimum std to avoid division by zero

        # ALS baseline parameters
        self.als_lambda = 1e6  # Smoothness (larger = smoother)
        self.als_p = 0.01  # Asymmetry (smaller = hugs bottom more)
        self.als_niter = 10  # Number of iterations

    def configure(self,
                 method: Literal['none', 'savgol', 'snv', 'baseline',
                                'savgol+snv', 'baseline+snv', 'full'] = 'full',
                 savgol_window: int = 11,
                 savgol_polyorder: int = 2,
                 als_lambda: float = 1e6,
                 als_p: float = 0.01,
                 als_niter: int = 10):
        """
        Configure preprocessing method and parameters.

        Args:
            method: Preprocessing pipeline to use:
                - 'none': No preprocessing
                - 'savgol': Savitzky-Golay smoothing only
                - 'snv': SNV normalization only
                - 'baseline': ALS baseline correction only
                - 'savgol+snv': Smoothing + normalization (RECOMMENDED for Phase 1)
                - 'baseline+snv': Advanced baseline + normalization
                - 'full': All steps (baseline + smoothing + SNV) (Phase 2)
            savgol_window: Savitzky-Golay window length (odd number, 7-15 recommended)
            savgol_polyorder: Polynomial order (2 recommended for LIBS)
            als_lambda: ALS smoothness parameter (1e6 recommended for LIBS)
            als_p: ALS asymmetry parameter (0.01 recommended for LIBS)
            als_niter: ALS iterations (10 usually sufficient)

        Raises:
            ValueError: If parameters are invalid
        """
        if method not in ['none', 'savgol', 'snv', 'baseline', 'savgol+snv', 'baseline+snv', 'full']:
            raise ValueError(f"Invalid method: {method}")

        if savgol_window % 2 == 0:
            raise ValueError("savgol_window must be odd")

        if savgol_window < 3:
            raise ValueError("savgol_window must be >= 3")

        if savgol_polyorder >= savgol_window:
            raise ValueError("savgol_polyorder must be < savgol_window")

        self.method = method
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        self.als_lambda = als_lambda
        self.als_p = als_p
        self.als_niter = als_niter

        logger.debug(f"Spectral preprocessing configured: method={method}")
        if method in ['savgol', 'savgol+snv', 'full']:
            logger.debug(f"  Savitzky-Golay: window={savgol_window}, poly={savgol_polyorder}")
        if method in ['baseline', 'baseline+snv', 'full']:
            logger.debug(f"  ALS baseline: lambda={als_lambda}, p={als_p}, niter={als_niter}")

    def preprocess(self,
                   spectrum: np.ndarray,
                   wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply configured preprocessing to a spectrum.

        Args:
            spectrum: 1D array of intensities (n_wavelengths,)
            wavelengths: Optional 1D array of wavelengths (for ALS baseline)
                        If None and ALS is needed, uses indices

        Returns:
            Preprocessed spectrum (same shape as input)

        Raises:
            ValueError: If spectrum is invalid
        """
        if spectrum.ndim != 1:
            raise ValueError("Spectrum must be 1D array")

        if self.method == 'none':
            return spectrum.copy()

        # Handle short spectra gracefully - skip preprocessing if too short
        spectrum_length = len(spectrum)
        if spectrum_length < 3:
            # Too short for any preprocessing
            logger.warning(f"Spectrum too short ({spectrum_length} points) - returning original spectrum")
            return spectrum.copy()

        result = spectrum.copy()

        # Determine if we can apply Savgol smoothing
        can_apply_savgol = True
        effective_window = self.savgol_window

        if self.method in ['savgol', 'savgol+snv', 'full']:
            if spectrum_length < self.savgol_window:
                # Auto-adjust window to largest valid odd number
                effective_window = spectrum_length if spectrum_length % 2 == 1 else spectrum_length - 1
                if effective_window < 3:
                    can_apply_savgol = False
                    logger.warning(
                        f"Spectrum too short ({spectrum_length} points) for Savgol (window={self.savgol_window}) - skipping smoothing"
                    )
                else:
                    logger.debug(
                        f"Auto-adjusted Savgol window from {self.savgol_window} to {effective_window} for short spectrum ({spectrum_length} points)"
                    )

        # Step 1: Baseline correction (if requested)
        if self.method in ['baseline', 'baseline+snv', 'full']:
            if wavelengths is None:
                wavelengths = np.arange(len(spectrum))
            result = self._apply_als_baseline(result, wavelengths)

        # Step 2: Savitzky-Golay smoothing (if requested and possible)
        if self.method in ['savgol', 'savgol+snv', 'full'] and can_apply_savgol:
            result = self._apply_savgol_filter(result, window=effective_window)

        # Step 3: SNV normalization (if requested)
        if self.method in ['snv', 'savgol+snv', 'baseline+snv', 'full']:
            result = self._apply_snv(result)

        return result

    def preprocess_batch(self,
                        spectra: np.ndarray,
                        wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply preprocessing to multiple spectra.

        Args:
            spectra: 2D array of shape (n_wavelengths, n_spectra)
            wavelengths: Optional 1D array of wavelengths

        Returns:
            Preprocessed spectra (same shape as input)
        """
        if spectra.ndim != 2:
            raise ValueError("Spectra must be 2D array (n_wavelengths, n_spectra)")

        n_wavelengths, n_spectra = spectra.shape
        preprocessed = np.zeros_like(spectra)

        for i in range(n_spectra):
            preprocessed[:, i] = self.preprocess(spectra[:, i], wavelengths)

        return preprocessed

    def _apply_savgol_filter(self, spectrum: np.ndarray, window: Optional[int] = None) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing filter.

        This method reduces noise while preserving peak shapes by fitting
        a polynomial to a moving window of data points.

        Args:
            spectrum: 1D array of intensities
            window: Optional custom window size (uses self.savgol_window if None)

        Returns:
            Smoothed spectrum
        """
        if window is None:
            window = self.savgol_window

        # Ensure polynomial order doesn't exceed window size
        polyorder = min(self.savgol_polyorder, window - 1)

        try:
            smoothed = savgol_filter(
                spectrum,
                window_length=window,
                polyorder=polyorder,
                mode='nearest'  # Handle edges by using nearest value
            )
            return smoothed
        except Exception as e:
            logger.warning(f"Savitzky-Golay filtering failed: {e}. Returning original spectrum.")
            return spectrum.copy()

    def _apply_snv(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Apply Standard Normal Variate (SNV) normalization.

        SNV removes multiplicative scatter and intensity variations by
        centering to zero mean and scaling to unit variance.

        This is crucial for LIBS as it corrects for:
        - Laser power drift between sessions
        - Instrument-to-instrument variations
        - Day-to-day intensity fluctuations

        Args:
            spectrum: 1D array of intensities

        Returns:
            SNV-normalized spectrum
        """
        mean = np.mean(spectrum)
        std = np.std(spectrum)

        # Avoid division by zero for flat spectra
        if std < self.snv_epsilon:
            logger.warning("Spectrum has near-zero std, SNV normalization skipped")
            return np.zeros_like(spectrum)

        return (spectrum - mean) / std

    def _apply_als_baseline(self,
                           spectrum: np.ndarray,
                           wavelengths: np.ndarray) -> np.ndarray:
        """
        Apply Asymmetric Least Squares (ALS) baseline correction.

        ALS fits a smooth baseline that stays BELOW peaks (asymmetric weighting).
        This is superior to linear baseline for LIBS because:
        - Handles curved continuum emission (bremsstrahlung)
        - Doesn't cut through peaks
        - Uses all wavelength points, not just edges

        Reference: Eilers & Boelens (2005), Analytical Chemistry

        Args:
            spectrum: 1D array of intensities
            wavelengths: 1D array of wavelength values (not used but kept for API)

        Returns:
            Baseline-corrected spectrum
        """
        try:
            L = len(spectrum)

            # Build difference matrix (2nd order)
            D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))

            # Initialize weights (all equal initially)
            w = np.ones(L)

            # Iteratively fit baseline
            for i in range(self.als_niter):
                # Build weighted least squares system
                W = sparse.spdiags(w, 0, L, L)
                Z = W + self.als_lambda * D.dot(D.transpose())

                # Solve for baseline
                z = spsolve(Z, w * spectrum)

                # Update weights (asymmetric)
                # Points above baseline get lower weight (p)
                # Points below baseline get higher weight (1-p)
                w = self.als_p * (spectrum > z) + (1 - self.als_p) * (spectrum < z)

            # Subtract baseline
            return spectrum - z

        except Exception as e:
            logger.warning(f"ALS baseline correction failed: {e}. Returning original spectrum.")
            return spectrum.copy()

    def get_config(self) -> dict:
        """
        Get current preprocessing configuration.

        Returns:
            Dictionary with current settings
        """
        return {
            'method': self.method,
            'savgol_window': self.savgol_window,
            'savgol_polyorder': self.savgol_polyorder,
            'als_lambda': self.als_lambda,
            'als_p': self.als_p,
            'als_niter': self.als_niter
        }


# Convenience functions for quick preprocessing

def preprocess_libs_spectrum(spectrum: np.ndarray,
                             wavelengths: Optional[np.ndarray] = None,
                             method: str = 'savgol+snv',
                             **kwargs) -> np.ndarray:
    """
    Convenience function for quick preprocessing of a single LIBS spectrum.

    Args:
        spectrum: 1D array of intensities
        wavelengths: Optional wavelength array
        method: Preprocessing method (see SpectralPreprocessor.configure)
        **kwargs: Additional parameters for SpectralPreprocessor.configure

    Returns:
        Preprocessed spectrum

    Example:
        >>> raw = np.random.randn(100) + 10
        >>> clean = preprocess_libs_spectrum(raw, method='savgol+snv')
    """
    preprocessor = SpectralPreprocessor()
    preprocessor.configure(method=method, **kwargs)
    return preprocessor.preprocess(spectrum, wavelengths)


def preprocess_libs_batch(spectra: np.ndarray,
                          wavelengths: Optional[np.ndarray] = None,
                          method: str = 'savgol+snv',
                          **kwargs) -> np.ndarray:
    """
    Convenience function for preprocessing multiple LIBS spectra.

    Args:
        spectra: 2D array (n_wavelengths, n_spectra)
        wavelengths: Optional wavelength array
        method: Preprocessing method
        **kwargs: Additional parameters

    Returns:
        Preprocessed spectra

    Example:
        >>> raw_batch = np.random.randn(100, 50) + 10  # 50 spectra
        >>> clean_batch = preprocess_libs_batch(raw_batch, method='full')
    """
    preprocessor = SpectralPreprocessor()
    preprocessor.configure(method=method, **kwargs)
    return preprocessor.preprocess_batch(spectra, wavelengths)


# Module-level instance for global configuration
_global_preprocessor = SpectralPreprocessor()


def configure_global_preprocessing(method: str = 'none', **kwargs):
    """
    Configure global preprocessing settings for the module.

    This allows setting preprocessing once and using it throughout the pipeline.

    Args:
        method: Preprocessing method
        **kwargs: Additional parameters

    Example:
        >>> from src.spectral_extraction.preprocessing import configure_global_preprocessing
        >>> configure_global_preprocessing(method='full', savgol_window=11)
    """
    _global_preprocessor.configure(method=method, **kwargs)


def get_global_preprocessor() -> SpectralPreprocessor:
    """
    Get the global preprocessor instance.

    Returns:
        Global SpectralPreprocessor instance
    """
    return _global_preprocessor
