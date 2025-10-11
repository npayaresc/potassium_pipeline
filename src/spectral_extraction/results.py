"""
Data structures for storing feature extraction results.

This module defines classes for organizing and accessing the results
of spectral feature extraction operations.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class PeakRegion:
    """Definition of a spectral peak region."""
    element: str
    lower_wavelength: float
    upper_wavelength: float
    center_wavelengths: List[float]
    
    @property
    def n_peaks(self) -> int:
        """Number of peaks defined in the region."""
        return len(self.center_wavelengths)
    
    def __post_init__(self):
        """Validate region definition."""
        if self.lower_wavelength >= self.upper_wavelength:
            raise ValueError("Lower wavelength must be less than upper wavelength")
        
        for center in self.center_wavelengths:
            if not (self.lower_wavelength <= center <= self.upper_wavelength):
                raise ValueError(f"Center wavelength {center} outside region bounds")


@dataclass
class PeakFitResult:
    """Results from fitting a single peak."""
    height: float
    width: float
    area: float
    center: float
    shape: str
    success: bool
    n_function_evaluations: int
    residuals: np.ndarray
    fitted_curve: np.ndarray

    # Physics-informed parameters (NEW)
    fwhm: float = 0.0  # Full Width at Half Maximum
    gamma: float = 0.0  # Lorentzian/Stark broadening parameter
    fit_quality: float = 0.0  # R² of the fit
    peak_asymmetry: float = 0.0  # Asymmetry index from residuals
    amplitude: float = 0.0  # Peak amplitude (for Lorentzian)
    kurtosis: float = 0.0  # Peak kurtosis (tailedness measure)

    @property
    def signal_to_noise(self) -> float:
        """Calculate signal-to-noise ratio."""
        if len(self.residuals) > 0:
            noise_std = np.std(self.residuals)
            return self.height / noise_std if noise_std > 0 else np.inf
        return np.inf

    @property
    def stark_broadening(self) -> float:
        """Alias for gamma parameter (Stark broadening indicator)."""
        return self.gamma

    @property
    def is_reliable_fit(self) -> float:
        """Check if fit is reliable based on R² threshold."""
        return self.fit_quality > 0.8


@dataclass
class RegionExtractionResult:
    """Results from extracting a single spectral region."""
    region: PeakRegion
    wavelengths: np.ndarray
    isolated_spectra: np.ndarray  # Shape: (n_wavelengths, n_spectra)
    baseline_corrected: bool
    area_normalized: bool
    noise_levels: Tuple[float, float]  # (start_noise, end_noise)
    
    @property
    def n_spectra(self) -> int:
        """Number of spectra in the region."""
        return self.isolated_spectra.shape[1]
    
    @property
    def n_wavelengths(self) -> int:
        """Number of wavelength points in the region."""
        return len(self.wavelengths)
    
    @property
    def average_spectrum(self) -> np.ndarray:
        """Calculate average spectrum across all shots."""
        return np.mean(self.isolated_spectra, axis=1)
    
    @property
    def spectral_range(self) -> float:
        """Wavelength range of the region."""
        return self.wavelengths[-1] - self.wavelengths[0]


@dataclass
class FittingResult:
    """Results from fitting peaks in a spectral region."""
    region_result: RegionExtractionResult
    fit_results: List[PeakFitResult]  # One per peak in region
    fitting_mode: str  # 'mean_first' or 'area_first'
    shape_used: str
    convergence_rate: float
    average_iterations: float
    
    @property
    def total_area(self) -> float:
        """Total area of all fitted peaks."""
        return sum(peak.area for peak in self.fit_results)
    
    @property
    def peak_heights(self) -> List[float]:
        """Heights of all fitted peaks."""
        return [peak.height for peak in self.fit_results]
    
    @property
    def peak_areas(self) -> List[float]:
        """Areas of all fitted peaks."""
        return [peak.area for peak in self.fit_results]
    
    @property
    def peak_widths(self) -> List[float]:
        """Widths of all fitted peaks."""
        return [peak.width for peak in self.fit_results]
    
    @property
    def peak_centers(self) -> List[float]:
        """Center positions of all fitted peaks."""
        return [peak.center for peak in self.fit_results]
    
    @property
    def successful_fits(self) -> int:
        """Number of successful peak fits."""
        return sum(1 for peak in self.fit_results if peak.success)


@dataclass
class FeatureExtractionResult:
    """Complete results from feature extraction on multiple regions."""
    region_results: List[RegionExtractionResult]
    fitting_results: List[FittingResult]
    processing_parameters: Dict[str, Any]
    
    @property
    def n_regions(self) -> int:
        """Number of processed regions."""
        return len(self.region_results)
    
    @property
    def element_names(self) -> List[str]:
        """Names of all processed elements."""
        return [result.region.element for result in self.region_results]
    
    @property
    def total_peaks_fitted(self) -> int:
        """Total number of peaks fitted across all regions."""
        return sum(len(result.fit_results) for result in self.fitting_results)
    
    @property
    def overall_success_rate(self) -> float:
        """Overall fitting success rate across all peaks."""
        if self.total_peaks_fitted == 0:
            return 0.0
        
        successful = sum(result.successful_fits for result in self.fitting_results)
        return successful / self.total_peaks_fitted
    
    def get_region_by_element(self, element: str) -> Optional[RegionExtractionResult]:
        """Get region result by element name."""
        for result in self.region_results:
            if result.region.element == element:
                return result
        return None
    
    def get_fitting_by_element(self, element: str) -> Optional[FittingResult]:
        """Get fitting result by element name."""
        for result in self.fitting_results:
            if result.region_result.region.element == element:
                return result
        return None
    
    def get_peak_areas_matrix(self) -> np.ndarray:
        """
        Get matrix of peak areas for all elements and peaks.
        
        Returns:
            2D array where rows are elements and columns are peak indices
        """
        max_peaks = max(len(result.fit_results) for result in self.fitting_results)
        matrix = np.full((len(self.fitting_results), max_peaks), np.nan)
        
        for i, result in enumerate(self.fitting_results):
            for j, peak in enumerate(result.fit_results):
                matrix[i, j] = peak.area
        
        return matrix
    
    def get_peak_heights_matrix(self) -> np.ndarray:
        """
        Get matrix of peak heights for all elements and peaks.
        
        Returns:
            2D array where rows are elements and columns are peak indices
        """
        max_peaks = max(len(result.fit_results) for result in self.fitting_results)
        matrix = np.full((len(self.fitting_results), max_peaks), np.nan)
        
        for i, result in enumerate(self.fitting_results):
            for j, peak in enumerate(result.fit_results):
                matrix[i, j] = peak.height
        
        return matrix
    
    def get_feature_vector(self, feature_type: str = 'area') -> np.ndarray:
        """
        Get flattened feature vector for machine learning applications.
        
        Args:
            feature_type: Type of feature ('area', 'height', 'width')
            
        Returns:
            1D array of features
        """
        if feature_type == 'area':
            matrix = self.get_peak_areas_matrix()
        elif feature_type == 'height':
            matrix = self.get_peak_heights_matrix()
        elif feature_type == 'width':
            max_peaks = max(len(result.fit_results) for result in self.fitting_results)
            matrix = np.full((len(self.fitting_results), max_peaks), np.nan)
            for i, result in enumerate(self.fitting_results):
                for j, peak in enumerate(result.fit_results):
                    matrix[i, j] = peak.width
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Flatten and remove NaN values
        flat = matrix.flatten()
        return flat[~np.isnan(flat)]
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert results to a summary dictionary."""
        return {
            'n_regions': self.n_regions,
            'elements': self.element_names,
            'total_peaks': self.total_peaks_fitted,
            'success_rate': self.overall_success_rate,
            'processing_parameters': self.processing_parameters,
            'region_summaries': [
                {
                    'element': result.region.element,
                    'wavelength_range': (result.region.lower_wavelength, result.region.upper_wavelength),
                    'n_spectra': result.n_spectra,
                    'noise_levels': result.noise_levels
                }
                for result in self.region_results
            ],
            'fitting_summaries': [
                {
                    'element': result.region_result.region.element,
                    'n_peaks': len(result.fit_results),
                    'convergence_rate': result.convergence_rate,
                    'peak_areas': result.peak_areas,
                    'peak_heights': result.peak_heights
                }
                for result in self.fitting_results
            ]
        }