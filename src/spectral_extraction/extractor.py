"""
Spectral Feature Extraction Module

This module provides standalone feature extraction functionality for spectral data,
extracted from the LIBSsa project. It implements peak isolation, baseline correction,
and multi-peak fitting with various mathematical models.
"""

import numpy as np
from scipy.optimize import least_squares, OptimizeResult
from typing import List, Dict, Any, Optional, Callable, Tuple
import warnings

from .peak_shapes import PeakShapes
from .results import (
    PeakRegion, PeakFitResult, RegionExtractionResult, 
    FittingResult, FeatureExtractionResult
)


class SpectralFeatureExtractor:
    """
    A class for extracting features from spectral data through peak isolation and fitting.
    
    This class implements the core functionality for:
    1. Isolating spectral regions based on wavelength ranges
    2. Applying baseline correction and normalization
    3. Fitting mathematical models to peaks
    4. Extracting quantitative features (height, width, area)
    
    Attributes:
        peak_shapes (PeakShapes): Instance for accessing peak shape functions
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.peak_shapes = PeakShapes()
        
        # Default fitting tolerances
        self.fit_tolerances = {
            'ftol': 1e-7,    # Function tolerance
            'gtol': 1e-7,    # Gradient tolerance  
            'xtol': 1e-7,    # Parameter tolerance
            'max_nfev': 1000 # Maximum function evaluations
        }
    
    def isolate_peaks(self, 
                     wavelengths: np.ndarray,
                     spectra: np.ndarray,
                     regions: List[PeakRegion],
                     baseline_correction: bool = True,
                     area_normalization: bool = False,
                     progress_callback: Optional[Callable[[int], None]] = None) -> List[RegionExtractionResult]:
        """
        Isolate spectral peaks based on defined regions.
        
        Args:
            wavelengths: 1D array of wavelength values
            spectra: 2D array of shape (n_wavelengths, n_spectra)
            regions: List of PeakRegion objects defining extraction regions
            baseline_correction: Whether to apply linear baseline correction
            area_normalization: Whether to normalize by baseline area
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of RegionExtractionResult objects
            
        Raises:
            ValueError: If wavelength and spectra dimensions don't match
        """
        if len(wavelengths) != spectra.shape[0]:
            raise ValueError("Wavelength and spectra dimensions don't match")
        
        results = []
        
        for i, region in enumerate(regions):
            # Find wavelength indices for this region
            mask = (wavelengths >= region.lower_wavelength) & (wavelengths <= region.upper_wavelength)
            region_indices = np.where(mask)[0]
            
            if len(region_indices) == 0:
                raise ValueError(f"No wavelength points found in region {region.element}")
            
            # Extract region data
            region_wavelengths = wavelengths[region_indices]
            region_spectra = spectra[region_indices, :]
            
            # Apply baseline correction if requested
            corrected_spectra = region_spectra.copy()
            if baseline_correction:
                corrected_spectra = self._apply_baseline_correction(
                    region_wavelengths, region_spectra, area_normalization
                )
            
            # Calculate noise levels from start and end of region
            noise_levels = self._calculate_noise_levels(corrected_spectra)
            
            # Create result object
            result = RegionExtractionResult(
                region=region,
                wavelengths=region_wavelengths,
                isolated_spectra=corrected_spectra,
                baseline_corrected=baseline_correction,
                area_normalized=area_normalization,
                noise_levels=noise_levels
            )
            
            results.append(result)
            
            if progress_callback:
                progress_callback(i)
        
        return results
    
    def fit_peaks(self,
                 region_results: List[RegionExtractionResult],
                 peak_shapes: List[str],
                 fitting_mode: str = 'mean_first',
                 asymmetry_params: Optional[List[float]] = None,
                 progress_callback: Optional[Callable[[int], None]] = None) -> List[FittingResult]:
        """
        Fit mathematical models to isolated peaks.
        
        Args:
            region_results: Results from peak isolation
            peak_shapes: List of peak shape names for each region
            fitting_mode: 'mean_first' or 'area_first'
            asymmetry_params: Asymmetry parameters for each region (for asymmetric shapes)
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of FittingResult objects
            
        Raises:
            ValueError: If inputs have mismatched lengths or invalid parameters
        """
        if len(region_results) != len(peak_shapes):
            raise ValueError("Number of regions and peak shapes must match")
        
        if fitting_mode not in ['mean_first', 'area_first']:
            raise ValueError("Fitting mode must be 'mean_first' or 'area_first'")
        
        if asymmetry_params is None:
            asymmetry_params = [0.0] * len(region_results)
        elif len(asymmetry_params) != len(region_results):
            raise ValueError("Number of asymmetry parameters must match number of regions")
        
        results = []
        
        for i, (region_result, shape, asymmetry) in enumerate(zip(region_results, peak_shapes, asymmetry_params)):
            
            if fitting_mode == 'mean_first':
                fit_result = self._fit_mean_first(region_result, shape, asymmetry)
            else:
                fit_result = self._fit_area_first(region_result, shape, asymmetry)
            
            results.append(fit_result)
            
            if progress_callback:
                progress_callback(i)
        
        return results
    
    def extract_features(self,
                        wavelengths: np.ndarray,
                        spectra: np.ndarray,
                        regions: List[PeakRegion],
                        peak_shapes: List[str],
                        baseline_correction: bool = True,
                        area_normalization: bool = False,
                        fitting_mode: str = 'mean_first',
                        asymmetry_params: Optional[List[float]] = None,
                        progress_callback: Optional[Callable[[int], None]] = None) -> FeatureExtractionResult:
        """
        Complete feature extraction workflow.
        
        Args:
            wavelengths: 1D array of wavelength values
            spectra: 2D array of shape (n_wavelengths, n_spectra)
            regions: List of PeakRegion objects
            peak_shapes: List of peak shape names
            baseline_correction: Whether to apply baseline correction
            area_normalization: Whether to normalize by baseline area
            fitting_mode: 'mean_first' or 'area_first'
            asymmetry_params: Asymmetry parameters for asymmetric shapes
            progress_callback: Optional callback for progress updates
            
        Returns:
            FeatureExtractionResult containing all results
        """
        # Step 1: Isolate peaks
        region_results = self.isolate_peaks(
            wavelengths, spectra, regions, baseline_correction, 
            area_normalization, progress_callback
        )
        
        # Step 2: Fit peaks
        fitting_results = self.fit_peaks(
            region_results, peak_shapes, fitting_mode, 
            asymmetry_params, progress_callback
        )
        
        # Step 3: Package results
        processing_params = {
            'baseline_correction': baseline_correction,
            'area_normalization': area_normalization,
            'fitting_mode': fitting_mode,
            'peak_shapes': peak_shapes,
            'asymmetry_params': asymmetry_params,
            'fit_tolerances': self.fit_tolerances
        }
        
        return FeatureExtractionResult(
            region_results=region_results,
            fitting_results=fitting_results,
            processing_parameters=processing_params
        )
    
    def _apply_baseline_correction(self, 
                                  wavelengths: np.ndarray, 
                                  spectra: np.ndarray,
                                  area_norm: bool = False) -> np.ndarray:
        """Apply linear baseline correction to spectral region."""
        corrected = spectra.copy()
        
        # Use first and last 2 points for baseline estimation
        x_baseline = np.concatenate([wavelengths[:2], wavelengths[-2:]])
        
        for i in range(spectra.shape[1]):  # For each spectrum
            y_baseline = np.concatenate([spectra[:2, i], spectra[-2:, i]])
            
            # Fit linear baseline
            coeffs = np.polyfit(x_baseline, y_baseline, 1)
            baseline = np.polyval(coeffs, wavelengths)
            
            # Subtract baseline
            corrected[:, i] -= baseline
            
            # Area normalization if requested
            if area_norm:
                baseline_area = np.trapezoid(baseline, wavelengths)
                if baseline_area != 0:
                    corrected[:, i] /= baseline_area
        
        return corrected
    
    def _calculate_noise_levels(self, spectra: np.ndarray) -> Tuple[float, float]:
        """Calculate noise levels from start and end of spectral region."""
        avg_spectrum = np.mean(spectra, axis=1)
        n_points = len(avg_spectrum)
        
        # Use 20% of points or minimum 2 points
        n_edge = max(2, int(0.2 * n_points))
        
        start_noise = np.std(avg_spectrum[:n_edge])
        end_noise = np.std(avg_spectrum[-n_edge:])
        
        return (start_noise, end_noise)
    
    def _fit_mean_first(self, 
                       region_result: RegionExtractionResult, 
                       shape: str, 
                       asymmetry: float) -> FittingResult:
        """Fit peaks by first averaging spectra, then fitting the mean."""
        
        # Special case for trapezoidal integration
        if shape.lower() == 'trapezoidal':
            return self._perform_trapezoidal_integration(region_result)

        avg_spectrum = region_result.average_spectrum
        x_data = region_result.wavelengths
        
        # Generate initial guess
        initial_guess = self._generate_initial_guess(
            x_data, avg_spectrum, region_result.region.center_wavelengths, shape, asymmetry
        )
        
        # Perform fitting
        try:
            fit_func = self.peak_shapes.get_shape_function(shape.lower())
            
            # Create fitting function
            def residual_func(params):
                model = fit_func(x_data, *params, centers=region_result.region.center_wavelengths)
                return avg_spectrum - model
            
            result = least_squares(
                residual_func,
                initial_guess,
                **self.fit_tolerances
            )
            
            # Extract peak parameters and create results
            peak_results = self._extract_peak_parameters(
                x_data, avg_spectrum, result, shape, len(region_result.region.center_wavelengths), region_result.region.center_wavelengths
            )
            
            success_rate = 1.0 if result.success else 0.0
            
        except Exception as e:
            warnings.warn(f"Fitting failed for {region_result.region.element}: {str(e)}")
            # Create dummy results for failed fit
            peak_results = [
                PeakFitResult(
                    height=0.0, width=0.0, area=0.0, center=center,
                    shape=shape, success=False, n_function_evaluations=0,
                    residuals=np.zeros_like(avg_spectrum),
                    fitted_curve=np.zeros_like(avg_spectrum)
                )
                for center in region_result.region.center_wavelengths
            ]
            success_rate = 0.0
            result = type('MockResult', (), {'nfev': 0})()
        
        return FittingResult(
            region_result=region_result,
            fit_results=peak_results,
            fitting_mode='mean_first',
            shape_used=shape,
            convergence_rate=success_rate,
            average_iterations=result.nfev
        )
    
    def _fit_area_first(self, 
                       region_result: RegionExtractionResult, 
                       shape: str, 
                       asymmetry: float) -> FittingResult:
        """Fit peaks using area-first approach (fit each spectrum individually)."""
        wavelengths = region_result.wavelengths
        spectra = region_result.isolated_spectra
        n_peaks = len(region_result.region.center_wavelengths)
        n_spectra = spectra.shape[1]
        
        all_peak_results = []
        total_iterations = 0
        successful_fits = 0
        
        for i in range(n_spectra):
            spectrum = spectra[:, i]
            
            # Generate initial guess
            initial_guess = self._generate_initial_guess(
                wavelengths, spectrum, region_result.region.center_wavelengths, shape, asymmetry
            )
            
            # Perform fitting
            try:
                fit_func = self.peak_shapes.get_shape_function(shape.lower())
                
                def residual_func(params):
                    model = fit_func(wavelengths, *params, centers=region_result.region.center_wavelengths)
                    return spectrum - model
                
                result = least_squares(
                    residual_func,
                    initial_guess,
                    **self.fit_tolerances
                )
                
                peak_results = self._extract_peak_parameters(
                    wavelengths, spectrum, result, shape, n_peaks, region_result.region.center_wavelengths
                )
                
                all_peak_results.append(peak_results)
                total_iterations += result.nfev
                if result.success:
                    successful_fits += 1
                    
            except Exception:
                # Create dummy results for failed fit
                dummy_results = [
                    PeakFitResult(
                        height=0.0, width=0.0, area=0.0, center=center,
                        shape=shape, success=False, n_function_evaluations=0,
                        residuals=np.zeros_like(spectrum),
                        fitted_curve=np.zeros_like(spectrum)
                    )
                    for center in region_result.region.center_wavelengths
                ]
                all_peak_results.append(dummy_results)
        
        # Average results across all spectra
        averaged_results = self._average_peak_results(all_peak_results)
        
        success_rate = successful_fits / n_spectra if n_spectra > 0 else 0.0
        avg_iterations = total_iterations / n_spectra if n_spectra > 0 else 0
        
        return FittingResult(
            region_result=region_result,
            fit_results=averaged_results,
            fitting_mode='area_first',
            shape_used=shape,
            convergence_rate=success_rate,
            average_iterations=avg_iterations
        )
    
    def _generate_initial_guess(self, 
                               wavelengths: np.ndarray, 
                               spectrum: np.ndarray, 
                               centers: List[float], 
                               shape: str, 
                               asymmetry: float) -> List[float]:
        """Generate initial parameter guess for fitting."""
        guess = []
        n_peaks = len(centers)
        wavelength_range = wavelengths[-1] - wavelengths[0]
        max_intensity = np.max(spectrum)
        
        for i, center in enumerate(centers):
            # Scaling factor based on peak number
            scale = 1 - 0.4 * (i / max(n_peaks, 1))
            
            if 'voigt' in shape.lower():
                # For Voigt: area, width_l, width_g, center
                area_guess = scale * max_intensity * wavelength_range / 2
                width_l_guess = scale * wavelength_range / 4
                width_g_guess = scale * wavelength_range / 4
                guess.extend([area_guess, width_l_guess, width_g_guess])
                if 'fixed' not in shape.lower():
                    guess.append(center)
            else:
                # For other shapes: height, width, center
                height_guess = scale * max_intensity
                width_guess = scale * wavelength_range / 4
                guess.extend([height_guess, width_guess])
                if 'fixed' not in shape.lower():
                    guess.append(center)
                
                # Add asymmetry parameter if needed
                if 'asymmetric' in shape.lower() or 'asym' in shape.lower():
                    guess.append(asymmetry)
        
        return guess
    
    def _extract_peak_parameters(self, 
                                wavelengths: np.ndarray,
                                spectrum: np.ndarray,
                                fit_result: OptimizeResult,
                                shape: str,
                                n_peaks: int,
                                centers: List[float]) -> List[PeakFitResult]:
        """Extract individual peak parameters from fit result."""
        if shape.lower() == 'trapezoidal':
            # Simple integration
            area = np.trapz(spectrum, wavelengths)
            height = np.max(spectrum)
            width = wavelengths[-1] - wavelengths[0]
            center = centers[0] if centers else np.mean(wavelengths)
            
            return [PeakFitResult(
                height=height,
                width=width,
                area=area,
                center=center,
                shape=shape,
                success=True,
                n_function_evaluations=0,
                residuals=np.zeros_like(spectrum),
                fitted_curve=spectrum.copy()
            )]
        
        # Parse fitted parameters
        params = np.abs(fit_result.x)
        
        # Determine parameter structure based on shape
        if 'voigt' in shape.lower():
            n_params_per_peak = 4 if 'fixed' not in shape.lower() else 3
        else:
            n_params_per_peak = 3 if 'fixed' not in shape.lower() else 2
            if 'asymmetric' in shape.lower() or 'asym' in shape.lower():
                n_params_per_peak += 1
        
        # Split parameters for each peak
        param_groups = np.array_split(params, n_peaks)
        
        results = []
        for i, peak_params in enumerate(param_groups):
            if 'voigt' in shape.lower():
                area, width_l, width_g = peak_params[:3]
                center = peak_params[3] if len(peak_params) > 3 else centers[i]
                width = self.peak_shapes.calculate_fwhm(width_l, width_g, shape)
                height = np.max(spectrum)  # Approximate for Voigt
            else:
                height, width = peak_params[:2]
                center = peak_params[2] if len(peak_params) > 2 else centers[i]
                area = self.peak_shapes.calculate_peak_area(height, width, shape)
            
            # Generate fitted curve for this peak
            try:
                if 'voigt' in shape.lower():
                    fitted_curve = self.peak_shapes.voigt(wavelengths, area, width_l, width_g, center)
                elif 'gaussian' in shape.lower():
                    fitted_curve = self.peak_shapes.gaussian(wavelengths, height, width, center)
                else:  # Lorentzian and variants
                    fitted_curve = self.peak_shapes.lorentzian(wavelengths, height, width, center)
            except:
                fitted_curve = np.zeros_like(wavelengths)
            
            # Calculate residuals
            residuals = spectrum - fitted_curve
            
            result = PeakFitResult(
                height=height,
                width=width,
                area=area,
                center=center,
                shape=shape,
                success=fit_result.success,
                n_function_evaluations=fit_result.nfev,
                residuals=residuals,
                fitted_curve=fitted_curve
            )
            
            results.append(result)
        
        return results
    
    def _average_peak_results(self, all_results: List[List[PeakFitResult]]) -> List[PeakFitResult]:
        """Average peak results across multiple spectra."""
        if not all_results:
            return []
        
        n_peaks = len(all_results[0])
        averaged = []
        
        for peak_idx in range(n_peaks):
            # Collect parameters for this peak across all spectra
            heights = [results[peak_idx].height for results in all_results]
            widths = [results[peak_idx].width for results in all_results]
            areas = [results[peak_idx].area for results in all_results]
            centers = [results[peak_idx].center for results in all_results]
            successes = [results[peak_idx].success for results in all_results]
            n_evals = [results[peak_idx].n_function_evaluations for results in all_results]
            
            # Calculate averages
            avg_result = PeakFitResult(
                height=np.mean(heights),
                width=np.mean(widths),
                area=np.mean(areas),
                center=np.mean(centers),
                shape=all_results[0][peak_idx].shape,
                success=np.mean(successes) > 0.5,
                n_function_evaluations=int(np.mean(n_evals)),
                residuals=np.mean([res[peak_idx].residuals for res in all_results], axis=0),
                fitted_curve=np.mean([res[peak_idx].fitted_curve for res in all_results], axis=0)
            )
            
            averaged.append(avg_result)
        
        return averaged

    def _perform_trapezoidal_integration(self, region_result: RegionExtractionResult) -> FittingResult:
        """Perform simple trapezoidal integration for a region."""
        x_data = region_result.wavelengths
        n_peaks = region_result.region.n_peaks
        centers = region_result.region.center_wavelengths

        all_peak_results = []
        for i in range(region_result.n_spectra):
            spectrum = region_result.isolated_spectra[:, i]
            
            peak_results_for_shot = []
            if n_peaks == 1:
                # Integrate the whole region for a single "peak"
                area = self.peak_shapes.trapezoidal_integration(x_data, spectrum)
                peak_results_for_shot.append(PeakFitResult(
                    height=np.max(spectrum) if len(spectrum) > 0 else 0.0,
                    width=x_data[-1] - x_data[0] if len(x_data) > 1 else 0.0,
                    area=area,
                    center=centers[0],
                    shape='trapezoidal',
                    success=True,
                    n_function_evaluations=0,
                    residuals=np.array([]),
                    fitted_curve=np.array([])
                ))
            else:
                # For multiple peaks, divide the region and integrate each part
                boundaries = np.linspace(x_data[0], x_data[-1], n_peaks + 1)
                for j in range(n_peaks):
                    mask = (x_data >= boundaries[j]) & (x_data <= boundaries[j+1])
                    sub_x = x_data[mask]
                    sub_y = spectrum[mask]
                    
                    if len(sub_x) < 2:
                        area = 0.0
                    else:
                        area = self.peak_shapes.trapezoidal_integration(sub_x, sub_y)

                    peak_results_for_shot.append(PeakFitResult(
                        height=np.max(sub_y) if len(sub_y) > 0 else 0.0,
                        width=sub_x[-1] - sub_x[0] if len(sub_x) > 1 else 0.0,
                        area=area,
                        center=centers[j],
                        shape='trapezoidal',
                        success=True,
                        n_function_evaluations=0,
                        residuals=np.array([]),
                        fitted_curve=np.array([])
                    ))
            all_peak_results.append(peak_results_for_shot)
        
        # Average results across all shots
        final_peak_results = self._average_peak_results(all_peak_results)

        return FittingResult(
            region_result=region_result,
            shape_used='trapezoidal',
            fitting_mode='mean_first', # or 'integration'
            fit_results=final_peak_results,
            convergence_rate=1.0, # Always converges
            average_iterations=0 # No iterations
        )