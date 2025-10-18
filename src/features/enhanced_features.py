"""
Enhanced Feature Engineering Module for Crop Potassium Prediction
Implements advanced spectral features including molecular bands, ratios, and plasma indicators.
"""
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class MolecularBandDetector:
    """Detects and quantifies molecular band features in LIBS spectra."""
    
    def detect_cn_bands(self, wavelengths: np.ndarray, intensities: np.ndarray, 
                       band_ranges: List[Tuple[float, float]]) -> Dict[str, float]:
        """Detect CN molecular bands in specified wavelength ranges."""
        features = {}
        
        for i, (lower, upper) in enumerate(band_ranges):
            mask = (wavelengths >= lower) & (wavelengths <= upper)
            if not np.any(mask):
                features[f'CN_band_{i}_intensity'] = np.nan
                features[f'CN_band_{i}_area'] = np.nan
                continue
                
            band_intensities = intensities[mask]
            band_wavelengths = wavelengths[mask]
            
            # Band intensity metrics
            features[f'CN_band_{i}_intensity'] = np.max(band_intensities)
            features[f'CN_band_{i}_area'] = np.trapezoid(band_intensities, band_wavelengths)
            
            # Band structure indicators
            if len(band_intensities) > 3:
                # Detect band head
                peaks, _ = signal.find_peaks(band_intensities, prominence=0.1*np.max(band_intensities))
                features[f'CN_band_{i}_n_peaks'] = len(peaks)
            else:
                features[f'CN_band_{i}_n_peaks'] = 0
                
        return features
    
    def calculate_molecular_atomic_ratio(self, molecular_intensity: float, 
                                       atomic_intensity: float) -> float:
        """Calculate molecular to atomic line intensity ratio."""
        if np.isnan(molecular_intensity) or np.isnan(atomic_intensity) or atomic_intensity < 1e-6:
            return np.nan
        return molecular_intensity / atomic_intensity


class AdvancedRatioCalculator:
    """Calculates advanced elemental ratios for potassium prediction."""

    def __init__(self):
        # Track missing elements to log only once
        self._missing_elements_logged = set()

    def calculate_nutrient_ratios(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate various nutrient ratios relevant to crop potassium."""
        ratios = {}

        # Define key features for ratio calculations
        key_features = ['K_I_simple_peak_height', 'K_I_peak_0', 'C_I_simple_peak_height', 'CA_I_help_simple_peak_height', 'N_I_help_simple_peak_height']

        # Helper function for safe division
        def safe_ratio(num: float, den: float, name: str) -> float:
            if np.isnan(num) or np.isnan(den) or den < 1e-12:
                return np.nan
            result = np.clip(num / den, -100, 100)
            return result

        # Helper function to get intensity from either full or simple features
        def get_intensity(features: Dict[str, float], element: str, metric: str = 'peak_height') -> float:
            """Get intensity from either {element}_peak_0 or {element}_simple_{metric} format."""
            # Try full feature format first
            full_key = f'{element}_peak_0'
            if full_key in features and not np.isnan(features[full_key]):
                return features[full_key]

            # Try simple feature format
            simple_key = f'{element}_simple_{metric}'
            if simple_key in features and not np.isnan(features[simple_key]):
                return features[simple_key]

            # Log what we couldn't find (only once per element per session)
            if element not in self._missing_elements_logged:
                logger.debug(f"[ENHANCED FEATURES] Element {element} not found in features - will use NaN")
                self._missing_elements_logged.add(element)
            return np.nan

        # Get peak intensities for potassium prediction using available feature format
        # Primary K I doublet (766.49, 769.90 nm) - the strongest K lines
        k_primary_intensity = get_intensity(features, 'K_I')  # Primary K line at 766.49 nm
        k_secondary_intensity = features.get('K_I_peak_1', get_intensity(features, 'K_I'))  # Secondary K line at 769.90 nm - use primary if peak_1 not available

        # Additional K lines at 404 nm region
        k_404_intensity = get_intensity(features, 'K_I_404')  # K line at 404.414 nm
        k_404_2_intensity = features.get('K_I_404_peak_1', np.nan)  # K line at 404.721 nm

        # Context elements for nutrient balance
        s_intensity = get_intensity(features, 'S_I')
        ca_intensity = get_intensity(features, 'CA_I_help')
        ca_ii_intensity = get_intensity(features, 'CA_II_393')  # Strong Ca II line at 393.37 nm
        p_intensity = get_intensity(features, 'P_I_secondary')  # P_I_secondary in config
        n_intensity = get_intensity(features, 'N_I_help')
        c_intensity = get_intensity(features, 'C_I')

        # Optional magnesium lines (for context, not primary target)
        mg_285_intensity = get_intensity(features, 'Mg_I_285')  # Mg I at 285.2 nm
        mg_383_intensity = get_intensity(features, 'Mg_I_383')  # Mg I at 383.8 nm
        mg_ii_intensity = get_intensity(features, 'Mg_II')  # Mg II ionic lines
        
        # Critical ratios for potassium prediction - K-centric approach
        # Note: K_C_ratio is already created in base feature engineering, so we skip it here
        
        # K/Ca ratios (cation competition)
        ratios['K_Ca_ratio'] = safe_ratio(k_primary_intensity, ca_intensity, 'K/Ca')  # Primary competition
        ratios['K_CaII_ratio'] = safe_ratio(k_primary_intensity, ca_ii_intensity, 'K/CaII')  # Ionic competition
        ratios['Ca_K_ratio'] = safe_ratio(ca_intensity, k_primary_intensity, 'Ca/K')  # Inverse for modeling
        
        # K line intensity ratios (plasma diagnostics)
        ratios['K_766_769_ratio'] = safe_ratio(k_primary_intensity, k_secondary_intensity, 'K766/K769')
        ratios['K_766_404_ratio'] = safe_ratio(k_primary_intensity, k_404_intensity, 'K766/K404')
        
        # K vs nutrients (nutrient balance indicators)
        ratios['K_N_ratio'] = safe_ratio(k_primary_intensity, n_intensity, 'K/N')  # Critical for plant metabolism
        ratios['K_P_ratio'] = safe_ratio(k_primary_intensity, p_intensity, 'K/P')  # Synergistic relationship
        ratios['K_S_ratio'] = safe_ratio(k_primary_intensity, s_intensity, 'K/S')  # Secondary nutrient balance
        
        # Combined cation ratios
        ca_total = ca_intensity + ca_ii_intensity if not (np.isnan(ca_intensity) or np.isnan(ca_ii_intensity)) else (ca_intensity if not np.isnan(ca_intensity) else ca_ii_intensity)
        ratios['K_Ca_total_ratio'] = safe_ratio(k_primary_intensity, ca_total, 'K/CaTotal')
        
        # Optional Mg ratios (only if Mg data is available - for context)
        if not np.isnan(mg_285_intensity):
            ratios['K_Mg_ratio'] = safe_ratio(k_primary_intensity, mg_285_intensity, 'K/Mg')
            ratios['Mg_K_ratio'] = safe_ratio(mg_285_intensity, k_primary_intensity, 'Mg/K')
        
        # Base saturation indicators (K-focused)
        # Total bases including available elements
        base_elements = [k_primary_intensity, ca_total]
        if not np.isnan(mg_285_intensity):
            base_elements.append(mg_285_intensity)
            
        total_bases = sum(x for x in base_elements if not np.isnan(x))
        if total_bases > 0:
            ratios['K_base_fraction'] = safe_ratio(k_primary_intensity, total_bases, 'K/TotalBases')
        
        # Combined K intensity (total potassium signal)
        k_total = sum(x for x in [k_primary_intensity, k_secondary_intensity, k_404_intensity, k_404_2_intensity] if not np.isnan(x))
        if k_total > 0:
            ratios['K_total_C_ratio'] = safe_ratio(k_total, c_intensity, 'KTotal/C')
            ratios['K_total_intensity'] = k_total  # Total K signal strength
        
        # Log-transformed ratios for non-linear relationships
        for key, value in list(ratios.items()):
            if not np.isnan(value) and value > 0:
                ratios[f'{key}_log'] = np.log1p(value)
            else:
                ratios[f'{key}_log'] = np.nan
                
        return ratios


class SpectralPatternAnalyzer:
    """Analyzes spectral patterns including FWHM, asymmetry, and line shapes."""
    
    def calculate_peak_fwhm(self, wavelengths: np.ndarray, intensities: np.ndarray,
                           peak_wavelength: float, window: float = 2.0) -> Dict[str, float]:
        """Calculate Full Width at Half Maximum for a spectral peak.

        Returns 0.0 for fwhm and asymmetry when calculation fails (e.g., weak/absent peaks)
        instead of NaN to avoid downstream issues in feature engineering.
        """
        # Handle 2D intensities by taking the mean across samples
        if intensities.ndim > 1:
            intensities = np.mean(intensities, axis=1)

        # Ensure we have matching dimensions
        if len(wavelengths) != len(intensities):
            logger.debug(f"[FWHM] Peak at {peak_wavelength:.2f} nm: dimension mismatch (wl={len(wavelengths)}, int={len(intensities)})")
            return {'fwhm': 0.0, 'asymmetry': 0.0}

        # Select region around peak
        mask = (wavelengths >= peak_wavelength - window) & (wavelengths <= peak_wavelength + window)
        n_points = np.sum(mask)

        if not np.any(mask) or n_points < 5:
            logger.debug(f"[FWHM] Peak at {peak_wavelength:.2f} nm: insufficient data points (n={n_points}, window=±{window} nm) - no measurable peak")
            return {'fwhm': 0.0, 'asymmetry': 0.0}

        peak_wl = wavelengths[mask]
        peak_int = intensities[mask]

        # Find peak maximum
        max_idx = np.argmax(peak_int)

        if max_idx >= len(peak_int):
            logger.debug(f"[FWHM] Peak at {peak_wavelength:.2f} nm: invalid peak maximum index - no measurable peak")
            return {'fwhm': 0.0, 'asymmetry': 0.0}

        max_intensity = peak_int[max_idx]
        half_max = max_intensity / 2

        # Find FWHM
        try:
            # Left side - only search if there are points to the left
            if max_idx > 0:
                left_indices = np.where(peak_int[:max_idx] <= half_max)[0]
                left_idx = left_indices[-1] if len(left_indices) > 0 else 0
            else:
                left_idx = 0

            # Right side - only search if there are points to the right
            if max_idx < len(peak_int) - 1:
                right_indices = np.where(peak_int[max_idx:] <= half_max)[0]
                right_idx = right_indices[0] + max_idx if len(right_indices) > 0 else len(peak_int) - 1
            else:
                right_idx = len(peak_int) - 1

            # Validate indices are within bounds
            left_idx = min(max(0, left_idx), len(peak_wl) - 1)
            right_idx = min(max(0, right_idx), len(peak_wl) - 1)
            max_idx = min(max(0, max_idx), len(peak_wl) - 1)

            fwhm = peak_wl[right_idx] - peak_wl[left_idx]

            # Calculate asymmetry
            left_width = peak_wl[max_idx] - peak_wl[left_idx]
            right_width = peak_wl[right_idx] - peak_wl[max_idx]
            asymmetry = (right_width - left_width) / (right_width + left_width) if (right_width + left_width) > 0 else 0.0

            # Validate results - return 0.0 for invalid values
            if fwhm <= 0 or np.isnan(fwhm):
                logger.debug(f"[FWHM] Peak at {peak_wavelength:.2f} nm: invalid FWHM value ({fwhm:.3f}) - no measurable peak")
                return {'fwhm': 0.0, 'asymmetry': 0.0}

            return {'fwhm': float(fwhm), 'asymmetry': float(asymmetry)}

        except Exception as e:
            logger.debug(f"[FWHM] Peak at {peak_wavelength:.2f} nm: calculation error ({type(e).__name__}) - no measurable peak")
            return {'fwhm': 0.0, 'asymmetry': 0.0}
    
    def calculate_continuum_level(self, wavelengths: np.ndarray, intensities: np.ndarray,
                                 line_free_regions: List[Tuple[float, float]]) -> float:
        """Estimate continuum background level from line-free regions."""
        continuum_points = []
        
        for lower, upper in line_free_regions:
            mask = (wavelengths >= lower) & (wavelengths <= upper)
            if np.any(mask):
                continuum_points.extend(intensities[mask])
        
        if len(continuum_points) > 0:
            return float(np.median(continuum_points))
        return np.nan


class InterferenceCorrector:
    """Handles spectral interference corrections."""
    
    def detect_fe_interference(self, wavelengths: np.ndarray, intensities: np.ndarray,
                              k_line: float = 766.49, fe_lines: Optional[List[float]] = None) -> Dict[str, float]:
        """Detect potential Fe interference on potassium lines."""
        if fe_lines is None:
            fe_lines = [766.0, 766.8, 767.0]  # Common Fe lines near K 766.49 nm

        features = {}
        window = 0.5  # nm window around each line

        # Get K line intensity
        k_mask = (wavelengths >= k_line - window) & (wavelengths <= k_line + window)
        k_intensity = np.max(intensities[k_mask]) if np.any(k_mask) else np.nan

        # Check Fe lines
        fe_intensities = []
        for fe_line in fe_lines:
            fe_mask = (wavelengths >= fe_line - 0.2) & (wavelengths <= fe_line + 0.2)
            if np.any(fe_mask):
                fe_intensities.append(np.max(intensities[fe_mask]))

        if fe_intensities:
            fe_level = np.mean(fe_intensities)
            features['fe_interference_level'] = fe_level

            # Calculate fe_k_ratio with safe division
            if np.isnan(fe_level) or np.isnan(k_intensity) or k_intensity < 1e-12:
                features['fe_k_ratio'] = np.nan
            else:
                features['fe_k_ratio'] = np.clip(fe_level / k_intensity, -100, 100)
        else:
            features['fe_interference_level'] = np.nan
            features['fe_k_ratio'] = np.nan
            
        return features
    
    def calculate_self_absorption_indicator(self, peak_intensity: float, peak_area: float,
                                          peak_fwhm: float) -> float:
        """Calculate indicator for self-absorption in strong lines.

        Returns 0.0 when calculation fails (e.g., missing inputs or invalid fwhm)
        instead of NaN to avoid downstream issues in feature engineering.
        """
        if np.isnan(peak_intensity) or np.isnan(peak_area) or np.isnan(peak_fwhm) or peak_fwhm < 1e-6:
            logger.debug(f"[SELF-ABSORPTION] Calculation failed: intensity={peak_intensity:.3f}, area={peak_area:.3f}, fwhm={peak_fwhm:.3f} - no measurable self-absorption")
            return 0.0

        # Expected area for Gaussian/Lorentzian peak
        expected_area = peak_intensity * peak_fwhm * np.sqrt(np.pi/np.log(2))

        # Self-absorption causes area deficit
        absorption_indicator = 1.0 - (peak_area / expected_area)
        return np.clip(absorption_indicator, 0, 1)


class PlasmaStateIndicators:
    """Calculate plasma state indicators from spectral features."""
    
    def calculate_ionic_neutral_ratio(self, ionic_intensity: float, neutral_intensity: float) -> float:
        """Calculate ionic to neutral line intensity ratio (indicates plasma temperature)."""
        if np.isnan(ionic_intensity) or np.isnan(neutral_intensity) or neutral_intensity < 1e-6:
            return np.nan
        return ionic_intensity / neutral_intensity
    
    def estimate_plasma_temperature_indicator(self, features: Dict[str, float]) -> float:
        """Estimate relative plasma temperature from multiple ionic/neutral ratios."""
        # Get ionic and neutral line intensities
        n_ii = features.get('N_II_help_peak_0', np.nan)  # N II (ionic)
        p_i = features.get('P_I_secondary_peak_0', np.nan)  # P I (neutral) - corrected name

        # Updated for new region names
        mg_ii = features.get('Mg_II_peak_0', np.nan)  # Mg II (ionic) at 279.55, 279.80, 280.27 nm
        mg_i = features.get('Mg_I_285_peak_0', np.nan)  # Mg I (neutral) at 285.2 nm - corrected name
        
        # Calculate ratios
        ratios = []
        if not (np.isnan(n_ii) or np.isnan(features.get('N_I_help_peak_0', np.nan))) and features.get('N_I_help_peak_0', 0) > 0:
            ratios.append(n_ii / features.get('N_I_help_peak_0', 1))
        if not (np.isnan(mg_ii) or np.isnan(mg_i)) and mg_i > 0:
            ratios.append(mg_ii / mg_i)
            
        if ratios:
            return float(np.mean(ratios))
        return np.nan


class EnhancedSpectralFeatures(BaseEstimator, TransformerMixin):
    """Main transformer that combines all enhanced features."""

    def __init__(self, config):
        self.config = config
        self.molecular_detector = MolecularBandDetector()
        self.ratio_calculator = AdvancedRatioCalculator()
        self.pattern_analyzer = SpectralPatternAnalyzer()
        self.interference_corrector = InterferenceCorrector()
        self.plasma_indicators = PlasmaStateIndicators()
        self.feature_names_ = []

        # Cache available elements from config to avoid repeated lookups
        self._available_elements = {region.element for region in self.config.all_regions}

    def fit(self, X, y=None):
        """Fit the transformer (determine feature names)."""
        # Feature names will be set during first transform
        return self

    def _element_exists_in_config(self, element: str) -> bool:
        """Check if an element exists in the configured regions."""
        return element in self._available_elements
        
    def transform(self, X_features: Dict[str, float], wavelengths: np.ndarray, 
                  intensities: np.ndarray) -> Dict[str, float]:
        """Transform existing features by adding enhanced features."""
        enhanced_features = {}
        
        # 1. Molecular band features
        if self.config.enable_molecular_bands:
            # CN bands
            cn_band_ranges = [(385, 390), (415, 425)]  # From config
            cn_features = self.molecular_detector.detect_cn_bands(wavelengths, intensities, cn_band_ranges)
            enhanced_features.update(cn_features)
            
            # Molecular/atomic ratios
            cn_intensity = cn_features.get('CN_band_0_intensity', np.nan)
            p_intensity = X_features.get('P_I_secondary_peak_0', np.nan)  # Corrected name
            enhanced_features['CN_P_molecular_ratio'] = self.molecular_detector.calculate_molecular_atomic_ratio(
                cn_intensity, p_intensity)
        
        # 2. Advanced ratios
        if self.config.enable_advanced_ratios:
            ratio_features = self.ratio_calculator.calculate_nutrient_ratios(X_features)
            enhanced_features.update(ratio_features)
        
        # 3. Spectral patterns
        if self.config.enable_spectral_patterns:
            # FWHM for key lines - dynamically based on configured regions
            # Priority elements for potassium prediction
            priority_elements = ['K_I', 'CA_I_help', 'P_I_secondary', 'Mg_I_285', 'Mg_II']

            # Element-specific window sizes (in nm) for FWHM calculation
            # Mg peaks need wider windows due to weaker signal and broader peak shapes
            element_windows = {
                'K_I': 2.0,
                'CA_I_help': 2.0,
                'P_I_secondary': 2.0,
                'Mg_I_285': 3.5,  # Wider window for Mg (285.2 nm line)
                'Mg_II': 3.5,     # Wider window for Mg II (279-280 nm lines)
            }

            for element in priority_elements:
                # Only process elements that exist in config
                if not self._element_exists_in_config(element):
                    continue

                peak_wl = self._get_peak_wavelength(element)
                if peak_wl:
                    window = element_windows.get(element, 2.0)  # Default to 2.0 nm if not specified
                    fwhm_features = self.pattern_analyzer.calculate_peak_fwhm(
                        wavelengths, intensities, peak_wl, window=window)
                    enhanced_features[f'{element}_fwhm'] = fwhm_features['fwhm']
                    enhanced_features[f'{element}_asymmetry'] = fwhm_features['asymmetry']

                    # Log if peak measurement failed (returned 0.0)
                    if fwhm_features['fwhm'] == 0.0:
                        logger.debug(f"[ENHANCED FEATURES] Element '{element}': no measurable peak at {peak_wl:.2f} nm (weak/absent signal)")
                else:
                    # Return 0.0 instead of NaN for missing peaks
                    logger.debug(f"[ENHANCED FEATURES] Element '{element}': no peak wavelength configured")
                    enhanced_features[f'{element}_fwhm'] = 0.0
                    enhanced_features[f'{element}_asymmetry'] = 0.0
            
            # Continuum level - using regions typically free of emission lines in plant LIBS
            # These regions avoid major elemental lines while capturing baseline
            line_free_regions = [
                (720, 725),  # Between K lines and molecular bands
                (790, 795),  # Between O lines
                (810, 815),  # Before C and S lines
            ]
            enhanced_features['continuum_level'] = self.pattern_analyzer.calculate_continuum_level(
                wavelengths, intensities, line_free_regions)
        
        # 4. Interference corrections
        if self.config.enable_interference_correction:
            fe_features = self.interference_corrector.detect_fe_interference(wavelengths, intensities)
            enhanced_features.update(fe_features)

            # Self-absorption for strong Mg lines (only if Mg regions are configured)
            # This is important at high Mg concentrations but not needed if Mg is not being analyzed
            if self._element_exists_in_config('Mg_I_285'):
                mg_intensity = X_features.get('Mg_I_285_peak_0', np.nan)
                mg_area = X_features.get('Mg_I_285_simple_peak_area', np.nan)
                mg_fwhm = enhanced_features.get('Mg_I_285_fwhm', np.nan)
                enhanced_features['Mg_self_absorption'] = self.interference_corrector.calculate_self_absorption_indicator(
                    mg_intensity, mg_area, mg_fwhm)

                # Check most prominent Mg line (285.2 nm) for self-absorption
                mg_285_intensity = X_features.get('Mg_I_285_peak_0', np.nan)
                mg_285_area = X_features.get('Mg_I_285_simple_peak_area', np.nan)
                mg_285_fwhm = enhanced_features.get('Mg_I_285_fwhm', np.nan)
                enhanced_features['Mg_285_self_absorption'] = self.interference_corrector.calculate_self_absorption_indicator(
                    mg_285_intensity, mg_285_area, mg_285_fwhm)

            # Check Mg II ionic lines for self-absorption (only if configured)
            if self._element_exists_in_config('Mg_II'):
                mg_ii_intensity = X_features.get('Mg_II_peak_0', np.nan)
                mg_ii_area = X_features.get('Mg_II_simple_peak_area', np.nan)
                mg_ii_fwhm = enhanced_features.get('Mg_II_fwhm', np.nan)
                enhanced_features['Mg_II_self_absorption'] = self.interference_corrector.calculate_self_absorption_indicator(
                    mg_ii_intensity, mg_ii_area, mg_ii_fwhm)
        
        # 5. Plasma indicators
        if self.config.enable_plasma_indicators:
            # Ionic/neutral ratios
            n_ii = X_features.get('N_II_help_peak_0', np.nan)
            n_i_help = X_features.get('N_I_help_peak_0', np.nan)
            enhanced_features['N_ionic_neutral_ratio'] = self.plasma_indicators.calculate_ionic_neutral_ratio(n_ii, n_i_help)
            
            # Temperature indicator
            enhanced_features['plasma_temp_indicator'] = self.plasma_indicators.estimate_plasma_temperature_indicator(X_features)
        
        return enhanced_features
    
    def _get_peak_wavelength(self, element: str) -> Optional[float]:
        """Get the center wavelength for an element from config."""
        for region in self.config.all_regions:
            if region.element == element:
                return region.center_wavelengths[0] if region.center_wavelengths else None
        return None