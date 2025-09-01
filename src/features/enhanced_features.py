"""
Enhanced Feature Engineering Module for Crop Magnesium Prediction
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
    """Calculates advanced elemental ratios for magnesium prediction."""
    
    def calculate_nutrient_ratios(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate various nutrient ratios relevant to crop magnesium."""
        ratios = {}
        
        # Helper function for safe division
        def safe_ratio(num: float, den: float, name: str) -> float:
            if np.isnan(num) or np.isnan(den) or den < 1e-6:
                return np.nan
            return np.clip(num / den, -100, 100)  # Clip to prevent extreme values
        
        # Get peak intensities (assuming peak_0 is the primary peak)
        mg_intensity = features.get('M_I_peak_0', np.nan)  # Primary Mg line at 516.77nm
        mg_secondary_intensity = features.get('Mg_secondary_peak_0', np.nan)  # Secondary at 280.3nm
        k_intensity = features.get('K_I_help_peak_0', np.nan)
        s_intensity = features.get('S_I_peak_0', np.nan)
        ca_intensity = features.get('CA_I_help_peak_0', np.nan)
        p_intensity = features.get('P_I_peak_0', np.nan)
        n_intensity = features.get('N_I_help_peak_0', np.nan)
        
        # Critical cation competition ratios for Mg uptake
        ratios['K_Mg_ratio'] = safe_ratio(k_intensity, mg_intensity, 'K/Mg')  # Ideal: 3-5
        ratios['Ca_Mg_ratio'] = safe_ratio(ca_intensity, mg_intensity, 'Ca/Mg')  # Ideal: 3-7
        ratios['Mg_K_ratio'] = safe_ratio(mg_intensity, k_intensity, 'Mg/K')  # Inverse for modeling
        ratios['Mg_Ca_ratio'] = safe_ratio(mg_intensity, ca_intensity, 'Mg/Ca')  # Inverse
        
        # Combined cation antagonism - critical for Mg availability
        kca_sum = k_intensity + ca_intensity if not (np.isnan(k_intensity) or np.isnan(ca_intensity)) else np.nan
        ratios['KCa_Mg_ratio'] = safe_ratio(kca_sum, mg_intensity, '(K+Ca)/Mg')  # Total antagonism
        ratios['Mg_KCa_ratio'] = safe_ratio(mg_intensity, kca_sum, 'Mg/(K+Ca)')  # Mg competitiveness
        
        # Nutrient balance ratios
        ratios['Mg_N_ratio'] = safe_ratio(mg_intensity, n_intensity, 'Mg/N')
        ratios['Mg_P_ratio'] = safe_ratio(mg_intensity, p_intensity, 'Mg/P')  # Synergistic
        ratios['Mg_S_ratio'] = safe_ratio(mg_intensity, s_intensity, 'Mg/S')
        
        # Secondary Mg line ratios (plasma diagnostics)
        ratios['Mg_primary_secondary_ratio'] = safe_ratio(mg_intensity, mg_secondary_intensity, 'Mg516/Mg280')
        
        # Base saturation indicator
        total_bases = mg_intensity + k_intensity + ca_intensity if not any(np.isnan([mg_intensity, k_intensity, ca_intensity])) else np.nan
        ratios['Mg_base_fraction'] = safe_ratio(mg_intensity, total_bases, 'Mg/TotalBases')
        
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
        """Calculate Full Width at Half Maximum for a spectral peak."""
        # Select region around peak
        mask = (wavelengths >= peak_wavelength - window) & (wavelengths <= peak_wavelength + window)
        if not np.any(mask) or np.sum(mask) < 5:
            return {'fwhm': np.nan, 'asymmetry': np.nan}
        
        peak_wl = wavelengths[mask]
        peak_int = intensities[mask]
        
        # Find peak maximum
        max_idx = np.argmax(peak_int)
        # Validate max_idx is within bounds
        if max_idx >= len(peak_int):
            return {'fwhm': np.nan, 'asymmetry': np.nan}
            
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
            asymmetry = (right_width - left_width) / (right_width + left_width) if (right_width + left_width) > 0 else 0
            
            return {'fwhm': float(fwhm), 'asymmetry': float(asymmetry)}
            
        except Exception as e:
            logger.debug("FWHM calculation failed: %s", e)
            return {'fwhm': np.nan, 'asymmetry': np.nan}
    
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
                              p_line: float = 213.6, fe_lines: Optional[List[float]] = None) -> Dict[str, float]:
        """Detect potential Fe interference on magnesium lines."""
        if fe_lines is None:
            fe_lines = [213.0, 213.8, 214.0]  # Common Fe lines near P 213.6 nm
        
        features = {}
        window = 0.5  # nm window around each line
        
        # Get P line intensity
        p_mask = (wavelengths >= p_line - window) & (wavelengths <= p_line + window)
        p_intensity = np.max(intensities[p_mask]) if np.any(p_mask) else np.nan
        
        # Check Fe lines
        fe_intensities = []
        for fe_line in fe_lines:
            fe_mask = (wavelengths >= fe_line - 0.2) & (wavelengths <= fe_line + 0.2)
            if np.any(fe_mask):
                fe_intensities.append(np.max(intensities[fe_mask]))
        
        if fe_intensities:
            features['fe_interference_level'] = np.mean(fe_intensities)
            features['fe_p_ratio'] = features['fe_interference_level'] / p_intensity if p_intensity > 0 else np.nan
        else:
            features['fe_interference_level'] = np.nan
            features['fe_p_ratio'] = np.nan
            
        return features
    
    def calculate_self_absorption_indicator(self, peak_intensity: float, peak_area: float,
                                          peak_fwhm: float) -> float:
        """Calculate indicator for self-absorption in strong lines."""
        if np.isnan(peak_intensity) or np.isnan(peak_area) or np.isnan(peak_fwhm) or peak_fwhm < 1e-6:
            return np.nan
        
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
        p_i = features.get('P_I_peak_0', np.nan)  # P I (neutral)
        
        mg_ii = features.get('Mg_II_peak_0', np.nan)  # Mg II (ionic)
        mg_i = features.get('Mg_I_peak_0', np.nan)  # Mg I (neutral)
        
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
        
    def fit(self, X, y=None):
        """Fit the transformer (determine feature names)."""
        # Feature names will be set during first transform
        return self
        
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
            p_intensity = X_features.get('P_I_peak_0', np.nan)
            enhanced_features['CN_P_molecular_ratio'] = self.molecular_detector.calculate_molecular_atomic_ratio(
                cn_intensity, p_intensity)
        
        # 2. Advanced ratios
        if self.config.enable_advanced_ratios:
            ratio_features = self.ratio_calculator.calculate_nutrient_ratios(X_features)
            enhanced_features.update(ratio_features)
        
        # 3. Spectral patterns
        if self.config.enable_spectral_patterns:
            # FWHM for key lines - focus on Mg and competing cations
            for element in ['M_I', 'Mg_secondary', 'CA_I_help', 'K_I_help', 'P_I']:
                peak_wl = self._get_peak_wavelength(element)
                if peak_wl:
                    fwhm_features = self.pattern_analyzer.calculate_peak_fwhm(
                        wavelengths, intensities, peak_wl)
                    enhanced_features[f'{element}_fwhm'] = fwhm_features['fwhm']
                    enhanced_features[f'{element}_asymmetry'] = fwhm_features['asymmetry']
            
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
            
            # Self-absorption for strong Mg lines (important at high concentrations)
            mg_intensity = X_features.get('M_I_peak_0', np.nan)
            mg_area = X_features.get('M_I_simple_peak_area', np.nan)
            mg_fwhm = enhanced_features.get('M_I_fwhm', np.nan)
            enhanced_features['Mg_self_absorption'] = self.interference_corrector.calculate_self_absorption_indicator(
                mg_intensity, mg_area, mg_fwhm)
            
            # Also check secondary Mg line for self-absorption
            mg2_intensity = X_features.get('Mg_secondary_peak_0', np.nan)
            mg2_area = X_features.get('Mg_secondary_simple_peak_area', np.nan)
            mg2_fwhm = enhanced_features.get('Mg_secondary_fwhm', np.nan)
            enhanced_features['Mg_secondary_self_absorption'] = self.interference_corrector.calculate_self_absorption_indicator(
                mg2_intensity, mg2_area, mg2_fwhm)
        
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