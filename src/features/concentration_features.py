"""
Concentration-Range Feature Engineering Module

This module adds concentration-aware features that help AutoGluon learn
patterns specific to different magnesium concentration ranges, replacing
the need for manual sample weighting.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TargetAwareTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper that allows a transformer to store target values from fit() 
    and use them during transform() for concentration features.
    """
    
    def __init__(self, transformer):
        self.transformer = transformer
        self.y_fit_ = None
        
    def fit(self, X, y=None):
        """Fit the wrapped transformer and store target values."""
        self.y_fit_ = y
        self.transformer.fit(X, y)
        return self
        
    def transform(self, X):
        """Transform using the wrapped transformer."""
        return self.transformer.transform(X)
        
    def get_feature_names_out(self, input_features=None):
        """Pass through feature names from wrapped transformer."""
        return self.transformer.get_feature_names_out(input_features)


class MinimalConcentrationFeatures(BaseEstimator, TransformerMixin):
    """
    Lightweight concentration-aware features for raw spectral mode.
    
    This transformer adds essential distribution-aware features to help models
    handle the uneven target distribution while maintaining the raw spectral philosophy.
    It focuses only on the most impactful concentration features without full
    feature engineering.
    """
    
    def __init__(self, 
                 use_target_percentiles: bool = True,
                 low_percentile: float = 25.0,
                 high_percentile: float = 75.0,
                 magnesium_wavelength: float = 214.914):
        """
        Initialize minimal concentration features.
        
        Args:
            use_target_percentiles: Calculate thresholds from target distribution
            low_percentile: Percentile for low concentration threshold
            high_percentile: Percentile for high concentration threshold
            magnesium_wavelength: Primary magnesium wavelength for intensity features
        """
        self.use_target_percentiles = use_target_percentiles
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.magnesium_wavelength = magnesium_wavelength
        
        # Will be learned during fit
        self.low_threshold_ = None
        self.high_threshold_ = None
        self.p_intensity_percentiles_ = None
        self.feature_names_out_ = []
        
    def fit(self, X, y=None):
        """
        Fit the transformer by learning distribution statistics.
        
        Args:
            X: Raw spectral features [n_samples, n_wavelengths]
            y: Target concentrations (required for threshold calculation)
        """
        if y is None:
            raise ValueError("Target values (y) are required for MinimalConcentrationFeatures.fit()")
            
        X = self._validate_input(X)
        y_array = np.array(y)
        
        # Calculate concentration thresholds from target distribution
        if self.use_target_percentiles:
            self.low_threshold_ = np.percentile(y_array, self.low_percentile)
            self.high_threshold_ = np.percentile(y_array, self.high_percentile)
            logger.info(f"Learned concentration thresholds: low={self.low_threshold_:.3f}, high={self.high_threshold_:.3f}")
        else:
            # Use fixed thresholds based on typical magnesium ranges
            self.low_threshold_ = 0.25
            self.high_threshold_ = 0.40
            
        # Calculate P intensity statistics if we can find the wavelength
        self.p_intensity_percentiles_ = self._calculate_p_intensity_percentiles(X)
        
        # Define feature names
        self.feature_names_out_ = [
            'concentration_range_low',      # Binary: is in low concentration range
            'concentration_range_high',     # Binary: is in high concentration range
            'concentration_range_mid',      # Binary: is in mid concentration range
            'p_intensity_weight',          # Concentration-based weighting feature
            'distribution_emphasis'         # Feature to emphasize rare concentration ranges
        ]
        
        logger.info(f"MinimalConcentrationFeatures fitted. Will generate {len(self.feature_names_out_)} features.")
        return self
        
    def transform(self, X):
        """
        Transform raw spectral data by adding minimal concentration features.
        
        Args:
            X: Raw spectral features [n_samples, n_wavelengths]
            
        Returns:
            X_enhanced: [n_samples, n_wavelengths + n_concentration_features]
        """
        X = self._validate_input(X)
        
        if self.low_threshold_ is None:
            raise ValueError("Transformer not fitted. Call fit() first.")
            
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Extract magnesium intensity estimate (best available proxy)
        m_intensity = self._extract_m_intensity_proxy(X_df)
        
        # Normalize p_intensity to a 0-1 scale based on fitted statistics
        if self.p_intensity_percentiles_ is not None:
            m_min = self.m_intensity_percentiles_['values'][0]
            m_max = self.m_intensity_percentiles_['values'][-1]
            # Handle edge cases to prevent divide by zero warnings
            m_range = m_max - m_min
            if m_range > 1e-8 and not np.isnan(m_range) and not np.isinf(m_range):
                m_intensity_norm = np.clip((m_intensity - m_min) / m_range, 0, 1)
            else:
                # If range is too small or invalid, use uniform values
                m_intensity_norm = np.full_like(m_intensity, 0.5)
        else:
            # Fallback normalization with better edge case handling
            m_min = m_intensity.min()
            m_max = m_intensity.max()
            m_range = m_max - m_min
            if m_range > 1e-8 and not np.isnan(m_range) and not np.isinf(m_range):
                m_intensity_norm = (m_intensity - m_min) / m_range
            else:
                # If all values are the same or invalid, use uniform values
                m_intensity_norm = np.full_like(m_intensity, 0.5)
        
        # Use normalized intensity for threshold comparisons
        # Map target thresholds to intensity scale
        low_intensity_threshold = 0.3  # Assume low concentrations have lower relative intensities
        high_intensity_threshold = 0.7  # Assume high concentrations have higher relative intensities
        
        # Add concentration range indicators
        # These help models learn different behaviors for different concentration ranges
        X_df['concentration_range_low'] = (m_intensity_norm <= low_intensity_threshold).astype(float)
        X_df['concentration_range_high'] = (m_intensity_norm >= high_intensity_threshold).astype(float)
        X_df['concentration_range_mid'] = ((m_intensity_norm > low_intensity_threshold) & 
                                          (m_intensity_norm < high_intensity_threshold)).astype(float)
        
        # Add P intensity weighting feature
        # This creates a smooth weighting that emphasizes rare concentration ranges
        # Use normalized intensity percentile for weighting
        intensity_percentile = m_intensity_norm
        
        # Create concentration-dependent weighting (higher weight for extreme values)
        m_intensity_weight = 1.0 + 2.0 * (np.abs(intensity_percentile - 0.5) ** 2)
        X_df['m_intensity_weight'] = m_intensity_weight

        # Add distribution emphasis feature
        # This helps models focus on underrepresented regions
        distribution_emphasis = np.where(
            X_df['concentration_range_low'] == 1, 2.5,  # High emphasis on low concentrations
            np.where(X_df['concentration_range_high'] == 1, 2.0,  # High emphasis on high concentrations
                    1.0)  # Normal emphasis on mid-range
        )
        X_df['distribution_emphasis'] = distribution_emphasis
        
        logger.debug(f"MinimalConcentrationFeatures: Added {len(self.feature_names_out_)} features to {X.shape[1]} raw features")
        
        # Return DataFrame to preserve feature names for models that need them (like LightGBM)
        return X_df
        
    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features."""
        if input_features is not None:
            # Combine input feature names with new concentration features
            if hasattr(input_features, 'tolist'):
                input_names = input_features.tolist()
            else:
                input_names = list(input_features)
            return input_names + self.feature_names_out_
        else:
            return self.feature_names_out_
            
    def _validate_input(self, X):
        """Validate and prepare input data."""
        if hasattr(X, 'values'):  # DataFrame
            return X.values if not isinstance(X, pd.DataFrame) else X
        return np.asarray(X)
        
    def _calculate_p_intensity_percentiles(self, X):
        """Calculate P intensity percentiles for weighting."""
        try:
            m_intensity = self._extract_m_intensity_proxy(X)
            percentiles = np.linspace(0, 100, 101)
            values = np.percentile(m_intensity, percentiles)
            return {
                'percentiles': percentiles / 100.0,  # Convert to 0-1 range
                'values': values
            }
        except Exception as e:
            logger.warning(f"Could not calculate M intensity percentiles: {e}")
            return None
            
    def _extract_m_intensity_proxy(self, X):
        """
        Extract a proxy for magnesium intensity from raw spectral data.
        
        Since we don't have wavelength information in raw mode, we use
        statistical proxies based on the spectral data distribution.
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Use the peak intensity in the spectrum as a proxy
        # This assumes higher peak intensities correlate with higher concentrations
        peak_intensity = np.max(X_array, axis=1)
        
        # Also use the 90th percentile intensity as additional signal
        percentile_90 = np.percentile(X_array, 90, axis=1)
        
        # Combine peak and high percentile for more robust proxy
        m_intensity_proxy = 0.7 * peak_intensity + 0.3 * percentile_90
        
        return m_intensity_proxy

class ConcentrationRangeFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer that adds concentration-aware features for improved AutoGluon performance.
    
    Instead of using sample weights, this transformer creates features that help
    the ensemble learn concentration-specific patterns, allowing different models
    in the ensemble to specialize in different concentration ranges.
    """
    
    def __init__(self, 
                 low_threshold: float = None,
                 high_threshold: float = None,
                 use_target_percentiles: bool = True,
                 low_percentile: float = 25.0,
                 high_percentile: float = 75.0,
                 enable_range_indicators: bool = True,
                 enable_spectral_modulation: bool = True,
                 enable_ratio_adjustments: bool = True,
                 enable_concentration_interactions: bool = True):
        """
        Initialize the concentration range feature transformer.
        
        Args:
            low_threshold: Fixed threshold for low concentration (if None, will be calculated from targets)
            high_threshold: Fixed threshold for high concentration (if None, will be calculated from targets)
            use_target_percentiles: If True, calculate thresholds from target distribution
            low_percentile: Percentile to use for low threshold (when use_target_percentiles=True)
            high_percentile: Percentile to use for high threshold (when use_target_percentiles=True)
            enable_range_indicators: Add binary and continuous range indicators
            enable_spectral_modulation: Add concentration-modulated spectral features
            enable_ratio_adjustments: Add concentration-adjusted ratio features
            enable_concentration_interactions: Add interaction features between concentration and spectral data
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.use_target_percentiles = use_target_percentiles
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.enable_range_indicators = enable_range_indicators
        self.enable_spectral_modulation = enable_spectral_modulation
        self.enable_ratio_adjustments = enable_ratio_adjustments
        self.enable_concentration_interactions = enable_concentration_interactions
        
        # These will be learned during fit()
        self.p_intensity_percentiles_ = None
        self.pc_ratio_percentiles_ = None
        self.feature_statistics_ = None
        self.feature_names_out_ = []
        
        # Target-derived thresholds (learned from actual concentration data)
        self.fitted_low_threshold_ = None
        self.fitted_high_threshold_ = None
        self.target_statistics_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the transformer by learning data distribution characteristics.
        
        Args:
            X: Feature matrix with spectral features
            y: Target concentrations (REQUIRED for target-aware thresholds)
        """
        # Learn target distribution characteristics first (if provided)
        if y is not None and self.use_target_percentiles:
            self._learn_target_characteristics(y)
            logger.info(f"Learned concentration thresholds from target data: "
                       f"low={self.fitted_low_threshold_:.3f}, high={self.fitted_high_threshold_:.3f}")
        else:
            # Fallback to fixed thresholds or defaults
            self.fitted_low_threshold_ = self.low_threshold or 0.25
            self.fitted_high_threshold_ = self.high_threshold or 0.40
            if y is None:
                logger.warning("No target concentrations provided. Using fixed thresholds.")
            logger.info(f"Using fixed concentration thresholds: "
                       f"low={self.fitted_low_threshold_:.3f}, high={self.fitted_high_threshold_:.3f}")
        
        # Learn data distribution characteristics for intelligent feature creation
        self._learn_data_characteristics(X, y)
        
        # Generate feature names based on input features and enabled options
        self._generate_feature_names(X)
        
        logger.info(f"ConcentrationRangeFeatures fitted. Will generate {len(self.feature_names_out_)} new features.")
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features by adding concentration-aware enhancements.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Enhanced feature matrix with concentration-aware features
        """
        if self.feature_statistics_ is None:
            raise ValueError("Transformer must be fitted before transform")
            
        X_enhanced = X.copy()
        
        # Add range indicator features
        if self.enable_range_indicators:
            X_enhanced = self._add_range_indicators(X_enhanced)
            
        # Add spectral modulation features
        if self.enable_spectral_modulation:
            X_enhanced = self._add_spectral_modulation(X_enhanced)
            
        # Add ratio adjustment features
        if self.enable_ratio_adjustments:
            X_enhanced = self._add_ratio_adjustments(X_enhanced)
            
        # Add concentration interaction features
        if self.enable_concentration_interactions:
            X_enhanced = self._add_concentration_interactions(X_enhanced)
            
        logger.debug(f"Added {len(X_enhanced.columns) - len(X.columns)} concentration-aware features")
        return X_enhanced
        
    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features."""
        if input_features is None:
            return self.feature_names_out_
        else:
            return list(input_features) + self.feature_names_out_
            
    def _learn_target_characteristics(self, y: pd.Series):
        """Learn concentration distribution characteristics from target data."""
        # Handle both pandas Series and numpy arrays
        if isinstance(y, pd.Series):
            y_clean = y.dropna()
        else:
            # For numpy arrays, remove NaN values
            y_clean = y[~np.isnan(y)]
        
        if len(y_clean) == 0:
            logger.warning("No valid target concentrations found. Using default thresholds.")
            self.fitted_low_threshold_ = self.low_threshold or 0.25
            self.fitted_high_threshold_ = self.high_threshold or 0.40
            return
        
        # Calculate thresholds based on target distribution percentiles
        self.fitted_low_threshold_ = np.percentile(y_clean, self.low_percentile)
        self.fitted_high_threshold_ = np.percentile(y_clean, self.high_percentile)
        
        # Store comprehensive target statistics
        self.target_statistics_ = {
            'min': np.min(y_clean),
            'max': np.max(y_clean),
            'mean': np.mean(y_clean),
            'std': np.std(y_clean),
            'p10': np.percentile(y_clean, 10),
            'p25': np.percentile(y_clean, 25),
            'p50': np.percentile(y_clean, 50),
            'p75': np.percentile(y_clean, 75),
            'p90': np.percentile(y_clean, 90),
            'count': len(y_clean)
        }
        
        # Ensure thresholds are reasonable
        concentration_range = self.target_statistics_['max'] - self.target_statistics_['min']
        min_gap = concentration_range * 0.1  # At least 10% of range between thresholds
        
        if self.fitted_high_threshold_ - self.fitted_low_threshold_ < min_gap:
            # Adjust thresholds to ensure meaningful separation
            center = (self.fitted_low_threshold_ + self.fitted_high_threshold_) / 2
            self.fitted_low_threshold_ = max(self.target_statistics_['min'], center - min_gap/2)
            self.fitted_high_threshold_ = min(self.target_statistics_['max'], center + min_gap/2)
            logger.info(f"Adjusted thresholds to ensure minimum separation: "
                       f"low={self.fitted_low_threshold_:.3f}, high={self.fitted_high_threshold_:.3f}")
    
    def _learn_data_characteristics(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Learn key statistics about the spectral data distribution."""
        self.feature_statistics_ = {}
        
        # Learn P intensity characteristics if available
        m_intensity_cols = [col for col in X.columns if 'M_I' in col and ('peak_area' in col or 'peak_height' in col)]
        if m_intensity_cols:
            m_intensities = X[m_intensity_cols].fillna(0).sum(axis=1)
            self.m_intensity_percentiles_ = {
                'm10': np.percentile(m_intensities, 10),
                'm25': np.percentile(m_intensities, 25),
                'm50': np.percentile(m_intensities, 50),
                'm75': np.percentile(m_intensities, 75),
                'm90': np.percentile(m_intensities, 90)
            }
            
        # Learn P/C ratio characteristics if available
        if 'M_C_ratio' in X.columns:
            mc_ratios = X['M_C_ratio'].fillna(0)
            self.mc_ratio_percentiles_ = {
                'm10': np.percentile(mc_ratios, 10),
                'm25': np.percentile(mc_ratios, 25),
                'm50': np.percentile(mc_ratios, 50),
                'm75': np.percentile(mc_ratios, 75),
                'm90': np.percentile(mc_ratios, 90)
            }
            
        # Only learn statistics for elements that are actually present in the transformed features
        # Build list of elements to track based on what's actually in the data
        elements_to_track = []
        
        # Always track P_I (magnesium) as it's the target element
        elements_to_track.append('M_I')
        
        # Check if C_I features exist (used for P/C ratio)
        if any('C_I' in col for col in X.columns):
            elements_to_track.append('C_I')
            
        # Only add other elements if they're actually present in the features
        # These would only be present if enable_oxygen_hydrogen or other flags are True
        for element in ['N_I', 'H_I', 'O_I', 'CA_I', 'K_I', 'S_I', 'P_I']:
            if any(element in col for col in X.columns):
                elements_to_track.append(element)
                logger.debug(f"Found {element} features in data, will track statistics")
        
        # Learn general feature statistics only for present spectral features
        spectral_cols = [col for col in X.columns if any(element in col for element in elements_to_track)]
        logger.info(f"Learning statistics for {len(spectral_cols)} spectral features from elements: {elements_to_track}")
        
        for col in spectral_cols:
            if X[col].notna().sum() > 10:  # Only if we have enough data
                self.feature_statistics_[col] = {
                    'mean': X[col].mean(),
                    'std': X[col].std(),
                    'p25': np.percentile(X[col].dropna(), 25),
                    'p75': np.percentile(X[col].dropna(), 75)
                }
                
    def _generate_feature_names(self, X: pd.DataFrame):
        """Generate names for the new features that will be created."""
        self.feature_names_out_ = []
        
        if self.enable_range_indicators:
            self.feature_names_out_.extend([
                'concentration_range_low',
                'concentration_range_medium', 
                'concentration_range_high',
                'concentration_intensity_score',
                'concentration_ratio_score'
            ])
            
        if self.enable_spectral_modulation and self.p_intensity_percentiles_:
            self.feature_names_out_.extend([
                'm_intensity_concentration_weight',
                'm_spectral_concentration_indicator'
            ])
            
        if self.enable_ratio_adjustments and 'P_C_ratio' in X.columns:
            self.feature_names_out_.extend([
                'mc_ratio_concentration_adjusted',
                'mc_ratio_extreme_indicator'
            ])
            
        if self.enable_concentration_interactions:
            # Add interaction features for key spectral features
            key_features = [col for col in X.columns if any(key in col for key in ['M_I_simple', 'M_C_ratio', 'MC_height_ratio'])][:3]
            for feature in key_features:
                self.feature_names_out_.append(f'{feature}_concentration_interaction')
                
    def _add_range_indicators(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add concentration range indicator features.
        
        These help AutoGluon identify and learn patterns specific to different
        concentration ranges without needing explicit sample weighting.
        """
        # Estimate concentration likelihood from spectral features
        concentration_score = self._estimate_concentration_likelihood(X)
        
        # Create soft range indicators (continuous values 0-1) using fitted thresholds
        X['concentration_range_low'] = 1.0 / (1.0 + np.exp(10 * (concentration_score - self.fitted_low_threshold_)))
        X['concentration_range_high'] = 1.0 / (1.0 + np.exp(-10 * (concentration_score - self.fitted_high_threshold_)))
        X['concentration_range_medium'] = 1.0 - X['concentration_range_low'] - X['concentration_range_high']
        
        # Add composite concentration indicators
        if self.p_intensity_percentiles_:
            p_intensity_cols = [col for col in X.columns if 'P_I' in col and ('peak_area' in col or 'peak_height' in col)]
            if p_intensity_cols:
                p_intensities = X[p_intensity_cols].fillna(0).sum(axis=1)
                # Normalize P intensity score based on learned distribution
                intensity_score = (p_intensities - self.p_intensity_percentiles_['p25']) / (
                    self.p_intensity_percentiles_['p75'] - self.p_intensity_percentiles_['p25'] + 1e-6)
                X['concentration_intensity_score'] = np.clip(intensity_score, 0, 2)
            else:
                X['concentration_intensity_score'] = 0.5  # Neutral score if no P intensity features
        else:
            X['concentration_intensity_score'] = 0.5
            
        # Add P/C ratio-based concentration score
        if 'P_C_ratio' in X.columns and self.pc_ratio_percentiles_:
            pc_ratio_score = (X['P_C_ratio'].fillna(0) - self.pc_ratio_percentiles_['p25']) / (
                self.pc_ratio_percentiles_['p75'] - self.pc_ratio_percentiles_['p25'] + 1e-6)
            X['concentration_ratio_score'] = np.clip(pc_ratio_score, 0, 2)
        else:
            X['concentration_ratio_score'] = 0.5
            
        return X
        
    def _add_spectral_modulation(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add features that modulate spectral intensity based on estimated concentration.
        
        This helps models learn that the same spectral intensity might indicate
        different concentrations depending on other contextual features.
        """
        if not self.m_intensity_percentiles_:
            return X
            
        # Calculate P intensity concentration weight
        m_intensity_cols = [col for col in X.columns if 'M_I' in col and ('peak_area' in col or 'peak_height' in col)]
        if m_intensity_cols:
            m_intensities = X[m_intensity_cols].fillna(0).sum(axis=1)
            
            # Create concentration-dependent weighting
            # High weights for extreme values (low/high concentrations)
            intensity_percentile = np.clip(
                (m_intensities - self.m_intensity_percentiles_['m10']) / 
                (self.m_intensity_percentiles_['m90'] - self.m_intensity_percentiles_['m10'] + 1e-6), 
                0, 1
            )
            
            # U-shaped weighting: higher weights for extreme percentiles
            concentration_weight = 1.0 + 2.0 * (np.abs(intensity_percentile - 0.5) ** 2)
            X['m_intensity_concentration_weight'] = concentration_weight
            
            # Binary indicator for extreme P intensities
            extreme_mask = (m_intensities <= self.m_intensity_percentiles_['m10']) | \
                          (m_intensities >= self.m_intensity_percentiles_['m90'])
            X['m_spectral_concentration_indicator'] = extreme_mask.astype(float)
        else:
            X['m_intensity_concentration_weight'] = 1.0
            X['m_spectral_concentration_indicator'] = 0.0
            
        return X
        
    def _add_ratio_adjustments(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add concentration-adjusted ratio features.
        
        These help capture non-linear relationships between ratios and concentration
        that might be missed by standard ratio features.
        """
        if 'M_C_ratio' not in X.columns or not self.mc_ratio_percentiles_:
            return X
            
        mc_ratio = X['M_C_ratio'].fillna(0)
        
        # Concentration-adjusted M/C ratio
        # Apply stronger adjustment for extreme values
        ratio_adjustment = 1.0 + 0.5 * np.abs(mc_ratio - self.mc_ratio_percentiles_['m50']) / (
            self.mc_ratio_percentiles_['m75'] - self.mc_ratio_percentiles_['m25'] + 1e-6)
        X['mc_ratio_concentration_adjusted'] = mc_ratio * ratio_adjustment
        
        # Extreme ratio indicator
        extreme_ratio_mask = (mc_ratio <= self.mc_ratio_percentiles_['m10']) | \
                            (mc_ratio >= self.mc_ratio_percentiles_['m90'])
        X['mc_ratio_extreme_indicator'] = extreme_ratio_mask.astype(float)
        
        return X
        
    def _add_concentration_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features between concentration indicators and key spectral features.
        
        This allows AutoGluon models to learn concentration-specific feature relationships.
        """
        # Get concentration likelihood estimate
        concentration_score = self._estimate_concentration_likelihood(X)
        
        # Add interactions with key features
        key_features = [col for col in X.columns if any(key in col for key in ['M_I_simple', 'M_C_ratio', 'MC_height_ratio'])][:3]
        
        for feature in key_features:
            if feature in X.columns:
                # Interaction: feature value modulated by concentration estimate
                interaction = X[feature].fillna(0) * concentration_score
                X[f'{feature}_concentration_interaction'] = interaction
                
        return X
        
    def _estimate_concentration_likelihood(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate relative concentration likelihood from spectral features.
        
        This is a heuristic based on P intensity and P/C ratio to create
        concentration-aware features without knowing actual concentrations.
        Now uses target-derived statistics when available.
        """
        # Use target mean if available, otherwise fallback to default
        default_concentration = self.target_statistics_['mean'] if self.target_statistics_ else 0.35
        concentration_range = self.fitted_high_threshold_ - self.fitted_low_threshold_
        
        score = np.full(len(X), default_concentration)
        
        # Use P intensity if available
        if self.m_intensity_percentiles_:
            m_intensity_cols = [col for col in X.columns if 'M_I' in col and ('peak_area' in col or 'peak_height' in col)]
            if m_intensity_cols:
                m_intensities = X[m_intensity_cols].fillna(0).sum(axis=1)
                # Normalize to target concentration range
                intensity_norm = (m_intensities - self.m_intensity_percentiles_['m10']) / (
                    self.m_intensity_percentiles_['m90'] - self.m_intensity_percentiles_['m10'] + 1e-6)
                # Map to actual target range instead of fixed range
                min_target = self.target_statistics_['min'] if self.target_statistics_ else 0.2
                max_target = self.target_statistics_['max'] if self.target_statistics_ else 0.5
                score = min_target + (max_target - min_target) * np.clip(intensity_norm, 0, 1)
                
        # Adjust with P/C ratio if available
        if 'M_C_ratio' in X.columns and self.mc_ratio_percentiles_:
            mc_ratio = X['M_C_ratio'].fillna(0)
            ratio_norm = (mc_ratio - self.mc_ratio_percentiles_['m25']) / (
                self.mc_ratio_percentiles_['m75'] - self.mc_ratio_percentiles_['m25'] + 1e-6)
            ratio_adjustment = concentration_range * 0.2 * np.clip(ratio_norm, -1, 1)  # Scale by actual range
            min_target = self.target_statistics_['min'] if self.target_statistics_ else 0.15
            max_target = self.target_statistics_['max'] if self.target_statistics_ else 0.5
            score = np.clip(score + ratio_adjustment, min_target, max_target)
            
        return score


def create_enhanced_feature_pipeline_with_concentration(config, strategy: str, exclude_scaler: bool = False, use_parallel: bool = False, n_jobs: int = -1):
    """
    Create a feature pipeline that includes concentration-aware features.
    
    This is a drop-in replacement for create_feature_pipeline that adds
    concentration range features for improved AutoGluon performance.
    
    Supports both raw spectral mode and feature engineering mode based on
    config.use_raw_spectral_data.
    
    Args:
        config: Pipeline configuration
        strategy: Feature strategy ('Mg_only', 'simple_only', 'full_context')
        exclude_scaler: Whether to exclude StandardScaler from pipeline
        use_parallel: Whether to use parallel processing for feature generation
        n_jobs: Number of parallel jobs (-1 for all cores, -2 for all but one)
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from src.features.feature_engineering import SpectralFeatureGenerator, RawSpectralTransformer
    from src.utils.helpers import OutlierClipper
    
    if config.use_raw_spectral_data:
        # Raw spectral mode - add minimal concentration features for distribution awareness
        logger.info("Using raw spectral data with minimal concentration features for distribution handling")
        
        steps = [
            ('raw_spectral', RawSpectralTransformer(config=config)),
            ('minimal_concentration', MinimalConcentrationFeatures())
        ]
    else:
        # Feature engineering mode - configure concentration features based on strategy
        if strategy == "Mg_only":
            # Focus on P-specific concentration patterns
            concentration_config = {
                'enable_range_indicators': True,
                'enable_spectral_modulation': True,
                'enable_ratio_adjustments': False,  # No P/C ratio in Mg_only
                'enable_concentration_interactions': True
            }
        elif strategy == "simple_only":
            # Balanced approach for simple features
            concentration_config = {
                'enable_range_indicators': True,
                'enable_spectral_modulation': True,
                'enable_ratio_adjustments': True,
                'enable_concentration_interactions': True
            }
        elif strategy == "full_context":
            # Full concentration-aware enhancement
            concentration_config = {
                'enable_range_indicators': True,
                'enable_spectral_modulation': True,
                'enable_ratio_adjustments': True,
                'enable_concentration_interactions': True
            }
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Use parallel or sequential feature generator based on parameters
        if use_parallel:
            from src.features.parallel_feature_engineering import ParallelSpectralFeatureGenerator
            feature_generator = ParallelSpectralFeatureGenerator(
                config=config, strategy=strategy, n_jobs=n_jobs
            )
            logger.info(f"Using PARALLEL feature generation with {n_jobs} jobs for concentration pipeline")
        else:
            feature_generator = SpectralFeatureGenerator(config=config, strategy=strategy)
            
        steps = [
            ('spectral_features', feature_generator),
            ('concentration_features', ConcentrationRangeFeatures(**concentration_config)),
            ('imputer', SimpleImputer(strategy='mean')),
            ('clipper', OutlierClipper())
        ]
    
    # Add StandardScaler unless explicitly excluded (e.g., for AutoGluon)
    if not exclude_scaler:
        steps.append(('scaler', StandardScaler()))
    
    pipeline = Pipeline(steps)
    scaler_info = "without StandardScaler" if exclude_scaler else "with StandardScaler"
    parallel_info = " (PARALLEL)" if use_parallel else ""
    
    if config.use_raw_spectral_data:
        logger.info(f"Created RAW SPECTRAL pipeline with minimal concentration features {scaler_info}.")
    else:
        logger.info(f"Created enhanced feature engineering pipeline{parallel_info} with concentration features for strategy: '{strategy}' {scaler_info}.")
    return pipeline