"""
Feature Engineering Helper Functions

This module contains the detailed logic for creating specific features,
adapted from the original nitrogen_regression_model.py script for magnesium prediction.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# This is a placeholder class to allow the function to run standalone.
# In the pipeline, the real class from the config is used.
class PeakRegion:
    def __init__(self, element, lower, upper, centers):
        self.element = element
        self.lower_wavelength = lower
        self.upper_wavelength = upper

logger = logging.getLogger(__name__)

def extract_full_simple_features(
    region: PeakRegion, wavelengths: np.ndarray, intensities: np.ndarray
) -> Dict[str, float]:
    """
    Extracts the full set of 8 simple features from a given peak region.
    This logic is identical to the original _extract_simple_region_features.
    """
    prefix = f"{region.element}_simple"
    nan_features = {
        f'{prefix}_peak_area': np.nan, f'{prefix}_peak_height': np.nan,
        f'{prefix}_peak_center_intensity': np.nan, f'{prefix}_baseline_avg': np.nan,
        f'{prefix}_signal_range': np.nan, f'{prefix}_total_intensity': np.nan,
        f'{prefix}_height_to_baseline': np.nan, f'{prefix}_normalized_area': np.nan
    }

    # Ensure arrays are numpy arrays
    wavelengths = np.asarray(wavelengths)
    intensities = np.asarray(intensities)
    
    mask = (wavelengths >= region.lower_wavelength) & (wavelengths <= region.upper_wavelength)
    
    # Handle both 1D (single sample) and 2D (multiple samples) arrays
    if intensities.ndim == 1:
        if not np.any(mask) or intensities.size == 0 or len(wavelengths[mask]) < 2:
            return nan_features
        avg_spectrum = intensities[mask]
    else:
        if not np.any(mask) or intensities.shape[1] == 0 or len(wavelengths[mask]) < 2:
            return nan_features
        avg_spectrum = np.mean(intensities[mask, :], axis=1)
    
    peak_area = np.trapz(avg_spectrum, wavelengths[mask])
    total_intensity = np.sum(avg_spectrum)
    
    features = {
        f'{prefix}_peak_area': peak_area,
        f'{prefix}_peak_height': np.max(avg_spectrum),
        f'{prefix}_peak_center_intensity': avg_spectrum[len(avg_spectrum)//2],
        f'{prefix}_baseline_avg': (avg_spectrum[0] + avg_spectrum[-1]) / 2,
        f'{prefix}_signal_range': np.max(avg_spectrum) - np.min(avg_spectrum),
        f'{prefix}_total_intensity': total_intensity
    }
    features[f'{prefix}_height_to_baseline'] = features[f'{prefix}_peak_height'] - features[f'{prefix}_baseline_avg']
    features[f'{prefix}_normalized_area'] = peak_area / total_intensity if total_intensity > 0 else 0
    
    return features

def generate_high_magnesium_features(df: pd.DataFrame, simple_feature_names: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generates the full set of enhanced features for high magnesium detection.
    This logic is adapted from the original _generate_high_nitrogen_features.
    """
    df_out = df.copy()
    index_to_use = df_out.index if hasattr(df_out, 'index') else None
    enhanced_features = pd.DataFrame(index=index_to_use)
    enhanced_feature_names = []
    
    # Apply reasonable bounds to PC_ratio to prevent extreme values
    mc_ratio_safe = df_out['M_C_ratio'].fillna(0.0)
    # Clip PC_ratio to reasonable bounds (e.g., -50 to 50) to prevent corruption
    mc_ratio_clipped = np.clip(mc_ratio_safe, -50.0, 50.0)
    
    enhanced_features['MC_ratio_squared'] = mc_ratio_clipped ** 2
    enhanced_feature_names.append('MC_ratio_squared')
    enhanced_features['MC_ratio_cubic'] = mc_ratio_clipped ** 3
    enhanced_feature_names.append('MC_ratio_cubic')
    enhanced_features['MC_ratio_log'] = np.log1p(np.abs(mc_ratio_clipped))
    enhanced_feature_names.append('MC_ratio_log')
    
    m_height_col, c_height_col = 'M_I_simple_peak_height', 'C_I_simple_peak_height'
    if m_height_col in df_out.columns and c_height_col in df_out.columns:
        c_heights_safe = df_out[c_height_col].replace(0, 1e-6).fillna(1e-6)
        height_ratio = df_out[m_height_col].fillna(0) / c_heights_safe
        enhanced_features['MC_height_ratio'] = height_ratio
        enhanced_feature_names.append('MC_height_ratio')
        enhanced_features['MC_height_ratio_squared'] = height_ratio ** 2
        enhanced_feature_names.append('MC_height_ratio_squared')

    m_base_col, m_total_col = 'M_I_simple_baseline_avg', 'M_I_simple_total_intensity'
    if m_base_col in df_out.columns and m_total_col in df_out.columns:
        base_safe = df_out[m_base_col].replace(0, 1e-6).fillna(1e-6)
        sbr = df_out[m_total_col].fillna(0) / base_safe
        enhanced_features['M_signal_baseline_ratio'] = sbr
        enhanced_feature_names.append('M_signal_baseline_ratio')
        enhanced_features['M_signal_baseline_log'] = np.log1p(np.abs(sbr))
        enhanced_feature_names.append('M_signal_baseline_log')

    m_area_col = 'M_I_simple_peak_area'
    if m_area_col in df_out.columns:
        m_area_safe = df_out[m_area_col].fillna(0)
        m_75th = np.percentile(m_area_safe[m_area_safe > 0], 75) if np.any(m_area_safe > 0) else 0
        m_90th = np.percentile(m_area_safe[m_area_safe > 0], 90) if np.any(m_area_safe > 0) else 0
        
        enhanced_features['high_M_indicator'] = 1 / (1 + np.exp(-(m_area_safe - m_75th) / (m_75th + 1e-6)))
        enhanced_feature_names.append('high_M_indicator')
        enhanced_features['very_high_M_indicator'] = 1 / (1 + np.exp(-(m_area_safe - m_90th) / (m_90th + 1e-6)))
        enhanced_feature_names.append('very_high_M_indicator')
        
    other_elements = set(name.split('_simple')[0] for name in simple_feature_names if '_simple_' in name and 'M_I' not in name and 'C_I' not in name)
    for element in list(other_elements)[:2]:
        element_area_col = f'{element}_simple_peak_area'
        if element_area_col in df_out.columns and m_area_col in df_out.columns:
            element_safe = df_out[element_area_col].replace(0, 1e-6).fillna(1e-6)
            ratio = df_out[m_area_col].fillna(0) / element_safe
            enhanced_features[f'M_{element}_ratio'] = ratio
            enhanced_feature_names.append(f'M_{element}_ratio')

    logger.info(f"Generated {len(enhanced_feature_names)} high-magnesium features.")
    return pd.concat([df_out, enhanced_features], axis=1), enhanced_feature_names

def generate_focused_magnesium_features(df: pd.DataFrame, simple_feature_names: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generates a focused set of magnesium-specific features based on spectroscopic domain knowledge.
    Only includes features directly relevant to magnesium detection and quantification.
    """
    df_out = df.copy()
    index_to_use = df_out.index if hasattr(df_out, 'index') else None
    enhanced_features = pd.DataFrame(index=index_to_use)
    enhanced_feature_names = []
    
    # M/C ratio transformations - fundamental for magnesium analysis
    if 'M_C_ratio' in df_out.columns:
        mc_ratio_safe = df_out['M_C_ratio'].fillna(0.0)
        mc_ratio_clipped = np.clip(mc_ratio_safe, -50.0, 50.0)
        
        # Non-linear transformations capture complex P-C relationships
        enhanced_features['MC_ratio_squared'] = mc_ratio_clipped ** 2
        enhanced_feature_names.append('MC_ratio_squared')
        enhanced_features['MC_ratio_log'] = np.log1p(np.abs(mc_ratio_clipped))
        enhanced_feature_names.append('MC_ratio_log')
    
    # M/C peak height ratio - important for spectral line intensity comparisons
    m_height_col, c_height_col = 'M_I_simple_peak_height', 'C_I_simple_peak_height'
    if m_height_col in df_out.columns and c_height_col in df_out.columns:
        c_heights_safe = df_out[c_height_col].replace(0, 1e-6).fillna(1e-6)
        height_ratio = df_out[m_height_col].fillna(0) / c_heights_safe
        enhanced_features['MC_height_ratio'] = height_ratio
        enhanced_feature_names.append('MC_height_ratio')
    
    # Signal-to-baseline ratio for magnesium - indicates spectral quality
    m_base_col, m_total_col = 'M_I_simple_baseline_avg', 'M_I_simple_total_intensity'
    if m_base_col in df_out.columns and m_total_col in df_out.columns:
        base_safe = df_out[m_base_col].replace(0, 1e-6).fillna(1e-6)
        sbr = df_out[m_total_col].fillna(0) / base_safe
        enhanced_features['M_signal_baseline_ratio'] = sbr
        enhanced_feature_names.append('M_signal_baseline_ratio')
    
    # P ratio to key interfering/related elements (N, K, Ca) using simple peak areas
    m_area_col = 'M_I_simple_peak_area'
    key_elements = ['N_I_help', 'K_I_help', 'CA_I_help']  # Elements that can interfere with P detection
    
    if m_area_col in df_out.columns:
        for element in key_elements:
            element_area_col = f'{element}_simple_peak_area'
            if element_area_col in df_out.columns:
                element_safe = df_out[element_area_col].replace(0, 1e-6).fillna(1e-6)
                ratio = df_out[m_area_col].fillna(0) / element_safe
                # Use 'area_ratio' suffix to distinguish from peak intensity ratios
                enhanced_features[f'M_{element.split("_")[0]}_area_ratio'] = ratio
                enhanced_feature_names.append(f'M_{element.split("_")[0]}_area_ratio')
    
    logger.info(f"Generated {len(enhanced_feature_names)} focused magnesium features.")
    return pd.concat([df_out, enhanced_features], axis=1), enhanced_feature_names