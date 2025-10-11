"""
Feature Engineering Helper Functions

This module contains the detailed logic for creating specific features,
adapted from the original nitrogen_regression_model.py script for potassium prediction.
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
    
    peak_area = np.trapezoid(avg_spectrum, wavelengths[mask])
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

def generate_high_potassium_features(df: pd.DataFrame, simple_feature_names: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generates the full set of enhanced features for high potassium detection.
    This logic is adapted from the original _generate_high_nitrogen_features.
    """
    df_out = df.copy()
    index_to_use = df_out.index if hasattr(df_out, 'index') else None
    enhanced_features = pd.DataFrame(index=index_to_use)
    enhanced_feature_names = []
    
    # Apply reasonable bounds to KC_ratio to prevent extreme values
    if 'K_C_ratio' not in df_out.columns:
        raise ValueError("K_C_ratio not found in features. Ensure potassium (K_I) features are properly extracted.")
    
    kc_ratio_safe = df_out['K_C_ratio'].fillna(0.0)
    # Clip ratio to reasonable bounds (e.g., -50 to 50) to prevent corruption
    kc_ratio_clipped = np.clip(kc_ratio_safe, -50.0, 50.0)
    
    enhanced_features['KC_ratio_squared'] = kc_ratio_clipped ** 2
    enhanced_feature_names.append('KC_ratio_squared')
    enhanced_features['KC_ratio_cubic'] = kc_ratio_clipped ** 3
    enhanced_feature_names.append('KC_ratio_cubic')
    enhanced_features['KC_ratio_log'] = np.log1p(np.abs(kc_ratio_clipped))
    enhanced_feature_names.append('KC_ratio_log')
    
    # K/C height ratio calculation
    k_height_col = 'K_I_simple_peak_height'
    c_height_col = 'C_I_simple_peak_height'
    
    if k_height_col not in df_out.columns:
        raise ValueError(f"{k_height_col} not found. Ensure potassium (K_I) features are properly extracted.")
    if c_height_col not in df_out.columns:
        raise ValueError(f"{c_height_col} not found. Ensure carbon (C_I) features are properly extracted.")
    
    c_heights_safe = df_out[c_height_col].replace(0, 1e-6).fillna(1e-6)
    height_ratio = df_out[k_height_col].fillna(0) / c_heights_safe
    enhanced_features['KC_height_ratio'] = height_ratio
    enhanced_feature_names.append('KC_height_ratio')
    enhanced_features['KC_height_ratio_squared'] = height_ratio ** 2
    enhanced_feature_names.append('KC_height_ratio_squared')

    # K signal-to-baseline ratio
    k_base_col = 'K_I_simple_baseline_avg'
    k_total_col = 'K_I_simple_total_intensity'
    
    if k_base_col not in df_out.columns:
        raise ValueError(f"{k_base_col} not found. Ensure potassium (K_I) features are properly extracted.")
    if k_total_col not in df_out.columns:
        raise ValueError(f"{k_total_col} not found. Ensure potassium (K_I) features are properly extracted.")
    
    base_safe = df_out[k_base_col].replace(0, 1e-6).fillna(1e-6)
    sbr = df_out[k_total_col].fillna(0) / base_safe
    enhanced_features['K_signal_baseline_ratio'] = sbr
    enhanced_feature_names.append('K_signal_baseline_ratio')
    enhanced_features['K_signal_baseline_log'] = np.log1p(np.abs(sbr))
    enhanced_feature_names.append('K_signal_baseline_log')

    # K area-based indicators
    k_area_col = 'K_I_simple_peak_area'
    
    if k_area_col not in df_out.columns:
        raise ValueError(f"{k_area_col} not found. Ensure potassium (K_I) features are properly extracted.")
    
    k_area_safe = df_out[k_area_col].fillna(0)
    k_75th = np.percentile(k_area_safe[k_area_safe > 0], 75) if np.any(k_area_safe > 0) else 0
    k_90th = np.percentile(k_area_safe[k_area_safe > 0], 90) if np.any(k_area_safe > 0) else 0
    
    enhanced_features['high_K_indicator'] = 1 / (1 + np.exp(-(k_area_safe - k_75th) / (k_75th + 1e-6)))
    enhanced_feature_names.append('high_K_indicator')
    enhanced_features['very_high_K_indicator'] = 1 / (1 + np.exp(-(k_area_safe - k_90th) / (k_90th + 1e-6)))
    enhanced_feature_names.append('very_high_K_indicator')
        
    # Exclude K_I and C_I from other elements
    other_elements = set(name.split('_simple')[0] for name in simple_feature_names if '_simple_' in name and 'K_I' not in name and 'C_I' not in name)
    for element in list(other_elements)[:2]:
        element_area_col = f'{element}_simple_peak_area'
        if element_area_col in df_out.columns and k_area_col in df_out.columns:
            element_safe = df_out[element_area_col].replace(0, 1e-6).fillna(1e-6)
            ratio = df_out[k_area_col].fillna(0) / element_safe
            enhanced_features[f'K_{element}_ratio'] = ratio
            enhanced_feature_names.append(f'K_{element}_ratio')

    logger.info(f"Generated {len(enhanced_feature_names)} high-potassium features.")
    return pd.concat([df_out, enhanced_features], axis=1), enhanced_feature_names

def generate_focused_potassium_features(df: pd.DataFrame, simple_feature_names: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generates a focused set of potassium-specific features based on spectroscopic domain knowledge.
    Only includes features directly relevant to potassium detection and quantification.
    """
    df_out = df.copy()
    index_to_use = df_out.index if hasattr(df_out, 'index') else None
    enhanced_features = pd.DataFrame(index=index_to_use)
    enhanced_feature_names = []
    
    # K/C ratio transformations - fundamental for potassium analysis
    if 'K_C_ratio' not in df_out.columns:
        raise ValueError("K_C_ratio not found in features. Ensure potassium (K_I) features are properly extracted.")
    
    kc_ratio_safe = df_out['K_C_ratio'].fillna(0.0)
    kc_ratio_clipped = np.clip(kc_ratio_safe, -50.0, 50.0)
    
    # Non-linear transformations capture complex K-C relationships
    enhanced_features['KC_ratio_squared'] = kc_ratio_clipped ** 2
    enhanced_feature_names.append('KC_ratio_squared')
    enhanced_features['KC_ratio_log'] = np.log1p(np.abs(kc_ratio_clipped))
    enhanced_feature_names.append('KC_ratio_log')
    
    # K/C peak height ratio - important for spectral line intensity comparisons
    k_height_col = 'K_I_simple_peak_height'
    c_height_col = 'C_I_simple_peak_height'
    
    if k_height_col not in df_out.columns:
        raise ValueError(f"{k_height_col} not found. Ensure potassium (K_I) features are properly extracted.")
    if c_height_col not in df_out.columns:
        raise ValueError(f"{c_height_col} not found. Ensure carbon (C_I) features are properly extracted.")
    
    c_heights_safe = df_out[c_height_col].replace(0, 1e-6).fillna(1e-6)
    height_ratio = df_out[k_height_col].fillna(0) / c_heights_safe
    enhanced_features['KC_height_ratio'] = height_ratio
    enhanced_feature_names.append('KC_height_ratio')
    
    # Signal-to-baseline ratio for potassium - indicates spectral quality
    k_base_col = 'K_I_simple_baseline_avg'
    k_total_col = 'K_I_simple_total_intensity'
    
    if k_base_col not in df_out.columns:
        raise ValueError(f"{k_base_col} not found. Ensure potassium (K_I) features are properly extracted.")
    if k_total_col not in df_out.columns:
        raise ValueError(f"{k_total_col} not found. Ensure potassium (K_I) features are properly extracted.")
    
    base_safe = df_out[k_base_col].replace(0, 1e-6).fillna(1e-6)
    sbr = df_out[k_total_col].fillna(0) / base_safe
    enhanced_features['K_signal_baseline_ratio'] = sbr
    enhanced_feature_names.append('K_signal_baseline_ratio')
    
    # K ratio to key interfering/related elements using simple peak areas
    k_area_col = 'K_I_simple_peak_area'
    
    if k_area_col not in df_out.columns:
        raise ValueError(f"{k_area_col} not found. Ensure potassium (K_I) features are properly extracted.")
    
    # Key elements that can interfere with potassium detection or are agronomically related
    # Priority order based on interference strength and agronomic importance:
    # 1. Ca and Mg - directly compete with K uptake in plants (antagonistic)
    # 2. P - important K/P balance for plant metabolism
    # 3. N - affects K availability and uptake
    # Note: We use the actual element names from PeakRegions
    key_elements = []
    
    # Always include Ca and Mg if available (strongest interference)
    if 'CA_I_help_simple_peak_area' in df_out.columns:
        key_elements.append('CA_I_help')
    if 'Mg_I_285_simple_peak_area' in df_out.columns:
        key_elements.append('Mg_I_285')
    
    # Include P and N for nutrient balance ratios
    if 'P_I_secondary_simple_peak_area' in df_out.columns:
        key_elements.append('P_I_secondary')
    elif 'P_I_simple_peak_area' in df_out.columns:
        key_elements.append('P_I')
    
    if 'N_I_help_simple_peak_area' in df_out.columns:
        key_elements.append('N_I_help')
    
    # Add Fe if available (spectral interference near K 404nm lines)
    if 'Fe_I_simple_peak_area' in df_out.columns:
        key_elements.append('Fe_I')
    
    # Add S if available (sulfate affects K availability)
    if 'S_I_simple_peak_area' in df_out.columns:
        key_elements.append('S_I')
    
    if k_area_col in df_out.columns:
        for element in key_elements:
            element_area_col = f'{element}_simple_peak_area'
            if element_area_col in df_out.columns:
                element_safe = df_out[element_area_col].replace(0, 1e-6).fillna(1e-6)
                ratio = df_out[k_area_col].fillna(0) / element_safe
                # Use 'area_ratio' suffix to distinguish from peak intensity ratios
                enhanced_features[f'K_{element.split("_")[0]}_area_ratio'] = ratio
                enhanced_feature_names.append(f'K_{element.split("_")[0]}_area_ratio')
    
    logger.info(f"Generated {len(enhanced_feature_names)} focused potassium features.")
    return pd.concat([df_out, enhanced_features], axis=1), enhanced_feature_names