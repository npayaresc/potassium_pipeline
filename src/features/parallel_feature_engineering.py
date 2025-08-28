"""
Parallel Feature Engineering Module: Optimized version using multiprocessing
for faster feature generation from spectral data.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.features.feature_helpers import (
    extract_full_simple_features,
    generate_high_magnesium_features,
    generate_focused_magnesium_features,
)
from src.config.pipeline_config import Config, PeakRegion
from src.spectral_extraction.extractor import SpectralFeatureExtractor
from src.spectral_extraction.results import PeakRegion as ResultsPeakRegion
from src.features.feature_helpers import PeakRegion as HelperPeakRegion
from src.features.enhanced_features import EnhancedSpectralFeatures

logger = logging.getLogger(__name__)


def _convert_to_results_regions(
    config_regions: List[PeakRegion],
) -> List[ResultsPeakRegion]:
    """Convert config PeakRegion objects to extractor.results PeakRegion objects."""
    return [
        ResultsPeakRegion(
            element=r.element,
            lower_wavelength=r.lower_wavelength,
            upper_wavelength=r.upper_wavelength,
            center_wavelengths=list(r.center_wavelengths),
        )
        for r in config_regions
    ]


def _extract_features_for_row(args: Tuple) -> Dict[str, Any]:
    """
    Extract features for a single row. This function is designed to be pickle-able
    for multiprocessing.
    
    Args:
        args: Tuple containing (idx, wavelengths, intensities, config_dict, strategy, use_enhanced)
    
    Returns:
        Dictionary with 'idx' and 'features' keys
    """
    idx, wavelengths, intensities, config_dict, strategy, use_enhanced = args
    
    try:
        # Reconstruct config from dict (configs aren't pickle-able directly)
        from src.config.pipeline_config import Config
        config = Config(**config_dict)
        
        features = {}
        wavelengths = np.asarray(wavelengths)
        intensities = np.asarray(intensities)
        
        if intensities.size == 0:
            logger.warning(f"Empty intensities found for sample {idx}")
            return {'idx': idx, 'features': {}}
        
        # Extract simple features for all regions
        for region in config.all_regions:
            helper_region = HelperPeakRegion(
                region.element,
                region.lower_wavelength,
                region.upper_wavelength,
                region.center_wavelengths,
            )
            features.update(
                extract_full_simple_features(helper_region, wavelengths, intensities)
            )
        
        # Extract complex features using SpectralFeatureExtractor
        extractor = SpectralFeatureExtractor()
        spectra_2d = intensities.reshape(-1, 1) if intensities.ndim == 1 else intensities
        
        fit_results = extractor.extract_features(
            wavelengths=wavelengths,
            spectra=spectra_2d,
            regions=_convert_to_results_regions(config.all_regions),
            peak_shapes=config.peak_shapes * len(config.all_regions),
        )
        
        for res in fit_results.fitting_results:
            element = res.region_result.region.element
            for i, area in enumerate(res.peak_areas):
                features[f"{element}_peak_{i}"] = area
        
        # Add enhanced features if enabled
        if use_enhanced:
            enhanced_features = EnhancedSpectralFeatures(config)
            enhanced_feats = enhanced_features.transform(
                features, wavelengths, intensities
            )
            features.update(enhanced_feats)
        
        return {'idx': idx, 'features': features}
        
    except Exception as e:
        logger.warning(f"Error processing sample {idx}: {e}")
        return {'idx': idx, 'features': {}}


class ParallelSpectralFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Parallel version of SpectralFeatureGenerator that uses multiprocessing
    to extract features from multiple samples simultaneously.
    """
    
    def __init__(self, config: Config, strategy: str = "simple_only", n_jobs: int = -1):
        """
        Initialize parallel feature generator.
        
        Args:
            config: Pipeline configuration
            strategy: Feature strategy ('Mg_only', 'simple_only', 'full_context')
            n_jobs: Number of parallel jobs. -1 uses all CPU cores, -2 uses all but one
        """
        self.config = config
        self.strategy = strategy
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count() + 1 + n_jobs
        self.feature_names_out_: List[str] = []
        self._all_simple_names = []
        self._high_p_names = []
        
        # Initialize enhanced features flag
        self._use_enhanced = any([
            config.enable_molecular_bands,
            config.enable_advanced_ratios,
            config.enable_spectral_patterns,
            config.enable_interference_correction,
            config.enable_plasma_indicators,
        ])
        
        logger.info(f"Initialized ParallelSpectralFeatureGenerator with {self.n_jobs} workers")
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits the transformer by determining the canonical feature names."""
        self._set_feature_names(X.iloc[0:1])
        logger.info(f"ParallelSpectralFeatureGenerator fitted for strategy '{self.strategy}'.")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw spectral data into the final feature matrix using parallel processing.
        """
        logger.info(f"Starting parallel feature extraction for {len(X)} samples with {self.n_jobs} workers")
        
        # Prepare arguments for parallel processing
        # Convert config to dict for pickling
        config_dict = self.config.model_dump()
        
        args_list = [
            (
                idx,
                row["wavelengths"],
                row["intensities"],
                config_dict,
                self.strategy,
                self._use_enhanced
            )
            for idx, row in X.iterrows()
        ]
        
        # Process samples in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all tasks
            future_to_args = {
                executor.submit(_extract_features_for_row, args): args[0] 
                for args in args_list
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_args):
                idx = future_to_args[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process sample {idx}: {e}")
                    results.append({'idx': idx, 'features': {}})
        
        # Sort results by index to maintain original order
        results.sort(key=lambda x: X.index.get_loc(x['idx']))
        
        # Extract features from results
        base_features_list = [r['features'] for r in results]
        base_features_df = pd.DataFrame(base_features_list, index=X.index)
        
        # Calculate P/C ratio
        p_area = (
            base_features_df["P_I_peak_0"]
            if "P_I_peak_0" in base_features_df
            else pd.Series(np.nan, index=base_features_df.index)
        )
        c_area = (
            base_features_df["C_I_peak_0"]
            if "C_I_peak_0" in base_features_df
            else pd.Series(np.nan, index=base_features_df.index)
        )
        
        pc_ratio_raw = p_area / c_area.replace(0, 1e-6)
        base_features_df["P_C_ratio"] = np.clip(pc_ratio_raw, -50.0, 50.0)
        
        # Generate additional features based on config
        if self.config.use_focused_magnesium_features:
            full_features_df, _ = generate_focused_magnesium_features(
                base_features_df, self._all_simple_names
            )
        else:
            full_features_df, _ = generate_high_magnesium_features(
                base_features_df, self._all_simple_names
            )
        
        # Validate feature names
        expected_features = self.get_feature_names_out()
        if len(expected_features) != len(set(expected_features)):
            from collections import Counter
            feature_counts = Counter(expected_features)
            duplicates = {
                name: count for name, count in feature_counts.items() if count > 1
            }
            raise ValueError(f"Duplicate feature names: {list(duplicates.keys())[:10]}")
        
        # Reindex to ensure all expected features are present
        final_df = full_features_df.reindex(
            columns=expected_features, fill_value=np.nan
        )
        final_df.index = X.index
        
        logger.info(
            f"Parallel transformation complete: {final_df.shape[1]} features for strategy '{self.strategy}'."
        )
        return final_df
    
    def _set_feature_names(self, X_sample):
        """
        Defines the canonical list of feature names for each strategy.
        This is identical to the original SpectralFeatureGenerator method.
        """
        # Define all possible base feature names
        all_complex_names = []
        for region in self.config.all_regions:
            for i in range(region.n_peaks):
                all_complex_names.append(f"{region.element}_peak_{i}")
        
        # Simple features (8 per region, 48 total for 6 regions)
        self._all_simple_names = []
        for region in self.config.all_regions:
            prefix = f"{region.element}_simple"
            self._all_simple_names.extend([
                f"{prefix}_peak_area",
                f"{prefix}_peak_height",
                f"{prefix}_peak_center_intensity",
                f"{prefix}_baseline_avg",
                f"{prefix}_signal_range",
                f"{prefix}_total_intensity",
                f"{prefix}_height_to_baseline",
                f"{prefix}_normalized_area",
            ])
        
        # Extract a sample to determine high_p_names dynamically
        sample_features = _extract_features_for_row((
            X_sample.index[0],
            X_sample.iloc[0]["wavelengths"],
            X_sample.iloc[0]["intensities"],
            self.config.model_dump(),
            self.strategy,
            False  # Don't use enhanced for name determination
        ))
        
        sample_base_df = pd.DataFrame([sample_features['features']])
        sample_base_df["P_C_ratio"] = 0.0
        
        # Get high magnesium feature names
        if self.config.use_focused_magnesium_features:
            _, self._high_p_names = generate_focused_magnesium_features(
                sample_base_df, self._all_simple_names
            )
        else:
            _, self._high_p_names = generate_high_magnesium_features(
                sample_base_df, self._all_simple_names
            )
        
        # Get enhanced feature names if enabled
        enhanced_names = []
        if self._use_enhanced:
            enhanced_features = EnhancedSpectralFeatures(self.config)
            sample_wavelengths = np.asarray(X_sample.iloc[0]["wavelengths"])
            sample_intensities = np.asarray(X_sample.iloc[0]["intensities"])
            enhanced_sample = enhanced_features.transform(
                sample_features['features'], sample_wavelengths, sample_intensities
            )
            enhanced_names = list(enhanced_sample.keys())
        
        # Set feature names based on strategy
        if self.strategy == "Mg_only":
            p_complex = [name for name in all_complex_names if name.startswith("P_I_")]
            p_simple = [
                name for name in self._all_simple_names if name.startswith("P_I_simple")
            ]
            p_enhanced = [
                name
                for name in enhanced_names
                if "Mg" in name or "magnesium" in name.lower()
            ]
            self.feature_names_out_ = p_complex + p_simple + p_enhanced
        elif self.strategy == "simple_only":
            self.feature_names_out_ = (
                self._all_simple_names
                + ["P_C_ratio"]
                + self._high_p_names
                + enhanced_names
            )
        elif self.strategy == "full_context":
            self.feature_names_out_ = (
                all_complex_names
                + self._all_simple_names
                + ["P_C_ratio"]
                + self._high_p_names
                + enhanced_names
            )
        else:
            raise ValueError(f"Unknown feature strategy: {self.strategy}")
        
        # Validate no duplicates
        from collections import Counter
        counts = Counter(self.feature_names_out_)
        duplicates = {name: count for name, count in counts.items() if count > 1}
        if duplicates:
            raise ValueError(f"Duplicate feature names: {list(duplicates.keys())}")
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features."""
        return self.feature_names_out_