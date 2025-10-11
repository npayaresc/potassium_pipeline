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
    generate_high_potassium_features,
    generate_focused_potassium_features,
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
        args: Tuple containing (idx, wavelengths, intensities, config_dict, strategy, use_enhanced, regions_list)

    Returns:
        Dictionary with 'idx' and 'features' keys
    """
    idx, wavelengths, intensities, config_dict, strategy, use_enhanced, regions_list = args

    try:
        # Reconstruct config from dict (configs aren't pickle-able directly)
        from src.config.pipeline_config import Config, PeakRegion
        config = Config(**config_dict)

        # Reconstruct regions from list of dicts
        regions = [PeakRegion(**r) for r in regions_list]

        features = {}
        wavelengths = np.asarray(wavelengths)
        intensities = np.asarray(intensities)

        if intensities.size == 0:
            logger.warning(f"Empty intensities found for sample {idx}")
            return {'idx': idx, 'features': {}}

        # Extract simple features for strategy-specific regions
        for region in regions:
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
        extractor = SpectralFeatureExtractor(
            enable_preprocessing=config.use_spectral_preprocessing,
            preprocessing_method=config.spectral_preprocessing_method
        )
        spectra_2d = intensities.reshape(-1, 1) if intensities.ndim == 1 else intensities

        fit_results = extractor.extract_features(
            wavelengths=wavelengths,
            spectra=spectra_2d,
            regions=_convert_to_results_regions(regions),
            peak_shapes=config.peak_shapes * len(regions),
        )
        
        for res in fit_results.fitting_results:
            element = res.region_result.region.element

            # Extract peak areas (original feature)
            for i, area in enumerate(res.peak_areas):
                features[f"{element}_peak_{i}"] = area

            # === EXTRACT PHYSICS-INFORMED FEATURES (NEW) ===
            for i, peak_fit in enumerate(res.fit_results):
                # FWHM - Full Width at Half Maximum
                features[f"{element}_fwhm_{i}"] = peak_fit.fwhm

                # Gamma (Stark broadening parameter)
                features[f"{element}_gamma_{i}"] = peak_fit.gamma

                # Fit quality (R²)
                features[f"{element}_fit_quality_{i}"] = peak_fit.fit_quality

                # Peak asymmetry (self-absorption indicator)
                features[f"{element}_asymmetry_{i}"] = peak_fit.peak_asymmetry

                # Amplitude (peak height for Lorentzian)
                features[f"{element}_amplitude_{i}"] = peak_fit.amplitude

                # Kurtosis (tailedness of peak distribution)
                features[f"{element}_kurtosis_{i}"] = peak_fit.kurtosis

                # Derived feature: FWHM × Asymmetry (absorption strength)
                features[f"{element}_absorption_index_{i}"] = peak_fit.fwhm * abs(peak_fit.peak_asymmetry)
        
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
            strategy: Feature strategy ('K_only', 'simple_only', 'full_context')
            n_jobs: Number of parallel jobs. -1 uses all CPU cores, -2 uses all but one
        """
        self.config = config
        self.strategy = strategy
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count() + 1 + n_jobs
        self.feature_names_out_: List[str] = []
        self._all_simple_names = []
        self._high_k_names = []

        # Strategy-optimized regions to avoid extracting unused features
        self._regions = self._get_strategy_regions()

        # Initialize enhanced features flag
        self._use_enhanced = any([
            config.enable_molecular_bands,
            config.enable_advanced_ratios,
            config.enable_spectral_patterns,
            config.enable_interference_correction,
            config.enable_plasma_indicators,
        ])

        logger.info(f"Initialized ParallelSpectralFeatureGenerator with {self.n_jobs} workers")

    def _get_strategy_regions(self) -> List[PeakRegion]:
        """Get regions optimized for current strategy."""
        if self.strategy == "K_only":
            # K regions + essential context elements for meaningful ratios
            k_regions = [self.config.potassium_region]
            k_regions.extend([r for r in self.config.context_regions if r.element.startswith("K_I")])

            # Add C_I for K_C_ratio calculation (baseline normalization)
            c_region = next((r for r in self.config.context_regions if r.element == "C_I"), None)
            if c_region:
                k_regions.append(c_region)

            # Add critical elements for agronomic ratios (Ca, N, P, S for nutrient balance)
            critical_elements = ['CA_I_help', 'CA_II_393', 'N_I_help', 'P_I_secondary']
            for element in critical_elements:
                region = next((r for r in self.config.context_regions if r.element == element), None)
                if region:
                    k_regions.append(region)

            # Add S_I from macro_elements (secondary nutrient)
            s_region = next((r for r in self.config.macro_elements if r.element == "S_I"), None)
            if s_region:
                k_regions.append(s_region)

            # Optional: Add Mg for complete cation analysis (can remove if too many features)
            mg_elements = ['Mg_I_285', 'Mg_II']
            for element in mg_elements:
                region = next((r for r in self.config.context_regions if r.element == element), None)
                if region:
                    k_regions.append(region)

            logger.info(f"K_only strategy: using {len(k_regions)} regions (vs {len(self.config.all_regions)} total)")
            logger.info(f"Regions: {[r.element for r in k_regions]}")
            return k_regions
        else:
            # Full regions for simple_only and full_context
            return self.config.all_regions
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits the transformer by determining the canonical feature names."""
        self._set_feature_names(X.iloc[0:1])
        logger.info(f"ParallelSpectralFeatureGenerator fitted for strategy '{self.strategy}'.")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw spectral data into the final feature matrix using parallel processing.
        """
        # Handle numpy arrays from SHAP or other tools
        if isinstance(X, np.ndarray):
            logger.debug("Converting numpy array to DataFrame for parallel feature extraction")
            # Check if this is a structured array with object dtype (array-valued columns)
            if X.dtype == object and X.shape[1] == 2:
                # Array-valued columns (wavelengths and intensities as arrays)
                X = pd.DataFrame({
                    "wavelengths": X[:, 0],
                    "intensities": X[:, 1]
                })
            elif X.shape[1] == 2:
                # Regular 2-column array
                X = pd.DataFrame(X, columns=["wavelengths", "intensities"])
            else:
                raise ValueError(f"Expected 2 columns (wavelengths, intensities), got {X.shape[1]}")

        logger.info(f"Starting parallel feature extraction for {len(X)} samples with {self.n_jobs} workers")

        # Prepare arguments for parallel processing
        # Convert config to dict for pickling
        config_dict = self.config.model_dump()

        # Convert regions to list of dicts for pickling
        regions_list = [r.model_dump() for r in self._regions]

        args_list = [
            (
                idx,
                row["wavelengths"],
                row["intensities"],
                config_dict,
                self.strategy,
                self._use_enhanced,
                regions_list
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
                    sample_features = result.get('features', {})

                    if not sample_features:
                        logger.warning("No features extracted for sample %s in parallel processing", idx)
                    else:
                        logger.debug("Extracted %d features for sample %s in parallel processing",
                                   len(sample_features), idx)

                    results.append(result)
                except Exception as e:
                    logger.error("Failed to process sample %s in parallel: %s", idx, e)
                    logger.error("Creating empty feature set for sample %s", idx)
                    results.append({'idx': idx, 'features': {}})
        
        # Sort results by index to maintain original order
        if hasattr(X, 'index'):
            try:
                results.sort(key=lambda x: X.index.get_loc(x['idx']))
            except (KeyError, ValueError):
                # If index sorting fails, maintain original order
                pass
        
        # Extract features from results
        base_features_list = [r['features'] for r in results]
        index_to_use = X.index if hasattr(X, 'index') else None
        base_features_df = pd.DataFrame(base_features_list, index=index_to_use)
        
        # Calculate K/C ratio
        k_area = (
            base_features_df["K_I_peak_0"].fillna(0.0)
            if "K_I_peak_0" in base_features_df
            else pd.Series(0.0, index=base_features_df.index)
        )
        c_area = (
            base_features_df["C_I_peak_0"].fillna(1e-6)
            if "C_I_peak_0" in base_features_df
            else pd.Series(1e-6, index=base_features_df.index)
        )
        
        kc_ratio_raw = k_area / c_area.replace(0, 1e-6)
        base_features_df["K_C_ratio"] = np.clip(kc_ratio_raw, -50.0, 50.0)
        
        # Generate additional features based on config
        if self.config.use_focused_potassium_features:
            full_features_df, _ = generate_focused_potassium_features(
                base_features_df, self._all_simple_names
            )
        else:
            full_features_df, _ = generate_high_potassium_features(
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
        
        # Check for missing features before reindexing
        missing_features = [f for f in expected_features if f not in full_features_df.columns]
        if missing_features:
            logger.warning("Missing %d features during parallel processing reindexing, filling with zeros: %s",
                         len(missing_features), missing_features[:10])

        # Reindex to ensure all expected features are present
        final_df = full_features_df.reindex(
            columns=expected_features, fill_value=0.0
        )

        # Final NaN check and cleanup for parallel processing
        if final_df.isnull().any().any():
            nan_count = final_df.isnull().sum().sum()
            nan_cols = final_df.columns[final_df.isnull().any()].tolist()
            logger.warning("[PARALLEL FEATURE ENGINEERING] Found %d NaN values in columns: %s",
                         nan_count, nan_cols[:10])
            logger.warning("Filling NaN values with 0.0 to prevent downstream issues")
            final_df = final_df.fillna(0.0)

        # Ensure the final dataframe maintains the original index
        if hasattr(X, 'index'):
            final_df.index = X.index

        logger.info(
            "Parallel transformation complete: %d features for strategy '%s'.",
            final_df.shape[1], self.strategy
        )
        return final_df
    
    def _set_feature_names(self, X_sample):
        """
        Defines the canonical list of feature names for each strategy.
        This is identical to the original SpectralFeatureGenerator method.
        """
        # Define all possible base feature names
        all_complex_names = []
        physics_informed_names = []  # NEW: Physics-informed features
        for region in self._regions:
            for i in range(region.n_peaks):
                # Original peak area feature
                all_complex_names.append(f"{region.element}_peak_{i}")

                # NEW: Physics-informed features from Lorentzian fits
                physics_informed_names.append(f"{region.element}_fwhm_{i}")
                physics_informed_names.append(f"{region.element}_gamma_{i}")
                physics_informed_names.append(f"{region.element}_fit_quality_{i}")
                physics_informed_names.append(f"{region.element}_asymmetry_{i}")
                physics_informed_names.append(f"{region.element}_amplitude_{i}")
                physics_informed_names.append(f"{region.element}_kurtosis_{i}")
                physics_informed_names.append(f"{region.element}_absorption_index_{i}")

        # Simple features (8 per region)
        self._all_simple_names = []
        for region in self._regions:
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
        sample_idx = X_sample.index[0] if hasattr(X_sample, 'index') else 0
        regions_list = [r.model_dump() for r in self._regions]
        sample_features = _extract_features_for_row((
            sample_idx,
            X_sample.iloc[0]["wavelengths"],
            X_sample.iloc[0]["intensities"],
            self.config.model_dump(),
            self.strategy,
            False,  # Don't use enhanced for name determination
            regions_list
        ))
        
        sample_base_df = pd.DataFrame([sample_features['features']])
        sample_base_df["K_C_ratio"] = 0.0
        
        # Get high potassium feature names
        if self.config.use_focused_potassium_features:
            _, self._high_k_names = generate_focused_potassium_features(
                sample_base_df, self._all_simple_names
            )
        else:
            _, self._high_k_names = generate_high_potassium_features(
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
        if self.strategy == "K_only":
            # Include all K_I features (K_I_, K_I_404_) and K_C_ratio
            k_complex = [name for name in all_complex_names if name.startswith("K_I")]
            k_simple = [
                name for name in self._all_simple_names if name.startswith("K_I")
            ]
            # Include all enhanced K features (ratios now have real values from extracted regions)
            k_enhanced = [
                name for name in enhanced_names
                if "K" in name or "potassium" in name.lower()
            ]
            # NEW: Include physics-informed features for K
            k_physics = [name for name in physics_informed_names if name.startswith("K_I")]

            # Always add K_C_ratio (critical potassium indicator, computed separately)
            self.feature_names_out_ = k_complex + k_physics + k_simple + k_enhanced + ["K_C_ratio"] + self._high_k_names
        elif self.strategy == "simple_only":
            self.feature_names_out_ = (
                self._all_simple_names
                + physics_informed_names  # NEW: Add physics-informed features
                + ["K_C_ratio"]
                + self._high_k_names
                + enhanced_names
            )
        elif self.strategy == "full_context":
            self.feature_names_out_ = (
                all_complex_names
                + physics_informed_names  # NEW: Add physics-informed features
                + self._all_simple_names
                + ["K_C_ratio"]
                + self._high_k_names
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