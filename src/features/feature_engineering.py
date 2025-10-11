"""
Feature Engineering Module: Defines reusable scikit-learn compatible transformers for
creating features from raw spectral data.
"""

import logging
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.features.feature_helpers import (
    extract_full_simple_features,
    generate_high_potassium_features,
    generate_focused_potassium_features,
)
from src.config.pipeline_config import Config, PeakRegion
from src.spectral_extraction.extractor import SpectralFeatureExtractor
from src.spectral_extraction.results import PeakRegion as ResultsPeakRegion
from src.features.feature_helpers import PeakRegion as HelperPeakRegion
from src.utils.helpers import OutlierClipper
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


class PandasStandardScaler(BaseEstimator, TransformerMixin):
    """StandardScaler that preserves pandas DataFrame structure and column names."""

    def __init__(self, **kwargs):
        self.scaler = StandardScaler(**kwargs)
        self.feature_names_in_: List[str] = []

    def fit(self, X, y=None, **fit_params):
        if isinstance(X, pd.DataFrame):
            # Ensure column names are strings for stable typing
            self.feature_names_in_ = [str(c) for c in X.columns.tolist()]
            self.scaler.fit(X)
        else:
            self.scaler.fit(X)
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        elif self.feature_names_in_:
            return pd.DataFrame(X_scaled, columns=pd.Index(self.feature_names_in_))
        else:
            return X_scaled

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features."""
        if self.feature_names_in_:
            return np.array(self.feature_names_in_)
        elif input_features is not None:
            return np.array(list(input_features))
        else:
            return None


class RawSpectralTransformer(BaseEstimator, TransformerMixin):
    """
    Extracts raw filtered intensities from PeakRegions without feature engineering.

    This transformer applies domain knowledge filtering (PeakRegions) but skips
    mathematical feature engineering, providing raw intensity values directly
    to machine learning models.
    """

    def __init__(self, config: Config):
        self.config = config
        self.extractor = SpectralFeatureExtractor(
            enable_preprocessing=config.use_spectral_preprocessing,
            preprocessing_method=config.spectral_preprocessing_method
        )
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit by determining actual feature names from a sample."""
        # Extract actual feature names by processing a sample
        sample_row = X.iloc[0]
        wavelengths = np.asarray(sample_row["wavelengths"])
        intensities = np.asarray(sample_row["intensities"])

        # Ensure intensities is 2D for the extractor
        if intensities.ndim == 1:
            intensities_2d = intensities.reshape(-1, 1)
        else:
            intensities_2d = intensities

        try:
            # Use isolate_peaks to get actual wavelengths
            region_results = self.extractor.isolate_peaks(
                wavelengths,
                intensities_2d,
                _convert_to_results_regions(self._regions),
                baseline_correction=self.config.baseline_correction,
                area_normalization=False,
            )

            # Generate feature names from actual extracted wavelengths
            self.feature_names_out_ = []
            for region_result in region_results:
                element = region_result.region.element
                region_wavelengths = region_result.wavelengths

                for wl in region_wavelengths:
                    self.feature_names_out_.append(f"{element}_wl_{wl:.2f}nm")

            logger.info(
                "RawSpectralTransformer fitted. Will extract %d raw intensity features.",
                len(self.feature_names_out_)
            )

        except Exception as e:
            logger.warning(
                "Could not extract actual wavelengths during fit, using estimates: %s", e
            )
            # Fallback to estimated feature names
            self.feature_names_out_ = []
            for region in self._regions:
                element = region.element
                lower_wl = region.lower_wavelength
                upper_wl = region.upper_wavelength
                wavelength_resolution = 0.34
                estimated_points = (
                    int((upper_wl - lower_wl) / wavelength_resolution) + 1
                )

                for i in range(estimated_points):
                    approx_wavelength = lower_wl + (i * wavelength_resolution)
                    self.feature_names_out_.append(
                        f"{element}_wl_{approx_wavelength:.2f}nm"
                    )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform spectral data into raw filtered intensities.

        Args:
            X: DataFrame with 'wavelengths' and 'intensities' columns

        Returns:
            DataFrame with raw intensity values from PeakRegions
        """
        # Handle numpy arrays from SHAP or other tools
        if isinstance(X, np.ndarray):
            logger.debug("Converting numpy array to DataFrame for raw spectral transformation")
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

        results = []

        for idx, row in X.iterrows():
            try:
                wavelengths = np.asarray(row["wavelengths"])
                intensities = np.asarray(row["intensities"])

                if wavelengths.size == 0 or intensities.size == 0:
                    logger.warning(
                        "Empty spectral data for sample %s, using NaN values", idx
                    )
                    raw_data = {name: np.nan for name in self.feature_names_out_}
                    results.append(raw_data)
                    continue

                # Use isolate_peaks to filter by PeakRegions (domain knowledge filtering)
                # Ensure intensities is 2D for the extractor (expects n_wavelengths x n_spectra)
                if intensities.ndim == 1:
                    intensities_2d = intensities.reshape(-1, 1)
                else:
                    intensities_2d = intensities

                # Process regions one by one to handle missing data gracefully
                raw_data = {}

                for region_config in self._regions:
                    try:
                        # Try to extract this specific region
                        region_results = self.extractor.isolate_peaks(
                            wavelengths,
                            intensities_2d,
                            _convert_to_results_regions([region_config]),
                            baseline_correction=self.config.baseline_correction,
                            area_normalization=False,  # Keep raw intensities
                        )

                        # Process the single region result
                        region_result = region_results[0]
                        element = region_result.region.element
                        region_wavelengths = region_result.wavelengths
                        region_spectra = region_result.isolated_spectra  # Shape: (n_wavelengths, n_shots)

                        # Average across shots (columns) to get single intensity per wavelength
                        if region_spectra.ndim == 2:
                            region_intensities = region_spectra.mean(axis=1)  # Average across shots
                        else:
                            region_intensities = region_spectra.flatten()

                        # Check for NaN values in the averaged intensities
                        if np.isnan(region_intensities).any():
                            nan_mask = np.isnan(region_intensities)
                            logger.debug("Found %d NaN values after averaging in %s region for sample %s",
                                       np.sum(nan_mask), element, idx)
                            region_intensities = np.nan_to_num(region_intensities, nan=0.0)

                        # Add each wavelength point as a separate feature
                        for wl, intensity in zip(region_wavelengths, region_intensities):
                            feature_name = f"{element}_wl_{wl:.2f}nm"
                            # Ensure no NaN values are stored
                            if np.isnan(intensity):
                                raw_data[feature_name] = 0.0
                            else:
                                raw_data[feature_name] = intensity

                    except ValueError as e:
                        # Region not found in spectrum - fill with zeros for this region
                        element = region_config.element
                        logger.debug("Region %s not found for sample %s: %s", element, idx, e)

                        # Fill expected features for this region with zeros
                        for feature_name in self.feature_names_out_:
                            if feature_name.startswith(f"{element}_wl_"):
                                raw_data[feature_name] = 0.0

                results.append(raw_data)

            except Exception as e:
                logger.warning("Error processing sample %s: %s, filling with zeros instead of NaN", idx, e)
                logger.warning("Failed to extract %d features for sample %s: %s",
                             len(self.feature_names_out_), idx, list(self.feature_names_out_)[:10])
                raw_data = {name: 0.0 for name in self.feature_names_out_}
                results.append(raw_data)

        # Create DataFrame with safe index handling
        index_to_use = X.index if hasattr(X, 'index') else None
        df_result = pd.DataFrame(results, index=index_to_use)

        # Ensure all expected columns are present (fill missing with zeros)
        missing_features = []
        for col in self.feature_names_out_:
            if col not in df_result.columns:
                df_result[col] = 0.0
                missing_features.append(col)

        if missing_features:
            logger.warning("Missing %d features during raw intensity extraction, filled with zeros: %s",
                         len(missing_features), missing_features[:10])

        # Reorder columns to match expected feature names
        df_result = df_result.reindex(
            columns=self.feature_names_out_, fill_value=0.0
        )

        # Final NaN check in raw spectral extraction
        if df_result.isnull().any().any():
            nan_count = df_result.isnull().sum().sum()
            nan_cols = df_result.columns[df_result.isnull().any()].tolist()

            # Group NaN columns by element for clearer reporting
            elements_with_nan = {}
            for col in nan_cols:
                element = col.split('_wl_')[0]
                if element not in elements_with_nan:
                    elements_with_nan[element] = []
                elements_with_nan[element].append(col)

            # Log a warning instead of error, as this is handled gracefully
            logger.warning("[RAW SPECTRAL] Found %d NaN values in %d features across %d elements",
                          nan_count, len(nan_cols), len(elements_with_nan))
            logger.debug("Elements with missing data: %s", list(elements_with_nan.keys()))

            # Fill NaN values with 0.0 (standard imputation for missing spectral regions)
            df_result = df_result.fillna(0.0)
            logger.info("Filled missing spectral regions with 0.0 (standard imputation)")

        logger.info(
            "Extracted %d raw intensity features from %d PeakRegions.",
            df_result.shape[1], len(self._regions)
        )
        return df_result

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features."""
        return self.feature_names_out_


class SpectralFeatureGenerator(BaseEstimator, TransformerMixin):
    """Extracts features from spectral data using the SpectralFeatureExtractor."""

    def __init__(self, config: Config, strategy: str = "simple_only"):
        self.config = config
        self.strategy = strategy
        self.extractor = SpectralFeatureExtractor(
            enable_preprocessing=config.use_spectral_preprocessing,
            preprocessing_method=config.spectral_preprocessing_method
        )
        self.feature_names_out_: List[str] = []
        self._all_simple_names = []
        self._high_k_names = []

        # Strategy-optimized regions to avoid extracting unused features
        self._regions = self._get_strategy_regions()

        # Initialize enhanced features if any are enabled
        self._use_enhanced = any(
            [
                config.enable_molecular_bands,
                config.enable_advanced_ratios,
                config.enable_spectral_patterns,
                config.enable_interference_correction,
                config.enable_plasma_indicators,
            ]
        )
        if self._use_enhanced:
            self.enhanced_features = EnhancedSpectralFeatures(config)

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
        self._set_feature_names(
            X.iloc[0:1]
        )  # Use a sample row to determine dynamic feature names
        logger.info("SpectralFeatureGenerator fitted for strategy '%s'.", self.strategy)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw spectral data into the final feature matrix."""
        # Handle numpy arrays from SHAP or other tools
        if isinstance(X, np.ndarray):
            logger.debug("Converting numpy array to DataFrame for feature transformation")
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

        base_features_list = [
            self._extract_base_features(row) for _, row in X.iterrows()
        ]
        # Create DataFrame with safe index handling
        index_to_use = X.index if hasattr(X, 'index') else None
        base_features_df = pd.DataFrame(base_features_list, index=index_to_use)

        # Extract K_I and C_I areas for ratio calculation
        if "K_I_peak_0" not in base_features_df:
            raise ValueError("K_I_peak_0 not found in base features. Ensure potassium (K_I) region is properly defined and extracted.")
        if "C_I_peak_0" not in base_features_df:
            raise ValueError("C_I_peak_0 not found in base features. Ensure carbon (C_I) region is properly defined and extracted.")
        
        k_area = base_features_df["K_I_peak_0"].fillna(0.0)
        c_area = base_features_df["C_I_peak_0"].fillna(1e-6)

        # Calculate KC_ratio with clipping to prevent extreme values
        kc_ratio_raw = k_area / c_area.replace(0, 1e-6)
        base_features_df["K_C_ratio"] = np.clip(kc_ratio_raw, -50.0, 50.0)

        # Choose feature generation method based on config
        if self.config.use_focused_potassium_features:
            full_features_df, _ = generate_focused_potassium_features(
                base_features_df, self._all_simple_names
            )
        else:
            full_features_df, _ = generate_high_potassium_features(
                base_features_df, self._all_simple_names
            )

        # Check for duplicate column names before reindexing
        expected_features = self.get_feature_names_out()
        if len(expected_features) != len(set(expected_features)):
            from collections import Counter

            feature_counts = Counter(expected_features)
            duplicates = {
                name: count for name, count in feature_counts.items() if count > 1
            }
            logger.error("Found duplicate feature names: %s", duplicates)
            logger.error(
                "Total features: %d, Unique: %d", len(expected_features), len(set(expected_features))
            )

            # Debug: Show where duplicates come from
            if self.strategy == "full_context":
                logger.error(
                    "Complex features count: %d",
                    len([n for n in expected_features if 'peak_' in n and 'simple' not in n])
                )
                logger.error("Simple features count: %d", len(self._all_simple_names))
                logger.error("High K features count: %d", len(self._high_k_names))
                logger.error(
                    "Enhanced features count: %d",
                    len([n for n in expected_features if any(x in n for x in ['ratio', 'fwhm', 'asymmetry'])])
                )

            raise ValueError(
                f"Cannot proceed with duplicate feature names: {list(duplicates.keys())[:10]}"
            )

        # Check for missing features before reindexing
        missing_features = [f for f in expected_features if f not in full_features_df.columns]
        if missing_features:
            logger.warning("Missing %d features during final reindexing, filling with zeros: %s",
                         len(missing_features), missing_features[:10])

        final_df = full_features_df.reindex(
            columns=expected_features, fill_value=0.0
        )

        # Ensure the final dataframe maintains the original index
        if hasattr(X, 'index'):
            final_df.index = X.index

        # Final NaN check and cleanup
        if final_df.isnull().any().any():
            nan_count = final_df.isnull().sum().sum()
            nan_cols = final_df.columns[final_df.isnull().any()].tolist()
            logger.warning(f"[FEATURE ENGINEERING] Found {nan_count} NaN values in columns: {nan_cols[:10]}...")
            logger.warning("Filling NaN values with 0.0 to prevent downstream issues")
            final_df = final_df.fillna(0.0)

        logger.info(
            "Transformed data into %d features for strategy '%s'.",
            final_df.shape[1], self.strategy
        )
        return final_df

    def _extract_base_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extracts simple and complex features for a single sample."""
        features = {}
        wavelengths = np.asarray(row["wavelengths"])
        intensities = np.asarray(row["intensities"])

        if intensities.size == 0:
            # Return empty dict for empty intensities
            sample_id = row.get(self.config.sample_id_column, "unknown")
            logger.warning(
                "Empty intensities found for sample %s, cannot extract any features", sample_id
            )
            logger.warning("Expected to extract features for %d regions: %s",
                         len(self._regions),
                         [r.element for r in self._regions])
            return {}

        sample_id = row.get(self.config.sample_id_column, "unknown")
        for region in self._regions:
            try:
                helper_region = HelperPeakRegion(
                    region.element,
                    region.lower_wavelength,
                    region.upper_wavelength,
                    region.center_wavelengths,
                )
                region_features = extract_full_simple_features(helper_region, wavelengths, intensities)
                features.update(region_features)

                if not region_features:
                    logger.warning("No features extracted for region %s (%.1f-%.1f nm) in sample %s",
                                 region.element, region.lower_wavelength, region.upper_wavelength, sample_id)

            except Exception as e:
                logger.error("Failed to extract features for region %s (%.1f-%.1f nm) in sample %s: %s",
                           region.element, region.lower_wavelength, region.upper_wavelength, sample_id, e)
                # Continue processing other regions

        # Ensure intensities is 2D for the extractor (expects n_wavelengths x n_spectra)
        spectra_2d = (
            intensities.reshape(-1, 1) if intensities.ndim == 1 else intensities
        )

        try:
            fit_results = self.extractor.extract_features(
                wavelengths=wavelengths,
                spectra=spectra_2d,
                regions=_convert_to_results_regions(self._regions),
                peak_shapes=self.config.peak_shapes * len(self._regions),
            )

            complex_features_extracted = 0
            for res in fit_results.fitting_results:
                element = res.region_result.region.element
                element_features_count = 0

                # Extract peak areas (original feature)
                for i, area in enumerate(res.peak_areas):
                    features[f"{element}_peak_{i}"] = area
                    element_features_count += 1
                    complex_features_extracted += 1

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

                if element_features_count == 0:
                    logger.warning("No complex peak features extracted for element %s in sample %s",
                                 element, sample_id)

            if complex_features_extracted == 0:
                logger.warning("No complex peak features extracted for any element in sample %s", sample_id)

        except Exception as e:
            logger.error("Failed to extract complex features for sample %s: %s", sample_id, e)

        # Add enhanced features if enabled
        if self._use_enhanced:
            try:
                enhanced_feats = self.enhanced_features.transform(
                    features, wavelengths, intensities
                )

                if enhanced_feats:
                    features.update(enhanced_feats)
                    logger.debug("Extracted %d enhanced features for sample %s",
                               len(enhanced_feats), sample_id)
                else:
                    logger.warning("No enhanced features extracted for sample %s", sample_id)

            except Exception as e:
                logger.error("Failed to extract enhanced features for sample %s: %s", sample_id, e)

        # Log summary of extracted features
        if features:
            logger.debug("Successfully extracted %d features for sample %s", len(features), sample_id)
        else:
            logger.warning("No features extracted for sample %s", sample_id)

        return features

    def _set_feature_names(self, X_sample):
        """
        Defines the canonical list of feature names for each strategy,
        matching the original script's logic exactly.
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

        # This list of 48 features is now correct
        self._all_simple_names = []
        for region in self._regions:
            prefix = f"{region.element}_simple"
            self._all_simple_names.extend(
                [
                    f"{prefix}_peak_area",
                    f"{prefix}_peak_height",
                    f"{prefix}_peak_center_intensity",
                    f"{prefix}_baseline_avg",
                    f"{prefix}_signal_range",
                    f"{prefix}_total_intensity",
                    f"{prefix}_height_to_baseline",
                    f"{prefix}_normalized_area",
                ]
            )

        # Dynamically determine the high_n_names by running a sample
        # This makes it robust to changes in the helper function
        # Temporarily disable enhanced features to avoid recursion
        use_enhanced_temp = self._use_enhanced
        self._use_enhanced = False
        sample_base_df = pd.DataFrame([self._extract_base_features(X_sample.iloc[0])])
        self._use_enhanced = use_enhanced_temp

        sample_base_df["K_C_ratio"] = 0.0
        # Choose feature generation method based on config for feature name determination
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
            # Use the sample features we already extracted
            sample_features = sample_base_df.iloc[0].to_dict()
            sample_wavelengths = np.asarray(X_sample.iloc[0]["wavelengths"])
            sample_intensities = np.asarray(X_sample.iloc[0]["intensities"])
            enhanced_sample = self.enhanced_features.transform(
                sample_features, sample_wavelengths, sample_intensities
            )
            enhanced_names = list(enhanced_sample.keys())

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

        # Debug: Check for duplicates at creation time
        from collections import Counter

        counts = Counter(self.feature_names_out_)
        duplicates = {name: count for name, count in counts.items() if count > 1}
        if duplicates:
            logger.error("Duplicates found during feature name generation:")
            logger.error("  Duplicates: %s", duplicates)
            logger.error("  Strategy: %s", self.strategy)
            logger.error("  Complex names count: %d", len(all_complex_names))
            logger.error("  Simple names count: %d", len(self._all_simple_names))
            logger.error("  High K names: %s", self._high_k_names)
            logger.error("  Enhanced names count: %d", len(enhanced_names))
            raise ValueError(
                f"Duplicate feature names generated: {list(duplicates.keys())}"
            )

    def _extract_features_for_sample(
        self, wavelengths: np.ndarray, intensities: np.ndarray
    ) -> Dict[str, Any]:
        """Core feature extraction logic for a single sample."""
        features = {}

        # Ensure intensities is 2D for the extractor (expects n_wavelengths x n_spectra)
        spectra_2d = (
            intensities.reshape(-1, 1) if intensities.ndim == 1 else intensities
        )

        fit_results = self.extractor.extract_features(
            wavelengths=wavelengths,
            spectra=spectra_2d,
            regions=_convert_to_results_regions(self._regions),
            peak_shapes=self.config.peak_shapes * len(self._regions),
            fitting_mode=self.config.fitting_mode,
            baseline_correction=self.config.baseline_correction,
        )
        element_to_areas = {
            res.region_result.region.element: res.peak_areas
            for res in fit_results.fitting_results
        }

        for region in self._regions:
            areas = element_to_areas.get(region.element, [np.nan] * region.n_peaks)
            for i, area in enumerate(areas):
                features[f"{region.element}_peak_{i}"] = area
            features.update(
                self._extract_simple_region_features(region, wavelengths, intensities)
            )

        k_area, c_area = (
            features.get("K_I_peak_0", np.nan),
            features.get("C_I_peak_0", np.nan),
        )
        kc_ratio = (
            k_area / c_area
            if not np.isnan(k_area) and not np.isnan(c_area) and c_area > 1e-6
            else np.nan
        )

        # Clip KC_ratio to reasonable bounds to prevent extreme values in derived features
        if not np.isnan(kc_ratio):
            kc_ratio = np.clip(kc_ratio, -50.0, 50.0)

        features.update(
            {
                "K_C_ratio": kc_ratio,
                "KC_ratio_squared": kc_ratio**2 if not np.isnan(kc_ratio) else np.nan,
                "KC_ratio_cubic": kc_ratio**3 if not np.isnan(kc_ratio) else np.nan,
                "KC_ratio_log": np.log1p(np.abs(kc_ratio))
                if not np.isnan(kc_ratio)
                else np.nan,
            }
        )
        return features

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

    def _get_empty_features(self) -> Dict[str, Any]:
        return {name: 0.0 for name in self.get_feature_names_out()}

    def _extract_simple_region_features(
        self, region: PeakRegion, wavelengths: np.ndarray, intensities: np.ndarray
    ) -> Dict[str, float]:
        """Simplified simple feature extractor."""
        prefix = f"{region.element}_simple"

        # Ensure arrays are numpy arrays
        wavelengths = np.asarray(wavelengths)
        intensities = np.asarray(intensities)

        mask = (wavelengths >= region.lower_wavelength) & (
            wavelengths <= region.upper_wavelength
        )

        # Handle both 1D (single sample) and 2D (multiple samples) arrays
        if intensities.ndim == 1:
            if not np.any(mask) or intensities.size == 0:
                return {
                    f"{prefix}_peak_area": np.nan,
                    f"{prefix}_peak_height": np.nan,
                    f"{prefix}_height_to_baseline": np.nan,
                }
            avg_spectrum = intensities[mask]
        else:
            if not np.any(mask) or intensities.shape[1] == 0:
                return {
                    f"{prefix}_peak_area": np.nan,
                    f"{prefix}_peak_height": np.nan,
                    f"{prefix}_height_to_baseline": np.nan,
                }
            avg_spectrum = np.mean(intensities[mask, :], axis=1)

        peak_height, baseline_avg = (
            np.max(avg_spectrum),
            (avg_spectrum[0] + avg_spectrum[-1]) / 2,
        )
        return {
            f"{prefix}_peak_area": float(np.trapezoid(avg_spectrum, wavelengths[mask])),
            f"{prefix}_peak_height": float(peak_height),
            f"{prefix}_height_to_baseline": float(peak_height - baseline_avg),
        }


def create_feature_pipeline(
    config: Config, strategy: str, exclude_scaler: bool = False, use_parallel: bool = False, n_jobs: int = -1
) -> Pipeline:
    """
    Creates a scikit-learn pipeline for the entire feature processing.

    Based on config.use_raw_spectral_data, returns either:
    - Raw spectral pipeline: PeakRegion filtering only (no feature engineering)
    - Feature engineering pipeline: Full mathematical feature extraction
    
    Args:
        config: Pipeline configuration
        strategy: Feature strategy ('K_only', 'simple_only', 'full_context')
        exclude_scaler: Whether to exclude StandardScaler from pipeline
        use_parallel: Whether to use parallel processing for feature generation
        n_jobs: Number of parallel jobs (-1 for all cores, -2 for all but one)
    """
    if config.use_raw_spectral_data:
        # Raw spectral data pipeline - domain knowledge filtering + minimal concentration features
        from src.features.concentration_features import MinimalConcentrationFeatures

        steps = [
            ("raw_spectral", RawSpectralTransformer(config=config)),
            ("minimal_concentration", MinimalConcentrationFeatures()),
        ]

        # Add StandardScaler unless explicitly excluded (only step needed for raw data)
        if not exclude_scaler:
            steps.append(("scaler", PandasStandardScaler()))

        pipeline = Pipeline(steps)

        scaler_info = (
            "with StandardScaler" if not exclude_scaler else "without StandardScaler"
        )
        logger.info(
            "Created RAW SPECTRAL pipeline with minimal concentration features %s.",
            scaler_info
        )
        logger.info(
            "Raw mode: ~%d PeakRegions -> 63 raw intensity features -> DIRECT to model",
            len(config.all_regions)
        )
        return pipeline
    else:
        # Traditional feature engineering pipeline
        if use_parallel:
            from src.features.parallel_feature_engineering import ParallelSpectralFeatureGenerator
            feature_generator = ParallelSpectralFeatureGenerator(
                config=config, strategy=strategy, n_jobs=n_jobs
            )
            logger.info(f"Using PARALLEL feature generation with {n_jobs} jobs")
        else:
            feature_generator = SpectralFeatureGenerator(config=config, strategy=strategy)
            
        steps = [
            ("spectral_features", feature_generator),
            ("imputer", SimpleImputer(strategy="mean")),
            ("clipper", OutlierClipper()),
        ]

        # Add StandardScaler unless explicitly excluded
        if not exclude_scaler:
            steps.append(("scaler", PandasStandardScaler()))

        pipeline = Pipeline(steps)

        scaler_info = (
            "without StandardScaler" if exclude_scaler else "with StandardScaler"
        )
        parallel_info = " (PARALLEL)" if use_parallel else ""
        logger.info(
            "Created FEATURE ENGINEERING pipeline%s for strategy: '%s' %s.",
            parallel_info, strategy, scaler_info
        )
        return pipeline
