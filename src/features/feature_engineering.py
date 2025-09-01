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
    generate_high_magnesium_features,
    generate_focused_magnesium_features,
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
        if self.feature_names_in_ is not None:
            return self.feature_names_in_
        elif input_features is not None:
            return list(input_features)
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
        self.extractor = SpectralFeatureExtractor()
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
                _convert_to_results_regions(self.config.all_regions),
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
            for region in self.config.all_regions:
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

                region_results = self.extractor.isolate_peaks(
                    wavelengths,
                    intensities_2d,
                    _convert_to_results_regions(self.config.all_regions),
                    baseline_correction=self.config.baseline_correction,
                    area_normalization=False,  # Keep raw intensities
                )

                # Extract raw intensities (no mathematical feature engineering)
                raw_data = {}

                for region_result in region_results:
                    element = region_result.region.element
                    region_wavelengths = region_result.wavelengths
                    region_spectra = (
                        region_result.isolated_spectra
                    )  # Shape: (n_wavelengths, n_shots)

                    # Average across shots (columns) to get single intensity per wavelength
                    if region_spectra.ndim == 2:
                        region_intensities = region_spectra.mean(
                            axis=1
                        )  # Average across shots
                    else:
                        region_intensities = region_spectra.flatten()

                    # Add each wavelength point as a separate feature
                    for i, (wl, intensity) in enumerate(
                        zip(region_wavelengths, region_intensities)
                    ):
                        feature_name = f"{element}_wl_{wl:.2f}nm"
                        raw_data[feature_name] = intensity

                results.append(raw_data)

            except Exception as e:
                logger.warning("Error processing sample %s: %s, using NaN values", idx, e)
                raw_data = {name: np.nan for name in self.feature_names_out_}
                results.append(raw_data)

        df_result = pd.DataFrame(results, index=X.index)

        # Ensure all expected columns are present (fill missing with NaN)
        for col in self.feature_names_out_:
            if col not in df_result.columns:
                df_result[col] = np.nan

        # Reorder columns to match expected feature names
        df_result = df_result.reindex(
            columns=self.feature_names_out_, fill_value=np.nan
        )

        logger.info(
            "Extracted %d raw intensity features from %d PeakRegions.",
            df_result.shape[1], len(self.config.all_regions)
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
        self.extractor = SpectralFeatureExtractor()
        self.feature_names_out_: List[str] = []
        self._all_simple_names = []
        self._high_p_names = []
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

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits the transformer by determining the canonical feature names."""
        self._set_feature_names(
            X.iloc[0:1]
        )  # Use a sample row to determine dynamic feature names
        logger.info("SpectralFeatureGenerator fitted for strategy '%s'.", self.strategy)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw spectral data into the final feature matrix."""
        base_features_list = [
            self._extract_base_features(row) for _, row in X.iterrows()
        ]
        base_features_df = pd.DataFrame(base_features_list, index=X.index)

        p_area = (
            base_features_df["M_I_peak_0"]
            if "M_I_peak_0" in base_features_df
            else pd.Series(np.nan, index=base_features_df.index)
        )
        c_area = (
            base_features_df["C_I_peak_0"]
            if "C_I_peak_0" in base_features_df
            else pd.Series(np.nan, index=base_features_df.index)
        )

        # Calculate PC_ratio with clipping to prevent extreme values
        mc_ratio_raw = p_area / c_area.replace(0, 1e-6)
        base_features_df["M_C_ratio"] = np.clip(mc_ratio_raw, -50.0, 50.0)

        # Choose feature generation method based on config
        if self.config.use_focused_magnesium_features:
            full_features_df, _ = generate_focused_magnesium_features(
                base_features_df, self._all_simple_names
            )
        else:
            full_features_df, _ = generate_high_magnesium_features(
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
                logger.error("High P features count: %d", len(self._high_p_names))
                logger.error(
                    "Enhanced features count: %d",
                    len([n for n in expected_features if any(x in n for x in ['ratio', 'fwhm', 'asymmetry'])])
                )

            raise ValueError(
                f"Cannot proceed with duplicate feature names: {list(duplicates.keys())[:10]}"
            )

        final_df = full_features_df.reindex(
            columns=expected_features, fill_value=np.nan
        )

        # Ensure the final dataframe maintains the original index
        final_df.index = X.index

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
            # Return NaN for all expected features instead of empty dict
            logger.warning(
                "Empty intensities found for sample, returning NaN features"
            )
            return {}

        for region in self.config.all_regions:
            helper_region = HelperPeakRegion(
                region.element,
                region.lower_wavelength,
                region.upper_wavelength,
                region.center_wavelengths,
            )
            features.update(
                extract_full_simple_features(helper_region, wavelengths, intensities)
            )

        # Ensure intensities is 2D for the extractor (expects n_wavelengths x n_spectra)
        spectra_2d = (
            intensities.reshape(-1, 1) if intensities.ndim == 1 else intensities
        )

        fit_results = self.extractor.extract_features(
            wavelengths=wavelengths,
            spectra=spectra_2d,
            regions=_convert_to_results_regions(self.config.all_regions),
            peak_shapes=self.config.peak_shapes * len(self.config.all_regions),
        )
        for res in fit_results.fitting_results:
            element = res.region_result.region.element
            for i, area in enumerate(res.peak_areas):
                features[f"{element}_peak_{i}"] = area

        # Add enhanced features if enabled
        if self._use_enhanced:
            enhanced_feats = self.enhanced_features.transform(
                features, wavelengths, intensities
            )
            features.update(enhanced_feats)

        return features

    def _set_feature_names(self, X_sample):
        """
        Defines the canonical list of feature names for each strategy,
        matching the original script's logic exactly.
        """
        # Define all possible base feature names
        all_complex_names = []
        for region in self.config.all_regions:
            for i in range(region.n_peaks):
                all_complex_names.append(f"{region.element}_peak_{i}")

        # This list of 48 features is now correct
        self._all_simple_names = []
        for region in self.config.all_regions:
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

        sample_base_df["M_C_ratio"] = 0.0
        # Choose feature generation method based on config for feature name determination
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
            # Use the sample features we already extracted
            sample_features = sample_base_df.iloc[0].to_dict()
            sample_wavelengths = np.asarray(X_sample.iloc[0]["wavelengths"])
            sample_intensities = np.asarray(X_sample.iloc[0]["intensities"])
            enhanced_sample = self.enhanced_features.transform(
                sample_features, sample_wavelengths, sample_intensities
            )
            enhanced_names = list(enhanced_sample.keys())

        if self.strategy == "Mg_only":
            m_complex = [name for name in all_complex_names if name.startswith("M_I_")]
            m_simple = [
                name for name in self._all_simple_names if name.startswith("M_I_simple")
            ]
            m_enhanced = [
                name
                for name in enhanced_names
                if "Mg" in name or "magnesium" in name.lower()
            ]
            self.feature_names_out_ = m_complex + m_simple + m_enhanced
        elif self.strategy == "simple_only":
            self.feature_names_out_ = (
                self._all_simple_names
                + ["M_C_ratio"]
                + self._high_p_names
                + enhanced_names
            )
        elif self.strategy == "full_context":
            self.feature_names_out_ = (
                all_complex_names
                + self._all_simple_names
                + ["M_C_ratio"]
                + self._high_p_names
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
            logger.error("  High P names: %s", self._high_p_names)
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
            regions=_convert_to_results_regions(self.config.all_regions),
            peak_shapes=self.config.peak_shapes * len(self.config.all_regions),
            fitting_mode=self.config.fitting_mode,
            baseline_correction=self.config.baseline_correction,
        )
        element_to_areas = {
            res.region_result.region.element: res.peak_areas
            for res in fit_results.fitting_results
        }

        for region in self.config.all_regions:
            areas = element_to_areas.get(region.element, [np.nan] * region.n_peaks)
            for i, area in enumerate(areas):
                features[f"{region.element}_peak_{i}"] = area
            features.update(
                self._extract_simple_region_features(region, wavelengths, intensities)
            )

        m_area, c_area = (
            features.get("M_I_peak_0", np.nan),
            features.get("C_I_peak_0", np.nan),
        )
        mc_ratio = (
            m_area / c_area
            if not np.isnan(m_area) and not np.isnan(c_area) and c_area > 1e-6
            else np.nan
        )

        # Clip PC_ratio to reasonable bounds to prevent extreme values in derived features
        if not np.isnan(mc_ratio):
            mc_ratio = np.clip(mc_ratio, -50.0, 50.0)

        features.update(
            {
                "M_C_ratio": mc_ratio,
                "MC_ratio_squared": mc_ratio**2 if not np.isnan(mc_ratio) else np.nan,
                "MC_ratio_cubic": mc_ratio**3 if not np.isnan(mc_ratio) else np.nan,
                "MC_ratio_log": np.log1p(np.abs(mc_ratio))
                if not np.isnan(mc_ratio)
                else np.nan,
            }
        )
        return features

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

    def _get_empty_features(self) -> Dict[str, Any]:
        return {name: np.nan for name in self.get_feature_names_out()}

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
        strategy: Feature strategy ('Mg_only', 'simple_only', 'full_context')
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
