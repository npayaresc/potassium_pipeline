"""
Mislabel Detection Module

This module implements clustering-based approaches to identify potentially mislabeled samples
in the spectral dataset, with a focus on samples in the lower concentration range.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

from src.config.pipeline_config import Config
from src.features.feature_engineering import create_feature_pipeline
from src.features.concentration_features import create_enhanced_feature_pipeline_with_concentration

logger = logging.getLogger(__name__)

class MislabelDetector:
    """
    Detects potentially mislabeled samples using clustering and outlier detection methods.

    The approach combines:
    1. Multiple clustering algorithms (K-means, DBSCAN, Hierarchical)
    2. Both raw spectral data and engineered features
    3. Outlier detection within clusters
    4. Focus on lower concentration range where mislabeling is more common
    """

    def __init__(self, config: Config, focus_range: Tuple[float, float] = (0.0, 0.5),
                 use_feature_parallel: bool = False, feature_n_jobs: int = 1,
                 use_data_parallel: bool = False, data_n_jobs: int = 1,
                 strategy_override: Optional[str] = None):
        """
        Initialize the mislabel detector.

        Args:
            config: Pipeline configuration
            focus_range: Concentration range to focus on for mislabel detection (min, max)
            use_feature_parallel: Enable parallel processing for feature engineering
            feature_n_jobs: Number of parallel jobs for feature processing
            use_data_parallel: Enable parallel processing for data operations
            data_n_jobs: Number of parallel jobs for data operations
            strategy_override: Override strategy from config (e.g., 'K_only', 'simple_only', 'full_context')
        """
        self.config = config
        self.focus_range = focus_range
        self.results = {}
        self.scaler = StandardScaler()
        self.strategy_override = strategy_override

        # Parallel processing configuration
        self.use_feature_parallel = use_feature_parallel
        self.feature_n_jobs = feature_n_jobs if feature_n_jobs > 0 else mp.cpu_count()
        self.use_data_parallel = use_data_parallel
        self.data_n_jobs = data_n_jobs if data_n_jobs > 0 else mp.cpu_count()

        logger.info(f"MislabelDetector initialized for range {focus_range[0]:.2f}-{focus_range[1]:.2f}")
        if self.use_feature_parallel:
            logger.info(f"Feature parallel processing enabled: {self.feature_n_jobs} jobs")
        if self.use_data_parallel:
            logger.info(f"Data parallel processing enabled: {self.data_n_jobs} jobs")

    def detect_mislabels(self,
                        dataset: pd.DataFrame,
                        use_features: bool = True,
                        clustering_methods: List[str] = ['kmeans', 'dbscan', 'hierarchical'],
                        n_clusters_range: Tuple[int, int] = (3, 8),
                        outlier_methods: List[str] = ['lof', 'isolation_forest'],
                        min_cluster_size: int = 5,
                        save_results: bool = True,
                        use_raw_spectral_override: Optional[bool] = None) -> Dict:
        """
        Main method to detect potentially mislabeled samples.

        Args:
            dataset: DataFrame with spectral data and target values
            use_features: Whether to use engineered features for clustering
            clustering_methods: List of clustering methods to use
            n_clusters_range: Range of cluster numbers to test for K-means
            outlier_methods: List of outlier detection methods
            min_cluster_size: Minimum cluster size for analysis
            save_results: Whether to save results and configuration to disk

        Returns:
            Dictionary with detection results and recommendations

        Note:
            Raw spectral analysis is automatically enabled/disabled based on
            config.use_raw_spectral_data setting, following pipeline conventions.
        """
        # Generate timestamp for this run
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine raw spectral usage (override takes precedence)
        if use_raw_spectral_override is not None:
            use_raw_spectral = use_raw_spectral_override
        else:
            use_raw_spectral = self.config.use_raw_spectral_data

        logger.info("Starting mislabel detection analysis...")
        if use_raw_spectral_override is not None:
            source = "command line override"
        else:
            source = "config.use_raw_spectral_data"
        logger.info(f"Raw spectral analysis: {'enabled' if use_raw_spectral else 'disabled'} (from {source})")

        # Filter dataset to focus range if specified
        if self.focus_range[1] < np.inf:
            focus_data = dataset[
                (dataset[self.config.target_column] >= self.focus_range[0]) &
                (dataset[self.config.target_column] <= self.focus_range[1])
            ].copy()
            logger.info(f"Focusing on {len(focus_data)} samples in range {self.focus_range[0]:.2f}-{self.focus_range[1]:.2f}")
        else:
            focus_data = dataset.copy()
            logger.info(f"Analyzing all {len(focus_data)} samples")

        if len(focus_data) < 10:
            logger.warning("Too few samples for meaningful clustering analysis")
            return {"error": "Insufficient samples"}

        results = {
            "run_info": {
                "timestamp": run_timestamp,
                "focus_range": self.focus_range,
                "use_features": use_features,
                "use_raw_spectral": use_raw_spectral,
                "raw_spectral_source": "config.use_raw_spectral_data",
                "clustering_methods": clustering_methods,
                "n_clusters_range": n_clusters_range,
                "outlier_methods": outlier_methods,
                "min_cluster_size": min_cluster_size,
                "use_feature_parallel": self.use_feature_parallel,
                "feature_n_jobs": self.feature_n_jobs,
                "use_data_parallel": self.use_data_parallel,
                "data_n_jobs": self.data_n_jobs
            },
            "dataset_info": {
                "total_samples": len(dataset),
                "focus_samples": len(focus_data),
                "focus_range": self.focus_range,
                "target_stats": focus_data[self.config.target_column].describe().to_dict()
            },
            "suspicious_samples": {},
            "cluster_analysis": {},
            "outlier_analysis": {},
            "recommendations": []
        }

        # 1. Feature-based clustering analysis (only if not using raw spectral data)
        if use_features and not use_raw_spectral:
            logger.info("Performing feature-based clustering analysis...")
            feature_results = self._analyze_with_features(focus_data, clustering_methods,
                                                        n_clusters_range, outlier_methods, min_cluster_size)
            results["cluster_analysis"]["features"] = feature_results

        # 2. Raw spectral clustering analysis (when raw spectral data is enabled)
        if use_raw_spectral:
            logger.info("Performing raw spectral clustering analysis...")
            spectral_results = self._analyze_with_raw_spectral(focus_data, clustering_methods,
                                                             n_clusters_range, outlier_methods, min_cluster_size)
            results["cluster_analysis"]["raw_spectral"] = spectral_results

        # Log the analysis mode being used
        if use_raw_spectral:
            logger.info("Analysis mode: Raw spectral data only (no feature engineering)")
        elif use_features:
            strategy = self.config.feature_strategies[0] if self.config.feature_strategies else "simple_only"
            logger.info(f"Analysis mode: Engineered features using strategy '{strategy}'")
        else:
            logger.warning("No analysis performed - both raw spectral and features are disabled")

        # 3. Combine results and identify consensus suspects
        results["suspicious_samples"] = self._combine_results(results["cluster_analysis"])

        # 4. Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        # Store results for visualization
        self.results = results
        self.focus_data = focus_data

        # 5. Save results and configuration if requested
        if save_results:
            self._save_results_and_config(results, run_timestamp)

        logger.info(f"Mislabel detection complete. Found {len(results['suspicious_samples'])} suspicious samples.")
        return results

    def _analyze_with_features(self, data: pd.DataFrame, clustering_methods: List[str],
                             n_clusters_range: Tuple[int, int], outlier_methods: List[str],
                             min_cluster_size: int) -> Dict:
        """Analyze using engineered features with strategy from config."""
        try:
            # Use strategy override if provided, otherwise get from config
            if self.strategy_override:
                strategy = self.strategy_override
                logger.info(f"Using strategy override from command line: {strategy}")
            else:
                strategy = self.config.feature_strategies[0] if self.config.feature_strategies else "simple_only"
                logger.info(f"Using feature strategy from config: {strategy}")

            # Create feature pipeline - use parallel processing if enabled
            if self.use_feature_parallel:
                logger.info(f"Using parallel feature engineering with {self.feature_n_jobs} jobs")
                from src.features.parallel_feature_engineering import ParallelSpectralFeatureGenerator

                # Use the parallel feature engineering system with config strategy
                parallel_generator = ParallelSpectralFeatureGenerator(
                    config=self.config,
                    strategy=strategy,
                    n_jobs=self.feature_n_jobs
                )
                X_features = parallel_generator.fit_transform(data)
            else:
                # Use standard feature pipeline with config strategy
                feature_pipeline = create_enhanced_feature_pipeline_with_concentration(
                    self.config, strategy=strategy
                )
                X_features = feature_pipeline.fit_transform(data)

            X_scaled = self.scaler.fit_transform(X_features)

            logger.info(f"Feature-based analysis ({strategy}): {X_features.shape[1]} features, {X_features.shape[0]} samples")

            # Perform clustering analysis
            return self._perform_clustering_analysis(
                X_scaled, data, "features", clustering_methods,
                n_clusters_range, outlier_methods, min_cluster_size
            )

        except Exception as e:
            logger.error(f"Feature-based analysis failed: {e}")
            return {"error": str(e)}

    def _analyze_with_raw_spectral(self, data: pd.DataFrame, clustering_methods: List[str],
                                 n_clusters_range: Tuple[int, int], outlier_methods: List[str],
                                 min_cluster_size: int) -> Dict:
        """Analyze using raw spectral data via RawSpectralTransformer (same as training pipeline)."""
        try:
            logger.info("Using raw spectral data mode via RawSpectralTransformer pipeline")

            # Use the same approach as training pipeline when use_raw_spectral_data=True
            # Create raw spectral pipeline with only RawSpectralTransformer (no concentration features for clustering)
            from src.features.feature_engineering import RawSpectralTransformer
            from sklearn.pipeline import Pipeline

            # Create raw spectral pipeline (only raw spectral transformer, no concentration features)
            # For clustering analysis, we don't need concentration-dependent features
            raw_spectral_transformer = RawSpectralTransformer(config=self.config)
            feature_pipeline = Pipeline([
                ("raw_spectral", raw_spectral_transformer)
            ])

            logger.info(f"Created raw spectral pipeline steps: {[step[0] for step in feature_pipeline.steps]}")

            # Transform the data using the raw spectral pipeline (no y needed)
            X_spectral = feature_pipeline.fit_transform(data)

            logger.info(f"Raw spectral pipeline output shape: {X_spectral.shape}")

            # Handle any NaN values that might exist
            if hasattr(X_spectral, 'isna') and X_spectral.isna().any().any():
                logger.warning("Found NaN values in raw spectral features, filling with median values")
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X_spectral_array = imputer.fit_transform(X_spectral)
            else:
                X_spectral_array = X_spectral.values if hasattr(X_spectral, 'values') else X_spectral

            # Scale the raw spectral data for clustering analysis
            X_scaled = self.scaler.fit_transform(X_spectral_array)

            logger.info(f"Raw spectral analysis: {X_spectral_array.shape[1]} spectral features, {X_spectral_array.shape[0]} samples")

            # Perform clustering analysis
            return self._perform_clustering_analysis(
                X_scaled, data, "raw_spectral", clustering_methods,
                n_clusters_range, outlier_methods, min_cluster_size
            )

        except Exception as e:
            logger.error(f"Raw spectral analysis failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}

    def _extract_spectral_features_sequential(self, data: pd.DataFrame) -> List[List[float]]:
        """Extract spectral features sequentially."""
        spectral_data = []
        for _, row in data.iterrows():
            features = self._extract_single_sample_features(row)
            spectral_data.append(features)
        return spectral_data

    def _extract_spectral_features_parallel(self, data: pd.DataFrame) -> List[List[float]]:
        """Extract spectral features in parallel."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Split data into chunks for parallel processing
        chunk_size = max(1, len(data) // self.data_n_jobs)
        data_chunks = [data.iloc[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        spectral_data = []
        with ProcessPoolExecutor(max_workers=self.data_n_jobs) as executor:
            # Submit chunks for processing
            futures = [executor.submit(self._process_data_chunk, chunk) for chunk in data_chunks]

            # Collect results
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    spectral_data.extend(chunk_results)
                except Exception as e:
                    logger.warning(f"Parallel spectral extraction chunk failed: {e}")
                    # Fall back to sequential processing for this chunk
                    continue

        return spectral_data

    @staticmethod
    def _process_data_chunk(data_chunk: pd.DataFrame) -> List[List[float]]:
        """Process a chunk of data for parallel spectral feature extraction."""
        chunk_results = []
        for _, row in data_chunk.iterrows():
            features = MislabelDetector._extract_single_sample_features(row)
            chunk_results.append(features)
        return chunk_results

    @staticmethod
    def _extract_single_sample_features(row: pd.Series) -> List[float]:
        """Extract spectral features from a single sample."""
        wavelengths = row['wavelengths']
        intensities = row['intensities']

        # Focus on potassium-relevant regions
        k_regions = [
            (766, 770),  # K I line
            (404, 405),  # K I line
            (769.9, 770.1)  # Narrow K line
        ]

        features = []
        for wl_min, wl_max in k_regions:
            mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
            if np.any(mask):
                region_intensities = intensities[mask]
                features.extend([
                    np.mean(region_intensities),
                    np.max(region_intensities),
                    np.std(region_intensities),
                    np.sum(region_intensities)
                ])
            else:
                features.extend([0, 0, 0, 0])  # Padding if region not found

        # Add some general spectral characteristics
        features.extend([
            np.mean(intensities),
            np.std(intensities),
            np.max(intensities),
            np.percentile(intensities, 95),
            np.percentile(intensities, 5)
        ])

        return features

    def _perform_clustering_analysis(self, X: np.ndarray, data: pd.DataFrame,
                                   analysis_type: str, clustering_methods: List[str],
                                   n_clusters_range: Tuple[int, int], outlier_methods: List[str],
                                   min_cluster_size: int) -> Dict:
        """Perform clustering analysis with multiple methods."""
        results = {
            "analysis_type": analysis_type,
            "data_shape": X.shape,
            "clustering_results": {},
            "outlier_results": {},
            "suspicious_samples": []
        }

        # Apply PCA for visualization and noise reduction
        pca = PCA(n_components=min(10, X.shape[1], X.shape[0]-1))
        X_pca = pca.fit_transform(X)
        results["pca_variance_explained"] = pca.explained_variance_ratio_.sum()

        # 1. Clustering Analysis - parallel if data processing is enabled
        if self.use_data_parallel and len(clustering_methods) > 1:
            logger.debug(f"Running {len(clustering_methods)} clustering methods in parallel")
            cluster_futures = {}
            with ProcessPoolExecutor(max_workers=min(self.data_n_jobs, len(clustering_methods))) as executor:
                for method in clustering_methods:
                    if method == 'kmeans':
                        future = executor.submit(self._kmeans_analysis_static, X_pca, data, n_clusters_range, min_cluster_size)
                    elif method == 'dbscan':
                        future = executor.submit(self._dbscan_analysis_static, X_pca, data, min_cluster_size)
                    elif method == 'hierarchical':
                        future = executor.submit(self._hierarchical_analysis_static, X_pca, data, n_clusters_range, min_cluster_size)
                    else:
                        continue
                    cluster_futures[method] = future

                # Collect results
                for method, future in cluster_futures.items():
                    try:
                        cluster_results = future.result()
                        results["clustering_results"][method] = cluster_results
                    except Exception as e:
                        logger.warning(f"Parallel clustering method {method} failed: {e}")
                        results["clustering_results"][method] = {"error": str(e)}
        else:
            # Sequential clustering analysis
            for method in clustering_methods:
                if method == 'kmeans':
                    cluster_results = self._kmeans_analysis(X_pca, data, n_clusters_range, min_cluster_size)
                elif method == 'dbscan':
                    cluster_results = self._dbscan_analysis(X_pca, data, min_cluster_size)
                elif method == 'hierarchical':
                    cluster_results = self._hierarchical_analysis(X_pca, data, n_clusters_range, min_cluster_size)
                else:
                    continue

                results["clustering_results"][method] = cluster_results

        # 2. Outlier Detection - parallel if data processing is enabled
        if self.use_data_parallel and len(outlier_methods) > 1:
            logger.debug(f"Running {len(outlier_methods)} outlier detection methods in parallel")
            outlier_futures = {}
            with ProcessPoolExecutor(max_workers=min(self.data_n_jobs, len(outlier_methods))) as executor:
                for method in outlier_methods:
                    if method == 'lof':
                        future = executor.submit(self._lof_analysis_static, X_pca, data)
                    elif method == 'isolation_forest':
                        future = executor.submit(self._isolation_forest_analysis_static, X_pca, data)
                    else:
                        continue
                    outlier_futures[method] = future

                # Collect results
                for method, future in outlier_futures.items():
                    try:
                        outlier_results = future.result()
                        results["outlier_results"][method] = outlier_results
                    except Exception as e:
                        logger.warning(f"Parallel outlier method {method} failed: {e}")
                        results["outlier_results"][method] = {"error": str(e)}
        else:
            # Sequential outlier detection
            for method in outlier_methods:
                if method == 'lof':
                    outlier_results = self._lof_analysis(X_pca, data)
                elif method == 'isolation_forest':
                    outlier_results = self._isolation_forest_analysis(X_pca, data)
                else:
                    continue

                results["outlier_results"][method] = outlier_results

        # 3. Identify suspicious samples from this analysis
        results["suspicious_samples"] = self._identify_suspicious_from_analysis(results, data)

        return results

    def _kmeans_analysis(self, X: np.ndarray, data: pd.DataFrame,
                        n_clusters_range: Tuple[int, int], min_cluster_size: int) -> Dict:
        """Perform K-means clustering with multiple cluster numbers."""
        results = {}
        best_score = -1
        best_n_clusters = n_clusters_range[0]

        for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
            if n_clusters >= len(X):
                break

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)

            # Calculate clustering quality metrics
            sil_score = silhouette_score(X, cluster_labels)
            ch_score = calinski_harabasz_score(X, cluster_labels)

            # Analyze clusters for concentration consistency
            cluster_analysis = self._analyze_cluster_consistency(cluster_labels, data, min_cluster_size)

            results[n_clusters] = {
                "silhouette_score": sil_score,
                "calinski_harabasz_score": ch_score,
                "cluster_labels": cluster_labels,
                "cluster_analysis": cluster_analysis,
                "n_valid_clusters": len([c for c in cluster_analysis if c["size"] >= min_cluster_size])
            }

            if sil_score > best_score:
                best_score = sil_score
                best_n_clusters = n_clusters

        results["best_n_clusters"] = best_n_clusters
        results["best_score"] = best_score

        return results

    def _dbscan_analysis(self, X: np.ndarray, data: pd.DataFrame, min_cluster_size: int) -> Dict:
        """Perform DBSCAN clustering."""
        # Try different eps values
        eps_values = np.arange(0.3, 2.0, 0.2)
        results = {}
        best_score = -1
        best_eps = eps_values[0]

        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_cluster_size)
            cluster_labels = dbscan.fit_predict(X)

            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)

            if n_clusters < 2:
                continue

            # Calculate silhouette score (excluding noise points)
            mask = cluster_labels != -1
            if mask.sum() > 0:
                sil_score = silhouette_score(X[mask], cluster_labels[mask])
            else:
                sil_score = -1

            cluster_analysis = self._analyze_cluster_consistency(cluster_labels, data, min_cluster_size)

            results[eps] = {
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_ratio": n_noise / len(cluster_labels),
                "silhouette_score": sil_score,
                "cluster_labels": cluster_labels,
                "cluster_analysis": cluster_analysis
            }

            if sil_score > best_score and n_clusters >= 2:
                best_score = sil_score
                best_eps = eps

        results["best_eps"] = best_eps
        results["best_score"] = best_score

        return results

    def _hierarchical_analysis(self, X: np.ndarray, data: pd.DataFrame,
                             n_clusters_range: Tuple[int, int], min_cluster_size: int) -> Dict:
        """Perform hierarchical clustering."""
        results = {}
        best_score = -1
        best_n_clusters = n_clusters_range[0]

        for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
            if n_clusters >= len(X):
                break

            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            cluster_labels = hierarchical.fit_predict(X)

            sil_score = silhouette_score(X, cluster_labels)
            ch_score = calinski_harabasz_score(X, cluster_labels)

            cluster_analysis = self._analyze_cluster_consistency(cluster_labels, data, min_cluster_size)

            results[n_clusters] = {
                "silhouette_score": sil_score,
                "calinski_harabasz_score": ch_score,
                "cluster_labels": cluster_labels,
                "cluster_analysis": cluster_analysis
            }

            if sil_score > best_score:
                best_score = sil_score
                best_n_clusters = n_clusters

        results["best_n_clusters"] = best_n_clusters
        results["best_score"] = best_score

        return results

    def _analyze_cluster_consistency(self, cluster_labels: np.ndarray,
                                   data: pd.DataFrame, min_cluster_size: int) -> List[Dict]:
        """Analyze the concentration consistency within each cluster."""
        cluster_analysis = []
        target_values = data[self.config.target_column].values
        sample_ids = data[self.config.sample_id_column].values

        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue

            cluster_mask = cluster_labels == cluster_id
            cluster_targets = target_values[cluster_mask]
            cluster_samples = sample_ids[cluster_mask]

            if len(cluster_targets) < min_cluster_size:
                continue

            # Calculate concentration statistics
            mean_conc = np.mean(cluster_targets)
            std_conc = np.std(cluster_targets)
            cv_conc = std_conc / mean_conc if mean_conc > 0 else np.inf

            # Identify potential outliers within cluster (concentration-wise)
            z_scores = np.abs((cluster_targets - mean_conc) / std_conc) if std_conc > 0 else np.zeros_like(cluster_targets)
            outlier_threshold = 2.0
            outliers = cluster_samples[z_scores > outlier_threshold]

            cluster_info = {
                "cluster_id": int(cluster_id),
                "size": len(cluster_targets),
                "mean_concentration": float(mean_conc),
                "std_concentration": float(std_conc),
                "cv_concentration": float(cv_conc),
                "min_concentration": float(np.min(cluster_targets)),
                "max_concentration": float(np.max(cluster_targets)),
                "concentration_range": float(np.max(cluster_targets) - np.min(cluster_targets)),
                "outlier_samples": outliers.tolist(),
                "all_samples": cluster_samples.tolist()
            }

            cluster_analysis.append(cluster_info)

        return cluster_analysis

    def _save_results_and_config(self, results: Dict, timestamp: str) -> None:
        """Save mislabel detection results and configuration to disk."""
        try:
            # Create mislabel analysis directory
            analysis_dir = Path(self.config.reports_dir) / "mislabel_analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)

            # Generate base filename with timestamp
            base_filename = f"mislabel_detection_{timestamp}"

            # 1. Save detailed results as JSON
            results_path = analysis_dir / f"{base_filename}_results.json"

            # Create a serializable copy of results
            serializable_results = self._make_json_serializable(results.copy())

            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)

            logger.info(f"Mislabel detection results saved to: {results_path}")

            # 2. Save configuration used for this run
            config_path = analysis_dir / f"{base_filename}_config.json"

            # Create configuration summary
            run_config = {
                "timestamp": timestamp,
                "pipeline_config": {
                    "project_name": self.config.project_name,
                    "target_column": self.config.target_column,
                    "sample_id_column": self.config.sample_id_column,
                    "use_raw_spectral_data": self.config.use_raw_spectral_data,
                    "focus_range": self.focus_range,
                    "use_feature_parallel": self.use_feature_parallel,
                    "feature_n_jobs": self.feature_n_jobs,
                    "use_data_parallel": self.use_data_parallel,
                    "data_n_jobs": self.data_n_jobs,
                    "all_regions_count": len(self.config.all_regions),
                    "potassium_region": {
                        "element": self.config.potassium_region.element,
                        "lower_wavelength": self.config.potassium_region.lower_wavelength,
                        "upper_wavelength": self.config.potassium_region.upper_wavelength,
                        "center_wavelengths": self.config.potassium_region.center_wavelengths
                    }
                },
                "analysis_parameters": results["run_info"]
            }

            with open(config_path, 'w') as f:
                json.dump(run_config, f, indent=2, default=str)

            logger.info(f"Mislabel detection configuration saved to: {config_path}")

            # 3. Save a summary CSV for easy review
            if results["suspicious_samples"]:
                summary_path = analysis_dir / f"{base_filename}_summary.csv"
                summary_data = []

                for sample_id, suspect_info in results["suspicious_samples"].items():
                    # Get concentration for this sample
                    if hasattr(self, 'focus_data'):
                        concentration = self.focus_data[
                            self.focus_data[self.config.sample_id_column] == sample_id
                        ][self.config.target_column].iloc[0] if len(self.focus_data[
                            self.focus_data[self.config.sample_id_column] == sample_id
                        ]) > 0 else "N/A"
                    else:
                        concentration = "N/A"

                    summary_data.append({
                        "sample_id": sample_id,
                        "concentration": concentration,
                        "suspicion_count": suspect_info["suspicion_count"],
                        "detection_methods": ", ".join(set(suspect_info["reasons"])),
                        "analysis_sources": ", ".join(set(suspect_info["analysis_sources"])),
                        "recommendation": "EXCLUDE" if suspect_info["suspicion_count"] >= 2 else "REVIEW",
                        "timestamp": timestamp
                    })

                # Sort by suspicion count
                summary_data.sort(key=lambda x: (-x["suspicion_count"], x["sample_id"]))

                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(summary_path, index=False)

                logger.info(f"Mislabel detection summary saved to: {summary_path}")

            logger.info(f"All mislabel detection files saved to: {analysis_dir}")

        except Exception as e:
            logger.error(f"Failed to save mislabel detection results: {e}")

    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj

    def _lof_analysis(self, X: np.ndarray, data: pd.DataFrame) -> Dict:
        """Perform Local Outlier Factor analysis."""
        lof = LocalOutlierFactor(n_neighbors=min(20, len(X)//2), contamination=0.1)
        outlier_labels = lof.fit_predict(X)
        outlier_scores = lof.negative_outlier_factor_

        # Get outlier samples
        outlier_mask = outlier_labels == -1
        outlier_samples = data[self.config.sample_id_column].values[outlier_mask]
        outlier_concentrations = data[self.config.target_column].values[outlier_mask]

        return {
            "outlier_samples": outlier_samples.tolist(),
            "outlier_concentrations": outlier_concentrations.tolist(),
            "outlier_scores": outlier_scores[outlier_mask].tolist(),
            "n_outliers": outlier_mask.sum(),
            "outlier_ratio": outlier_mask.mean()
        }

    def _isolation_forest_analysis(self, X: np.ndarray, data: pd.DataFrame) -> Dict:
        """Perform Isolation Forest analysis."""
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(X)
        outlier_scores = iso_forest.decision_function(X)

        # Get outlier samples
        outlier_mask = outlier_labels == -1
        outlier_samples = data[self.config.sample_id_column].values[outlier_mask]
        outlier_concentrations = data[self.config.target_column].values[outlier_mask]

        return {
            "outlier_samples": outlier_samples.tolist(),
            "outlier_concentrations": outlier_concentrations.tolist(),
            "outlier_scores": outlier_scores[outlier_mask].tolist(),
            "n_outliers": outlier_mask.sum(),
            "outlier_ratio": outlier_mask.mean()
        }

    def _identify_suspicious_from_analysis(self, analysis_results: Dict, data: pd.DataFrame) -> List[Dict]:
        """Identify suspicious samples from a single analysis."""
        suspicious = []

        # From clustering analysis - focus on concentration outliers within clusters
        for method, results in analysis_results["clustering_results"].items():
            if "error" in results:
                continue

            best_key = results.get("best_n_clusters") or results.get("best_eps")
            if best_key and best_key in results:
                cluster_analysis = results[best_key]["cluster_analysis"]

                for cluster in cluster_analysis:
                    for sample_id in cluster["outlier_samples"]:
                        suspicious.append({
                            "sample_id": sample_id,
                            "reason": f"{method}_cluster_concentration_outlier",
                            "cluster_id": cluster["cluster_id"],
                            "cluster_mean": cluster["mean_concentration"],
                            "cluster_std": cluster["std_concentration"]
                        })

        # From outlier analysis
        for method, results in analysis_results["outlier_results"].items():
            for i, sample_id in enumerate(results["outlier_samples"]):
                suspicious.append({
                    "sample_id": sample_id,
                    "reason": f"{method}_outlier",
                    "outlier_score": results["outlier_scores"][i],
                    "concentration": results["outlier_concentrations"][i]
                })

        return suspicious

    def _combine_results(self, cluster_analysis: Dict) -> Dict:
        """Combine results from different analyses to identify consensus suspects."""
        all_suspicious = {}

        # Collect all suspicious samples with their reasons
        for analysis_type, results in cluster_analysis.items():
            if "error" in results:
                continue

            for suspect in results["suspicious_samples"]:
                sample_id = suspect["sample_id"]
                if sample_id not in all_suspicious:
                    all_suspicious[sample_id] = {
                        "sample_id": sample_id,
                        "suspicion_count": 0,
                        "reasons": [],
                        "analysis_sources": []
                    }

                all_suspicious[sample_id]["suspicion_count"] += 1
                all_suspicious[sample_id]["reasons"].append(suspect["reason"])
                all_suspicious[sample_id]["analysis_sources"].append(analysis_type)

        # Sort by suspicion count (samples flagged by multiple methods)
        sorted_suspects = sorted(all_suspicious.values(),
                               key=lambda x: x["suspicion_count"], reverse=True)

        return {sample["sample_id"]: sample for sample in sorted_suspects}

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations based on the analysis."""
        recommendations = []

        n_suspicious = len(results["suspicious_samples"])
        total_samples = results["dataset_info"]["focus_samples"]

        if n_suspicious == 0:
            recommendations.append("✅ No obviously mislabeled samples detected in the focus range.")
        else:
            recommendations.append(f"🔍 Found {n_suspicious} potentially mislabeled samples ({n_suspicious/total_samples:.1%} of focus range).")

            # Focus on high-confidence suspects (flagged by multiple methods)
            high_confidence = [s for s in results["suspicious_samples"].values() if s["suspicion_count"] >= 2]
            if high_confidence:
                recommendations.append(f"⚠️  {len(high_confidence)} samples flagged by multiple methods - high confidence suspects.")
                recommendations.append("   → Review these samples first for potential exclusion from training.")

            # Check concentration distribution of suspects
            if results["cluster_analysis"]:
                low_conc_suspects = [s for s in results["suspicious_samples"].values()
                                   if any("concentration" in r for r in results["suspicious_samples"].values())]
                if low_conc_suspects:
                    recommendations.append("📊 Most suspects are in lower concentration range - consistent with expectation.")

        recommendations.append("💡 Use visualize_results() to inspect clustering patterns and suspect locations.")
        recommendations.append("📋 Use export_suspect_list() to get sample IDs for exclusion from next training run.")

        return recommendations

    def visualize_results(self, save_dir: Optional[Path] = None) -> None:
        """Create visualizations of the clustering and outlier detection results."""
        if not hasattr(self, 'results') or not hasattr(self, 'focus_data'):
            logger.error("No results to visualize. Run detect_mislabels() first.")
            return

        if save_dir is None:
            save_dir = Path(self.config.reports_dir) / "mislabel_analysis"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Mislabel Detection Analysis Results', fontsize=16)

        # 1. Concentration distribution with suspects highlighted
        ax = axes[0, 0]
        concentrations = self.focus_data[self.config.target_column]
        ax.hist(concentrations, bins=30, alpha=0.7, label='All samples')

        suspect_ids = list(self.results["suspicious_samples"].keys())
        if suspect_ids:
            suspect_concentrations = self.focus_data[
                self.focus_data[self.config.sample_id_column].isin(suspect_ids)
            ][self.config.target_column]
            ax.hist(suspect_concentrations, bins=30, alpha=0.8, color='red', label='Suspicious samples')

        ax.set_xlabel('Concentration')
        ax.set_ylabel('Count')
        ax.set_title('Concentration Distribution')
        ax.legend()

        # 2. Suspicion count distribution
        ax = axes[0, 1]
        if suspect_ids:
            suspicion_counts = [self.results["suspicious_samples"][sid]["suspicion_count"] for sid in suspect_ids]
            ax.hist(suspicion_counts, bins=max(1, max(suspicion_counts)), alpha=0.7)
            ax.set_xlabel('Number of Methods Flagging Sample')
            ax.set_ylabel('Count')
            ax.set_title('Suspicion Confidence Distribution')
        else:
            ax.text(0.5, 0.5, 'No suspicious samples found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Suspicion Confidence Distribution')

        # 3. Method comparison
        ax = axes[0, 2]
        method_counts = {}
        for suspect in self.results["suspicious_samples"].values():
            for reason in suspect["reasons"]:
                method = reason.split('_')[0]
                method_counts[method] = method_counts.get(method, 0) + 1

        if method_counts:
            methods = list(method_counts.keys())
            counts = list(method_counts.values())
            ax.bar(methods, counts)
            ax.set_xlabel('Detection Method')
            ax.set_ylabel('Suspicious Samples Count')
            ax.set_title('Detection Method Effectiveness')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No detections', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Detection Method Effectiveness')

        # 4. Sample ID vs Concentration with suspects highlighted
        ax = axes[1, 0]
        sample_indices = range(len(self.focus_data))
        concentrations = self.focus_data[self.config.target_column].values
        ax.scatter(sample_indices, concentrations, alpha=0.6, label='All samples')

        if suspect_ids:
            suspect_mask = self.focus_data[self.config.sample_id_column].isin(suspect_ids)
            suspect_indices = np.array(sample_indices)[suspect_mask]
            suspect_concs = concentrations[suspect_mask]
            ax.scatter(suspect_indices, suspect_concs, color='red', s=60, label='Suspicious samples')

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Concentration')
        ax.set_title('Sample Distribution with Suspects')
        ax.legend()

        # 5. Analysis summary text
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = f"""
Analysis Summary:
• Total samples analyzed: {self.results['dataset_info']['focus_samples']}
• Suspicious samples found: {len(suspect_ids)}
• Focus range: {self.focus_range[0]:.2f} - {self.focus_range[1]:.2f}
• High-confidence suspects: {len([s for s in self.results['suspicious_samples'].values() if s['suspicion_count'] >= 2])}

Concentration Statistics:
• Mean: {concentrations.mean():.3f}
• Std: {concentrations.std():.3f}
• Range: {concentrations.min():.3f} - {concentrations.max():.3f}
        """
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')

        # 6. Recommendations
        ax = axes[1, 2]
        ax.axis('off')
        rec_text = "Recommendations:\n" + "\n".join(self.results["recommendations"])
        ax.text(0.1, 0.9, rec_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', wrap=True)

        plt.tight_layout()

        # Save the plot
        plot_path = save_dir / "mislabel_detection_summary.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(f"Visualization saved to: {plot_path}")

    def export_suspect_list(self, save_dir: Optional[Path] = None,
                          min_suspicion_count: int = 1) -> Path:
        """
        Export list of suspicious samples for exclusion from training.

        Args:
            save_dir: Directory to save the results
            min_suspicion_count: Minimum number of methods that must flag a sample

        Returns:
            Path to the exported CSV file
        """
        if not hasattr(self, 'results'):
            logger.error("No results to export. Run detect_mislabels() first.")
            return None

        if save_dir is None:
            save_dir = Path(self.config.reports_dir) / "mislabel_analysis"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Filter suspects by confidence level
        suspects = [
            s for s in self.results["suspicious_samples"].values()
            if s["suspicion_count"] >= min_suspicion_count
        ]

        if not suspects:
            logger.warning(f"No suspects found with suspicion count >= {min_suspicion_count}")
            return None

        # Create detailed export
        export_data = []
        for suspect in suspects:
            sample_id = suspect["sample_id"]

            # Get concentration for this sample
            concentration = self.focus_data[
                self.focus_data[self.config.sample_id_column] == sample_id
            ][self.config.target_column].iloc[0]

            export_data.append({
                "sample_id": sample_id,
                "concentration": concentration,
                "suspicion_count": suspect["suspicion_count"],
                "detection_methods": ", ".join(set(suspect["reasons"])),
                "analysis_sources": ", ".join(set(suspect["analysis_sources"])),
                "recommendation": "EXCLUDE" if suspect["suspicion_count"] >= 2 else "REVIEW"
            })

        # Sort by suspicion count and then by concentration
        export_data.sort(key=lambda x: (-x["suspicion_count"], x["concentration"]))

        # Save to CSV
        df_export = pd.DataFrame(export_data)
        export_path = save_dir / f"suspicious_samples_min_confidence_{min_suspicion_count}.csv"
        df_export.to_csv(export_path, index=False)

        # Also save just the sample IDs for easy exclusion
        sample_ids_path = save_dir / f"exclude_sample_ids_min_confidence_{min_suspicion_count}.txt"
        with open(sample_ids_path, 'w') as f:
            for sample_id in df_export["sample_id"]:
                f.write(f"{sample_id}\n")

        logger.info(f"Exported {len(export_data)} suspicious samples to: {export_path}")
        logger.info(f"Sample IDs for exclusion saved to: {sample_ids_path}")

        return export_path

    # Static methods for parallel processing
    @staticmethod
    def _kmeans_analysis_static(X: np.ndarray, data: pd.DataFrame,
                               n_clusters_range: Tuple[int, int], min_cluster_size: int) -> Dict:
        """Static version of K-means analysis for parallel processing."""
        return MislabelDetector._kmeans_analysis_pure(X, data, n_clusters_range, min_cluster_size)

    @staticmethod
    def _dbscan_analysis_static(X: np.ndarray, data: pd.DataFrame, min_cluster_size: int) -> Dict:
        """Static version of DBSCAN analysis for parallel processing."""
        return MislabelDetector._dbscan_analysis_pure(X, data, min_cluster_size)

    @staticmethod
    def _hierarchical_analysis_static(X: np.ndarray, data: pd.DataFrame,
                                     n_clusters_range: Tuple[int, int], min_cluster_size: int) -> Dict:
        """Static version of hierarchical analysis for parallel processing."""
        return MislabelDetector._hierarchical_analysis_pure(X, data, n_clusters_range, min_cluster_size)

    @staticmethod
    def _lof_analysis_static(X: np.ndarray, data: pd.DataFrame) -> Dict:
        """Static version of LOF analysis for parallel processing."""
        return MislabelDetector._lof_analysis_pure(X, data)

    @staticmethod
    def _isolation_forest_analysis_static(X: np.ndarray, data: pd.DataFrame) -> Dict:
        """Static version of Isolation Forest analysis for parallel processing."""
        return MislabelDetector._isolation_forest_analysis_pure(X, data)

    @staticmethod
    def _kmeans_analysis_pure(X: np.ndarray, data: pd.DataFrame,
                             n_clusters_range: Tuple[int, int], min_cluster_size: int) -> Dict:
        """Pure static K-means analysis without config dependency."""
        from src.config.pipeline_config import config
        results = {}
        best_score = -1
        best_n_clusters = n_clusters_range[0]

        for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
            if n_clusters >= len(X):
                break

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)

            # Calculate clustering quality metrics
            sil_score = silhouette_score(X, cluster_labels)
            ch_score = calinski_harabasz_score(X, cluster_labels)

            # Analyze clusters for concentration consistency
            cluster_analysis = MislabelDetector._analyze_cluster_consistency_pure(
                cluster_labels, data, min_cluster_size, config.target_column, config.sample_id_column
            )

            results[n_clusters] = {
                "silhouette_score": sil_score,
                "calinski_harabasz_score": ch_score,
                "cluster_labels": cluster_labels,
                "cluster_analysis": cluster_analysis,
                "n_valid_clusters": len([c for c in cluster_analysis if c["size"] >= min_cluster_size])
            }

            if sil_score > best_score:
                best_score = sil_score
                best_n_clusters = n_clusters

        results["best_n_clusters"] = best_n_clusters
        results["best_score"] = best_score
        return results

    @staticmethod
    def _dbscan_analysis_pure(X: np.ndarray, data: pd.DataFrame, min_cluster_size: int) -> Dict:
        """Pure static DBSCAN analysis without config dependency."""
        from src.config.pipeline_config import config
        eps_values = np.arange(0.3, 2.0, 0.2)
        results = {}
        best_score = -1
        best_eps = eps_values[0]

        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_cluster_size)
            cluster_labels = dbscan.fit_predict(X)

            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)

            if n_clusters < 2:
                continue

            # Calculate silhouette score (excluding noise points)
            mask = cluster_labels != -1
            if mask.sum() > 0:
                sil_score = silhouette_score(X[mask], cluster_labels[mask])
            else:
                sil_score = -1

            cluster_analysis = MislabelDetector._analyze_cluster_consistency_pure(
                cluster_labels, data, min_cluster_size, config.target_column, config.sample_id_column
            )

            results[eps] = {
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_ratio": n_noise / len(cluster_labels),
                "silhouette_score": sil_score,
                "cluster_labels": cluster_labels,
                "cluster_analysis": cluster_analysis
            }

            if sil_score > best_score and n_clusters >= 2:
                best_score = sil_score
                best_eps = eps

        results["best_eps"] = best_eps
        results["best_score"] = best_score
        return results

    @staticmethod
    def _hierarchical_analysis_pure(X: np.ndarray, data: pd.DataFrame,
                                   n_clusters_range: Tuple[int, int], min_cluster_size: int) -> Dict:
        """Pure static hierarchical analysis without config dependency."""
        from src.config.pipeline_config import config
        results = {}
        best_score = -1
        best_n_clusters = n_clusters_range[0]

        for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
            if n_clusters >= len(X):
                break

            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            cluster_labels = hierarchical.fit_predict(X)

            sil_score = silhouette_score(X, cluster_labels)
            ch_score = calinski_harabasz_score(X, cluster_labels)

            cluster_analysis = MislabelDetector._analyze_cluster_consistency_pure(
                cluster_labels, data, min_cluster_size, config.target_column, config.sample_id_column
            )

            results[n_clusters] = {
                "silhouette_score": sil_score,
                "calinski_harabasz_score": ch_score,
                "cluster_labels": cluster_labels,
                "cluster_analysis": cluster_analysis
            }

            if sil_score > best_score:
                best_score = sil_score
                best_n_clusters = n_clusters

        results["best_n_clusters"] = best_n_clusters
        results["best_score"] = best_score
        return results

    @staticmethod
    def _lof_analysis_pure(X: np.ndarray, data: pd.DataFrame) -> Dict:
        """Pure static LOF analysis without config dependency."""
        from src.config.pipeline_config import config
        lof = LocalOutlierFactor(n_neighbors=min(20, len(X)//2), contamination=0.1)
        outlier_labels = lof.fit_predict(X)
        outlier_scores = lof.negative_outlier_factor_

        # Get outlier samples
        outlier_mask = outlier_labels == -1
        outlier_samples = data[config.sample_id_column].values[outlier_mask]
        outlier_concentrations = data[config.target_column].values[outlier_mask]

        return {
            "outlier_samples": outlier_samples.tolist(),
            "outlier_concentrations": outlier_concentrations.tolist(),
            "outlier_scores": outlier_scores[outlier_mask].tolist(),
            "n_outliers": outlier_mask.sum(),
            "outlier_ratio": outlier_mask.mean()
        }

    @staticmethod
    def _isolation_forest_analysis_pure(X: np.ndarray, data: pd.DataFrame) -> Dict:
        """Pure static Isolation Forest analysis without config dependency."""
        from src.config.pipeline_config import config
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(X)
        outlier_scores = iso_forest.decision_function(X)

        # Get outlier samples
        outlier_mask = outlier_labels == -1
        outlier_samples = data[config.sample_id_column].values[outlier_mask]
        outlier_concentrations = data[config.target_column].values[outlier_mask]

        return {
            "outlier_samples": outlier_samples.tolist(),
            "outlier_concentrations": outlier_concentrations.tolist(),
            "outlier_scores": outlier_scores[outlier_mask].tolist(),
            "n_outliers": outlier_mask.sum(),
            "outlier_ratio": outlier_mask.mean()
        }

    @staticmethod
    def _analyze_cluster_consistency_pure(cluster_labels: np.ndarray, data: pd.DataFrame,
                                         min_cluster_size: int, target_column: str,
                                         sample_id_column: str) -> List[Dict]:
        """Pure static cluster consistency analysis without config dependency."""
        cluster_analysis = []
        target_values = data[target_column].values
        sample_ids = data[sample_id_column].values

        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue

            cluster_mask = cluster_labels == cluster_id
            cluster_targets = target_values[cluster_mask]
            cluster_samples = sample_ids[cluster_mask]

            if len(cluster_targets) < min_cluster_size:
                continue

            # Calculate concentration statistics
            mean_conc = np.mean(cluster_targets)
            std_conc = np.std(cluster_targets)
            cv_conc = std_conc / mean_conc if mean_conc > 0 else np.inf

            # Identify potential outliers within cluster (concentration-wise)
            z_scores = np.abs((cluster_targets - mean_conc) / std_conc) if std_conc > 0 else np.zeros_like(cluster_targets)
            outlier_threshold = 2.0
            outliers = cluster_samples[z_scores > outlier_threshold]

            cluster_info = {
                "cluster_id": int(cluster_id),
                "size": len(cluster_targets),
                "mean_concentration": float(mean_conc),
                "std_concentration": float(std_conc),
                "cv_concentration": float(cv_conc),
                "min_concentration": float(np.min(cluster_targets)),
                "max_concentration": float(np.max(cluster_targets)),
                "concentration_range": float(np.max(cluster_targets) - np.min(cluster_targets)),
                "outlier_samples": outliers.tolist(),
                "all_samples": cluster_samples.tolist()
            }

            cluster_analysis.append(cluster_info)

        return cluster_analysis

