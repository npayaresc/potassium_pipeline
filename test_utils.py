"""
Test utilities and helper functions for predictor testing

Provides common fixtures, mock data generators, and assertion helpers
for all predictor test modules.
"""
import numpy as np
import pandas as pd
import tempfile
import joblib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from unittest.mock import Mock


class SpectralDataGenerator:
    """Generates realistic spectral data for testing"""

    @staticmethod
    def generate_potassium_spectrum(wavelength_range: Tuple[float, float] = (400, 800),
                                  n_points: int = 200,
                                  n_measurements: int = 3,
                                  potassium_concentration: float = 0.3,
                                  noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic potassium spectral data with characteristic peaks.

        Args:
            wavelength_range: (min, max) wavelength range in nm
            n_points: Number of wavelength points
            n_measurements: Number of intensity measurements
            potassium_concentration: Simulated potassium concentration (affects peak intensity)
            noise_level: Relative noise level (0.1 = 10% noise)

        Returns:
            Tuple of (wavelengths, intensities) arrays
        """
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_points)

        # Base spectrum with typical LIBS characteristics
        base_intensity = 1000 + 500 * np.exp(-((wavelengths - 500) / 100) ** 2)

        # Add potassium spectral lines
        k_lines = [766.5, 769.9, 404.4]  # Main K lines in nm

        intensities = np.zeros((n_points, n_measurements))
        for i in range(n_measurements):
            spectrum = base_intensity.copy()

            # Add potassium peaks
            for k_line in k_lines:
                if wavelength_range[0] <= k_line <= wavelength_range[1]:
                    # Gaussian peak
                    peak_intensity = potassium_concentration * 2000 * np.exp(-((wavelengths - k_line) / 0.5) ** 2)
                    spectrum += peak_intensity

            # Add noise
            noise = np.random.normal(0, noise_level * np.mean(spectrum), n_points)
            spectrum += noise

            # Ensure positive values
            spectrum = np.maximum(spectrum, 100)

            intensities[:, i] = spectrum

        return wavelengths, intensities

    @staticmethod
    def create_spectral_file(file_path: Path,
                           wavelength_range: Tuple[float, float] = (400, 800),
                           n_points: int = 200,
                           n_measurements: int = 3,
                           potassium_concentration: float = 0.3) -> Path:
        """
        Create a spectral data file in the expected CSV format.

        Args:
            file_path: Path where to save the file
            wavelength_range: (min, max) wavelength range in nm
            n_points: Number of wavelength points
            n_measurements: Number of intensity measurements
            potassium_concentration: Simulated potassium concentration

        Returns:
            Path to the created file
        """
        wavelengths, intensities = SpectralDataGenerator.generate_potassium_spectrum(
            wavelength_range, n_points, n_measurements, potassium_concentration
        )

        # Create DataFrame in expected format
        data = {'Wavelength': wavelengths}
        for i in range(n_measurements):
            data[f'Intensity_{i+1}'] = intensities[:, i]

        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

        return file_path

    @staticmethod
    def create_batch_spectral_files(base_dir: Path,
                                  sample_prefixes: List[str],
                                  files_per_sample: int = 2,
                                  **spectrum_kwargs) -> Dict[str, List[Path]]:
        """
        Create multiple spectral files for batch testing.

        Args:
            base_dir: Directory to create files in
            sample_prefixes: List of sample ID prefixes
            files_per_sample: Number of files per sample
            **spectrum_kwargs: Additional arguments for spectrum generation

        Returns:
            Dictionary mapping sample IDs to list of file paths
        """
        files_by_sample = {}

        for prefix in sample_prefixes:
            files_by_sample[prefix] = []
            for i in range(files_per_sample):
                file_path = base_dir / f"{prefix}_{i:02d}.csv.txt"
                SpectralDataGenerator.create_spectral_file(file_path, **spectrum_kwargs)
                files_by_sample[prefix].append(file_path)

        return files_by_sample


class ModelMockFactory:
    """Factory for creating various types of model mocks"""

    @staticmethod
    def create_sklearn_pipeline_mock(model_type: str = "Ridge",
                                   prediction_value: float = 0.25) -> Mock:
        """Create mock scikit-learn pipeline"""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Create actual pipeline structure for more realistic testing
        if model_type.lower() == "ridge":
            from sklearn.linear_model import Ridge
            model = Ridge()
        elif model_type.lower() == "randomforest":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=10)
        else:
            model = Mock()

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        # Fit with dummy data
        X_dummy = np.random.random((10, 20))
        y_dummy = np.random.random(10)

        try:
            pipeline.fit(X_dummy, y_dummy)
        except:
            # If fitting fails, create a mock
            pipeline = Mock()
            pipeline.predict.return_value = np.array([prediction_value])

        return pipeline

    @staticmethod
    def create_autogluon_model_directory(base_dir: Path,
                                       model_name: str = "test_autogluon",
                                       include_calibration: bool = True,
                                       include_feature_selector: bool = True,
                                       include_dimension_reducer: bool = True) -> Path:
        """Create mock AutoGluon model directory structure"""
        model_dir = base_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Feature pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer

        feature_pipeline = ColumnTransformer([
            ('scaler', StandardScaler(), slice(None))
        ])
        feature_pipeline.fit(np.random.random((10, 50)))
        joblib.dump(feature_pipeline, model_dir / "feature_pipeline.pkl")

        # Optional feature selector
        if include_feature_selector:
            from sklearn.feature_selection import SelectKBest
            feature_selector = SelectKBest(k=30)
            feature_selector.fit(np.random.random((10, 50)), np.random.random(10))
            joblib.dump(feature_selector, model_dir / "feature_selector.pkl")

        # Optional dimension reducer
        if include_dimension_reducer:
            from sklearn.decomposition import PCA
            dimension_reducer = PCA(n_components=20)
            dimension_reducer.fit(np.random.random((10, 30)))
            joblib.dump(dimension_reducer, model_dir / "pca_transformer.pkl")

        # Optional calibrator
        if include_calibration:
            from sklearn.preprocessing import StandardScaler
            calibrator = StandardScaler()
            calibrator.fit(np.random.random((10, 1)))
            joblib.dump(calibrator, model_dir / "calibrator.pkl")

        # AutoGluon predictor directory
        predictor_dir = model_dir / "predictor"
        predictor_dir.mkdir()

        return model_dir

    @staticmethod
    def create_neural_network_pipeline_mock(prediction_value: float = 0.35) -> Mock:
        """Create mock neural network pipeline"""
        pipeline = Mock()

        # Add neural network specific attributes
        pipeline.neural_network = Mock()
        pipeline.feature_pipeline = Mock()
        pipeline.predict.return_value = np.array([prediction_value])

        return pipeline


class ConfigFactory:
    """Factory for creating test configurations"""

    @staticmethod
    def create_test_config(temp_dir: Path,
                         enable_gpu: bool = False,
                         enable_wavelength_standardization: bool = False,
                         use_raw_spectral_data: bool = False,
                         use_sample_weights: bool = True) -> 'Config':
        """Create test configuration with specified settings"""
        from src.config.pipeline_config import Config

        config = Config()

        # Directory setup
        config.data_dir = temp_dir / "data"
        config.models_dir = temp_dir / "models"
        config.reports_dir = temp_dir / "reports"
        config.bad_prediction_files_dir = temp_dir / "bad_prediction_files"
        config.logs_dir = temp_dir / "logs"

        # Create directories
        for dir_path in [config.data_dir, config.models_dir, config.reports_dir,
                        config.bad_prediction_files_dir, config.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Feature settings
        config.enable_wavelength_standardization = enable_wavelength_standardization
        config.use_raw_spectral_data = use_raw_spectral_data
        config.use_sample_weights = use_sample_weights
        config.use_concentration_features = True

        # AutoGluon settings
        config.autogluon.time_limit = 30
        config.autogluon.use_gpu = enable_gpu
        config.autogluon.use_improved_config = True
        if enable_gpu:
            config.autogluon.num_gpus = 1

        # Sample weight settings
        if use_sample_weights:
            config.sample_weight_method = 'inverse_frequency'

        return config


class AssertionHelpers:
    """Helper functions for common test assertions"""

    @staticmethod
    def assert_valid_prediction(result: float, expected_range: Tuple[float, float] = (0.0, 5.0)):
        """Assert that a prediction result is valid"""
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert not np.isnan(result), "Prediction should not be NaN"
        assert not np.isinf(result), "Prediction should not be infinite"
        assert expected_range[0] <= result <= expected_range[1], \
            f"Prediction {result} outside expected range {expected_range}"

    @staticmethod
    def assert_valid_batch_results(results_df: pd.DataFrame,
                                 expected_samples: int,
                                 required_columns: List[str] = None):
        """Assert that batch prediction results are valid"""
        if required_columns is None:
            required_columns = ['sampleId', 'PredictedValue', 'Status']

        assert isinstance(results_df, pd.DataFrame), "Results should be a DataFrame"
        assert len(results_df) == expected_samples, \
            f"Expected {expected_samples} samples, got {len(results_df)}"

        for col in required_columns:
            assert col in results_df.columns, f"Missing required column: {col}"

        # Check for valid sample IDs
        assert results_df['sampleId'].notna().all(), "All samples should have valid IDs"

        # Check success/failure status
        valid_statuses = ['Success', 'Failed', 'Failed - Outlier']
        status_mask = results_df['Status'].str.contains('Success|Failed')
        assert status_mask.all(), "All samples should have valid status"

    @staticmethod
    def assert_model_loading(model, needs_manual_features: bool, expected_model_type: str = None):
        """Assert that model loading was successful"""
        assert model is not None, "Model should not be None"
        assert hasattr(model, 'predict'), "Model should have predict method"
        assert isinstance(needs_manual_features, bool), \
            "needs_manual_features should be boolean"

        if expected_model_type:
            assert expected_model_type.lower() in str(type(model)).lower(), \
                f"Expected model type {expected_model_type}, got {type(model)}"

    @staticmethod
    def assert_calibration_status(has_sample_cal: bool, has_post_cal: bool,
                                expected_sample_cal: bool = None,
                                expected_post_cal: bool = None):
        """Assert calibration status detection"""
        assert isinstance(has_sample_cal, bool), "Sample calibration status should be boolean"
        assert isinstance(has_post_cal, bool), "Post calibration status should be boolean"

        if expected_sample_cal is not None:
            assert has_sample_cal == expected_sample_cal, \
                f"Expected sample calibration {expected_sample_cal}, got {has_sample_cal}"

        if expected_post_cal is not None:
            assert has_post_cal == expected_post_cal, \
                f"Expected post calibration {expected_post_cal}, got {has_post_cal}"


class TestDataSets:
    """Predefined test datasets for consistent testing"""

    @staticmethod
    def get_low_concentration_samples() -> Dict[str, float]:
        """Get sample data with low potassium concentrations"""
        return {
            'low_sample_001': 0.05,
            'low_sample_002': 0.10,
            'low_sample_003': 0.15
        }

    @staticmethod
    def get_medium_concentration_samples() -> Dict[str, float]:
        """Get sample data with medium potassium concentrations"""
        return {
            'med_sample_001': 0.25,
            'med_sample_002': 0.35,
            'med_sample_003': 0.45
        }

    @staticmethod
    def get_high_concentration_samples() -> Dict[str, float]:
        """Get sample data with high potassium concentrations"""
        return {
            'high_sample_001': 0.60,
            'high_sample_002': 0.80,
            'high_sample_003': 1.20
        }

    @staticmethod
    def get_mixed_concentration_samples() -> Dict[str, float]:
        """Get mixed concentration sample data for comprehensive testing"""
        samples = {}
        samples.update(TestDataSets.get_low_concentration_samples())
        samples.update(TestDataSets.get_medium_concentration_samples())
        samples.update(TestDataSets.get_high_concentration_samples())
        return samples


# Convenience functions for common test patterns
def create_temporary_test_environment():
    """Create temporary test environment with common structure"""
    temp_dir = Path(tempfile.mkdtemp())
    config = ConfigFactory.create_test_config(temp_dir)

    return temp_dir, config


def cleanup_test_environment(temp_dir: Path):
    """Clean up temporary test environment"""
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


# Context manager for test environments
class TestEnvironment:
    """Context manager for creating and cleaning up test environments"""

    def __init__(self, **config_kwargs):
        self.config_kwargs = config_kwargs
        self.temp_dir = None
        self.config = None

    def __enter__(self):
        self.temp_dir, self.config = create_temporary_test_environment()

        # Apply any additional config settings
        for key, value in self.config_kwargs.items():
            setattr(self.config, key, value)

        return self.temp_dir, self.config

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir:
            cleanup_test_environment(self.temp_dir)