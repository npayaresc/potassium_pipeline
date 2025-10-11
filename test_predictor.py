"""
Comprehensive test cases for the Predictor class

Tests both single file prediction and batch prediction functionality
with various model types and error handling scenarios.
"""
import os
import pytest
import tempfile
import shutil
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.config.pipeline_config import Config
from src.models.predictor import Predictor
from src.utils.custom_exceptions import PipelineError


class TestPredictorSetup:
    """Test fixtures and setup for predictor tests"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration"""
        config = Config()
        config.data_dir = temp_dir / "data"
        config.models_dir = temp_dir / "models"
        config.reports_dir = temp_dir / "reports"
        config.bad_prediction_files_dir = temp_dir / "bad_prediction_files"
        config.logs_dir = temp_dir / "logs"

        # Create directories
        for dir_path in [config.data_dir, config.models_dir, config.reports_dir,
                        config.bad_prediction_files_dir, config.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Set up spectral processing config
        config.enable_wavelength_standardization = False
        config.use_concentration_features = False
        config.use_raw_spectral_data = False

        return config

    @pytest.fixture
    def predictor(self, config):
        """Create Predictor instance"""
        return Predictor(config)

    @pytest.fixture
    def sample_spectral_file(self, temp_dir):
        """Create a sample spectral data file"""
        file_path = temp_dir / "test_sample_001.csv.txt"

        # Create realistic spectral data
        wavelengths = np.linspace(400, 800, 100)
        intensities = np.random.uniform(1000, 5000, (100, 3))  # 3 measurements

        # Create DataFrame with proper format
        df = pd.DataFrame({
            'Wavelength': wavelengths,
            'Intensity_1': intensities[:, 0],
            'Intensity_2': intensities[:, 1],
            'Intensity_3': intensities[:, 2]
        })

        df.to_csv(file_path, index=False)
        return file_path

    @pytest.fixture
    def mock_sklearn_model(self, temp_dir):
        """Create a mock scikit-learn model"""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge

        # Create a simple pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge())
        ])

        # Fit with dummy data
        X_dummy = np.random.random((10, 5))
        y_dummy = np.random.random(10)
        pipeline.fit(X_dummy, y_dummy)

        # Save model
        model_path = temp_dir / "test_sklearn_model.pkl"
        joblib.dump(pipeline, model_path)
        return model_path

    @pytest.fixture
    def mock_autogluon_model_dir(self, temp_dir):
        """Create a mock AutoGluon model directory structure"""
        model_dir = temp_dir / "test_autogluon_model"
        model_dir.mkdir()

        # Create feature pipeline mock
        from sklearn.preprocessing import StandardScaler
        feature_pipeline = StandardScaler()
        feature_pipeline.fit(np.random.random((10, 5)))
        joblib.dump(feature_pipeline, model_dir / "feature_pipeline.pkl")

        # Create AutoGluon predictor mock directory structure
        predictor_dir = model_dir / "predictor"
        predictor_dir.mkdir(parents=True)

        return model_dir


class TestPredictorModelLoading(TestPredictorSetup):
    """Test model loading functionality"""

    def test_load_sklearn_model_success(self, predictor, mock_sklearn_model):
        """Test successful loading of scikit-learn model"""
        model, needs_manual_features = predictor._load_model(mock_sklearn_model)

        assert model is not None
        assert needs_manual_features is False  # sklearn pipelines handle features internally
        assert hasattr(model, 'predict')

    def test_load_model_file_not_found(self, predictor, temp_dir):
        """Test loading non-existent model file"""
        non_existent_path = temp_dir / "non_existent_model.pkl"

        with pytest.raises(PipelineError, match="Invalid model path"):
            predictor._load_model(non_existent_path)

    def test_load_model_invalid_extension(self, predictor, temp_dir):
        """Test loading model with invalid extension"""
        invalid_path = temp_dir / "model.txt"
        invalid_path.touch()

        with pytest.raises(PipelineError, match="Invalid model path"):
            predictor._load_model(invalid_path)

    @patch('src.models.predictor.AUTOGLUON_AVAILABLE', True)
    @patch('src.models.predictor.TabularPredictor')
    def test_load_autogluon_model_success(self, mock_tabular_predictor, predictor, mock_autogluon_model_dir):
        """Test successful loading of AutoGluon model"""
        # Mock TabularPredictor.load
        mock_predictor_instance = Mock()
        mock_tabular_predictor.load.return_value = mock_predictor_instance

        model, needs_manual_features = predictor._load_model(mock_autogluon_model_dir)

        assert model is not None
        assert needs_manual_features is True  # AutoGluon models need manual feature transformation
        mock_tabular_predictor.load.assert_called_once()

    @patch('src.models.predictor.AUTOGLUON_AVAILABLE', False)
    def test_load_autogluon_model_not_available(self, predictor, mock_autogluon_model_dir):
        """Test loading AutoGluon model when not available"""
        with pytest.raises(ImportError, match="AutoGluon is not installed"):
            predictor._load_model(mock_autogluon_model_dir)


class TestPredictorSinglePrediction(TestPredictorSetup):
    """Test single file prediction functionality"""

    @patch.object(Predictor, '_load_model')
    @patch.object(Predictor, '_detect_model_calibration_status')
    def test_single_prediction_success(self, mock_calibration, mock_load_model,
                                     predictor, sample_spectral_file, mock_sklearn_model):
        """Test successful single file prediction"""
        # Setup mocks
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.25])
        mock_load_model.return_value = (mock_model, False)
        mock_calibration.return_value = (False, False)

        # Make prediction
        result = predictor.make_prediction(sample_spectral_file, mock_sklearn_model)

        assert isinstance(result, float)
        assert result == 0.25
        mock_load_model.assert_called_once_with(mock_sklearn_model)
        mock_model.predict.assert_called_once()

    def test_single_prediction_file_not_found(self, predictor, temp_dir, mock_sklearn_model):
        """Test single prediction with non-existent input file"""
        non_existent_file = temp_dir / "non_existent.csv.txt"

        with pytest.raises(FileNotFoundError, match="Input file not found"):
            predictor.make_prediction(non_existent_file, mock_sklearn_model)

    @patch.object(Predictor, '_load_model')
    @patch('src.models.predictor.logger')
    def test_single_prediction_no_data_after_cleaning(self, mock_logger, mock_load_model,
                                                     predictor, sample_spectral_file, mock_sklearn_model):
        """Test single prediction when no data remains after cleaning"""
        # Setup mock to simulate failed cleaning
        mock_model = Mock()
        mock_load_model.return_value = (mock_model, False)

        # Mock data cleanser to return empty array
        with patch.object(predictor.data_cleanser, 'clean_spectra') as mock_clean:
            mock_clean.return_value = np.array([])

            with pytest.raises(PipelineError, match="No data remaining"):
                predictor.make_prediction(sample_spectral_file, mock_sklearn_model)

    @patch.object(Predictor, '_load_model')
    @patch.object(Predictor, '_detect_model_calibration_status')
    def test_single_prediction_with_calibration_logging(self, mock_calibration, mock_load_model,
                                                       predictor, sample_spectral_file, mock_sklearn_model):
        """Test single prediction with calibration status logging"""
        # Setup mocks
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.30])
        mock_load_model.return_value = (mock_model, False)
        mock_calibration.return_value = (True, True)  # Both calibrations enabled

        with patch('src.models.predictor.logger') as mock_logger:
            result = predictor.make_prediction(sample_spectral_file, mock_sklearn_model)

            # Check logging calls
            mock_logger.info.assert_any_call("Single prediction will use sample-weight calibrated model - correcting for sample weight bias")
            mock_logger.info.assert_any_call("Single prediction will use post-calibrated model - enhanced accuracy for target metrics")


class TestPredictorBatchPrediction(TestPredictorSetup):
    """Test batch prediction functionality"""

    @pytest.fixture
    def batch_input_dir(self, temp_dir):
        """Create directory with multiple spectral files for batch testing"""
        batch_dir = temp_dir / "batch_input"
        batch_dir.mkdir()

        # Create multiple sample files with different prefixes
        sample_prefixes = ["sample_001", "sample_002", "sample_003"]

        for prefix in sample_prefixes:
            for i in range(2):  # 2 files per sample
                file_path = batch_dir / f"{prefix}_{i:02d}.csv.txt"

                # Create spectral data
                wavelengths = np.linspace(400, 800, 100)
                intensities = np.random.uniform(1000, 5000, (100, 3))

                df = pd.DataFrame({
                    'Wavelength': wavelengths,
                    'Intensity_1': intensities[:, 0],
                    'Intensity_2': intensities[:, 1],
                    'Intensity_3': intensities[:, 2]
                })

                df.to_csv(file_path, index=False)

        return batch_dir

    @patch.object(Predictor, '_load_model')
    @patch.object(Predictor, '_detect_model_calibration_status')
    def test_batch_prediction_success(self, mock_calibration, mock_load_model,
                                    predictor, batch_input_dir, mock_sklearn_model):
        """Test successful batch prediction"""
        # Setup mocks
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.20, 0.25, 0.30])  # 3 predictions
        mock_load_model.return_value = (mock_model, False)
        mock_calibration.return_value = (False, False)

        # Run batch prediction
        results_df = predictor.make_batch_predictions(batch_input_dir, mock_sklearn_model)

        # Verify results
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 3  # 3 unique samples
        assert all(col in results_df.columns for col in ['sampleId', 'PredictedValue', 'Status'])
        assert all(results_df['Status'] == 'Success')

    @patch.object(Predictor, '_load_model')
    def test_batch_prediction_empty_directory(self, mock_load_model, predictor, temp_dir, mock_sklearn_model):
        """Test batch prediction with empty input directory"""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        mock_model = Mock()
        mock_load_model.return_value = (mock_model, False)

        results_df = predictor.make_batch_predictions(empty_dir, mock_sklearn_model)

        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 0

    @patch.object(Predictor, '_load_model')
    @patch.object(Predictor, '_detect_model_calibration_status')
    def test_batch_prediction_with_failures(self, mock_calibration, mock_load_model,
                                          predictor, batch_input_dir, mock_sklearn_model):
        """Test batch prediction with some sample failures"""
        # Setup mocks
        mock_model = Mock()
        # Simulate partial failure - only 2 successful predictions instead of 3
        mock_model.predict.side_effect = Exception("Prediction failed")
        mock_load_model.return_value = (mock_model, False)
        mock_calibration.return_value = (False, False)

        # Mock fallback individual predictions
        with patch.object(predictor, '_fallback_individual_predictions') as mock_fallback:
            mock_fallback.return_value = [
                {'sampleId': 'sample_001', 'PredictedValue': 0.20, 'Status': 'Success'},
                {'sampleId': 'sample_002', 'PredictedValue': np.nan, 'Status': 'Failed - Test error'}
            ]

            results_df = predictor.make_batch_predictions(batch_input_dir, mock_sklearn_model)

            # Verify fallback was called
            mock_fallback.assert_called_once()
            assert isinstance(results_df, pd.DataFrame)

    @patch.object(Predictor, '_load_model')
    @patch.object(Predictor, '_detect_model_calibration_status')
    def test_batch_prediction_with_outlier_samples(self, mock_calibration, mock_load_model,
                                                  predictor, batch_input_dir, mock_sklearn_model):
        """Test batch prediction when some samples are flagged as outliers"""
        # Setup mocks
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.25, 0.30])  # Only 2 successful
        mock_load_model.return_value = (mock_model, False)
        mock_calibration.return_value = (False, False)

        # Mock data cleanser to simulate one sample being flagged as outlier
        original_clean_spectra = predictor.data_cleanser.clean_spectra
        call_count = 0

        def mock_clean_spectra(sample_id, intensities):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return np.array([])  # First sample flagged as outlier
            else:
                return original_clean_spectra(sample_id, intensities)

        with patch.object(predictor.data_cleanser, 'clean_spectra', side_effect=mock_clean_spectra):
            results_df = predictor.make_batch_predictions(batch_input_dir, mock_sklearn_model)

            # Should have mixed results - some failed, some successful
            assert isinstance(results_df, pd.DataFrame)
            failed_samples = results_df[results_df['Status'].str.contains('Failed')]
            successful_samples = results_df[results_df['Status'] == 'Success']

            assert len(failed_samples) >= 1
            assert len(successful_samples) >= 1


class TestPredictorErrorHandling(TestPredictorSetup):
    """Test error handling scenarios"""

    def test_wavelength_metadata_loading_success(self, predictor, mock_sklearn_model, temp_dir):
        """Test successful loading of wavelength metadata"""
        # Create metadata file
        metadata_path = mock_sklearn_model.with_suffix('.wavelength_metadata.json')
        metadata = {
            'global_wavelength_range': {'min': 400.0, 'max': 800.0},
            'wavelength_standardization_enabled': True,
            'interpolation_method': 'linear',
            'wavelength_resolution': 0.5
        }

        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        # Test metadata loading
        predictor._load_wavelength_metadata(mock_sklearn_model)

        assert predictor.config.enable_wavelength_standardization is True

    def test_wavelength_metadata_loading_failure(self, predictor, mock_sklearn_model, temp_dir):
        """Test handling of corrupted wavelength metadata"""
        # Create corrupted metadata file
        metadata_path = mock_sklearn_model.with_suffix('.wavelength_metadata.json')
        with open(metadata_path, 'w') as f:
            f.write("invalid json content")

        # Should not raise exception, just log warning
        with patch('src.models.predictor.logger') as mock_logger:
            predictor._load_wavelength_metadata(mock_sklearn_model)
            mock_logger.warning.assert_called()

    @patch.object(Predictor, '_load_model')
    def test_prediction_model_prediction_failure(self, mock_load_model, predictor,
                                               sample_spectral_file, mock_sklearn_model):
        """Test handling of model prediction failures"""
        # Setup mock to fail during prediction
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model prediction failed")
        mock_load_model.return_value = (mock_model, False)

        with pytest.raises(Exception, match="Model prediction failed"):
            predictor.make_prediction(sample_spectral_file, mock_sklearn_model)

    def test_calibration_status_detection(self, predictor, mock_sklearn_model):
        """Test calibration status detection for different model types"""
        # Test with non-calibrated model
        mock_model = Mock()
        has_sample_cal, has_post_cal = predictor._detect_model_calibration_status(mock_model, mock_sklearn_model)

        assert has_sample_cal is False
        assert has_post_cal is False

        # Test with model that has calibrator attribute
        mock_model.calibrator = Mock()
        has_sample_cal, has_post_cal = predictor._detect_model_calibration_status(mock_model, mock_sklearn_model)

        assert has_sample_cal is True
        assert has_post_cal is False


class TestPredictorFeatureHandling(TestPredictorSetup):
    """Test feature handling for different model types"""

    @patch.object(Predictor, '_load_model')
    @patch.object(Predictor, '_detect_model_calibration_status')
    def test_manual_feature_transformation(self, mock_calibration, mock_load_model,
                                         predictor, sample_spectral_file, mock_sklearn_model):
        """Test manual feature transformation for legacy AutoGluon models"""
        # Setup mock for legacy format
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.35])
        mock_load_model.return_value = (mock_model, True)  # needs_manual_features = True
        mock_calibration.return_value = (False, False)

        # Add mock feature pipeline
        predictor.feature_pipeline = Mock()
        predictor.feature_pipeline.transform.return_value = np.random.random((1, 10))
        predictor.feature_selector = None
        predictor.dimension_reducer = None

        result = predictor.make_prediction(sample_spectral_file, mock_sklearn_model)

        assert isinstance(result, float)
        assert result == 0.35
        predictor.feature_pipeline.transform.assert_called_once()

    @patch.object(Predictor, '_load_model')
    @patch.object(Predictor, '_detect_model_calibration_status')
    def test_raw_spectral_mode(self, mock_calibration, mock_load_model,
                              predictor, sample_spectral_file, mock_sklearn_model):
        """Test raw spectral mode feature extraction"""
        # Enable raw spectral mode
        predictor.config.use_raw_spectral_data = True

        # Setup mock for legacy format with raw spectral mode
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.40])
        mock_load_model.return_value = (mock_model, True)  # needs_manual_features = True
        mock_calibration.return_value = (False, False)

        # Mock raw spectral feature extraction
        with patch.object(predictor, '_extract_raw_spectral_features') as mock_extract:
            mock_extract.return_value = np.random.random((1, 50))
            predictor.dimension_reducer = None

            result = predictor.make_prediction(sample_spectral_file, mock_sklearn_model)

            assert isinstance(result, float)
            assert result == 0.40
            mock_extract.assert_called_once()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])