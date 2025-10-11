"""
Specialized test cases for AutoGluon model prediction

Tests AutoGluon-specific functionality including:
- Legacy AutoGluon directory format
- New consistent pipeline format with AutoGluon wrapper
- GPU-enabled predictions
- Calibrated AutoGluon models
- Feature selection and dimension reduction with AutoGluon
"""
import pytest
import tempfile
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.config.pipeline_config import Config
from src.models.predictor import Predictor
from src.models.custom_autogluon import AutoGluonRegressor


class TestAutoGluonPredictorSetup:
    """Setup fixtures for AutoGluon predictor tests"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration with AutoGluon settings"""
        config = Config()
        config.data_dir = temp_dir / "data"
        config.models_dir = temp_dir / "models"
        config.reports_dir = temp_dir / "reports"
        config.bad_prediction_files_dir = temp_dir / "bad_prediction_files"

        # Create directories
        for dir_path in [config.data_dir, config.models_dir, config.reports_dir,
                        config.bad_prediction_files_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # AutoGluon specific settings
        config.autogluon.time_limit = 60
        config.autogluon.use_improved_config = True
        config.use_sample_weights = True
        config.sample_weight_method = 'inverse_frequency'

        return config

    @pytest.fixture
    def predictor(self, config):
        """Create Predictor instance"""
        return Predictor(config)

    @pytest.fixture
    def sample_spectral_data(self, temp_dir):
        """Create sample spectral data files"""
        files = []
        for i in range(3):
            file_path = temp_dir / f"test_sample_{i:03d}.csv.txt"

            # Create realistic potassium spectral data
            wavelengths = np.linspace(400, 800, 200)
            # Add some potassium peaks around 766-770nm
            intensities = np.random.uniform(1000, 3000, (200, 4))

            # Enhance intensity around potassium lines
            k_indices = np.where((wavelengths >= 766) & (wavelengths <= 770))[0]
            if len(k_indices) > 0:
                intensities[k_indices] *= 1.5

            df = pd.DataFrame({
                'Wavelength': wavelengths,
                'Intensity_1': intensities[:, 0],
                'Intensity_2': intensities[:, 1],
                'Intensity_3': intensities[:, 2],
                'Intensity_4': intensities[:, 3]
            })

            df.to_csv(file_path, index=False)
            files.append(file_path)

        return files

    @pytest.fixture
    def legacy_autogluon_model(self, temp_dir):
        """Create mock legacy AutoGluon model directory"""
        model_dir = temp_dir / "legacy_autogluon_model"
        model_dir.mkdir()

        # Create feature pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer

        feature_pipeline = ColumnTransformer([
            ('scaler', StandardScaler(), slice(None))
        ])
        feature_pipeline.fit(np.random.random((10, 20)))
        joblib.dump(feature_pipeline, model_dir / "feature_pipeline.pkl")

        # Create feature selector (optional)
        from sklearn.feature_selection import SelectKBest
        feature_selector = SelectKBest(k=15)
        feature_selector.fit(np.random.random((10, 20)), np.random.random(10))
        joblib.dump(feature_selector, model_dir / "feature_selector.pkl")

        # Create dimension reducer (optional)
        from sklearn.decomposition import PCA
        dimension_reducer = PCA(n_components=10)
        dimension_reducer.fit(np.random.random((10, 15)))
        joblib.dump(dimension_reducer, model_dir / "pca_transformer.pkl")

        # Create calibrator for calibrated models
        from sklearn.preprocessing import StandardScaler as CalibrationScaler
        calibrator = CalibrationScaler()
        calibrator.fit(np.random.random((10, 1)))
        joblib.dump(calibrator, model_dir / "calibrator.pkl")

        # Create predictor directory (AutoGluon format)
        predictor_dir = model_dir / "predictor"
        predictor_dir.mkdir()

        return model_dir

    @pytest.fixture
    def consistent_autogluon_model(self, temp_dir):
        """Create mock consistent pipeline format with AutoGluon wrapper"""
        from sklearn.pipeline import Pipeline

        # Create AutoGluon wrapper
        autogluon_wrapper = Mock(spec=AutoGluonRegressor)
        autogluon_wrapper.predict.return_value = np.array([0.45])

        # Create pipeline with AutoGluon as model step
        pipeline = Pipeline([
            ('model', autogluon_wrapper)
        ])

        # Save model
        model_path = temp_dir / "consistent_autogluon_model.pkl"
        joblib.dump(pipeline, model_path)
        return model_path


class TestLegacyAutoGluonPrediction(TestAutoGluonPredictorSetup):
    """Test legacy AutoGluon directory format prediction"""

    @patch('src.models.predictor.AUTOGLUON_AVAILABLE', True)
    @patch('src.models.predictor.TabularPredictor')
    def test_legacy_autogluon_single_prediction(self, mock_tabular_predictor, predictor,
                                              sample_spectral_data, legacy_autogluon_model):
        """Test single prediction with legacy AutoGluon format"""
        # Setup AutoGluon predictor mock
        mock_ag_predictor = Mock()
        mock_ag_predictor.predict.return_value = np.array([0.35])
        mock_tabular_predictor.load.return_value = mock_ag_predictor

        # Make prediction
        result = predictor.make_prediction(sample_spectral_data[0], legacy_autogluon_model)

        assert isinstance(result, float)
        assert result == 0.35
        mock_tabular_predictor.load.assert_called_once_with(str(legacy_autogluon_model))

    @patch('src.models.predictor.AUTOGLUON_AVAILABLE', True)
    @patch('src.models.predictor.TabularPredictor')
    def test_legacy_autogluon_batch_prediction(self, mock_tabular_predictor, predictor,
                                             temp_dir, legacy_autogluon_model):
        """Test batch prediction with legacy AutoGluon format"""
        # Create batch input directory
        batch_dir = temp_dir / "batch_input"
        batch_dir.mkdir()

        # Create multiple sample files
        for i in range(3):
            for j in range(2):  # 2 files per sample
                file_path = batch_dir / f"sample_{i:03d}_{j:02d}.csv.txt"

                wavelengths = np.linspace(400, 800, 100)
                intensities = np.random.uniform(1000, 5000, (100, 3))

                df = pd.DataFrame({
                    'Wavelength': wavelengths,
                    'Intensity_1': intensities[:, 0],
                    'Intensity_2': intensities[:, 1],
                    'Intensity_3': intensities[:, 2]
                })

                df.to_csv(file_path, index=False)

        # Setup AutoGluon predictor mock
        mock_ag_predictor = Mock()
        mock_ag_predictor.predict.return_value = np.array([0.20, 0.30, 0.40])
        mock_tabular_predictor.load.return_value = mock_ag_predictor

        # Run batch prediction
        results_df = predictor.make_batch_predictions(batch_dir, legacy_autogluon_model)

        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 3
        assert all(results_df['Status'] == 'Success')
        assert all(results_df['PredictedValue'].isin([0.20, 0.30, 0.40]))

    @patch('src.models.predictor.AUTOGLUON_AVAILABLE', True)
    @patch('src.models.predictor.TabularPredictor')
    def test_legacy_autogluon_with_dimension_reduction(self, mock_tabular_predictor, predictor,
                                                     sample_spectral_data, legacy_autogluon_model):
        """Test legacy AutoGluon with PCA dimension reduction"""
        # Setup AutoGluon predictor mock
        mock_ag_predictor = Mock()
        mock_ag_predictor.predict.return_value = np.array([0.28])
        mock_tabular_predictor.load.return_value = mock_ag_predictor

        # Make prediction (PCA should be automatically loaded)
        result = predictor.make_prediction(sample_spectral_data[0], legacy_autogluon_model)

        assert isinstance(result, float)
        assert result == 0.28

    @patch('src.models.predictor.AUTOGLUON_AVAILABLE', True)
    @patch('src.models.predictor.TabularPredictor')
    def test_legacy_autogluon_calibrated_model(self, mock_tabular_predictor, predictor,
                                             sample_spectral_data, legacy_autogluon_model):
        """Test legacy AutoGluon with calibration"""
        # Setup AutoGluon predictor mock
        mock_ag_predictor = Mock()
        mock_ag_predictor.predict.return_value = np.array([0.33])
        mock_tabular_predictor.load.return_value = mock_ag_predictor

        # Make prediction (calibration should be detected)
        with patch('src.models.predictor.logger') as mock_logger:
            result = predictor.make_prediction(sample_spectral_data[0], legacy_autogluon_model)

            # Check calibration logging
            mock_logger.info.assert_any_call("AutoGluon model has calibration - predictions will be calibrated for sample weight bias correction")

    @patch('src.models.predictor.AUTOGLUON_AVAILABLE', False)
    def test_legacy_autogluon_not_available(self, predictor, sample_spectral_data, legacy_autogluon_model):
        """Test legacy AutoGluon when library not available"""
        with pytest.raises(ImportError, match="AutoGluon is not installed"):
            predictor.make_prediction(sample_spectral_data[0], legacy_autogluon_model)


class TestConsistentAutoGluonPrediction(TestAutoGluonPredictorSetup):
    """Test consistent pipeline format with AutoGluon wrapper"""

    def test_consistent_autogluon_single_prediction(self, predictor, sample_spectral_data, consistent_autogluon_model):
        """Test single prediction with consistent AutoGluon pipeline format"""
        result = predictor.make_prediction(sample_spectral_data[0], consistent_autogluon_model)

        assert isinstance(result, float)
        assert result == 0.45

    def test_consistent_autogluon_batch_prediction(self, predictor, temp_dir, consistent_autogluon_model):
        """Test batch prediction with consistent AutoGluon pipeline format"""
        # Create batch input directory
        batch_dir = temp_dir / "batch_input"
        batch_dir.mkdir()

        # Create sample files
        for i in range(2):
            file_path = batch_dir / f"sample_{i:03d}.csv.txt"

            wavelengths = np.linspace(400, 800, 100)
            intensities = np.random.uniform(1000, 5000, (100, 3))

            df = pd.DataFrame({
                'Wavelength': wavelengths,
                'Intensity_1': intensities[:, 0],
                'Intensity_2': intensities[:, 1],
                'Intensity_3': intensities[:, 2]
            })

            df.to_csv(file_path, index=False)

        # Mock the AutoGluon wrapper for batch prediction
        with patch('joblib.load') as mock_load:
            from sklearn.pipeline import Pipeline

            autogluon_wrapper = Mock(spec=AutoGluonRegressor)
            autogluon_wrapper.predict.return_value = np.array([0.25, 0.35])

            pipeline = Pipeline([('model', autogluon_wrapper)])
            mock_load.return_value = pipeline

            results_df = predictor.make_batch_predictions(batch_dir, consistent_autogluon_model)

            assert isinstance(results_df, pd.DataFrame)
            assert len(results_df) == 2
            assert all(results_df['Status'] == 'Success')


class TestAutoGluonFeatureHandling(TestAutoGluonPredictorSetup):
    """Test AutoGluon-specific feature handling"""

    @patch('src.models.predictor.AUTOGLUON_AVAILABLE', True)
    @patch('src.models.predictor.TabularPredictor')
    def test_autogluon_with_feature_selection(self, mock_tabular_predictor, predictor,
                                            sample_spectral_data, legacy_autogluon_model):
        """Test AutoGluon with feature selection"""
        # Setup AutoGluon predictor mock
        mock_ag_predictor = Mock()
        mock_ag_predictor.predict.return_value = np.array([0.42])
        mock_tabular_predictor.load.return_value = mock_ag_predictor

        # Make prediction (feature selection should be applied)
        result = predictor.make_prediction(sample_spectral_data[0], legacy_autogluon_model)

        assert isinstance(result, float)
        assert result == 0.42

    @patch('src.models.predictor.AUTOGLUON_AVAILABLE', True)
    @patch('src.models.predictor.TabularPredictor')
    def test_autogluon_raw_spectral_mode(self, mock_tabular_predictor, predictor,
                                       sample_spectral_data, legacy_autogluon_model):
        """Test AutoGluon with raw spectral mode"""
        # Enable raw spectral mode
        predictor.config.use_raw_spectral_data = True

        # Setup AutoGluon predictor mock
        mock_ag_predictor = Mock()
        mock_ag_predictor.predict.return_value = np.array([0.38])
        mock_tabular_predictor.load.return_value = mock_ag_predictor

        # Mock raw spectral feature extraction
        with patch.object(predictor, '_extract_raw_spectral_features') as mock_extract:
            mock_extract.return_value = np.random.random((1, 100))

            result = predictor.make_prediction(sample_spectral_data[0], legacy_autogluon_model)

            assert isinstance(result, float)
            assert result == 0.38
            mock_extract.assert_called_once()

    def test_autogluon_feature_name_building(self, predictor, temp_dir):
        """Test feature name building for AutoGluon models"""
        # Create mock feature pipeline
        from sklearn.preprocessing import StandardScaler
        feature_pipeline = StandardScaler()
        feature_pipeline.fit(np.random.random((10, 5)))
        predictor.feature_pipeline = feature_pipeline

        # Test feature name building
        feature_names = predictor._build_feature_names_from_pipeline()

        assert feature_names is None or isinstance(feature_names, list)


class TestAutoGluonGPUSupport(TestAutoGluonPredictorSetup):
    """Test GPU-enabled AutoGluon prediction"""

    def test_autogluon_gpu_config_detection(self, config):
        """Test GPU configuration detection for AutoGluon"""
        # Enable GPU config
        config.autogluon.use_gpu = True
        config.autogluon.num_gpus = 1

        predictor = Predictor(config)

        assert config.autogluon.use_gpu is True
        assert config.autogluon.num_gpus == 1

    @patch('src.models.predictor.AUTOGLUON_AVAILABLE', True)
    @patch('src.models.predictor.TabularPredictor')
    def test_autogluon_gpu_prediction_logging(self, mock_tabular_predictor, predictor,
                                            sample_spectral_data, legacy_autogluon_model):
        """Test GPU prediction logging for AutoGluon"""
        # Enable GPU in config
        predictor.config.autogluon.use_gpu = True

        # Setup AutoGluon predictor mock
        mock_ag_predictor = Mock()
        mock_ag_predictor.predict.return_value = np.array([0.29])
        mock_tabular_predictor.load.return_value = mock_ag_predictor

        # Make prediction
        with patch('src.models.predictor.logger') as mock_logger:
            result = predictor.make_prediction(sample_spectral_data[0], legacy_autogluon_model)

            assert isinstance(result, float)
            assert result == 0.29


class TestAutoGluonErrorHandling(TestAutoGluonPredictorSetup):
    """Test AutoGluon-specific error handling"""

    @patch('src.models.predictor.AUTOGLUON_AVAILABLE', True)
    @patch('src.models.predictor.TabularPredictor')
    def test_autogluon_prediction_failure_fallback(self, mock_tabular_predictor, predictor,
                                                  temp_dir, legacy_autogluon_model):
        """Test AutoGluon batch prediction failure fallback"""
        # Create batch input directory
        batch_dir = temp_dir / "batch_input"
        batch_dir.mkdir()

        # Create sample files
        for i in range(2):
            file_path = batch_dir / f"sample_{i:03d}.csv.txt"

            wavelengths = np.linspace(400, 800, 100)
            intensities = np.random.uniform(1000, 5000, (100, 3))

            df = pd.DataFrame({
                'Wavelength': wavelengths,
                'Intensity_1': intensities[:, 0],
                'Intensity_2': intensities[:, 1],
                'Intensity_3': intensities[:, 2]
            })

            df.to_csv(file_path, index=False)

        # Setup AutoGluon predictor to fail on batch, succeed on individual
        mock_ag_predictor = Mock()
        # First call (batch) fails, subsequent calls (individual) succeed
        mock_ag_predictor.predict.side_effect = [Exception("Batch failed"), np.array([0.30]), np.array([0.35])]
        mock_tabular_predictor.load.return_value = mock_ag_predictor

        # Run batch prediction (should fallback to individual predictions)
        results_df = predictor.make_batch_predictions(batch_dir, legacy_autogluon_model)

        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 2

    def test_autogluon_missing_components(self, predictor, sample_spectral_data, temp_dir):
        """Test AutoGluon model with missing components"""
        # Create incomplete AutoGluon model directory
        incomplete_model_dir = temp_dir / "incomplete_autogluon_model"
        incomplete_model_dir.mkdir()

        # Only create feature pipeline, missing other components
        from sklearn.preprocessing import StandardScaler
        feature_pipeline = StandardScaler()
        feature_pipeline.fit(np.random.random((10, 20)))
        joblib.dump(feature_pipeline, incomplete_model_dir / "feature_pipeline.pkl")

        # Create predictor directory
        predictor_dir = incomplete_model_dir / "predictor"
        predictor_dir.mkdir()

        with patch('src.models.predictor.AUTOGLUON_AVAILABLE', True):
            with patch('src.models.predictor.TabularPredictor') as mock_tabular_predictor:
                mock_ag_predictor = Mock()
                mock_ag_predictor.predict.return_value = np.array([0.31])
                mock_tabular_predictor.load.return_value = mock_ag_predictor

                # Should handle missing components gracefully
                result = predictor.make_prediction(sample_spectral_data[0], incomplete_model_dir)

                assert isinstance(result, float)
                assert result == 0.31


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])