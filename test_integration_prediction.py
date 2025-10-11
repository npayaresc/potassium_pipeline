"""
Integration tests for end-to-end prediction workflows

Tests the complete prediction pipeline from raw spectral files to final predictions,
including data loading, cleaning, feature engineering, model loading, and prediction.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, Mock
import tempfile
import shutil

from src.config.pipeline_config import Config
from src.models.predictor import Predictor
from test_utils import (
    SpectralDataGenerator, ModelMockFactory, ConfigFactory, AssertionHelpers,
    TestDataSets, TestEnvironment
)


class TestEndToEndPredictionWorkflow:
    """Integration tests for complete prediction workflows"""

    def test_complete_single_prediction_workflow(self):
        """Test complete single file prediction workflow"""
        with TestEnvironment() as (temp_dir, config):
            # 1. Create test data
            sample_file = SpectralDataGenerator.create_spectral_file(
                temp_dir / "test_sample.csv.txt",
                potassium_concentration=0.30
            )

            # 2. Create mock model
            mock_model_path = temp_dir / "test_model.pkl"
            mock_model = ModelMockFactory.create_sklearn_pipeline_mock(
                model_type="Ridge", prediction_value=0.28
            )

            import joblib
            joblib.dump(mock_model, mock_model_path)

            # 3. Create predictor and make prediction
            predictor = Predictor(config)
            result = predictor.make_prediction(sample_file, mock_model_path)

            # 4. Verify results
            AssertionHelpers.assert_valid_prediction(result, expected_range=(0.0, 2.0))

    def test_complete_batch_prediction_workflow(self):
        """Test complete batch prediction workflow"""
        with TestEnvironment() as (temp_dir, config):
            # 1. Create batch test data
            batch_dir = temp_dir / "batch_input"
            batch_dir.mkdir()

            sample_concentrations = TestDataSets.get_mixed_concentration_samples()
            files_by_sample = {}

            for sample_id, concentration in sample_concentrations.items():
                sample_files = []
                for i in range(2):  # 2 files per sample
                    file_path = SpectralDataGenerator.create_spectral_file(
                        batch_dir / f"{sample_id}_{i:02d}.csv.txt",
                        potassium_concentration=concentration
                    )
                    sample_files.append(file_path)
                files_by_sample[sample_id] = sample_files

            # 2. Create mock model
            mock_model_path = temp_dir / "batch_test_model.pkl"
            expected_predictions = [0.20, 0.30, 0.40, 0.25, 0.35, 0.45, 0.60, 0.80, 1.20]

            mock_model = Mock()
            mock_model.predict.return_value = np.array(expected_predictions)
            mock_model.named_steps = {'model': Mock()}

            import joblib
            joblib.dump(mock_model, mock_model_path)

            # 3. Run batch prediction
            predictor = Predictor(config)
            results_df = predictor.make_batch_predictions(batch_dir, mock_model_path)

            # 4. Verify results
            AssertionHelpers.assert_valid_batch_results(
                results_df,
                expected_samples=len(sample_concentrations)
            )

    def test_end_to_end_with_data_cleaning_failures(self):
        """Test end-to-end workflow with some samples failing data cleaning"""
        with TestEnvironment() as (temp_dir, config):
            # 1. Create batch test data
            batch_dir = temp_dir / "batch_with_failures"
            batch_dir.mkdir()

            # Create good and bad samples
            good_samples = ['good_001', 'good_002']
            bad_samples = ['bad_001', 'bad_002']

            all_samples = good_samples + bad_samples
            for sample_id in all_samples:
                SpectralDataGenerator.create_spectral_file(
                    batch_dir / f"{sample_id}.csv.txt",
                    potassium_concentration=0.25
                )

            # 2. Create mock model
            mock_model_path = temp_dir / "model_with_failures.pkl"
            mock_model = ModelMockFactory.create_sklearn_pipeline_mock(
                prediction_value=0.25
            )
            import joblib
            joblib.dump(mock_model, mock_model_path)

            # 3. Mock data cleaner to simulate failures for bad samples
            predictor = Predictor(config)

            def mock_clean_spectra(sample_id, intensities):
                if 'bad' in sample_id:
                    return np.array([])  # Empty array = cleaning failure
                else:
                    return intensities.mean(axis=1)  # Simple averaging for good samples

            with patch.object(predictor.data_cleanser, 'clean_spectra', side_effect=mock_clean_spectra):
                results_df = predictor.make_batch_predictions(batch_dir, mock_model_path)

                # 4. Verify mixed results
                successful = results_df[results_df['Status'] == 'Success']
                failed = results_df[results_df['Status'].str.contains('Failed')]

                assert len(successful) == len(good_samples)
                assert len(failed) == len(bad_samples)

    def test_end_to_end_with_wavelength_standardization(self):
        """Test end-to-end workflow with wavelength standardization enabled"""
        with TestEnvironment(enable_wavelength_standardization=True) as (temp_dir, config):
            # 1. Create test data with non-standard wavelength grid
            sample_file = temp_dir / "non_standard_wavelengths.csv.txt"

            # Create irregular wavelength grid
            wavelengths = np.concatenate([
                np.linspace(400, 500, 50),
                np.linspace(502, 600, 30),  # Gap and different spacing
                np.linspace(605, 800, 70)
            ])

            intensities = np.random.uniform(1000, 5000, (len(wavelengths), 3))

            df = pd.DataFrame({
                'Wavelength': wavelengths,
                'Intensity_1': intensities[:, 0],
                'Intensity_2': intensities[:, 1],
                'Intensity_3': intensities[:, 2]
            })
            df.to_csv(sample_file, index=False)

            # 2. Create mock model with wavelength metadata
            mock_model_path = temp_dir / "wavelength_aware_model.pkl"
            mock_model = ModelMockFactory.create_sklearn_pipeline_mock(prediction_value=0.32)

            # Create wavelength metadata
            import json
            metadata = {
                'global_wavelength_range': {'min': 400.0, 'max': 800.0},
                'wavelength_standardization_enabled': True,
                'interpolation_method': 'linear',
                'wavelength_resolution': 1.0
            }
            metadata_path = mock_model_path.with_suffix('.wavelength_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            import joblib
            joblib.dump(mock_model, mock_model_path)

            # 3. Run prediction
            predictor = Predictor(config)
            result = predictor.make_prediction(sample_file, mock_model_path)

            # 4. Verify wavelength standardization was applied
            AssertionHelpers.assert_valid_prediction(result)
            assert config.enable_wavelength_standardization is True

    def test_end_to_end_autogluon_legacy_format(self):
        """Test end-to-end workflow with legacy AutoGluon model format"""
        with TestEnvironment() as (temp_dir, config):
            # 1. Create test data
            sample_file = SpectralDataGenerator.create_spectral_file(
                temp_dir / "autogluon_test.csv.txt",
                potassium_concentration=0.35
            )

            # 2. Create legacy AutoGluon model directory
            autogluon_model_dir = ModelMockFactory.create_autogluon_model_directory(
                temp_dir,
                model_name="legacy_autogluon_test",
                include_calibration=True
            )

            # 3. Mock AutoGluon prediction
            with patch('src.models.predictor.AUTOGLUON_AVAILABLE', True):
                with patch('src.models.predictor.TabularPredictor') as mock_tabular:
                    mock_ag_predictor = Mock()
                    mock_ag_predictor.predict.return_value = np.array([0.33])
                    mock_tabular.load.return_value = mock_ag_predictor

                    # 4. Run prediction
                    predictor = Predictor(config)
                    result = predictor.make_prediction(sample_file, autogluon_model_dir)

                    # 5. Verify results
                    AssertionHelpers.assert_valid_prediction(result)
                    assert result == 0.33

    def test_end_to_end_with_raw_spectral_mode(self):
        """Test end-to-end workflow with raw spectral mode enabled"""
        with TestEnvironment(use_raw_spectral_data=True) as (temp_dir, config):
            # 1. Create test data focused on potassium regions
            sample_file = SpectralDataGenerator.create_spectral_file(
                temp_dir / "raw_spectral_test.csv.txt",
                wavelength_range=(760, 780),  # Focus on K line region
                potassium_concentration=0.40
            )

            # 2. Create mock model for raw spectral mode
            mock_model_path = temp_dir / "raw_spectral_model.pkl"
            mock_model = ModelMockFactory.create_sklearn_pipeline_mock(prediction_value=0.38)
            import joblib
            joblib.dump(mock_model, mock_model_path)

            # 3. Mock raw spectral feature extraction
            predictor = Predictor(config)

            with patch.object(predictor, '_extract_raw_spectral_features') as mock_extract:
                mock_extract.return_value = np.random.random((1, 50))

                # 4. Run prediction
                result = predictor.make_prediction(sample_file, mock_model_path)

                # 5. Verify raw spectral extraction was used
                AssertionHelpers.assert_valid_prediction(result)
                mock_extract.assert_called_once()

    def test_end_to_end_error_recovery(self):
        """Test end-to-end workflow with error recovery mechanisms"""
        with TestEnvironment() as (temp_dir, config):
            # 1. Create test data
            batch_dir = temp_dir / "error_recovery_test"
            batch_dir.mkdir()

            samples = ['sample_001', 'sample_002', 'sample_003']
            for sample_id in samples:
                SpectralDataGenerator.create_spectral_file(
                    batch_dir / f"{sample_id}.csv.txt",
                    potassium_concentration=0.30
                )

            # 2. Create mock model
            mock_model_path = temp_dir / "error_recovery_model.pkl"
            mock_model = Mock()

            # Mock to fail on batch prediction but succeed on individual
            mock_model.predict.side_effect = [
                Exception("Batch prediction failed"),  # First call fails
                np.array([0.25]),  # Individual predictions succeed
                np.array([0.30]),
                np.array([0.35])
            ]
            mock_model.named_steps = {'model': Mock()}

            import joblib
            joblib.dump(mock_model, mock_model_path)

            # 3. Run batch prediction (should trigger fallback)
            predictor = Predictor(config)
            results_df = predictor.make_batch_predictions(batch_dir, mock_model_path)

            # 4. Verify fallback recovery worked
            AssertionHelpers.assert_valid_batch_results(results_df, len(samples))
            # Should have some successful predictions from fallback
            successful = results_df[results_df['Status'] == 'Success']
            assert len(successful) > 0

    def test_end_to_end_performance_with_large_batch(self):
        """Test end-to-end performance with larger batch sizes"""
        with TestEnvironment() as (temp_dir, config):
            # 1. Create larger batch dataset
            batch_dir = temp_dir / "large_batch_test"
            batch_dir.mkdir()

            n_samples = 20  # Moderate size for testing
            sample_concentrations = np.random.uniform(0.1, 1.0, n_samples)

            for i, concentration in enumerate(sample_concentrations):
                sample_id = f"large_batch_sample_{i:03d}"
                SpectralDataGenerator.create_spectral_file(
                    batch_dir / f"{sample_id}.csv.txt",
                    potassium_concentration=concentration
                )

            # 2. Create mock model
            mock_model_path = temp_dir / "large_batch_model.pkl"
            expected_predictions = np.random.uniform(0.15, 0.95, n_samples)

            mock_model = Mock()
            mock_model.predict.return_value = expected_predictions
            mock_model.named_steps = {'model': Mock()}

            import joblib
            joblib.dump(mock_model, mock_model_path)

            # 3. Run batch prediction
            predictor = Predictor(config)

            import time
            start_time = time.time()
            results_df = predictor.make_batch_predictions(batch_dir, mock_model_path)
            end_time = time.time()

            # 4. Verify performance and results
            processing_time = end_time - start_time
            print(f"Processed {n_samples} samples in {processing_time:.2f} seconds")

            AssertionHelpers.assert_valid_batch_results(results_df, n_samples)
            assert processing_time < 30  # Should complete within 30 seconds

    def test_end_to_end_calibration_workflow(self):
        """Test end-to-end workflow with model calibration"""
        with TestEnvironment(use_sample_weights=True) as (temp_dir, config):
            # 1. Create test data
            sample_file = SpectralDataGenerator.create_spectral_file(
                temp_dir / "calibration_test.csv.txt",
                potassium_concentration=0.28
            )

            # 2. Create calibrated model mock
            from src.models.predictor import CalibratedModelWrapper
            base_model = Mock()
            base_model.predict.return_value = np.array([0.30])

            calibrator = Mock()
            calibrator.transform.return_value = np.array([0.27])  # Calibrated value

            calibrated_model = CalibratedModelWrapper(base_model, calibrator)

            from sklearn.pipeline import Pipeline
            pipeline = Pipeline([('model', calibrated_model)])

            mock_model_path = temp_dir / "calibrated_model.pkl"
            import joblib
            joblib.dump(pipeline, mock_model_path)

            # 3. Run prediction
            predictor = Predictor(config)

            with patch('src.models.predictor.logger') as mock_logger:
                result = predictor.make_prediction(sample_file, mock_model_path)

                # 4. Verify calibration was detected and applied
                AssertionHelpers.assert_valid_prediction(result)

                # Check that calibration logging occurred
                calibration_logged = any(
                    'calibrated' in str(call)
                    for call in mock_logger.info.call_args_list
                )
                assert calibration_logged, "Calibration should be logged"

    def test_end_to_end_gpu_workflow_simulation(self):
        """Test end-to-end workflow with GPU configuration (simulated)"""
        with TestEnvironment(enable_gpu=True) as (temp_dir, config):
            # 1. Verify GPU configuration
            assert config.autogluon.use_gpu is True
            assert config.autogluon.num_gpus == 1

            # 2. Create test data
            sample_file = SpectralDataGenerator.create_spectral_file(
                temp_dir / "gpu_test.csv.txt",
                potassium_concentration=0.42
            )

            # 3. Create mock GPU-enabled model
            mock_model_path = temp_dir / "gpu_model.pkl"
            mock_model = ModelMockFactory.create_sklearn_pipeline_mock(prediction_value=0.41)
            import joblib
            joblib.dump(mock_model, mock_model_path)

            # 4. Run prediction
            predictor = Predictor(config)
            result = predictor.make_prediction(sample_file, mock_model_path)

            # 5. Verify GPU-aware prediction
            AssertionHelpers.assert_valid_prediction(result)

    def test_comprehensive_integration_scenario(self):
        """Comprehensive integration test combining multiple features"""
        with TestEnvironment(
            enable_wavelength_standardization=True,
            use_sample_weights=True
        ) as (temp_dir, config):

            # 1. Create comprehensive test dataset
            batch_dir = temp_dir / "comprehensive_test"
            batch_dir.mkdir()

            # Mix of sample types and concentrations
            test_scenarios = {
                'low_conc_001': 0.05,
                'medium_conc_001': 0.35,
                'high_conc_001': 0.85,
                'outlier_candidate': 0.25  # Will be marked as outlier
            }

            for sample_id, concentration in test_scenarios.items():
                SpectralDataGenerator.create_spectral_file(
                    batch_dir / f"{sample_id}.csv.txt",
                    potassium_concentration=concentration,
                    wavelength_range=(390, 810),  # Slightly extended range
                    n_points=250  # Higher resolution
                )

            # 2. Create sophisticated mock model with calibration
            mock_model_path = temp_dir / "comprehensive_model.pkl"

            # Predictions roughly matching input concentrations with some calibration effect
            expected_predictions = [0.06, 0.33, 0.82]  # Excluding outlier

            mock_model = Mock()
            mock_model.predict.return_value = np.array(expected_predictions)
            mock_model.named_steps = {'model': Mock()}

            import joblib
            joblib.dump(mock_model, mock_model_path)

            # Add wavelength metadata
            import json
            metadata = {
                'global_wavelength_range': {'min': 390.0, 'max': 810.0},
                'wavelength_standardization_enabled': True,
                'interpolation_method': 'cubic',
                'wavelength_resolution': 0.5
            }
            metadata_path = mock_model_path.with_suffix('.wavelength_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            # 3. Setup predictor with selective sample failures
            predictor = Predictor(config)

            def selective_clean_spectra(sample_id, intensities):
                if 'outlier_candidate' in sample_id:
                    return np.array([])  # Mark as outlier
                else:
                    return intensities.mean(axis=1)

            # 4. Run comprehensive prediction
            with patch.object(predictor.data_cleanser, 'clean_spectra', side_effect=selective_clean_spectra):
                results_df = predictor.make_batch_predictions(batch_dir, mock_model_path)

                # 5. Comprehensive verification
                AssertionHelpers.assert_valid_batch_results(results_df, len(test_scenarios))

                # Check mixed success/failure results
                successful = results_df[results_df['Status'] == 'Success']
                failed = results_df[results_df['Status'].str.contains('Failed')]

                assert len(successful) == 3  # 3 should succeed
                assert len(failed) == 1     # 1 should fail (outlier)

                # Verify prediction quality for successful samples
                for _, row in successful.iterrows():
                    AssertionHelpers.assert_valid_prediction(
                        row['PredictedValue'],
                        expected_range=(0.0, 2.0)
                    )


class TestIntegrationEdgeCases:
    """Integration tests for edge cases and boundary conditions"""

    def test_empty_directory_integration(self):
        """Test integration with empty input directory"""
        with TestEnvironment() as (temp_dir, config):
            empty_dir = temp_dir / "empty_input"
            empty_dir.mkdir()

            mock_model_path = temp_dir / "empty_test_model.pkl"
            mock_model = ModelMockFactory.create_sklearn_pipeline_mock()
            import joblib
            joblib.dump(mock_model, mock_model_path)

            predictor = Predictor(config)
            results_df = predictor.make_batch_predictions(empty_dir, mock_model_path)

            assert isinstance(results_df, pd.DataFrame)
            assert len(results_df) == 0

    def test_corrupted_spectral_files_integration(self):
        """Test integration with corrupted spectral files"""
        with TestEnvironment() as (temp_dir, config):
            # Create directory with mix of good and corrupted files
            mixed_dir = temp_dir / "mixed_quality"
            mixed_dir.mkdir()

            # Good file
            SpectralDataGenerator.create_spectral_file(
                mixed_dir / "good_sample.csv.txt",
                potassium_concentration=0.30
            )

            # Corrupted file (invalid CSV)
            with open(mixed_dir / "corrupted_sample.csv.txt", 'w') as f:
                f.write("invalid,csv,content\nwith,wrong,format")

            # Empty file
            (mixed_dir / "empty_sample.csv.txt").touch()

            mock_model_path = temp_dir / "mixed_test_model.pkl"
            mock_model = ModelMockFactory.create_sklearn_pipeline_mock(prediction_value=0.29)
            import joblib
            joblib.dump(mock_model, mock_model_path)

            predictor = Predictor(config)
            results_df = predictor.make_batch_predictions(mixed_dir, mock_model_path)

            # Should handle errors gracefully
            assert isinstance(results_df, pd.DataFrame)
            # At least one sample should succeed (the good one)
            successful = results_df[results_df['Status'] == 'Success']
            assert len(successful) >= 1


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])  # -s to show print outputs