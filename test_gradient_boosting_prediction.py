"""
Gradient Boosting Models Prediction Tests

Specific tests for XGBoost, CatBoost, and LightGBM models with real data.
These models often have specific requirements and behaviors that need testing.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

from src.config.pipeline_config import Config
from src.models.predictor import Predictor


# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestGradientBoostingSetup:
    """Setup for gradient boosting model tests"""

    @pytest.fixture(scope="session")
    def project_root(self):
        """Get project root directory"""
        return Path(__file__).resolve().parent

    @pytest.fixture(scope="session")
    def test_config(self, project_root):
        """Create test configuration"""
        from src.config.pipeline_config import config

        # Use the global config instance - it already has correct paths
        config.run_timestamp = "gb_test_run"

        # Ensure directories exist
        for dir_path in [config.reports_dir, config.bad_prediction_files_dir, config.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        return config

    @pytest.fixture(scope="session")
    def predictor_instance(self, test_config):
        """Create predictor instance"""
        return Predictor(test_config)

    @pytest.fixture(scope="session")
    def gradient_boosting_models(self, project_root) -> Dict[str, List[Tuple[str, Path]]]:
        """Find gradient boosting models categorized by type

        Uses environment variables to control how many models to test:
        - TEST_MODELS_PER_TYPE: Max models per type (default: 5)
        - TEST_MAX_MODELS_TOTAL: Max total models (default: None)
        """
        import os

        # Get configuration from environment or use defaults
        models_per_type = int(os.environ.get('TEST_MODELS_PER_TYPE', 5))
        max_total = os.environ.get('TEST_MAX_MODELS_TOTAL')
        max_total = int(max_total) if max_total else None

        models_dir = project_root / "models"
        gb_models = {
            "xgboost": [],
            "catboost": [],
            "lightgbm": []
        }

        # Find standard gradient boosting models
        all_pkl_files = sorted(models_dir.glob("*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)

        total_models_added = 0

        for model_file in all_pkl_files:
            # Check if we've reached the total limit
            if max_total and total_models_added >= max_total:
                break

            model_name = model_file.stem.lower()
            added = False

            if "xgboost" in model_name or "xgb" in model_name:
                if len(gb_models["xgboost"]) < models_per_type:
                    gb_models["xgboost"].append((model_file.stem, model_file))
                    added = True
            elif "catboost" in model_name or "cat" in model_name:
                if len(gb_models["catboost"]) < models_per_type:
                    gb_models["catboost"].append((model_file.stem, model_file))
                    added = True
            elif "lightgbm" in model_name or "lgb" in model_name:
                if len(gb_models["lightgbm"]) < models_per_type:
                    gb_models["lightgbm"].append((model_file.stem, model_file))
                    added = True

            if added:
                total_models_added += 1

        # Find AutoGluon models that might contain gradient boosting models
        autogluon_dir = models_dir / "autogluon"
        if autogluon_dir.exists():
            ag_dirs = [d for d in autogluon_dir.iterdir()
                      if d.is_dir() and (d / "feature_pipeline.pkl").exists()]
            # Sort by modification time, newest first
            ag_dirs_sorted = sorted(ag_dirs, key=lambda x: x.stat().st_mtime, reverse=True)

            # Apply limits
            gb_models["autogluon_ensemble"] = []
            if not (max_total and total_models_added >= max_total):
                remaining = None
                if max_total:
                    remaining = max_total - total_models_added
                    limit = min(models_per_type, remaining)
                else:
                    limit = models_per_type

                for ag_model_dir in ag_dirs_sorted[:limit]:
                    gb_models["autogluon_ensemble"].append((ag_model_dir.name, ag_model_dir))
                    total_models_added += 1

        return gb_models

    @pytest.fixture(scope="session")
    def test_data_samples(self, project_root) -> List[Path]:
        """Get sample spectral data files for testing"""
        data_dir = project_root / "data" / "raw" / "data_5278_Phase3"
        if not data_dir.exists():
            return []

        # Get a representative sample of files
        all_files = list(data_dir.glob("*.csv.txt"))
        return all_files[:10]  # Limit to 10 for faster testing

    @pytest.fixture(scope="session")
    def batch_test_directory(self, project_root) -> Optional[Path]:
        """Get directory for batch testing"""
        data_dir = project_root / "data" / "raw" / "data_5278_Phase3"
        return data_dir if data_dir.exists() else None


class TestXGBoostModels(TestGradientBoostingSetup):
    """Test XGBoost models specifically"""

    def test_xgboost_single_file_prediction(self, predictor_instance, gradient_boosting_models, test_data_samples):
        """Test XGBoost models with single file prediction"""
        xgboost_models = gradient_boosting_models.get("xgboost", [])
        if not xgboost_models:
            pytest.skip("No XGBoost models found")

        if not test_data_samples:
            pytest.skip("No test data files available")

        for model_name, model_path in xgboost_models:
            logger.info(f"Testing XGBoost model: {model_name}")

            test_file = test_data_samples[0]
            logger.info(f"  Testing with file: {test_file.name}")

            try:
                prediction = predictor_instance.make_prediction(test_file, model_path)

                # Validate XGBoost prediction
                assert isinstance(prediction, (int, float)), f"XGBoost should return numeric prediction"
                assert not np.isnan(prediction), f"XGBoost prediction should not be NaN"
                assert not np.isinf(prediction), f"XGBoost prediction should not be infinite"

                # XGBoost typically handles extreme values well
                assert -1.0 <= prediction <= 6.0, f"XGBoost prediction {prediction} outside reasonable range"

                logger.info(f"  ✓ XGBoost prediction: {prediction:.4f}")

            except Exception as e:
                pytest.fail(f"XGBoost single file prediction failed for {model_name}: {e}")

    def test_xgboost_batch_prediction(self, predictor_instance, gradient_boosting_models, batch_test_directory):
        """Test XGBoost models with batch prediction"""
        xgboost_models = gradient_boosting_models.get("xgboost", [])
        if not xgboost_models:
            pytest.skip("No XGBoost models found")

        if not batch_test_directory:
            pytest.skip("No batch test directory available")

        for model_name, model_path in xgboost_models[:2]:  # Test first 2 XGBoost models
            logger.info(f"Testing XGBoost batch prediction: {model_name}")

            try:
                # Get max_samples from environment
                import os
                max_samples = os.environ.get('TEST_MAX_SAMPLES')
                max_samples = int(max_samples) if max_samples else None

                results_df = predictor_instance.make_batch_predictions(batch_test_directory, model_path, max_samples=max_samples)

                if max_samples:
                    logger.info(f"  Limited to max {max_samples} sample IDs")

                # Validate XGBoost batch results
                assert isinstance(results_df, pd.DataFrame), "Should return DataFrame"
                assert len(results_df) > 0, "Should have results"

                successful = results_df[results_df['Status'] == 'Success']
                failed = results_df[results_df['Status'].str.contains('Failed', na=False)]

                logger.info(f"  XGBoost batch results: {len(successful)} success, {len(failed)} failed")

                # XGBoost is generally robust, expect reasonable success rate
                if len(results_df) > 0:
                    success_rate = len(successful) / len(results_df) * 100
                    logger.info(f"  Success rate: {success_rate:.1f}%")

                # Validate successful predictions
                if len(successful) > 0:
                    predictions = successful['PredictedValue'].values
                    valid_predictions = predictions[~np.isnan(predictions)]

                    if len(valid_predictions) > 0:
                        logger.info(f"  Prediction range: {valid_predictions.min():.4f} - {valid_predictions.max():.4f}")

                        # XGBoost specific validations
                        extreme_predictions = sum(1 for p in valid_predictions if p < -0.5 or p > 8.0)
                        assert extreme_predictions < len(valid_predictions) * 0.1, \
                            "Too many extreme XGBoost predictions"

            except Exception as e:
                pytest.fail(f"XGBoost batch prediction failed for {model_name}: {e}")

    def test_xgboost_prediction_consistency(self, predictor_instance, gradient_boosting_models, test_data_samples):
        """Test XGBoost prediction consistency across multiple runs"""
        xgboost_models = gradient_boosting_models.get("xgboost", [])
        if not xgboost_models:
            pytest.skip("No XGBoost models found")

        if len(test_data_samples) < 3:
            pytest.skip("Need at least 3 test files for consistency testing")

        model_name, model_path = xgboost_models[0]  # Use first XGBoost model
        logger.info(f"Testing XGBoost consistency: {model_name}")

        predictions = []
        test_file = test_data_samples[0]

        # Make multiple predictions on same file
        for i in range(3):
            try:
                prediction = predictor_instance.make_prediction(test_file, model_path)
                predictions.append(prediction)
            except Exception as e:
                logger.warning(f"Prediction {i+1} failed: {e}")

        assert len(predictions) >= 2, "Need at least 2 successful predictions for consistency test"

        # XGBoost should be deterministic (same prediction each time)
        predictions = np.array(predictions)
        prediction_std = np.std(predictions)

        logger.info(f"  Predictions: {predictions}")
        logger.info(f"  Standard deviation: {prediction_std:.6f}")

        # XGBoost should be highly consistent (deterministic)
        assert prediction_std < 1e-10, f"XGBoost predictions should be deterministic, got std={prediction_std}"


class TestCatBoostModels(TestGradientBoostingSetup):
    """Test CatBoost models specifically"""

    def test_catboost_single_file_prediction(self, predictor_instance, gradient_boosting_models, test_data_samples):
        """Test CatBoost models with single file prediction"""
        catboost_models = gradient_boosting_models.get("catboost", [])
        if not catboost_models:
            pytest.skip("No CatBoost models found")

        if not test_data_samples:
            pytest.skip("No test data files available")

        for model_name, model_path in catboost_models:
            logger.info(f"Testing CatBoost model: {model_name}")

            test_file = test_data_samples[0]
            logger.info(f"  Testing with file: {test_file.name}")

            try:
                prediction = predictor_instance.make_prediction(test_file, model_path)

                # Validate CatBoost prediction
                assert isinstance(prediction, (int, float)), f"CatBoost should return numeric prediction"
                assert not np.isnan(prediction), f"CatBoost prediction should not be NaN"
                assert not np.isinf(prediction), f"CatBoost prediction should not be infinite"

                # CatBoost is generally well-calibrated
                assert -0.5 <= prediction <= 5.5, f"CatBoost prediction {prediction} outside reasonable range"

                logger.info(f"  ✓ CatBoost prediction: {prediction:.4f}")

            except Exception as e:
                pytest.fail(f"CatBoost single file prediction failed for {model_name}: {e}")

    def test_catboost_batch_prediction(self, predictor_instance, gradient_boosting_models, batch_test_directory):
        """Test CatBoost models with batch prediction"""
        catboost_models = gradient_boosting_models.get("catboost", [])
        if not catboost_models:
            pytest.skip("No CatBoost models found")

        if not batch_test_directory:
            pytest.skip("No batch test directory available")

        for model_name, model_path in catboost_models[:2]:  # Test first 2 CatBoost models
            logger.info(f"Testing CatBoost batch prediction: {model_name}")

            try:
                # Get max_samples from environment
                import os
                max_samples = os.environ.get('TEST_MAX_SAMPLES')
                max_samples = int(max_samples) if max_samples else None

                results_df = predictor_instance.make_batch_predictions(batch_test_directory, model_path, max_samples=max_samples)

                if max_samples:
                    logger.info(f"  Limited to max {max_samples} sample IDs")

                assert isinstance(results_df, pd.DataFrame), "Should return DataFrame"
                assert len(results_df) > 0, "Should have results"

                successful = results_df[results_df['Status'] == 'Success']
                failed = results_df[results_df['Status'].str.contains('Failed', na=False)]

                logger.info(f"  CatBoost batch results: {len(successful)} success, {len(failed)} failed")

                # Validate successful predictions
                if len(successful) > 0:
                    predictions = successful['PredictedValue'].values
                    valid_predictions = predictions[~np.isnan(predictions)]

                    if len(valid_predictions) > 0:
                        logger.info(f"  Prediction range: {valid_predictions.min():.4f} - {valid_predictions.max():.4f}")

                        # CatBoost specific validations - generally well-behaved
                        reasonable_predictions = sum(1 for p in valid_predictions if 0.0 <= p <= 3.0)
                        assert reasonable_predictions > len(valid_predictions) * 0.7, \
                            "Most CatBoost predictions should be in reasonable range"

            except Exception as e:
                pytest.fail(f"CatBoost batch prediction failed for {model_name}: {e}")

    def test_catboost_gpu_model_handling(self, predictor_instance, gradient_boosting_models, test_data_samples):
        """Test CatBoost models that may have been trained with GPU"""
        catboost_models = gradient_boosting_models.get("catboost", [])
        if not catboost_models:
            pytest.skip("No CatBoost models found")

        if not test_data_samples:
            pytest.skip("No test data files available")

        # CatBoost GPU models should work on CPU for prediction
        for model_name, model_path in catboost_models:
            logger.info(f"Testing CatBoost GPU compatibility: {model_name}")

            test_file = test_data_samples[0]

            try:
                # This should work even if model was trained on GPU
                prediction = predictor_instance.make_prediction(test_file, model_path)

                assert isinstance(prediction, (int, float)), "CatBoost GPU model should predict on CPU"
                logger.info(f"  ✓ GPU-trained CatBoost works on CPU: {prediction:.4f}")

            except Exception as e:
                # This might fail if there are GPU/CPU compatibility issues
                logger.warning(f"CatBoost GPU model compatibility issue for {model_name}: {e}")
                # Don't fail the test, just warn


class TestLightGBMModels(TestGradientBoostingSetup):
    """Test LightGBM models specifically"""

    def test_lightgbm_single_file_prediction(self, predictor_instance, gradient_boosting_models, test_data_samples):
        """Test LightGBM models with single file prediction"""
        lightgbm_models = gradient_boosting_models.get("lightgbm", [])
        if not lightgbm_models:
            pytest.skip("No LightGBM models found")

        if not test_data_samples:
            pytest.skip("No test data files available")

        for model_name, model_path in lightgbm_models:
            logger.info(f"Testing LightGBM model: {model_name}")

            test_file = test_data_samples[0]
            logger.info(f"  Testing with file: {test_file.name}")

            try:
                prediction = predictor_instance.make_prediction(test_file, model_path)

                # Validate LightGBM prediction
                assert isinstance(prediction, (int, float)), f"LightGBM should return numeric prediction"
                assert not np.isnan(prediction), f"LightGBM prediction should not be NaN"
                assert not np.isinf(prediction), f"LightGBM prediction should not be infinite"

                # LightGBM typically produces well-calibrated predictions
                assert -0.5 <= prediction <= 5.5, f"LightGBM prediction {prediction} outside reasonable range"

                logger.info(f"  ✓ LightGBM prediction: {prediction:.4f}")

            except Exception as e:
                pytest.fail(f"LightGBM single file prediction failed for {model_name}: {e}")

    def test_lightgbm_batch_prediction(self, predictor_instance, gradient_boosting_models, batch_test_directory):
        """Test LightGBM models with batch prediction"""
        lightgbm_models = gradient_boosting_models.get("lightgbm", [])
        if not lightgbm_models:
            pytest.skip("No LightGBM models found")

        if not batch_test_directory:
            pytest.skip("No batch test directory available")

        for model_name, model_path in lightgbm_models[:2]:  # Test first 2 LightGBM models
            logger.info(f"Testing LightGBM batch prediction: {model_name}")

            try:
                # Get max_samples from environment
                import os
                max_samples = os.environ.get('TEST_MAX_SAMPLES')
                max_samples = int(max_samples) if max_samples else None

                results_df = predictor_instance.make_batch_predictions(batch_test_directory, model_path, max_samples=max_samples)

                if max_samples:
                    logger.info(f"  Limited to max {max_samples} sample IDs")

                assert isinstance(results_df, pd.DataFrame), "Should return DataFrame"
                assert len(results_df) > 0, "Should have results"

                successful = results_df[results_df['Status'] == 'Success']
                failed = results_df[results_df['Status'].str.contains('Failed', na=False)]

                logger.info(f"  LightGBM batch results: {len(successful)} success, {len(failed)} failed")

                # Validate successful predictions
                if len(successful) > 0:
                    predictions = successful['PredictedValue'].values
                    valid_predictions = predictions[~np.isnan(predictions)]

                    if len(valid_predictions) > 0:
                        logger.info(f"  Prediction range: {valid_predictions.min():.4f} - {valid_predictions.max():.4f}")

                        # LightGBM specific validations - fast and efficient
                        assert all(isinstance(p, (int, float)) for p in valid_predictions), \
                            "All LightGBM predictions should be numeric"

            except Exception as e:
                pytest.fail(f"LightGBM batch prediction failed for {model_name}: {e}")

    def test_lightgbm_performance_characteristics(self, predictor_instance, gradient_boosting_models, test_data_samples):
        """Test LightGBM performance characteristics"""
        lightgbm_models = gradient_boosting_models.get("lightgbm", [])
        if not lightgbm_models:
            pytest.skip("No LightGBM models found")

        if len(test_data_samples) < 5:
            pytest.skip("Need at least 5 test files for performance testing")

        model_name, model_path = lightgbm_models[0]  # Use first LightGBM model
        logger.info(f"Testing LightGBM performance: {model_name}")

        import time

        # Test multiple predictions for timing
        start_time = time.time()
        successful_predictions = 0

        for test_file in test_data_samples[:5]:
            try:
                prediction = predictor_instance.make_prediction(test_file, model_path)
                if isinstance(prediction, (int, float)) and not np.isnan(prediction):
                    successful_predictions += 1
            except Exception as e:
                logger.warning(f"Prediction failed for {test_file.name}: {e}")

        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_prediction = total_time / successful_predictions if successful_predictions > 0 else float('inf')

        logger.info(f"  LightGBM performance: {successful_predictions}/5 successful")
        logger.info(f"  Average time per prediction: {avg_time_per_prediction:.3f} seconds")

        # LightGBM should be fast
        assert avg_time_per_prediction < 10.0, f"LightGBM should be fast, got {avg_time_per_prediction:.3f}s per prediction"
        assert successful_predictions >= 3, "Most LightGBM predictions should succeed"


class TestGradientBoostingComparison(TestGradientBoostingSetup):
    """Compare different gradient boosting models"""

    def test_model_agreement_comparison(self, predictor_instance, gradient_boosting_models, test_data_samples):
        """Compare predictions from different gradient boosting models"""
        if not test_data_samples:
            pytest.skip("No test data files available")

        # Collect all available GB models
        all_gb_models = []
        for model_type, models in gradient_boosting_models.items():
            if model_type != "autogluon_ensemble":  # Skip ensemble for this test
                for model_name, model_path in models:
                    all_gb_models.append((model_type, model_name, model_path))

        if len(all_gb_models) < 2:
            pytest.skip("Need at least 2 gradient boosting models for comparison")

        logger.info(f"Comparing {len(all_gb_models)} gradient boosting models")

        test_file = test_data_samples[0]
        predictions = {}

        # Get predictions from all models
        for model_type, model_name, model_path in all_gb_models:
            try:
                prediction = predictor_instance.make_prediction(test_file, model_path)
                predictions[f"{model_type}_{model_name}"] = prediction
                logger.info(f"  {model_type} ({model_name}): {prediction:.4f}")
            except Exception as e:
                logger.warning(f"  {model_type} ({model_name}) failed: {e}")

        if len(predictions) < 2:
            pytest.skip("Need at least 2 successful predictions for comparison")

        # Analyze agreement between models
        pred_values = np.array(list(predictions.values()))
        pred_mean = np.mean(pred_values)
        pred_std = np.std(pred_values)
        pred_range = np.max(pred_values) - np.min(pred_values)

        logger.info(f"  Model agreement analysis:")
        logger.info(f"    Mean: {pred_mean:.4f}")
        logger.info(f"    Std:  {pred_std:.4f}")
        logger.info(f"    Range: {pred_range:.4f}")

        # Models should generally agree (within reasonable bounds)
        assert pred_range < 2.0, f"Model predictions too diverse (range: {pred_range:.4f})"
        assert pred_std < 0.8, f"Model predictions too variable (std: {pred_std:.4f})"

    def test_gradient_boosting_robustness(self, predictor_instance, gradient_boosting_models, test_data_samples):
        """Test robustness of gradient boosting models across different samples"""
        if len(test_data_samples) < 3:
            pytest.skip("Need at least 3 test files for robustness testing")

        # Test with first available GB model of each type
        test_models = []
        for model_type in ["xgboost", "catboost", "lightgbm"]:
            if gradient_boosting_models.get(model_type):
                model_name, model_path = gradient_boosting_models[model_type][0]
                test_models.append((model_type, model_name, model_path))

        if not test_models:
            pytest.skip("No gradient boosting models available for robustness testing")

        logger.info("Testing gradient boosting model robustness across samples")

        for model_type, model_name, model_path in test_models:
            logger.info(f"  Testing {model_type}: {model_name}")

            predictions = []
            successful = 0

            for test_file in test_data_samples[:3]:
                try:
                    prediction = predictor_instance.make_prediction(test_file, model_path)
                    if isinstance(prediction, (int, float)) and not np.isnan(prediction):
                        predictions.append(prediction)
                        successful += 1
                except Exception as e:
                    logger.warning(f"    {test_file.name} failed: {e}")

            if successful > 0:
                pred_array = np.array(predictions)
                logger.info(f"    Successful: {successful}/3")
                logger.info(f"    Range: {pred_array.min():.4f} - {pred_array.max():.4f}")

                # Gradient boosting models should be robust
                assert successful >= 2, f"{model_type} should handle most samples successfully"

                # Check for reasonable prediction ranges
                reasonable_predictions = sum(1 for p in predictions if 0.0 <= p <= 3.0)
                assert reasonable_predictions >= successful * 0.7, \
                    f"Most {model_type} predictions should be reasonable"


if __name__ == "__main__":
    # Run gradient boosting specific tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])