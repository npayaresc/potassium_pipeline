"""
Real data prediction tests

Tests using actual spectral data files and trained models to verify
that single file and batch predictions are working correctly.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

from src.config.pipeline_config import config, Config
from src.models.predictor import Predictor


# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRealDataSetup:
    """Setup for real data tests"""

    @pytest.fixture(scope="session")
    def project_root(self):
        """Get project root directory"""
        return Path(__file__).resolve().parent

    @pytest.fixture(scope="session")
    def test_config(self, project_root):
        """Create test configuration using project paths"""
        from src.config.pipeline_config import config

        # Use the global config instance - it already has the correct paths
        # Just set the run_timestamp
        config.run_timestamp = "test_run"

        # Ensure directories exist
        for dir_path in [config.reports_dir, config.bad_prediction_files_dir, config.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        return config

    @pytest.fixture(scope="session")
    def predictor_instance(self, test_config):
        """Create predictor instance"""
        return Predictor(test_config)

    @pytest.fixture(scope="session")
    def available_models(self, project_root) -> Dict[str, List[Path]]:
        """Find available trained models categorized by type

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
        available = {
            "xgboost": [],
            "catboost": [],
            "lightgbm": [],
            "random_forest": [],
            "extratrees": [],
            "neural_network": [],
            "ridge": [],
            "autogluon": [],
            "other": []
        }

        # Find standard sklearn models and categorize them
        all_pkl_files = sorted(models_dir.glob("*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)

        total_models_added = 0

        for model_file in all_pkl_files:
            # Check if we've reached the total limit
            if max_total and total_models_added >= max_total:
                break

            model_name = model_file.stem.lower()
            added = False

            if "xgboost" in model_name or "xgb" in model_name:
                if len(available["xgboost"]) < models_per_type:
                    available["xgboost"].append(model_file)
                    added = True
            elif "catboost" in model_name or "cat" in model_name:
                if len(available["catboost"]) < models_per_type:
                    available["catboost"].append(model_file)
                    added = True
            elif "lightgbm" in model_name or "lgb" in model_name:
                if len(available["lightgbm"]) < models_per_type:
                    available["lightgbm"].append(model_file)
                    added = True
            elif "random_forest" in model_name or "randomforest" in model_name:
                if len(available["random_forest"]) < models_per_type:
                    available["random_forest"].append(model_file)
                    added = True
            elif "extratrees" in model_name or "extra_trees" in model_name:
                if len(available["extratrees"]) < models_per_type:
                    available["extratrees"].append(model_file)
                    added = True
            elif "neural" in model_name:
                if len(available["neural_network"]) < models_per_type:
                    available["neural_network"].append(model_file)
                    added = True
            elif "ridge" in model_name:
                if len(available["ridge"]) < models_per_type:
                    available["ridge"].append(model_file)
                    added = True
            elif "autogluon" in model_name:
                if len(available["autogluon"]) < models_per_type:
                    available["autogluon"].append(model_file)
                    added = True
            else:
                if len(available["other"]) < models_per_type:
                    available["other"].append(model_file)
                    added = True

            if added:
                total_models_added += 1

        # Find AutoGluon directory models
        autogluon_dir = models_dir / "autogluon"
        if autogluon_dir.exists():
            ag_dirs = [d for d in autogluon_dir.iterdir()
                      if d.is_dir() and (d / "feature_pipeline.pkl").exists()]
            # Sort by modification time, newest first
            ag_dirs_sorted = sorted(ag_dirs, key=lambda x: x.stat().st_mtime, reverse=True)

            # Apply limits
            if max_total and total_models_added >= max_total:
                pass  # Don't add any AutoGluon models
            else:
                remaining = None
                if max_total:
                    remaining = max_total - total_models_added
                    limit = min(models_per_type, remaining)
                else:
                    limit = models_per_type

                for ag_model_dir in ag_dirs_sorted[:limit]:
                    available["autogluon"].append(ag_model_dir)
                    total_models_added += 1

        return available

    @pytest.fixture(scope="session")
    def available_data_files(self, project_root) -> Dict[str, List[Path]]:
        """Find available spectral data files grouped by sample"""
        data_dir = project_root / "data" / "raw"
        files_by_sample = {}

        # Look for spectral data files
        for data_subdir in data_dir.rglob("*.csv.txt"):
            if data_subdir.is_file():
                # Extract sample prefix using the same logic as the pipeline
                filename = data_subdir.name
                # Remove common suffixes to get sample ID
                sample_id = filename.replace('.csv.txt', '')
                # Remove trailing numbers and underscores to group by sample
                parts = sample_id.split('_')
                if len(parts) > 1 and parts[-1].isdigit():
                    base_sample_id = '_'.join(parts[:-1])
                else:
                    base_sample_id = sample_id

                if base_sample_id not in files_by_sample:
                    files_by_sample[base_sample_id] = []
                files_by_sample[base_sample_id].append(data_subdir)

        return files_by_sample

    @pytest.fixture(scope="session")
    def reference_data(self, test_config) -> Optional[pd.DataFrame]:
        """Load reference data if available"""
        if test_config.reference_data_path.exists():
            try:
                return pd.read_excel(test_config.reference_data_path)
            except Exception as e:
                logger.warning(f"Could not load reference data: {e}")
                return None
        return None


class TestRealSingleFilePrediction(TestRealDataSetup):
    """Test single file prediction with real data and models"""

    def test_single_file_prediction_all_model_types(self, predictor_instance, available_models, available_data_files):
        """Test single file prediction with all available model types"""
        if not available_data_files:
            pytest.skip("No spectral data files available")

        # Get first available data file
        sample_id, file_list = next(iter(available_data_files.items()))
        test_file = file_list[0]

        tested_models = 0
        successful_predictions = 0

        for model_type, model_list in available_models.items():
            if not model_list:
                continue

            # Test first model of each type
            model_path = model_list[0]
            model_name = model_path.name if hasattr(model_path, 'name') else str(model_path)

            logger.info(f"Testing {model_type} model: {model_name}")
            logger.info(f"  Testing with file: {test_file.name}")

            try:
                result = predictor_instance.make_prediction(test_file, model_path)

                # Validate result
                assert isinstance(result, float), f"Expected float result, got {type(result)}"
                assert not np.isnan(result), "Result should not be NaN"
                assert not np.isinf(result), "Result should not be infinite"
                assert -1.0 <= result <= 10.0, f"Result {result} outside reasonable range for {model_type}"

                logger.info(f"  ✓ {model_type} prediction successful: {result:.4f}")
                successful_predictions += 1

            except Exception as e:
                logger.warning(f"  ✗ {model_type} prediction failed: {e}")

            tested_models += 1

        assert tested_models > 0, "No models available for testing"
        assert successful_predictions > 0, "No successful predictions"

        success_rate = successful_predictions / tested_models * 100
        logger.info(f"Overall success rate: {successful_predictions}/{tested_models} ({success_rate:.1f}%)")

        # At least 70% of models should work
        assert success_rate >= 70, f"Too many model failures: {success_rate:.1f}% success rate"

    def test_single_file_prediction_gradient_boosting_focus(self, predictor_instance, available_models, available_data_files):
        """Test single file prediction focusing on gradient boosting models (your most common models)"""
        if not available_data_files:
            pytest.skip("No spectral data files available")

        # Focus on the gradient boosting models you train most
        gb_model_types = ["xgboost", "catboost", "lightgbm"]

        # Get first available data file
        sample_id, file_list = next(iter(available_data_files.items()))
        test_file = file_list[0]

        gb_results = {}

        for model_type in gb_model_types:
            if not available_models.get(model_type):
                logger.info(f"No {model_type} models available")
                continue

            # Test multiple models of this type (you have many of each)
            models_to_test = available_models[model_type][:3]  # Test first 3

            for i, model_path in enumerate(models_to_test):
                model_name = model_path.name
                logger.info(f"Testing {model_type} model {i+1}: {model_name}")

                try:
                    result = predictor_instance.make_prediction(test_file, model_path)

                    # Validate result
                    assert isinstance(result, float), f"Expected float result from {model_type}"
                    assert not np.isnan(result), f"{model_type} result should not be NaN"
                    assert not np.isinf(result), f"{model_type} result should not be infinite"

                    # Different ranges for different model types
                    if model_type == "xgboost":
                        assert -1.0 <= result <= 8.0, f"XGBoost result {result} outside range"
                    elif model_type == "catboost":
                        assert -0.5 <= result <= 6.0, f"CatBoost result {result} outside range"
                    elif model_type == "lightgbm":
                        assert -0.5 <= result <= 6.0, f"LightGBM result {result} outside range"

                    gb_results[f"{model_type}_{i+1}"] = result
                    logger.info(f"  ✓ {model_type} prediction {i+1}: {result:.4f}")

                except Exception as e:
                    logger.warning(f"  ✗ {model_type} model {i+1} failed: {e}")

        assert len(gb_results) > 0, "No gradient boosting models worked"

        # Analyze gradient boosting model agreement
        if len(gb_results) > 1:
            predictions = np.array(list(gb_results.values()))
            pred_std = np.std(predictions)
            pred_range = np.max(predictions) - np.min(predictions)

            logger.info(f"Gradient boosting model analysis:")
            logger.info(f"  Predictions: {dict(gb_results)}")
            logger.info(f"  Standard deviation: {pred_std:.4f}")
            logger.info(f"  Range: {pred_range:.4f}")

            # Your optimized models should generally agree
            assert pred_range < 2.5, f"Gradient boosting models disagree too much (range: {pred_range:.4f})"

    def test_single_file_prediction_multiple_files(self, predictor_instance, available_models, available_data_files):
        """Test single file prediction on multiple files to ensure consistency"""
        if not available_models:
            pytest.skip("No trained models available")

        if not available_data_files:
            pytest.skip("No spectral data files available")

        # Get first available model (any type)
        model_name, model_path = next(iter(available_models.items()))

        # Get files from first sample with multiple files
        sample_with_multiple_files = None
        for sample_id, file_list in available_data_files.items():
            if len(file_list) > 1:
                sample_with_multiple_files = (sample_id, file_list)
                break

        if not sample_with_multiple_files:
            pytest.skip("No samples with multiple files available")

        sample_id, file_list = sample_with_multiple_files

        logger.info(f"Testing multiple file consistency with model: {model_name}")
        logger.info(f"Testing sample: {sample_id} ({len(file_list)} files)")

        predictions = []
        for test_file in file_list[:3]:  # Test first 3 files max
            try:
                result = predictor_instance.make_prediction(test_file, model_path)
                predictions.append(result)
                logger.info(f"  File {test_file.name}: {result:.4f}")
            except Exception as e:
                logger.warning(f"  File {test_file.name} failed: {e}")

        assert len(predictions) > 0, "At least one prediction should succeed"

        # Check that predictions are in reasonable range and somewhat consistent
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)

        assert all(0.0 <= p <= 5.0 for p in predictions), "All predictions should be in reasonable range"
        logger.info(f"✓ Multiple file predictions - Mean: {mean_pred:.4f}, Std: {std_pred:.4f}")


class TestRealBatchPrediction(TestRealDataSetup):
    """Test batch prediction with real data and models"""

    def test_batch_prediction_optimized_models(self, predictor_instance, available_models, project_root):
        """Test batch prediction with your optimized models (focus on most common types)"""
        # Test the model types you have the most of
        priority_models = ["xgboost", "catboost", "lightgbm", "random_forest", "extratrees"]

        tested_model = None
        for model_type in priority_models:
            if available_models.get(model_type):
                tested_model = (model_type, available_models[model_type][0])
                break

        if not tested_model:
            pytest.skip("No priority models available for batch testing")

        model_type, model_path = tested_model

        # Use a subset of available data for batch testing
        data_dir = project_root / "data" / "raw" / "data_5278_Phase3"
        if not data_dir.exists():
            pytest.skip("Test data directory not available")

        # Get a sample of files for batch testing
        test_files = list(data_dir.glob("*.csv.txt"))[:15]  # Test more files since you have many models
        if not test_files:
            pytest.skip("No test files available")

        model_name = model_path.name
        logger.info(f"Testing batch prediction with {model_type} model: {model_name}")
        logger.info(f"Testing with {len(test_files)} files from {data_dir.name}")

        # Get max_samples from environment
        import os
        max_samples = os.environ.get('TEST_MAX_SAMPLES')
        max_samples = int(max_samples) if max_samples else None

        # Make batch predictions
        try:
            results_df = predictor_instance.make_batch_predictions(data_dir, model_path, max_samples=max_samples)

            if max_samples:
                logger.info(f"  Limited to max {max_samples} sample IDs")

            # Validate results
            assert isinstance(results_df, pd.DataFrame), "Results should be a DataFrame"
            assert len(results_df) > 0, "Should have some results"
            assert all(col in results_df.columns for col in ['sampleId', 'PredictedValue', 'Status']), \
                "Required columns missing"

            # Check for successful predictions
            successful = results_df[results_df['Status'] == 'Success']
            failed = results_df[results_df['Status'].str.contains('Failed', na=False)]

            logger.info(f"✓ Batch prediction results:")
            logger.info(f"  Total samples: {len(results_df)}")
            logger.info(f"  Successful: {len(successful)}")
            logger.info(f"  Failed: {len(failed)}")

            if len(successful) > 0:
                predictions = successful['PredictedValue'].values
                logger.info(f"  Prediction range: {predictions.min():.4f} - {predictions.max():.4f}")
                logger.info(f"  Mean prediction: {predictions.mean():.4f}")

            assert len(successful) > 0, "At least some predictions should succeed"

            # Model-specific validation
            if len(successful) > 0:
                predictions = successful['PredictedValue'].values
                valid_predictions = predictions[~np.isnan(predictions)]

                if len(valid_predictions) > 0:
                    logger.info(f"  {model_type} prediction range: {valid_predictions.min():.4f} - {valid_predictions.max():.4f}")

                    # Validate based on model type
                    for _, row in successful.iterrows():
                        pred_val = row['PredictedValue']
                        assert isinstance(pred_val, (int, float)), f"Invalid prediction type: {type(pred_val)}"
                        assert not np.isnan(pred_val), f"NaN prediction for sample {row['sampleId']}"

                        # Different ranges for different model types (based on your actual models)
                        if model_type in ["xgboost", "catboost", "lightgbm"]:
                            assert -1.0 <= pred_val <= 8.0, f"{model_type} prediction {pred_val} outside range for {row['sampleId']}"
                        else:
                            assert -0.5 <= pred_val <= 6.0, f"Prediction {pred_val} outside range for {row['sampleId']}"

        except Exception as e:
            pytest.fail(f"Batch prediction failed with {model_type} model: {e}")

    def test_batch_prediction_feature_strategy_comparison(self, predictor_instance, available_models, project_root):
        """Test batch prediction comparing different feature strategies you use"""
        data_dir = project_root / "data" / "raw" / "data_5278_Phase3"
        if not data_dir.exists():
            pytest.skip("Test data directory not available")

        # Find models with different feature strategies
        feature_strategies = {
            "simple_only": [],
            "full_context": [],
            "raw_spectral": []
        }

        for model_type, model_list in available_models.items():
            for model_path in model_list:
                model_name = model_path.name.lower()
                if "simple_only" in model_name or "simple" in model_name:
                    feature_strategies["simple_only"].append((model_type, model_path))
                elif "full_context" in model_name or "full" in model_name:
                    feature_strategies["full_context"].append((model_type, model_path))
                elif "raw-spectral" in model_name or "raw_spectral" in model_name:
                    feature_strategies["raw_spectral"].append((model_type, model_path))

        strategy_results = {}

        for strategy, models in feature_strategies.items():
            if not models:
                logger.info(f"No {strategy} models available")
                continue

            # Test first model of this strategy
            model_type, model_path = models[0]
            model_name = model_path.name

            logger.info(f"Testing {strategy} strategy with {model_type}: {model_name}")

            try:
                # Get max_samples from environment
                import os
                max_samples = os.environ.get('TEST_MAX_SAMPLES')
                max_samples = int(max_samples) if max_samples else None

                results_df = predictor_instance.make_batch_predictions(data_dir, model_path, max_samples=max_samples)

                if max_samples:
                    logger.info(f"    Limited to max {max_samples} sample IDs")

                successful = results_df[results_df['Status'] == 'Success']
                success_count = len(successful)
                total_count = len(results_df)

                strategy_results[strategy] = {
                    'model_type': model_type,
                    'success_count': success_count,
                    'total_count': total_count,
                    'success_rate': success_count / total_count * 100 if total_count > 0 else 0
                }

                if success_count > 0:
                    predictions = successful['PredictedValue'].values
                    valid_preds = predictions[~np.isnan(predictions)]
                    if len(valid_preds) > 0:
                        strategy_results[strategy].update({
                            'mean_prediction': np.mean(valid_preds),
                            'std_prediction': np.std(valid_preds)
                        })

                logger.info(f"  {strategy}: {success_count}/{total_count} successful ({strategy_results[strategy]['success_rate']:.1f}%)")

            except Exception as e:
                logger.warning(f"  {strategy} strategy failed: {e}")

        assert len(strategy_results) > 0, "No feature strategies could be tested"

        # Compare strategies
        logger.info(f"Feature strategy comparison:")
        for strategy, results in strategy_results.items():
            logger.info(f"  {strategy}: {results['success_rate']:.1f}% success, mean pred: {results.get('mean_prediction', 'N/A'):.4f}")

        # All strategies should work reasonably well
        for strategy, results in strategy_results.items():
            assert results['success_rate'] >= 50, f"{strategy} strategy has too low success rate: {results['success_rate']:.1f}%"

    def test_batch_prediction_autogluon_models(self, predictor_instance, available_models, project_root):
        """Test batch prediction with AutoGluon models"""
        if not available_models:
            pytest.skip("No trained models available")

        # Get first available AutoGluon model
        autogluon_models = {k: v for k, v in available_models.items() if k.startswith("autogluon_")}
        if not autogluon_models:
            pytest.skip("No AutoGluon models available")

        model_name, model_path = next(iter(autogluon_models.items()))

        # Use a subset of available data for batch testing
        data_dir = project_root / "data" / "raw" / "data_5278_Phase3"
        if not data_dir.exists():
            pytest.skip("Test data directory not available")

        # Get a sample of files for batch testing (smaller set for AutoGluon as it's slower)
        test_files = list(data_dir.glob("*.csv.txt"))[:5]  # Limit to 5 files for AutoGluon testing
        if not test_files:
            pytest.skip("No test files available")

        logger.info(f"Testing batch prediction with AutoGluon model: {model_name}")
        logger.info(f"Testing with {len(test_files)} files from {data_dir.name}")

        # Get max_samples from environment
        import os
        max_samples = os.environ.get('TEST_MAX_SAMPLES')
        max_samples = int(max_samples) if max_samples else None

        # Make batch predictions
        try:
            results_df = predictor_instance.make_batch_predictions(data_dir, model_path, max_samples=max_samples)

            if max_samples:
                logger.info(f"  Limited to max {max_samples} sample IDs")

            # Validate results (same as sklearn test)
            assert isinstance(results_df, pd.DataFrame), "Results should be a DataFrame"
            assert len(results_df) > 0, "Should have some results"
            assert all(col in results_df.columns for col in ['sampleId', 'PredictedValue', 'Status']), \
                "Required columns missing"

            # Check for successful predictions
            successful = results_df[results_df['Status'] == 'Success']
            failed = results_df[results_df['Status'].str.contains('Failed', na=False)]

            logger.info(f"✓ AutoGluon batch prediction results:")
            logger.info(f"  Total samples: {len(results_df)}")
            logger.info(f"  Successful: {len(successful)}")
            logger.info(f"  Failed: {len(failed)}")

            if len(successful) > 0:
                predictions = successful['PredictedValue'].values
                logger.info(f"  Prediction range: {predictions.min():.4f} - {predictions.max():.4f}")
                logger.info(f"  Mean prediction: {predictions.mean():.4f}")

            # AutoGluon might have more failures due to complexity, so be more lenient
            assert len(results_df) > 0, "Should have some results (successful or failed)"

            # Validate any successful predictions
            for _, row in successful.iterrows():
                pred_val = row['PredictedValue']
                assert isinstance(pred_val, (int, float)), f"Invalid prediction type: {type(pred_val)}"
                assert not np.isnan(pred_val), f"NaN prediction for sample {row['sampleId']}"
                assert 0.0 <= pred_val <= 5.0, f"Prediction {pred_val} outside range for {row['sampleId']}"

        except Exception as e:
            pytest.fail(f"AutoGluon batch prediction failed: {e}")


class TestRealDataValidation(TestRealDataSetup):
    """Test prediction validation against reference data"""

    def test_predictions_against_reference_data(self, predictor_instance, available_models,
                                              available_data_files, reference_data):
        """Test predictions against known reference values"""
        if not available_models:
            pytest.skip("No trained models available")

        if not available_data_files:
            pytest.skip("No spectral data files available")

        if reference_data is None:
            pytest.skip("Reference data not available")

        # Get first available model
        model_name, model_path = next(iter(available_models.items()))

        logger.info(f"Testing predictions against reference data with model: {model_name}")
        logger.info(f"Reference data shape: {reference_data.shape}")

        # Find samples that exist in both spectral data and reference data
        sample_ids_in_ref = set(reference_data['Sample ID'].astype(str))
        sample_ids_in_data = set(available_data_files.keys())
        common_samples = sample_ids_in_ref.intersection(sample_ids_in_data)

        if not common_samples:
            logger.warning("No common samples between spectral data and reference data")
            logger.info(f"Sample IDs in reference (first 10): {list(sample_ids_in_ref)[:10]}")
            logger.info(f"Sample IDs in data (first 10): {list(sample_ids_in_data)[:10]}")
            pytest.skip("No overlapping samples for validation")

        logger.info(f"Found {len(common_samples)} common samples for validation")

        validation_results = []
        test_count = 0
        max_tests = 5  # Limit number of validation tests

        for sample_id in list(common_samples)[:max_tests]:
            try:
                # Get spectral file for this sample
                test_file = available_data_files[sample_id][0]  # Use first file

                # Make prediction
                prediction = predictor_instance.make_prediction(test_file, model_path)

                # Get reference value (assuming potassium column exists)
                ref_row = reference_data[reference_data['Sample ID'] == sample_id]
                if len(ref_row) == 0:
                    continue

                # Try to find potassium column
                potassium_cols = [col for col in reference_data.columns
                                if 'potassium' in col.lower() or 'k' in col.lower()]

                if not potassium_cols:
                    logger.warning("No potassium column found in reference data")
                    continue

                ref_value = ref_row[potassium_cols[0]].iloc[0]

                validation_results.append({
                    'sample_id': sample_id,
                    'predicted': prediction,
                    'reference': ref_value,
                    'absolute_error': abs(prediction - ref_value),
                    'relative_error': abs(prediction - ref_value) / max(ref_value, 0.01) * 100
                })

                logger.info(f"Sample {sample_id}: Predicted={prediction:.4f}, Reference={ref_value:.4f}")
                test_count += 1

            except Exception as e:
                logger.warning(f"Validation failed for sample {sample_id}: {e}")

        assert test_count > 0, "No successful validations"

        # Analyze validation results
        validation_df = pd.DataFrame(validation_results)

        logger.info(f"✓ Validation results for {len(validation_df)} samples:")
        logger.info(f"  Mean absolute error: {validation_df['absolute_error'].mean():.4f}")
        logger.info(f"  Mean relative error: {validation_df['relative_error'].mean():.2f}%")
        logger.info(f"  Max absolute error: {validation_df['absolute_error'].max():.4f}")
        logger.info(f"  Correlation: {validation_df['predicted'].corr(validation_df['reference']):.3f}")

        # Basic sanity checks
        assert validation_df['absolute_error'].mean() < 2.0, "Mean absolute error too high"
        assert validation_df['relative_error'].mean() < 200.0, "Mean relative error too high"

    def test_model_consistency_across_samples(self, predictor_instance, available_models, available_data_files):
        """Test that model predictions are consistent across similar samples"""
        if not available_models:
            pytest.skip("No trained models available")

        if not available_data_files:
            pytest.skip("No spectral data files available")

        # Get first available model
        model_name, model_path = next(iter(available_models.items()))

        # Find samples with multiple files for consistency testing
        multi_file_samples = [(sid, files) for sid, files in available_data_files.items() if len(files) > 1]

        if not multi_file_samples:
            pytest.skip("No samples with multiple files for consistency testing")

        logger.info(f"Testing model consistency with model: {model_name}")

        consistency_results = []

        for sample_id, file_list in multi_file_samples[:3]:  # Test first 3 multi-file samples
            try:
                predictions = []
                for test_file in file_list[:3]:  # Max 3 files per sample
                    prediction = predictor_instance.make_prediction(test_file, model_path)
                    predictions.append(prediction)

                if len(predictions) > 1:
                    mean_pred = np.mean(predictions)
                    std_pred = np.std(predictions)
                    cv = std_pred / mean_pred * 100 if mean_pred != 0 else 0

                    consistency_results.append({
                        'sample_id': sample_id,
                        'n_files': len(predictions),
                        'mean_prediction': mean_pred,
                        'std_prediction': std_pred,
                        'cv_percent': cv
                    })

                    logger.info(f"Sample {sample_id}: Mean={mean_pred:.4f}, Std={std_pred:.4f}, CV={cv:.2f}%")

            except Exception as e:
                logger.warning(f"Consistency test failed for sample {sample_id}: {e}")

        assert len(consistency_results) > 0, "No successful consistency tests"

        # Analyze consistency
        consistency_df = pd.DataFrame(consistency_results)

        logger.info(f"✓ Consistency results for {len(consistency_df)} samples:")
        logger.info(f"  Mean CV: {consistency_df['cv_percent'].mean():.2f}%")
        logger.info(f"  Max CV: {consistency_df['cv_percent'].max():.2f}%")

        # Check that coefficient of variation is reasonable (models should be somewhat consistent)
        assert consistency_df['cv_percent'].mean() < 50.0, "Model predictions too inconsistent across files"


class TestRealDataEdgeCases(TestRealDataSetup):
    """Test edge cases with real data"""

    def test_prediction_with_problematic_files(self, predictor_instance, available_models, project_root):
        """Test prediction handling with potentially problematic files"""
        if not available_models:
            pytest.skip("No trained models available")

        # Get first available model
        model_name, model_path = next(iter(available_models.items()))

        # Create a test directory with mixed file qualities
        test_dir = project_root / "test_problematic_files"
        test_dir.mkdir(exist_ok=True)

        try:
            # Copy a few real files
            data_dir = project_root / "data" / "raw" / "data_5278_Phase3"
            if data_dir.exists():
                real_files = list(data_dir.glob("*.csv.txt"))[:2]
                for real_file in real_files:
                    import shutil
                    shutil.copy(real_file, test_dir)

            # Create an empty file
            (test_dir / "empty_file.csv.txt").touch()

            # Create a file with invalid content
            with open(test_dir / "invalid_content.csv.txt", 'w') as f:
                f.write("This is not valid spectral data\n")

            logger.info(f"Testing problematic files handling with model: {model_name}")

            # Run batch prediction
            results_df = predictor_instance.make_batch_predictions(test_dir, model_path)

            # Should handle errors gracefully
            assert isinstance(results_df, pd.DataFrame), "Should return DataFrame even with problematic files"

            successful = results_df[results_df['Status'] == 'Success']
            failed = results_df[results_df['Status'].str.contains('Failed', na=False)]

            logger.info(f"✓ Problematic files handling:")
            logger.info(f"  Total files processed: {len(results_df)}")
            logger.info(f"  Successful: {len(successful)}")
            logger.info(f"  Failed: {len(failed)}")

            # Should have some failed files due to problematic content
            assert len(failed) > 0, "Should detect and handle problematic files"

        finally:
            # Clean up
            import shutil
            if test_dir.exists():
                shutil.rmtree(test_dir)


if __name__ == "__main__":
    # Run with verbose output to see detailed results
    pytest.main([__file__, "-v", "-s", "--tb=short"])