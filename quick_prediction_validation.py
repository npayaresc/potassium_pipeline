#!/usr/bin/env python3
"""
Quick prediction validation script

A simple script to quickly test that both single file and batch predictions
are working with real data and models. This is useful for quick smoke testing
before running the full test suite.
"""
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.config.pipeline_config import Config
from src.models.predictor import Predictor


def setup_logging():
    """Setup simple logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    return logging.getLogger(__name__)


def find_available_resources(logger, models_per_type=3, max_total=None):
    """Find available models and data files

    Args:
        models_per_type: Number of models to include per type (default: 3 for quick validation)
        max_total: Maximum total number of models (default: None)
    """
    models_dir = project_root / "models"
    data_dir = project_root / "data" / "raw"

    # Find models and categorize them
    available_models = []

    # Categorize .pkl models (get latest models of each type)
    model_categories = {
        "xgboost": [],
        "catboost": [],
        "lightgbm": [],
        "random_forest": [],
        "extratrees": [],
        "neural_network": [],
        "ridge": [],
        "autogluon_pkl": [],
        "other": []
    }

    # Sort by modification time, newest first
    all_pkl_files = sorted(models_dir.glob("*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)

    total_models_added = 0

    for pkl_file in all_pkl_files:
        if max_total and total_models_added >= max_total:
            break

        model_name = pkl_file.stem.lower()
        added = False

        if "xgboost" in model_name and len(model_categories["xgboost"]) < models_per_type:
            model_categories["xgboost"].append(("xgboost", pkl_file.name, pkl_file))
            added = True
        elif "catboost" in model_name and len(model_categories["catboost"]) < models_per_type:
            model_categories["catboost"].append(("catboost", pkl_file.name, pkl_file))
            added = True
        elif "lightgbm" in model_name and len(model_categories["lightgbm"]) < models_per_type:
            model_categories["lightgbm"].append(("lightgbm", pkl_file.name, pkl_file))
            added = True
        elif "random_forest" in model_name and len(model_categories["random_forest"]) < models_per_type:
            model_categories["random_forest"].append(("random_forest", pkl_file.name, pkl_file))
            added = True
        elif "extratrees" in model_name and len(model_categories["extratrees"]) < models_per_type:
            model_categories["extratrees"].append(("extratrees", pkl_file.name, pkl_file))
            added = True
        elif "neural" in model_name and len(model_categories["neural_network"]) < models_per_type:
            model_categories["neural_network"].append(("neural_network", pkl_file.name, pkl_file))
            added = True
        elif "ridge" in model_name and len(model_categories["ridge"]) < models_per_type:
            model_categories["ridge"].append(("ridge", pkl_file.name, pkl_file))
            added = True
        elif "autogluon" in model_name and len(model_categories["autogluon_pkl"]) < models_per_type:
            model_categories["autogluon_pkl"].append(("autogluon_pkl", pkl_file.name, pkl_file))
            added = True
        elif len(model_categories["other"]) < models_per_type:
            model_categories["other"].append(("other", pkl_file.name, pkl_file))
            added = True

        if added:
            total_models_added += 1

    # Flatten the categorized models
    for category_models in model_categories.values():
        available_models.extend(category_models)

    # AutoGluon directory models
    autogluon_dir = models_dir / "autogluon"
    if autogluon_dir.exists():
        ag_dirs = [d for d in autogluon_dir.iterdir()
                  if d.is_dir() and (d / "feature_pipeline.pkl").exists()]
        # Sort by modification time, newest first
        ag_dirs_sorted = sorted(ag_dirs, key=lambda x: x.stat().st_mtime, reverse=True)

        # Apply limits
        if max_total and total_models_added >= max_total:
            pass  # Don't add AutoGluon models
        else:
            remaining = None
            if max_total:
                remaining = max_total - total_models_added
                limit = min(models_per_type, remaining)
            else:
                limit = models_per_type

            for ag_dir in ag_dirs_sorted[:limit]:
                available_models.append(("autogluon_dir", ag_dir.name, ag_dir))
                total_models_added += 1

    # Find data files
    available_data_dirs = []
    if data_dir.exists():
        for subdir in data_dir.iterdir():
            if subdir.is_dir():
                spectral_files = list(subdir.glob("*.csv.txt"))
                if spectral_files:
                    available_data_dirs.append((subdir.name, subdir, len(spectral_files)))

    logger.info(f"Found {len(available_models)} models:")
    for model_type, name, path in available_models[:3]:
        logger.info(f"  {model_type}: {name}")
    if len(available_models) > 3:
        logger.info(f"  ... and {len(available_models) - 3} more")

    logger.info(f"Found {len(available_data_dirs)} data directories:")
    for name, path, file_count in available_data_dirs[:3]:
        logger.info(f"  {name}: {file_count} files")
    if len(available_data_dirs) > 3:
        logger.info(f"  ... and {len(available_data_dirs) - 3} more")

    return available_models, available_data_dirs


def create_test_config():
    """Create configuration for testing"""
    from src.config.pipeline_config import config

    # Use the global config - it already has correct paths
    config.run_timestamp = "quick_validation"

    # Create directories if they don't exist
    for dir_path in [config.reports_dir, config.bad_prediction_files_dir, config.log_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    return config


def test_single_file_prediction(predictor, model_info, data_dirs, logger):
    """Test single file prediction"""
    model_type, model_name, model_path = model_info

    # Find a test file
    test_file = None
    for dir_name, dir_path, file_count in data_dirs:
        spectral_files = list(dir_path.glob("*.csv.txt"))
        if spectral_files:
            test_file = spectral_files[0]
            break

    if not test_file:
        logger.error("No spectral files found for testing")
        return False

    logger.info(f"Testing single file prediction...")
    logger.info(f"  Model: {model_type} - {model_name}")
    logger.info(f"  File: {test_file.name}")

    try:
        prediction = predictor.make_prediction(test_file, model_path)

        # Validate prediction
        if not isinstance(prediction, (int, float)):
            logger.error(f"Invalid prediction type: {type(prediction)}")
            return False

        if np.isnan(prediction) or np.isinf(prediction):
            logger.error(f"Invalid prediction value: {prediction}")
            return False

        if not (0.0 <= prediction <= 5.0):
            logger.warning(f"Prediction outside expected range: {prediction}")

        logger.info(f"  ✓ Prediction: {prediction:.4f}")
        return True

    except Exception as e:
        logger.error(f"Single file prediction failed: {e}")
        return False


def test_batch_prediction(predictor, model_info, data_dirs, logger, max_samples=None):
    """Test batch prediction"""
    model_type, model_name, model_path = model_info

    # Use the first available data directory with limited files
    if not data_dirs:
        logger.error("No data directories available for batch testing")
        return False

    test_dir_name, test_dir_path, file_count = data_dirs[0]

    logger.info(f"Testing batch prediction...")
    logger.info(f"  Model: {model_type} - {model_name}")
    logger.info(f"  Directory: {test_dir_name} ({file_count} files)")
    if max_samples:
        logger.info(f"  Max samples: {max_samples}")

    try:
        results_df = predictor.make_batch_predictions(test_dir_path, model_path, max_samples=max_samples)

        # Validate results
        if not isinstance(results_df, pd.DataFrame):
            logger.error(f"Expected DataFrame, got {type(results_df)}")
            return False

        if len(results_df) == 0:
            logger.error("No results returned from batch prediction")
            return False

        # Check required columns
        required_cols = ['sampleId', 'PredictedValue', 'Status']
        if not all(col in results_df.columns for col in required_cols):
            logger.error(f"Missing required columns. Found: {list(results_df.columns)}")
            return False

        # Analyze results
        successful = results_df[results_df['Status'] == 'Success']
        failed = results_df[results_df['Status'].str.contains('Failed', na=False)]

        logger.info(f"  ✓ Processed {len(results_df)} samples:")
        logger.info(f"    Successful: {len(successful)}")
        logger.info(f"    Failed: {len(failed)}")

        if len(successful) > 0:
            predictions = successful['PredictedValue'].values
            predictions = predictions[~np.isnan(predictions)]  # Remove NaN values

            if len(predictions) > 0:
                logger.info(f"    Prediction range: {predictions.min():.4f} - {predictions.max():.4f}")
                logger.info(f"    Mean prediction: {predictions.mean():.4f}")

                # Basic validation
                invalid_predictions = sum(1 for p in predictions if p < 0.0 or p > 10.0)
                if invalid_predictions > 0:
                    logger.warning(f"    {invalid_predictions} predictions outside reasonable range")

        return len(successful) > 0  # Success if at least one prediction worked

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return False


def main():
    """Main validation function"""
    import argparse

    parser = argparse.ArgumentParser(description="Quick prediction validation")
    parser.add_argument(
        "--models-per-type",
        type=int,
        default=3,
        help="Number of models to test per type (default: 3)"
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=None,
        help="Maximum total number of models to test"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum number of sample IDs to test (default: 50)"
    )

    args = parser.parse_args()

    logger = setup_logging()

    print("Quick Prediction Validation")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Models per type: {args.models_per_type}")
    if args.max_total:
        print(f"  Max total models: {args.max_total}")
    print(f"  Max sample IDs: {args.max_samples}")
    print()

    # Find available resources
    logger.info("Scanning for models and data...")
    available_models, available_data_dirs = find_available_resources(
        logger,
        models_per_type=args.models_per_type,
        max_total=args.max_total
    )

    if not available_models:
        logger.error("No trained models found!")
        print("\nPlease ensure you have trained models in the ./models/ directory")
        return 1

    if not available_data_dirs:
        logger.error("No spectral data found!")
        print("\nPlease ensure you have spectral data files in ./data/raw/ directories")
        return 1

    # Create predictor
    config = create_test_config()
    predictor = Predictor(config)

    # Test with first available model
    test_model = available_models[0]
    logger.info(f"\nTesting with: {test_model[0]} - {test_model[1]}")

    print("\n" + "-" * 50)

    # Test single file prediction
    single_success = test_single_file_prediction(predictor, test_model, available_data_dirs, logger)

    print("\n" + "-" * 30)

    # Test batch prediction
    batch_success = test_batch_prediction(predictor, test_model, available_data_dirs, logger, max_samples=args.max_samples)

    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)

    print(f"Single file prediction: {'✅ PASS' if single_success else '❌ FAIL'}")
    print(f"Batch prediction:       {'✅ PASS' if batch_success else '❌ FAIL'}")

    if single_success and batch_success:
        print("\n🎉 All basic prediction functionality is working!")
        print("\nYou can now run the full test suite with:")
        print("  python run_real_data_tests.py")
        print("  python run_predictor_tests.py")
        return 0
    else:
        print("\n⚠️  Some functionality is not working correctly.")
        print("Please check the error messages above and ensure:")
        print("  - Models are properly trained and saved")
        print("  - Data files are in the correct format")
        print("  - All dependencies are installed")
        print("  - Configuration is correct")
        return 1


if __name__ == "__main__":
    sys.exit(main())