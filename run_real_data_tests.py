#!/usr/bin/env python3
"""
Test runner for real data prediction tests

Runs tests using actual trained models and spectral data to verify
that predictions are working correctly in production environment.
"""
import sys
import subprocess
import argparse
from pathlib import Path
import logging


def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def check_prerequisites():
    """Check if necessary data and models are available"""
    project_root = Path(__file__).parent

    # Check for data
    data_dir = project_root / "data" / "raw"
    if not data_dir.exists():
        print("❌ No raw data directory found")
        return False

    spectral_files = list(data_dir.rglob("*.csv.txt"))
    if not spectral_files:
        print("❌ No spectral data files found")
        return False

    print(f"✓ Found {len(spectral_files)} spectral data files")

    # Check for models
    models_dir = project_root / "models"
    if not models_dir.exists():
        print("❌ No models directory found")
        return False

    # Check for different model types
    model_counts = {
        "xgboost": 0,
        "catboost": 0,
        "lightgbm": 0,
        "random_forest": 0,
        "extratrees": 0,
        "neural": 0,
        "ridge": 0,
        "autogluon_pkl": 0,
        "other": 0
    }

    sklearn_models = list(models_dir.glob("*.pkl"))
    for model_file in sklearn_models:
        model_name = model_file.stem.lower()
        if "xgboost" in model_name:
            model_counts["xgboost"] += 1
        elif "catboost" in model_name:
            model_counts["catboost"] += 1
        elif "lightgbm" in model_name:
            model_counts["lightgbm"] += 1
        elif "random_forest" in model_name:
            model_counts["random_forest"] += 1
        elif "extratrees" in model_name:
            model_counts["extratrees"] += 1
        elif "neural" in model_name:
            model_counts["neural"] += 1
        elif "ridge" in model_name:
            model_counts["ridge"] += 1
        elif "autogluon" in model_name:
            model_counts["autogluon_pkl"] += 1
        else:
            model_counts["other"] += 1

    # Check for AutoGluon directory models
    autogluon_dir = models_dir / "autogluon"
    autogluon_dir_models = 0
    if autogluon_dir.exists():
        autogluon_dir_models = len([d for d in autogluon_dir.iterdir()
                                   if d.is_dir() and (d / "feature_pipeline.pkl").exists()])

    print(f"✓ Found trained models:")
    for model_type, count in model_counts.items():
        if count > 0:
            print(f"  {model_type}: {count}")
    if autogluon_dir_models > 0:
        print(f"  autogluon (directories): {autogluon_dir_models}")

    total_models = sum(model_counts.values()) + autogluon_dir_models
    if total_models == 0:
        print("❌ No trained models found")
        return False

    # Check for reference data
    ref_data_path = project_root / "data" / "reference_data" / "Final_Lab_Data_Nico_New.xlsx"
    if ref_data_path.exists():
        print("✓ Found reference data for validation")
    else:
        print("⚠ No reference data found - validation tests will be skipped")

    return True


def run_pytest_command(test_patterns, verbose=True, show_output=True):
    """Run pytest with specified test patterns"""
    cmd = ["python", "-m", "pytest"]

    # Add test patterns
    cmd.extend(test_patterns)

    # Add pytest options
    if verbose:
        cmd.append("-v")

    if show_output:
        cmd.append("-s")  # Don't capture output

    # Add other useful options
    cmd.extend([
        "--tb=short",  # Shorter traceback format
        "--disable-warnings",  # Disable warnings for cleaner output
        "--maxfail=5",  # Stop after 5 failures
        "-x"  # Stop on first failure for real data tests
    ])

    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)

    return subprocess.run(cmd, cwd=Path(__file__).parent)


def main():
    parser = argparse.ArgumentParser(description="Run real data prediction tests")

    # Test selection
    parser.add_argument(
        "--test-type",
        choices=["single", "batch", "validation", "edge-cases", "all"],
        default="all",
        help="Type of tests to run"
    )

    parser.add_argument(
        "--model-type",
        choices=["xgboost", "catboost", "lightgbm", "random_forest", "extratrees", "neural", "ridge", "autogluon", "all"],
        default="all",
        help="Model types to test"
    )

    # Model count configuration
    parser.add_argument(
        "--models-per-type",
        type=int,
        default=5,
        help="Number of models to test per type (default: 5)"
    )

    parser.add_argument(
        "--max-models-total",
        type=int,
        default=None,
        help="Maximum total number of models to test across all types"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum number of sample IDs to test (default: 50)"
    )

    # Options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (fewer samples)"
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check prerequisites, don't run tests"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    print("Real Data Prediction Tests")
    print("=" * 50)
    print(f"\nConfiguration:")
    print(f"  Models per type: {args.models_per_type}")
    if args.max_models_total:
        print(f"  Max total models: {args.max_models_total}")
    print(f"  Max sample IDs: {args.max_samples}")

    # Check prerequisites
    print("\n1. Checking prerequisites...")
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please ensure you have:")
        print("  - Trained models in ./models/ directory")
        print("  - Spectral data files in ./data/raw/ directory")
        print("  - Reference data (optional) in ./data/reference_data/")
        return 1

    if args.check_only:
        print("\n✅ All prerequisites met!")
        return 0

    print("\n✅ Prerequisites met!")

    # Set environment variables for test configuration
    import os
    os.environ['TEST_MODELS_PER_TYPE'] = str(args.models_per_type)
    if args.max_models_total:
        os.environ['TEST_MAX_MODELS_TOTAL'] = str(args.max_models_total)
    os.environ['TEST_MAX_SAMPLES'] = str(args.max_samples)

    # Build test patterns based on arguments
    test_patterns = []
    base_file = "test_real_data_prediction.py"

    if args.test_type == "single" or args.test_type == "all":
        test_patterns.append(f"{base_file}::TestRealSingleFilePrediction")
        if args.model_type == "sklearn":
            test_patterns[-1] += "::test_single_file_prediction_sklearn_models"
        elif args.model_type == "autogluon":
            test_patterns[-1] += "::test_single_file_prediction_autogluon_models"

    if args.test_type == "batch" or args.test_type == "all":
        test_patterns.append(f"{base_file}::TestRealBatchPrediction")
        if args.model_type == "sklearn":
            test_patterns[-1] += "::test_batch_prediction_sklearn_models"
        elif args.model_type == "autogluon":
            test_patterns[-1] += "::test_batch_prediction_autogluon_models"

    if args.test_type == "validation" or args.test_type == "all":
        test_patterns.append(f"{base_file}::TestRealDataValidation")

    if args.test_type == "edge-cases" or args.test_type == "all":
        test_patterns.append(f"{base_file}::TestRealDataEdgeCases")

    # If no specific patterns, test everything
    if not test_patterns:
        test_patterns = [base_file]

    print(f"\n2. Running tests...")
    print(f"Test patterns: {test_patterns}")

    # Run tests
    result = run_pytest_command(
        test_patterns,
        verbose=args.verbose,
        show_output=True
    )

    print("\n" + "=" * 80)
    if result.returncode == 0:
        print("✅ All tests passed!")
        print("\nYour prediction pipeline is working correctly with real data and models.")
    else:
        print("❌ Some tests failed!")
        print("\nPlease check the output above for details on what went wrong.")
        print("Common issues:")
        print("  - Model file corruption or incompatibility")
        print("  - Data file format changes")
        print("  - Missing dependencies")
        print("  - Configuration mismatches")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())