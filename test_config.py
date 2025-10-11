#!/usr/bin/env python3
"""
Test Configuration Script

This script provides a convenient interface for configuring and running tests
with different numbers of models.
"""
import sys
import subprocess
import argparse
from pathlib import Path


def run_quick_validation(models_per_type=3, max_total=None, max_samples=None):
    """Run quick prediction validation with specified model counts"""
    cmd = ["python", "quick_prediction_validation.py"]
    cmd.extend(["--models-per-type", str(models_per_type)])

    if max_total:
        cmd.extend(["--max-total", str(max_total)])

    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])

    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=Path(__file__).parent)


def run_real_data_tests(models_per_type=5, max_total=None, test_type="all", verbose=False, max_samples=None):
    """Run real data tests with specified model counts"""
    cmd = ["python", "run_real_data_tests.py"]
    cmd.extend(["--models-per-type", str(models_per_type)])
    cmd.extend(["--test-type", test_type])

    if max_total:
        cmd.extend(["--max-models-total", str(max_total)])

    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])

    if verbose:
        cmd.append("--verbose")

    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=Path(__file__).parent)


def run_pytest_tests(models_per_type=5, max_total=None, test_file=None, max_samples=None):
    """Run pytest with specified model counts"""
    import os

    # Set environment variables for pytest
    os.environ['TEST_MODELS_PER_TYPE'] = str(models_per_type)
    if max_total:
        os.environ['TEST_MAX_MODELS_TOTAL'] = str(max_total)
    if max_samples:
        os.environ['TEST_MAX_SAMPLES'] = str(max_samples)

    cmd = ["python", "-m", "pytest"]

    if test_file:
        cmd.append(test_file)
    else:
        # Run all real data tests
        cmd.extend([
            "test_real_data_prediction.py",
            "test_gradient_boosting_prediction.py"
        ])

    cmd.extend(["-v", "-s", "--tb=short"])

    print(f"Running: {' '.join(cmd)}")
    print(f"Environment: TEST_MODELS_PER_TYPE={models_per_type}")
    if max_total:
        print(f"Environment: TEST_MAX_MODELS_TOTAL={max_total}")
    if max_samples:
        print(f"Environment: TEST_MAX_SAMPLES={max_samples}")

    return subprocess.run(cmd, cwd=Path(__file__).parent)


def main():
    parser = argparse.ArgumentParser(
        description="Configure and run prediction tests with flexible model counts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick validation with 2 models per type and 30 samples
  python test_config.py quick --models-per-type 2 --max-samples 30

  # Run all tests with max 10 total models and 50 samples
  python test_config.py tests --max-total 10 --max-samples 50

  # Run only single file tests with 3 models per type
  python test_config.py tests --test-type single --models-per-type 3

  # Run pytest directly with 1 model per type and 20 samples
  python test_config.py pytest --models-per-type 1 --max-samples 20

  # Test specific file with limited models and samples
  python test_config.py pytest --models-per-type 2 --test-file test_real_data_prediction.py --max-samples 25
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Test command to run")

    # Quick validation command
    quick_parser = subparsers.add_parser("quick", help="Run quick prediction validation")
    quick_parser.add_argument("--models-per-type", type=int, default=3,
                            help="Models per type (default: 3)")
    quick_parser.add_argument("--max-total", type=int,
                            help="Max total models")
    quick_parser.add_argument("--max-samples", type=int, default=50,
                            help="Max sample IDs to test (default: 50)")

    # Real data tests command
    tests_parser = subparsers.add_parser("tests", help="Run real data tests via runner")
    tests_parser.add_argument("--models-per-type", type=int, default=5,
                            help="Models per type (default: 5)")
    tests_parser.add_argument("--max-total", type=int,
                            help="Max total models")
    tests_parser.add_argument("--max-samples", type=int, default=50,
                            help="Max sample IDs to test (default: 50)")
    tests_parser.add_argument("--test-type",
                            choices=["single", "batch", "validation", "edge-cases", "all"],
                            default="all", help="Type of tests to run")
    tests_parser.add_argument("--verbose", action="store_true",
                            help="Verbose output")

    # Pytest command
    pytest_parser = subparsers.add_parser("pytest", help="Run pytest directly")
    pytest_parser.add_argument("--models-per-type", type=int, default=5,
                             help="Models per type (default: 5)")
    pytest_parser.add_argument("--max-total", type=int,
                             help="Max total models")
    pytest_parser.add_argument("--max-samples", type=int, default=50,
                             help="Max sample IDs to test (default: 50)")
    pytest_parser.add_argument("--test-file",
                             help="Specific test file to run")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    print("Test Configuration")
    print("=" * 50)
    print(f"Command: {args.command}")
    print(f"Models per type: {args.models_per_type}")
    if hasattr(args, 'max_total') and args.max_total:
        print(f"Max total models: {args.max_total}")
    if hasattr(args, 'max_samples') and args.max_samples:
        print(f"Max sample IDs: {args.max_samples}")
    print()

    result = None
    if args.command == "quick":
        result = run_quick_validation(
            models_per_type=args.models_per_type,
            max_total=getattr(args, 'max_total', None),
            max_samples=getattr(args, 'max_samples', None)
        )
    elif args.command == "tests":
        result = run_real_data_tests(
            models_per_type=args.models_per_type,
            max_total=getattr(args, 'max_total', None),
            test_type=args.test_type,
            verbose=args.verbose,
            max_samples=getattr(args, 'max_samples', None)
        )
    elif args.command == "pytest":
        result = run_pytest_tests(
            models_per_type=args.models_per_type,
            max_total=getattr(args, 'max_total', None),
            test_file=getattr(args, 'test_file', None),
            max_samples=getattr(args, 'max_samples', None)
        )

    return result.returncode if result else 1


if __name__ == "__main__":
    sys.exit(main())