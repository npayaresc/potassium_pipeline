#!/usr/bin/env python3
"""
Test runner for predictor test suite

Provides convenient interface for running different test suites
with appropriate configurations and reporting.
"""
import sys
import subprocess
from pathlib import Path
import argparse


def run_pytest_command(test_files, verbose=True, coverage=False, parallel=False):
    """Run pytest with specified options"""
    cmd = ["python", "-m", "pytest"]

    # Add test files
    cmd.extend(test_files)

    # Add options
    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=src/models", "--cov-report=html", "--cov-report=term"])

    if parallel:
        cmd.extend(["-n", "auto"])  # Requires pytest-xdist

    # Add other useful options
    cmd.extend([
        "--tb=short",  # Shorter traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings",  # Disable warnings for cleaner output
    ])

    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=Path(__file__).parent)


def main():
    parser = argparse.ArgumentParser(description="Run predictor test suite")

    # Test suite selection
    parser.add_argument(
        "--suite",
        choices=["unit", "autogluon", "integration", "all"],
        default="all",
        help="Test suite to run"
    )

    # Test options
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only fast tests (excludes integration tests)"
    )

    args = parser.parse_args()

    # Define test files
    test_files = []

    if args.suite == "unit" or args.suite == "all":
        test_files.append("test_predictor.py")

    if args.suite == "autogluon" or args.suite == "all":
        test_files.append("test_autogluon_prediction.py")

    if args.suite == "integration" or args.suite == "all":
        if not args.quick:
            test_files.append("test_integration_prediction.py")

    if not test_files:
        print("No test files selected!")
        return 1

    print(f"Running test suite: {args.suite}")
    print(f"Test files: {', '.join(test_files)}")

    # Check if test files exist
    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"Warning: Test file {test_file} not found!")

    # Run tests
    result = run_pytest_command(
        test_files,
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel
    )

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())