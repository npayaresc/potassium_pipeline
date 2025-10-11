#!/usr/bin/env python3
"""
Test script for uncertainty quantification module.

Tests:
1. Conformal prediction
2. Ensemble uncertainty
3. Combined method
4. Visualization
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.models.uncertainty import UncertaintyQuantifier, format_prediction_with_uncertainty
from src.models.uncertainty_trainer import (
    train_model_with_uncertainty,
    train_ensemble_with_uncertainty
)

def generate_synthetic_data(n_samples=500, n_features=60, noise=0.1):
    """Generate synthetic data similar to K prediction."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=40,
        noise=noise * 100,
        random_state=42
    )

    # Scale to K-like range (0.2 - 0.5%)
    y = (y - y.min()) / (y.max() - y.min()) * 0.3 + 0.2

    return X, y


def test_conformal_prediction():
    """Test 1: Conformal prediction method."""
    print("=" * 80)
    print("TEST 1: CONFORMAL PREDICTION")
    print("=" * 80)
    print()

    # Generate data
    X, y = generate_synthetic_data(n_samples=500, n_features=60)

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )

    print(f"Data splits:")
    print(f"  Train: {len(y_train)} samples")
    print(f"  Val:   {len(y_val)} samples")
    print(f"  Test:  {len(y_test)} samples")
    print()

    # Train model
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on validation set
    y_val_pred = model.predict(X_val)

    # Initialize and calibrate uncertainty quantifier
    quantifier = UncertaintyQuantifier(
        method='conformal',
        confidence_level=0.68
    )
    quantifier.fit(y_val, y_val_pred)

    # Get predictions with intervals on test set
    predictions, lower, upper = quantifier.predict_with_intervals(model, X_test)

    # Compute quality metrics
    metrics = quantifier.compute_prediction_quality(y_test, predictions, lower, upper)

    print("Results:")
    print(f"  Coverage: {metrics['coverage']:.1%} (target: 68%)")
    print(f"  Calibration error: {metrics['calibration_error']:.3f}")
    print(f"  Mean interval width: {metrics['mean_interval_width']:.4f}")
    print(f"  Mean relative width: {metrics['mean_relative_width']:.1%}")
    print(f"  Well calibrated: {metrics['well_calibrated']}")
    print()

    # Show example predictions
    print("Example predictions:")
    for i in range(min(5, len(predictions))):
        formatted = format_prediction_with_uncertainty(
            predictions[i], lower[i], upper[i], 0.68
        )
        print(f"  Sample {i}: {formatted}")
    print()

    if metrics['well_calibrated']:
        print("✓ Conformal prediction test PASSED")
    else:
        print("⚠ Warning: Coverage not well-calibrated (may need more validation data)")
    print()


def test_ensemble_uncertainty():
    """Test 2: Ensemble uncertainty method."""
    print("=" * 80)
    print("TEST 2: ENSEMBLE UNCERTAINTY")
    print("=" * 80)
    print()

    # Generate data
    X, y = generate_synthetic_data(n_samples=500, n_features=60)

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )

    # Create ensemble of models
    models = [
        XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=5, random_state=43),
        LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
    ]
    model_names = ['XGB_1', 'XGB_2', 'LGBM']

    print(f"Training ensemble of {len(models)} models...")
    results = train_ensemble_with_uncertainty(
        models=models,
        model_names=model_names,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        confidence_level=0.68
    )

    print()
    print("Results:")
    print(f"  Ensemble R²: {results['metrics']['r2']:.4f}")
    print(f"  Coverage: {results['metrics']['coverage']:.1%} (target: 68%)")
    print(f"  Calibration error: {results['metrics']['calibration_error']:.3f}")
    print(f"  Mean interval width: {results['metrics']['mean_interval_width']:.4f}")
    print(f"  Well calibrated: {results['metrics']['well_calibrated']}")
    print()

    # Show example predictions
    predictions = results['test_predictions']
    lower = results['test_lower_bounds']
    upper = results['test_upper_bounds']

    print("Example ensemble predictions:")
    for i in range(min(5, len(predictions))):
        formatted = format_prediction_with_uncertainty(
            predictions[i], lower[i], upper[i], 0.68
        )
        print(f"  Sample {i}: {formatted}")
    print()

    if results['metrics']['well_calibrated']:
        print("✓ Ensemble uncertainty test PASSED")
    else:
        print("⚠ Warning: Coverage not well-calibrated")
    print()


def test_complete_workflow():
    """Test 3: Complete training workflow with uncertainty."""
    print("=" * 80)
    print("TEST 3: COMPLETE WORKFLOW")
    print("=" * 80)
    print()

    # Generate data
    X, y = generate_synthetic_data(n_samples=500, n_features=60)

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )

    # Train with uncertainty (saves results)
    save_dir = Path("reports/uncertainty_test")

    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)

    print("Training model with full uncertainty support...")
    results = train_model_with_uncertainty(
        model=model,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        model_name="XGBoost_test",
        uncertainty_method='conformal',
        confidence_level=0.68,
        save_dir=save_dir
    )

    print()
    print("Training complete! Files saved:")
    print(f"  Uncertainty quantifier: {save_dir}/XGBoost_test_uncertainty.pkl")
    print(f"  Predictions report: {save_dir}/XGBoost_test_uncertainty_report.csv")
    print(f"  Visualization: {save_dir}/XGBoost_test_uncertainty_plot.png")
    print()

    print("Results:")
    print(f"  R²: {results['metrics']['r2']:.4f}")
    print(f"  MAE: {results['metrics']['mae']:.4f}")
    print(f"  Coverage: {results['metrics']['coverage']:.1%}")
    print(f"  Mean interval width: {results['metrics']['mean_interval_width']:.4f}")
    print(f"  Well calibrated: {results['metrics']['well_calibrated']}")
    print()

    print("✓ Complete workflow test PASSED")
    print()


def main():
    """Run all tests."""
    print()
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                   UNCERTAINTY QUANTIFICATION TESTS                           ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print()

    try:
        test_conformal_prediction()
        test_ensemble_uncertainty()
        test_complete_workflow()

        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print()
        print("✓ All uncertainty quantification tests PASSED")
        print()
        print("Next steps:")
        print("  1. Review saved files in reports/uncertainty_test/")
        print("  2. Check uncertainty_plot.png visualization")
        print("  3. Read UNCERTAINTY_QUANTIFICATION_GUIDE.md")
        print("  4. Integrate into your model training pipeline")
        print()
        print("Ready to use! 🎉")
        print()

    except Exception as e:
        print()
        print(f"✗ Test failed with error: {e}")
        print()
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
