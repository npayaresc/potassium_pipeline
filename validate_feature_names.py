#!/usr/bin/env python3
"""
Validation script to verify feature names are correctly saved and match the model.

Usage:
    python validate_feature_names.py models/simple_only_xgboost_20251006_000443.pkl
"""
import argparse
import json
import sys
from pathlib import Path
import joblib


def validate_feature_names(model_path: Path):
    """
    Validate that saved feature names match the model's expectations.

    Args:
        model_path: Path to the saved model .pkl file
    """
    print(f"Validating feature names for: {model_path}")
    print("=" * 80)

    # Load the model
    try:
        model = joblib.load(model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    # Load the feature names file
    feature_names_path = model_path.with_suffix('.feature_names.json')
    if not feature_names_path.exists():
        print(f"✗ Feature names file not found: {feature_names_path}")
        return False

    try:
        with open(feature_names_path) as f:
            feature_info = json.load(f)
        print(f"✓ Feature names file loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load feature names: {e}")
        return False

    # Display basic info
    print(f"\n📊 Model Information:")
    print(f"  Model name: {feature_info['model_name']}")
    print(f"  Strategy: {feature_info['strategy']}")
    print(f"  Model type: {feature_info['model_type']}")
    print(f"  Timestamp: {feature_info['timestamp']}")
    print(f"  Pipeline steps: {', '.join(feature_info['pipeline_steps'])}")

    # Display transformation info
    transformations = feature_info.get('transformations', {})
    print(f"\n🔄 Transformations Applied:")

    if transformations.get('feature_selection'):
        fs = transformations['feature_selection']
        print(f"  Feature Selection:")
        print(f"    Method: {fs['method']}")
        print(f"    Features selected: {fs['n_features_selected']}")
    else:
        print(f"  Feature Selection: None")

    if transformations.get('dimension_reduction'):
        dr = transformations['dimension_reduction']
        print(f"  Dimension Reduction:")
        print(f"    Method: {dr['method']}")
        print(f"    Components: {dr['n_components']}")
    else:
        print(f"  Dimension Reduction: None")

    # Validate feature count and names
    print(f"\n📝 Feature Information:")
    feature_count = feature_info.get('feature_count')
    feature_names = feature_info.get('feature_names')

    if feature_count is not None:
        print(f"  Feature count: {feature_count}")
    else:
        print(f"  Feature count: Not available (dimension reduction present)")

    if feature_names:
        print(f"  Feature names: Available ({len(feature_names)} features)")
        print(f"  Sample features (first 10):")
        for name in feature_names[:10]:
            print(f"    - {name}")
        if len(feature_names) > 10:
            print(f"    ... and {len(feature_names) - 10} more")
    else:
        print(f"  Feature names: Not available (dimension reduction transforms features)")

    # Validate against model pipeline
    print(f"\n🔍 Validation:")
    validation_passed = True

    # Check if model has nested 'model' pipeline
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        model_pipeline = model.named_steps['model']

        # Check for feature selection
        if hasattr(model_pipeline, 'named_steps') and 'feature_selection' in model_pipeline.named_steps:
            selector = model_pipeline.named_steps['feature_selection']
            if hasattr(selector, 'get_selected_features'):
                actual_features = selector.get_selected_features()
                if feature_names:
                    if len(actual_features) == len(feature_names):
                        print(f"  ✓ Feature count matches model's feature selector")
                    else:
                        print(f"  ✗ Feature count mismatch: {len(actual_features)} (model) vs {len(feature_names)} (saved)")
                        validation_passed = False

                    if actual_features == feature_names:
                        print(f"  ✓ Feature names match model's feature selector")
                    else:
                        print(f"  ✗ Feature names do not match")
                        validation_passed = False
                else:
                    print(f"  ⚠ Feature names not saved, but feature selector exists")
                    validation_passed = False

        # Check for dimension reduction
        if hasattr(model_pipeline, 'named_steps') and 'dimension_reduction' in model_pipeline.named_steps:
            if feature_names is not None:
                print(f"  ✗ Feature names should be None when dimension reduction is present")
                validation_passed = False
            else:
                print(f"  ✓ Correctly marked feature names as unavailable (dimension reduction present)")

    # Final result
    print(f"\n{'=' * 80}")
    if validation_passed:
        print(f"✅ VALIDATION PASSED - Feature names are correctly saved!")
    else:
        print(f"❌ VALIDATION FAILED - Issues detected!")

    return validation_passed


def main():
    parser = argparse.ArgumentParser(description='Validate feature names for a trained model')
    parser.add_argument('model_path', type=Path, help='Path to the model .pkl file')
    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    success = validate_feature_names(args.model_path)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
