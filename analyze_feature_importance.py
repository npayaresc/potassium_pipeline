#!/usr/bin/env python3
"""
SHAP-based Feature Importance Analysis for LIBS Spectral Models

This script uses SHAP (SHapley Additive exPlanations) for robust, model-agnostic
feature importance analysis. SHAP provides:
- Consistent importance across model types
- Correct handling of correlated features (critical for spectral data)
- Individual prediction explanations
- Feature interaction analysis

SHAP is superior to tree-based importance for spectral analysis because:
1. Spectral features are highly correlated (neighboring wavelengths, physics features)
2. Tree importance can be biased toward high-cardinality features
3. SHAP provides mathematically consistent attribution
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
from typing import Optional, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import SHAP with proper error handling
try:
    import shap
    shap.initjs()  # Initialize JavaScript visualization
except ImportError:
    logger.error("SHAP not installed. Install with: uv pip install shap")
    sys.exit(1)


def detect_model_type(model: Any) -> str:
    """
    Detect the type of model for SHAP explainer selection.

    Args:
        model: Trained model object

    Returns:
        Model type string: 'xgboost', 'lightgbm', 'catboost', 'sklearn', 'torch', 'pipeline'
    """
    model_class = type(model).__name__
    model_module = type(model).__module__

    logger.info(f"Detecting model type: class={model_class}, module={model_module}")

    # Check for sklearn Pipeline (common in our pipeline)
    if 'Pipeline' in model_class:
        # Get the final estimator from pipeline
        if hasattr(model, 'named_steps'):
            final_step = list(model.named_steps.values())[-1]
            return detect_model_type(final_step)
        return 'pipeline'

    # Tree-based models (optimized SHAP explainers)
    if 'xgboost' in model_module.lower() or 'XGB' in model_class or 'xgb' in model_class.lower():
        return 'xgboost'
    elif 'lightgbm' in model_module.lower() or 'LGB' in model_class or 'lgb' in model_class.lower():
        return 'lightgbm'
    elif 'catboost' in model_module.lower() or 'catboost' in model_class.lower():
        return 'catboost'

    # PyTorch models
    elif 'torch' in model_module.lower() or hasattr(model, 'forward'):
        return 'torch'

    # Sklearn models (RandomForest, ExtraTrees, etc.)
    elif 'sklearn' in model_module.lower():
        return 'sklearn'

    # AutoGluon (TabularPredictor)
    elif 'autogluon' in model_module.lower():
        return 'autogluon'

    logger.warning(f"Model type not recognized - returning 'unknown' for class={model_class}, module={model_module}")
    return 'unknown'


def extract_model_from_pipeline(model: Any) -> Any:
    """
    Extract the actual model from sklearn Pipeline if wrapped.

    Args:
        model: Model object (may be Pipeline or raw model)

    Returns:
        Unwrapped model
    """
    # Check if it's a CalibratedModelWrapper (from model_trainer.py)
    if hasattr(model, 'base_model'):
        logger.info("Extracting base model from CalibratedModelWrapper")
        model = model.base_model

    # Check if it's a sklearn Pipeline
    if hasattr(model, 'named_steps'):
        # Get the last step (the actual model)
        steps = list(model.named_steps.values())
        if len(steps) > 0:
            logger.info(f"Extracting model from Pipeline (last step of {len(steps)} steps)")
            # Recursively extract in case the last step is also a wrapper
            return extract_model_from_pipeline(steps[-1])

    # Check if it's a Pipeline with _final_estimator
    if hasattr(model, '_final_estimator'):
        logger.info("Extracting model from Pipeline (_final_estimator)")
        return extract_model_from_pipeline(model._final_estimator)

    # Not a pipeline, return as is
    return model


def create_shap_explainer(model: Any, X_background: np.ndarray, model_type: str, feature_names: Optional[list] = None):
    """
    Create appropriate SHAP explainer based on model type.

    Args:
        model: Trained model
        X_background: Background dataset for SHAP (sampled from training data)
        model_type: Model type string from detect_model_type()
        feature_names: List of feature names (required for AutoGluon)

    Returns:
        SHAP explainer object
    """
    logger.info(f"Creating SHAP explainer for model type: {model_type}")

    # Store feature_names as function attribute for AutoGluon handler
    create_shap_explainer._feature_names = feature_names

    # Extract actual model from pipeline if needed
    actual_model = extract_model_from_pipeline(model)

    if model_type in ['xgboost', 'lightgbm', 'catboost']:
        # TreeExplainer is fastest and most accurate for tree models
        # Uses optimized algorithms for tree ensembles
        explainer = shap.TreeExplainer(actual_model)
        logger.info("✓ Using TreeExplainer (optimized for tree ensembles)")

    elif model_type == 'sklearn':
        # For sklearn models, try TreeExplainer first (if tree-based)
        if hasattr(actual_model, 'estimators_') or 'Forest' in type(actual_model).__name__ or 'Tree' in type(actual_model).__name__:
            explainer = shap.TreeExplainer(actual_model)
            logger.info("✓ Using TreeExplainer for sklearn tree model")
        else:
            # For non-tree sklearn models, use KernelExplainer
            # Use the original model's predict method if it's a pipeline (includes preprocessing)
            explainer = shap.KernelExplainer(model.predict, X_background)
            logger.info("✓ Using KernelExplainer for sklearn model")

    elif model_type == 'torch':
        # DeepExplainer for PyTorch neural networks
        explainer = shap.DeepExplainer(actual_model, X_background)
        logger.info("✓ Using DeepExplainer for PyTorch model")

    elif model_type == 'autogluon':
        # AutoGluon TabularPredictor in pipeline needs DataFrames, not numpy arrays
        logger.warning("AutoGluon detected - using KernelExplainer with DataFrame wrapper")
        # Note: feature_names will be added as parameter in the function signature
        feature_names = getattr(create_shap_explainer, '_feature_names', None)

        if feature_names is None:
            logger.error("❌ Feature names required for AutoGluon but not available")
            logger.error("   This is likely because the model's pipeline needs column names")
            raise ValueError("AutoGluon models require feature_names parameter")

        # Convert background data to DataFrame
        X_bg_df = pd.DataFrame(X_background, columns=feature_names)

        # Create wrapper that converts numpy to DataFrame before prediction
        def predict_wrapper(X):
            if not isinstance(X, pd.DataFrame):
                X_df = pd.DataFrame(X, columns=feature_names)
            else:
                X_df = X
            return model.predict(X_df)

        explainer = shap.KernelExplainer(predict_wrapper, X_bg_df)
        logger.info("✓ Created DataFrame wrapper for AutoGluon pipeline")

    else:
        # Fallback to model-agnostic KernelExplainer
        logger.warning(f"Unknown model type '{model_type}' - using KernelExplainer (may be slow)")

        # Check if we have already processed features (i.e., not raw spectral data)
        # If so, use the extracted model directly instead of the full pipeline
        if hasattr(model, 'named_steps'):
            logger.info("Pipeline detected with processed features - using extracted model")
            # Try TreeExplainer first for the extracted model (it might be tree-based)
            try:
                explainer = shap.TreeExplainer(actual_model)
                logger.info("✓ Using TreeExplainer on extracted model")
            except Exception as e:
                logger.info(f"TreeExplainer failed ({e}), falling back to KernelExplainer on extracted model")
                explainer = shap.KernelExplainer(actual_model.predict, X_background)
        else:
            # Use full model prediction
            explainer = shap.KernelExplainer(model.predict, X_background)

    return explainer


def load_training_data(model_path: str, max_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load training data for SHAP analysis.

    Attempts to find and load the corresponding training data file.
    Falls back to generating synthetic data if training data not found.

    Args:
        model_path: Path to the saved model
        max_samples: Maximum number of samples to use (for computational efficiency)

    Returns:
        Tuple of (X_train, y_train, feature_names)
    """
    model_dir = Path(model_path).parent
    model_name = Path(model_path).stem

    # Try to find corresponding training data
    # Pattern: reports/training_summary_*.csv or saved in model directory
    reports_dir = Path("reports")

    # Look for training data in reports
    training_files = sorted(reports_dir.glob("*training_summary*.csv"))

    if training_files:
        # Use most recent training summary
        data_file = training_files[-1]
        logger.info(f"Loading training data from: {data_file}")

        # This would need to be implemented based on your data structure
        # For now, return None and we'll use model-based sampling
        logger.warning("Training data file found but loading not implemented")

    # If no training data found, use model to generate representative samples
    logger.warning("No training data found. SHAP analysis requires actual training data.")
    logger.warning("Please provide training data file path or implement data loading.")

    return None, None, None


def analyze_shap_importance(
    model_path: str,
    X_data: Optional[np.ndarray] = None,
    y_data: Optional[np.ndarray] = None,
    feature_names: Optional[list] = None,
    top_n: int = 30,
    save_plots: bool = True,
    background_samples: int = 100,
    explain_samples: int = 500
) -> pd.DataFrame:
    """
    Comprehensive SHAP-based feature importance analysis.

    Args:
        model_path: Path to saved model file
        X_data: Training/validation features (required for SHAP)
        y_data: Training/validation targets (optional, for analysis)
        feature_names: List of feature names
        top_n: Number of top features to display
        save_plots: Whether to save visualization plots
        background_samples: Number of background samples for SHAP
        explain_samples: Number of samples to explain

    Returns:
        DataFrame with SHAP-based feature importance
    """
    print("="*100)
    print("SHAP-BASED FEATURE IMPORTANCE ANALYSIS")
    print(f"Model: {Path(model_path).name}")
    print("="*100)

    # Load model
    try:
        model = joblib.load(model_path)
        logger.info(f"✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

    # Try to load feature names from .feature_names.json file
    feature_names_file = Path(model_path).with_suffix('.feature_names.json')
    if feature_names is None and feature_names_file.exists():
        try:
            import json
            with open(feature_names_file, 'r') as f:
                feature_data = json.load(f)
                if 'feature_names' in feature_data:
                    feature_names = feature_data['feature_names']
                    logger.info(f"✓ Loaded {len(feature_names)} feature names from {feature_names_file.name}")
                    logger.info(f"  Feature selection: {feature_data.get('transformations', {}).get('feature_selection', {}).get('method', 'None')}")
        except Exception as e:
            logger.warning(f"Failed to load feature names from {feature_names_file}: {e}")

    # Detect model type
    model_type = detect_model_type(model)
    print(f"\n🔍 Model Type Detected: {model_type}")

    # Check if data is provided
    if X_data is None:
        logger.error("❌ X_data is required for SHAP analysis")
        logger.error("Please provide training/validation data:")
        logger.error("  analyze_feature_importance.py <model_path> <data_path>")
        return None

    # Get feature names
    if feature_names is None:
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        else:
            feature_names = [f"feature_{i}" for i in range(X_data.shape[1])]
            logger.warning("⚠️  No feature names found, using generic names")

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {X_data.shape[0]}")
    print(f"   Total features: {X_data.shape[1]}")
    print(f"   Background samples for SHAP: {background_samples}")
    print(f"   Samples to explain: {min(explain_samples, X_data.shape[0])}")

    # Sample background data for SHAP
    # Use k-means clustering to get representative samples
    if X_data.shape[0] > background_samples:
        logger.info(f"Sampling {background_samples} background samples using k-means...")
        X_background = shap.kmeans(X_data, background_samples).data
    else:
        X_background = X_data

    # Sample data to explain
    explain_indices = np.random.choice(X_data.shape[0], min(explain_samples, X_data.shape[0]), replace=False)
    X_explain = X_data[explain_indices]

    # Create SHAP explainer
    try:
        explainer = create_shap_explainer(model, X_background, model_type, feature_names=feature_names)
    except Exception as e:
        logger.error(f"Failed to create SHAP explainer: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Calculate SHAP values
    print(f"\n⚙️  Calculating SHAP values...")
    print(f"   This may take a few minutes depending on model complexity...")

    # For AutoGluon, convert to DataFrame for explanation
    X_explain_for_shap = X_explain
    if model_type == 'autogluon' and feature_names is not None:
        X_explain_for_shap = pd.DataFrame(X_explain, columns=feature_names)
        logger.info("Converting X_explain to DataFrame for AutoGluon")

    try:
        shap_values = explainer(X_explain_for_shap)
        logger.info("✓ SHAP values calculated successfully")
    except Exception as e:
        logger.error(f"Failed to calculate SHAP values: {e}")
        logger.info("Trying alternative calculation method...")
        try:
            # Fallback: calculate without Explanation object wrapper
            shap_values_raw = explainer.shap_values(X_explain_for_shap)

            # Handle multi-output models
            if isinstance(shap_values_raw, list):
                shap_values_raw = shap_values_raw[0]

            # Create Explanation object manually
            shap_values = shap.Explanation(
                values=shap_values_raw,
                base_values=np.full(X_explain.shape[0], explainer.expected_value),
                data=X_explain,
                feature_names=feature_names
            )
            logger.info("✓ SHAP values calculated using fallback method")
        except Exception as e2:
            logger.error(f"Fallback method also failed: {e2}")
            return None

    # Calculate global feature importance (mean absolute SHAP values)
    # This is the CORRECT way to measure feature importance with SHAP
    shap_importance = np.abs(shap_values.values).mean(axis=0)

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': shap_importance
    })

    # Sort by importance
    importance_df = importance_df.sort_values('shap_importance', ascending=False)

    # Calculate statistics
    total_importance = importance_df['shap_importance'].sum()
    cumsum = importance_df['shap_importance'].cumsum() / total_importance * 100
    importance_df['cumulative_%'] = cumsum
    importance_df['%_of_total'] = importance_df['shap_importance'] / total_importance * 100

    # Display overall statistics
    print(f"\n📈 SHAP Importance Statistics:")
    print(f"   Total features: {len(importance_df)}")
    print(f"   Mean SHAP importance: {importance_df['shap_importance'].mean():.6f}")
    print(f"   Max SHAP importance: {importance_df['shap_importance'].max():.6f}")
    print(f"   Min SHAP importance: {importance_df['shap_importance'].min():.6f}")

    # Display top N features
    print(f"\n🏆 Top {top_n} Most Important Features (SHAP):")
    print("-"*100)
    print(f"{'Rank':<6} {'Feature':<50} {'SHAP Value':<15} {'% of Total':<12} {'Cumulative %':<15}")
    print("-"*100)

    for idx, row in importance_df.head(top_n).iterrows():
        rank = list(importance_df.index).index(idx) + 1
        print(f"{rank:<6} {row['feature']:<50} {row['shap_importance']:<15.6f} {row['%_of_total']:<12.2f} {row['cumulative_%']:<15.2f}")

    # Categorize features
    print(f"\n📂 Feature Categories (SHAP-based):")
    print("-"*100)

    categories = {
        'Physics-Informed': ['fwhm', 'gamma', 'asymmetry', 'absorption_index', 'fit_quality', 'amplitude', 'kurtosis'],
        'Peak Area': ['_peak_'],
        'Simple Features': ['_simple_'],
        'Ratios': ['ratio', 'K_C_ratio'],
        'Enhanced Features': ['KC_ratio', 'KH_ratio', 'squared', 'cubic', 'log']
    }

    for cat_name, keywords in categories.items():
        cat_features = importance_df[importance_df['feature'].str.contains('|'.join(keywords), case=False, na=False)]
        if len(cat_features) > 0:
            cat_total_importance = cat_features['shap_importance'].sum()
            cat_pct = cat_total_importance / total_importance * 100
            print(f"\n{cat_name}: {len(cat_features)} features ({cat_pct:.1f}% of total SHAP importance)")

            # Show top 5 from this category
            top_cat = cat_features.head(5)
            for idx, row in top_cat.iterrows():
                rank = list(importance_df.index).index(idx) + 1
                print(f"  #{rank:<4} {row['feature']:<45} {row['shap_importance']:.6f} ({row['%_of_total']:.2f}%)")

    # Physics-informed feature breakdown
    print(f"\n🔬 Physics-Informed Features Breakdown (SHAP):")
    print("-"*100)

    physics_types = {
        'FWHM': 'fwhm',
        'Gamma (Stark)': 'gamma',
        'Asymmetry': 'asymmetry',
        'Kurtosis': 'kurtosis',
        'Absorption Index': 'absorption_index',
        'Fit Quality': 'fit_quality',
        'Amplitude': 'amplitude'
    }

    for name, keyword in physics_types.items():
        features = importance_df[importance_df['feature'].str.contains(keyword, case=False, na=False)]
        if len(features) > 0:
            total_imp = features['shap_importance'].sum()
            pct = total_imp / total_importance * 100
            print(f"\n{name}: {len(features)} features ({pct:.2f}% of total SHAP importance)")

            # Show top 3
            for idx, row in features.head(3).iterrows():
                rank = list(importance_df.index).index(idx) + 1
                print(f"  #{rank:<4} {row['feature']:<45} {row['shap_importance']:.6f}")

    # Cumulative importance milestones
    print(f"\n📊 Cumulative SHAP Importance Milestones:")
    print("-"*100)
    for threshold in [50, 80, 90, 95]:
        n_features = (importance_df['cumulative_%'] <= threshold).sum() + 1
        print(f"  Top {n_features:3d} features cover {threshold}% of total SHAP importance")

    # Generate visualizations
    if save_plots:
        output_dir = Path(model_path).parent / "shap_analysis"
        output_dir.mkdir(exist_ok=True)

        print(f"\n📊 Generating SHAP visualizations...")

        # 1. Summary plot (beeswarm)
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False, max_display=top_n)
            plt.tight_layout()
            summary_file = output_dir / f"{Path(model_path).stem}_shap_summary.png"
            plt.savefig(summary_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ Summary plot saved: {summary_file}")
        except Exception as e:
            logger.warning(f"Failed to create summary plot: {e}")

        # 2. Bar plot (global importance)
        try:
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_explain, feature_names=feature_names,
                            plot_type="bar", show=False, max_display=top_n)
            plt.tight_layout()
            bar_file = output_dir / f"{Path(model_path).stem}_shap_bar.png"
            plt.savefig(bar_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ Bar plot saved: {bar_file}")
        except Exception as e:
            logger.warning(f"Failed to create bar plot: {e}")

        # 3. Custom importance bar plot with categories
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            top_features = importance_df.head(top_n)

            # Color by category
            colors = []
            for feature in top_features['feature']:
                if any(kw in feature.lower() for kw in ['fwhm', 'gamma', 'asymmetry', 'kurtosis', 'absorption']):
                    colors.append('#1f77b4')  # Blue for physics
                elif '_peak_' in feature:
                    colors.append('#ff7f0e')  # Orange for peak area
                elif 'ratio' in feature.lower():
                    colors.append('#2ca02c')  # Green for ratios
                else:
                    colors.append('#d62728')  # Red for others

            ax.barh(range(len(top_features)), top_features['shap_importance'], color=colors)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'], fontsize=8)
            ax.set_xlabel('Mean |SHAP value| (average impact on model output)', fontsize=10)
            ax.set_title(f'Top {top_n} Features by SHAP Importance', fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            plt.tight_layout()

            custom_file = output_dir / f"{Path(model_path).stem}_shap_custom_bar.png"
            plt.savefig(custom_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ Custom bar plot saved: {custom_file}")
        except Exception as e:
            logger.warning(f"Failed to create custom bar plot: {e}")

        print(f"\n💾 Visualizations saved to: {output_dir}")

    # Save importance table to CSV
    output_file = Path(model_path).parent / f"{Path(model_path).stem}_shap_importance.csv"
    importance_df.to_csv(output_file, index=False)
    logger.info(f"✓ SHAP importance table saved: {output_file}")

    return importance_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SHAP-based feature importance analysis for LIBS spectral models"
    )
    parser.add_argument(
        "model_path",
        nargs='?',
        help="Path to trained model file (default: latest optimized model)"
    )
    parser.add_argument(
        "--data",
        help="Path to training data CSV file (required for SHAP)",
        default=None
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top features to display (default: 30)"
    )
    parser.add_argument(
        "--background-samples",
        type=int,
        default=100,
        help="Number of background samples for SHAP (default: 100)"
    )
    parser.add_argument(
        "--explain-samples",
        type=int,
        default=500,
        help="Number of samples to explain (default: 500)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating visualization plots"
    )

    args = parser.parse_args()

    # Find model if not specified
    if args.model_path is None:
        models_dir = Path("models")
        pattern = "optimized_xgboost_simple_only_*.pkl"
        models = sorted(models_dir.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)

        if not models:
            logger.error(f"❌ No models found matching {pattern}")
            logger.error("Please specify model path explicitly")
            sys.exit(1)

        args.model_path = str(models[0])
        logger.info(f"Using latest model: {args.model_path}")

    # Load data if provided
    X_data = None
    y_data = None
    feature_names = None

    if args.data:
        logger.info(f"Loading data from: {args.data}")
        try:
            df = pd.read_csv(args.data)
            # Assume last column is target, rest are features
            y_data = df.iloc[:, -1].values
            X_data = df.iloc[:, :-1].values
            feature_names = list(df.columns[:-1])
            logger.info(f"✓ Loaded {X_data.shape[0]} samples with {X_data.shape[1]} features")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            sys.exit(1)
    else:
        logger.error("❌ Data file is required for SHAP analysis")
        logger.error("Usage: python analyze_feature_importance.py <model_path> --data <data.csv>")
        sys.exit(1)

    # Run analysis
    importance_df = analyze_shap_importance(
        model_path=args.model_path,
        X_data=X_data,
        y_data=y_data,
        feature_names=feature_names,
        top_n=args.top_n,
        save_plots=not args.no_plots,
        background_samples=args.background_samples,
        explain_samples=args.explain_samples
    )

    if importance_df is not None:
        print("\n" + "="*100)
        print("✅ SHAP analysis completed successfully!")
        print("="*100)
