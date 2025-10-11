#!/usr/bin/env python3
"""
Extract training data for SHAP analysis.

This script loads processed training data and saves it in a format
suitable for SHAP feature importance analysis.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.pipeline_config import config  # Import global config instance
from src.data_management.data_manager import DataManager
from src.features.feature_engineering import create_feature_pipeline
from src.features.concentration_features import create_enhanced_feature_pipeline_with_concentration
from src.cleansing.data_cleanser import DataCleanser

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def extract_training_data(strategy: str = "simple_only", output_file: str = None, max_samples: int = None, feature_names_file: str = None):
    """
    Extract and save training data for SHAP analysis.

    Args:
        strategy: Feature strategy ('K_only', 'simple_only', 'full_context')
        output_file: Output CSV file path
        max_samples: Maximum number of samples to extract (for large datasets)
    """
    logger.info("="*80)
    logger.info("EXTRACTING TRAINING DATA FOR SHAP ANALYSIS")
    logger.info("="*80)
    logger.info(f"Strategy: {strategy}")

    # Initialize data manager
    data_manager = DataManager(config)

    # Load metadata and training files
    logger.info("Loading metadata and training files...")
    metadata = data_manager.load_and_prepare_metadata()
    training_files = data_manager.get_training_data_paths()
    logger.info(f"✓ Found {len(training_files)} training files")

    # Load and clean data (same as load_and_clean_data in main.py)
    data_cleanser = DataCleanser(config)
    processed_data_for_training = []

    logger.info("Processing spectral data...")
    for i, file_path in enumerate(training_files):
        if i % 100 == 0:
            logger.info(f"  Processing {i}/{len(training_files)} files...")

        wavelengths, intensities = data_manager.load_spectral_file(file_path)

        # Ensure intensity is 2D
        if intensities.ndim == 1:
            intensities = intensities.reshape(-1, 1)

        clean_intensities = data_cleanser.clean_spectra(str(file_path), intensities)

        if clean_intensities.size > 0:
            # Average across remaining shots
            avg_intensities = np.mean(clean_intensities, axis=1)

            # Get sample ID from file name
            sample_id = file_path.stem.replace('.csv', '')
            processed_data_for_training.append({
                config.sample_id_column: sample_id,
                'wavelengths': wavelengths,
                'intensities': avg_intensities
            })

    # Create DataFrame
    full_dataset = pd.DataFrame(processed_data_for_training)

    # Merge with metadata to get target values
    full_dataset = full_dataset.merge(metadata, on=config.sample_id_column, how='inner')
    logger.info(f"✓ Loaded {len(full_dataset)} samples after merging with metadata")

    # Split data
    logger.info("Splitting data into train/validation sets...")
    train_df, val_df = data_manager.create_reproducible_splits(full_dataset)
    logger.info(f"✓ Train: {len(train_df)}, Validation: {len(val_df)}")

    # Combine train and validation for SHAP (more representative)
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    logger.info(f"Combined train+val: {len(combined_df)} samples")

    # Limit samples if requested
    if max_samples and len(combined_df) > max_samples:
        logger.info(f"Sampling {max_samples} random samples from {len(combined_df)}")
        combined_df = combined_df.sample(n=max_samples, random_state=42)

    # Create feature pipeline with concentration-aware features (matches training)
    # Use parallel processing to match training feature generation (70 vs 112 features difference)
    logger.info(f"Creating enhanced feature pipeline with concentration features for strategy: {strategy}")
    pipeline = create_enhanced_feature_pipeline_with_concentration(config, strategy, exclude_scaler=True, use_parallel=True, n_jobs=-1)

    # Prepare features
    logger.info("Extracting features...")
    X_combined = combined_df[['wavelengths', 'intensities']].copy()
    y_combined = combined_df[config.target_column].values

    try:
        # Fit and transform
        X_features = pipeline.fit_transform(X_combined, y_combined)
        logger.info(f"✓ Extracted {X_features.shape[1]} features from {X_features.shape[0]} samples")
    except Exception as e:
        logger.error(f"Failed to extract features: {e}")
        return None

    # Get feature names
    if hasattr(X_features, 'columns'):
        feature_names = list(X_features.columns)
        X_array = X_features.values
    else:
        # Try to get from pipeline - collect from all feature generators
        feature_names = None
        try:
            # First try standard method
            feature_names = list(pipeline.get_feature_names_out())
            logger.info(f"Got feature names from pipeline.get_feature_names_out()")
        except Exception as e1:
            logger.debug(f"get_feature_names_out() failed: {e1}, trying fallback")
            # Fallback: collect from all feature generation steps
            try:
                all_feature_names = []
                for step_name, transformer in pipeline.steps:
                    if 'spectral' in step_name.lower() or 'feature' in step_name.lower():
                        if hasattr(transformer, 'get_feature_names_out'):
                            try:
                                names = transformer.get_feature_names_out()
                                step_feature_names = names.tolist() if hasattr(names, 'tolist') else list(names)
                                all_feature_names.extend(step_feature_names)
                                logger.info(f"Got {len(step_feature_names)} feature names from '{step_name}'")
                            except Exception as e_inner:
                                logger.debug(f"Failed to get feature names from {step_name}: {e_inner}")
                                continue

                if all_feature_names:
                    feature_names = all_feature_names
                    logger.info(f"Combined {len(feature_names)} feature names from all feature steps")
            except Exception as e2:
                logger.warning(f"Could not extract feature names from pipeline: {e2}")

        # Final fallback to generic names
        if feature_names is None or len(feature_names) != X_features.shape[1]:
            logger.warning(f"Using generic feature names (got {len(feature_names) if feature_names else 0}, need {X_features.shape[1]})")
            feature_names = [f"feature_{i}" for i in range(X_features.shape[1])]

        X_array = X_features

    logger.info(f"Feature names: {len(feature_names)}")

    # Create output DataFrame
    output_df = pd.DataFrame(X_array, columns=feature_names)
    output_df[config.target_column] = y_combined

    # Filter features if feature_names_file is provided (for feature selection)
    if feature_names_file:
        import json
        try:
            logger.info(f"Loading feature selection from: {feature_names_file}")
            with open(feature_names_file, 'r') as f:
                feature_data = json.load(f)

            if 'feature_names' in feature_data:
                selected_features = feature_data['feature_names']
                original_count = len(output_df.columns) - 1  # Exclude target column

                # Filter to only selected features (preserve target column)
                available_features = [f for f in selected_features if f in output_df.columns]
                missing_features = [f for f in selected_features if f not in output_df.columns]

                if missing_features:
                    logger.warning(f"⚠️  {len(missing_features)} selected features not found in extracted data:")
                    for f in missing_features[:5]:  # Show first 5
                        logger.warning(f"  - {f}")
                    if len(missing_features) > 5:
                        logger.warning(f"  ... and {len(missing_features) - 5} more")

                # Keep only selected features + target
                output_df = output_df[available_features + [config.target_column]]
                logger.info(f"✓ Feature selection applied: {original_count} → {len(available_features)} features")

                # Update feature_names for reporting
                feature_names = available_features
        except Exception as e:
            logger.warning(f"Failed to apply feature selection from {feature_names_file}: {e}")
            logger.info("Continuing with all features")

    # Save to CSV
    if output_file is None:
        output_file = f"data/training_data_{strategy}_for_shap.csv"

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved training data to: {output_path}")
    logger.info(f"  Samples: {len(output_df)}")
    logger.info(f"  Features: {len(feature_names)}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Display sample statistics
    logger.info("\n📊 Target Distribution:")
    logger.info(f"  Mean: {y_combined.mean():.4f}")
    logger.info(f"  Std:  {y_combined.std():.4f}")
    logger.info(f"  Min:  {y_combined.min():.4f}")
    logger.info(f"  Max:  {y_combined.max():.4f}")

    logger.info("\n✅ Data extraction complete!")
    logger.info(f"Use this file for SHAP analysis:")
    logger.info(f"  python analyze_feature_importance.py <model> --data {output_path}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract training data for SHAP feature importance analysis"
    )
    parser.add_argument(
        "--strategy",
        default="simple_only",
        choices=["K_only", "simple_only", "full_context"],
        help="Feature engineering strategy (default: simple_only)"
    )
    parser.add_argument(
        "--output",
        help="Output CSV file path (default: data/training_data_<strategy>_for_shap.csv)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to extract (default: all)"
    )
    parser.add_argument(
        "--feature-names-file",
        help="Path to .feature_names.json file to apply feature selection (optional)"
    )

    args = parser.parse_args()

    extract_training_data(
        strategy=args.strategy,
        output_file=args.output,
        max_samples=args.max_samples,
        feature_names_file=args.feature_names_file
    )
