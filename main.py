"""
Main Orchestration Script for the Potassium Prediction Pipeline.

This script serves as the entry point to run different stages of the ML pipeline,
including data preparation, standard training, AutoGluon training, hyperparameter
tuning, and prediction.

Usage:
  - python main.py train            (Trains all standard models from the config)
  - python main.py train --models xgboost lightgbm  (Trains only specified models)
  - python main.py autogluon        (Runs the automated AutoGluon pipeline)
  - python main.py tune             (Runs Optuna hyperparameter optimization)
  - python main.py tune --models xgboost lightgbm   (Tunes only specified models)
  - python main.py predict --input-file ... --model-path ... (Makes a prediction)
"""
import argparse
import base64
import binascii
import json
import logging
import os
import tempfile
import yaml
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import shutil

from src.config.pipeline_config import config, Config
from src.config.config_manager import config_manager

def create_config_from_env():
    """Create a temporary configuration file from environment variables starting with CONFIG_"""
    config_dict = {}
    
    # Collect all CONFIG_* environment variables
    for key, value in os.environ.items():
        if key.startswith('CONFIG_'):
            # Remove CONFIG_ prefix and convert to lowercase
            config_key = key[7:].lower()
            
            # Handle nested configurations (e.g., CONFIG_AUTOGLUON_TIME_LIMIT -> autogluon.time_limit)
            parts = config_key.split('_')
            
            # Convert string values to appropriate types
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif '.' in value and value.replace('.', '').isdigit():
                value = float(value)
            
            # Build nested dictionary structure
            current = config_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
    
    # Create temporary YAML file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config_dict, temp_file, default_flow_style=False, indent=2)
    temp_file.close()
    
    return temp_file.name

def create_config_from_json(json_string):
    """Create a temporary configuration file from JSON string"""
    try:
        import json
        config_dict = json.loads(json_string)
        
        # Create temporary YAML file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config_dict, temp_file, default_flow_style=False, indent=2)
        temp_file.close()
        
        return temp_file.name
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON configuration: {e}")

def create_config_from_base64(base64_string):
    """Create a temporary configuration file from base64-encoded JSON string"""
    try:
        import json
        import base64
        # Decode base64 to get JSON string
        json_string = base64.b64decode(base64_string.encode('utf-8')).decode('utf-8')
        config_dict = json.loads(json_string)
        
        # Create temporary YAML file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config_dict, temp_file, default_flow_style=False, indent=2)
        temp_file.close()
        
        return temp_file.name
    except (json.JSONDecodeError, binascii.Error) as e:
        raise ValueError(f"Invalid base64-encoded JSON configuration: {e}")

# Ensure directories exist for cloud deployments
config.ensure_paths_exist(create_dirs=True)
from src.data_management.data_manager import DataManager
from src.cleansing.data_cleanser import DataCleanser
from src.models.model_trainer import ModelTrainer
from src.models.autogluon_trainer import AutoGluonTrainer
from src.models.model_tuner import ModelTuner
from src.models.predictor import Predictor
from src.models.optimize_xgboost import XGBoostOptimizer
from src.models.optimize_all_models import UnifiedModelOptimizer
from src.models.optimize_range_specialist_neural_net import RangeSpecialistOptimizer
from src.reporting.reporter import Reporter
from src.utils.helpers import calculate_regression_metrics, setup_logging
from src.utils.custom_exceptions import DataValidationError, PipelineError
from src.analysis.mislabel_detector import MislabelDetector



logger = logging.getLogger(__name__)

def get_display_strategy_name(strategy: str, use_raw_spectral_data: bool) -> str:
    """
    Returns the strategy name to use for display purposes.
    When raw_spectral_data is enabled, returns 'raw-spectral' instead of the original strategy.
    
    Args:
        strategy: The original strategy name
        use_raw_spectral_data: Whether raw spectral data mode is enabled
        
    Returns:
        The strategy name to use for display
    """
    if use_raw_spectral_data:
        return "raw-spectral"
    return strategy

def setup_pipeline_config(use_gpu: bool = False, use_raw_spectral: Optional[bool] = None, 
                          validation_dir: Optional[str] = None, 
                          config_path: Optional[str] = None) -> Config:
    """Sets up dynamic configuration values and creates all necessary directories."""
    # Load saved config if provided
    if config_path:
        stored_config_path = Path(config_path)
        base_config = config_manager.apply_config(config, stored_config_path)
        logger.info(f"Loaded configuration from: {config_path}")
    else:
        base_config = config
    
    # Create updated config with CLI overrides using model_copy
    update_dict = {
        "run_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "use_gpu": use_gpu
    }
    
    # Only override use_raw_spectral_data if explicitly provided via command line
    if use_raw_spectral is not None:
        update_dict["use_raw_spectral_data"] = use_raw_spectral
    
    # Only override custom_validation_dir if explicitly provided
    if validation_dir is not None:
        update_dict["custom_validation_dir"] = validation_dir
    
    updated_config = base_config.model_copy(update=update_dict)
    
    if use_gpu:
        logger.info("GPU mode enabled - models will use CUDA acceleration where available")
    else:
        logger.info("GPU mode disabled - models will use CPU only")
        
    if use_raw_spectral:
        logger.info("RAW SPECTRAL mode enabled - using raw intensities from PeakRegions DIRECT to models")
        logger.info(f"Raw mode: 63 raw intensity features from {len(updated_config.all_regions)} PeakRegions -> DIRECT to models (no imputation/clipping/concentration features)")
    else:
        logger.info("FEATURE ENGINEERING mode enabled - using mathematical feature extraction")
    
    if updated_config.custom_validation_dir:
        logger.info(f"Custom validation directory configured: {updated_config.custom_validation_dir}")
    project_root = Path(__file__).resolve().parents[0]
    
    # Define all paths relative to the project root
    data_dir = project_root / "data"
    raw_data_dir = data_dir / "raw" / "data_5278_Phase3"
    processed_data_dir = data_dir / "processed"
    averaged_files_dir = data_dir / "averaged_files_per_sample"
    cleansed_files_dir = data_dir / "cleansed_files_per_sample"
    model_dir = project_root / "models"
    reports_dir = project_root / "reports"
    log_dir = project_root / "logs"
    bad_files_dir = project_root / "bad_files"
    bad_prediction_files_dir = project_root / "bad_prediction_files"
    reference_data_path = project_root / "data" / "reference_data" / "Final_Lab_Data_Nico_New.xlsx"
    
    # Create all directories first, before assigning to config
    for dir_path in [
        processed_data_dir, averaged_files_dir, cleansed_files_dir,
        model_dir, reports_dir, log_dir, bad_files_dir, bad_prediction_files_dir
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Now assign paths to updated_config after directories exist (using private attributes)
    updated_config._data_dir = str(data_dir)
    updated_config._raw_data_dir = str(raw_data_dir)
    updated_config._processed_data_dir = str(processed_data_dir)
    updated_config._averaged_files_dir = str(averaged_files_dir)
    updated_config._cleansed_files_dir = str(cleansed_files_dir)
    updated_config._model_dir = str(model_dir)
    updated_config._reports_dir = str(reports_dir)
    updated_config._log_dir = str(log_dir)
    updated_config._bad_files_dir = str(bad_files_dir)
    updated_config._bad_prediction_files_dir = str(bad_prediction_files_dir)
    updated_config._reference_data_path = str(reference_data_path)
    updated_config.sample_id_column = "Sample ID"
    
    setup_logging()
    return updated_config

def run_data_preparation(cfg: Config, use_data_parallel: bool = False, data_n_jobs: int = -1):
    """Orchestrates the initial data preparation: averaging raw files."""
    # Clean up any existing averaged files to ensure consistent state
    import shutil
    if cfg.averaged_files_dir.exists():
        shutil.rmtree(cfg.averaged_files_dir)
    cfg.averaged_files_dir.mkdir(parents=True, exist_ok=True)
    
    data_manager = DataManager(cfg)
    if use_data_parallel:
        from src.data_management.parallel_data_manager import parallel_average_raw_files
        parallel_average_raw_files(data_manager, n_jobs=data_n_jobs)
    else:
        data_manager.average_raw_files()

def load_and_clean_data(cfg: Config, use_data_parallel: bool = False, data_n_jobs: int = -1, exclude_suspects_file: Optional[str] = None) -> Tuple[pd.DataFrame, DataManager]:
    """Loads averaged data, cleans it, saves the clean version, and moves bad files.

    Args:
        cfg: Pipeline configuration
        use_data_parallel: Enable parallel processing for data operations
        data_n_jobs: Number of parallel jobs (-1 uses all cores)
        exclude_suspects_file: Path to CSV file containing sample IDs to exclude
    """
    data_manager = DataManager(cfg)
    metadata = data_manager.load_and_prepare_metadata()
    training_files = data_manager.get_training_data_paths()
    
    # Determine global wavelength range from raw files if standardization is enabled
    if cfg.enable_wavelength_standardization:
        data_manager._global_wavelength_range = data_manager.determine_global_wavelength_range_from_raw_files()

    # Use parallel processing if requested
    if use_data_parallel:
        from src.data_management.parallel_data_manager import parallel_load_and_clean_data
        processed_data_for_training = parallel_load_and_clean_data(
            cfg, training_files, metadata, 
            global_wavelength_range=getattr(data_manager, '_global_wavelength_range', None),
            n_jobs=data_n_jobs
        )
    else:
        # Serial processing
        data_cleanser = DataCleanser(cfg)
        processed_data_for_training = []

        logger.info(f"Cleansing {len(training_files)} averaged files...")
        for file_path in training_files:
            wavelengths, intensities = data_manager.load_spectral_file(file_path)
            
            # Apply wavelength standardization if enabled
            if cfg.enable_wavelength_standardization:
                wavelengths, intensities = data_manager.standardize_wavelength_grid(
                    wavelengths, intensities, 
                    interpolation_method=cfg.wavelength_interpolation_method
                )
            
            # Ensure intensity is 2D for the cleanser, as averaged files might have 1D intensity
            if intensities.ndim == 1:
                intensities = intensities.reshape(-1, 1)

            clean_intensities = data_cleanser.clean_spectra(str(file_path), intensities)
            
            if clean_intensities.size > 0:
                # Save the cleansed file for auditing and reuse
                cleansed_path = cfg.cleansed_files_dir / file_path.name
                # Start DataFrame with the Wavelength column
                cleansed_df = pd.DataFrame({'Wavelength': wavelengths})
                
                # Add each remaining intensity column (shot) one by one
                for i in range(clean_intensities.shape[1]):
                    cleansed_df[f'Intensity{i+1}'] = clean_intensities[:, i]
                    
                cleansed_df.to_csv(cleansed_path, index=False)

                # Prepare the data for in-memory use in the current pipeline run
                sample_id = file_path.name.replace('.csv.txt', '')
                target = metadata.loc[metadata[cfg.sample_id_column] == sample_id, cfg.target_column].values[0]
                processed_data_for_training.append({
                     cfg.sample_id_column: sample_id, "wavelengths": wavelengths,
                    "intensities": clean_intensities, cfg.target_column: target
                })
            else:
                # File had too many outliers; move the *averaged* source file to bad_files
                try:
                    destination = cfg.bad_files_dir / file_path.name
                    shutil.move(str(file_path), str(destination))
                    logger.debug(f"Moved bad averaged file {file_path.name} to {cfg.bad_files_dir}")
                except Exception as e:
                    logger.error(f"Failed to move bad file {file_path.name}: {e}")
    
    if not processed_data_for_training:
        raise DataValidationError("No data left after cleansing. Aborting pipeline.")

    logger.info(f"Data cleaning complete. {len(processed_data_for_training)} files are ready for training.")

    # Exclude suspicious samples if specified (before DataFrame conversion to avoid corruption)
    if exclude_suspects_file:
        logger.info("=" * 80)
        logger.info("SAMPLE EXCLUSION PROCESS")
        logger.info("=" * 80)
        logger.info(f"Exclusion file specified: {exclude_suspects_file}")

        if Path(exclude_suspects_file).exists():
            try:
                suspects_df = pd.read_csv(exclude_suspects_file)
                logger.info(f"Loaded exclusion file with {len(suspects_df)} entries")

                if 'sample_id' in suspects_df.columns:
                    suspects_to_exclude = set(suspects_df['sample_id'].astype(str))
                    initial_count = len(processed_data_for_training)
                    logger.info(f"Initial sample count (before exclusion): {initial_count}")
                    logger.info(f"Number of unique samples to exclude: {len(suspects_to_exclude)}")

                    # Filter the list directly to avoid DataFrame corruption of numpy arrays
                    filtered_data = []
                    excluded_samples = []
                    for sample_data in processed_data_for_training:
                        sample_id = sample_data[cfg.sample_id_column]
                        if sample_id not in suspects_to_exclude:
                            filtered_data.append(sample_data)
                        else:
                            excluded_samples.append(sample_id)

                    processed_data_for_training = filtered_data
                    excluded_count = initial_count - len(processed_data_for_training)

                    logger.info(f"Actually excluded: {excluded_count} samples")
                    logger.info(f"Final sample count (after exclusion): {len(processed_data_for_training)}")

                    if excluded_count > 0:
                        logger.info(f"Examples of excluded samples: {excluded_samples[:5]}")

                    # Calculate percentage
                    exclusion_percentage = (excluded_count / initial_count) * 100 if initial_count > 0 else 0
                    logger.info(f"Exclusion percentage: {exclusion_percentage:.1f}%")
                    logger.info("=" * 80)

                    # Validate that we still have data
                    if len(processed_data_for_training) == 0:
                        raise DataValidationError("No samples remaining after exclusion. Check your exclusion file.")

                else:
                    logger.warning(f"Exclusion file {exclude_suspects_file} does not have 'sample_id' column. Skipping exclusion.")
            except Exception as e:
                logger.error(f"Failed to load exclusion file {exclude_suspects_file}: {e}")
        else:
            logger.warning(f"Exclusion file not found: {exclude_suspects_file}")
    else:
        logger.info("No sample exclusion file specified - using all available samples")

    # Convert to DataFrame after filtering to preserve data integrity
    df = pd.DataFrame(processed_data_for_training)

    return df, data_manager

def process_validation_data(cfg: Config, validation_dir: Path, global_data_manager: Optional[DataManager] = None) -> pd.DataFrame:
    """Process raw validation files through averaging and cleansing."""
    logger.info(f"Processing validation data from: {validation_dir}")
    
    # Create temporary directories for validation data processing
    val_averaged_dir = cfg.data_dir / f"validation_averaged_{cfg.run_timestamp}"
    val_cleansed_dir = cfg.data_dir / f"validation_cleansed_{cfg.run_timestamp}"
    val_averaged_dir.mkdir(parents=True, exist_ok=True)
    val_cleansed_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the global data manager if provided (for wavelength range consistency)
    if global_data_manager is not None:
        data_manager = global_data_manager
    else:
        data_manager = DataManager(cfg)
    data_cleanser = DataCleanser(cfg)
    metadata = data_manager.load_and_prepare_metadata()
    
    # Step 1: Average validation files
    logger.info("Averaging validation files...")
    file_groups = defaultdict(list)
    for file_path in validation_dir.glob('*.csv.txt'):
        prefix = data_manager._extract_file_prefix(file_path.name)
        file_groups[prefix].append(file_path)
    
    for prefix, file_paths in file_groups.items():
        logger.info(f"Processing validation group: {prefix} ({len(file_paths)} files)")
        wavelengths, averaged_data = data_manager.average_files_in_memory(file_paths)
        
        if wavelengths is not None and averaged_data is not None:
            output_path = val_averaged_dir / f"{prefix}.csv.txt"
            output_df = pd.DataFrame()
            output_df['Wavelength'] = wavelengths
            for i in range(averaged_data.shape[1]):
                output_df[f'Intensity{i+1}'] = averaged_data[:, i]
            output_df.to_csv(output_path, index=False)
    
    # Step 2: Clean averaged validation files and prepare dataset
    logger.info("Cleaning validation files...")
    processed_validation_data = []
    
    for file_path in val_averaged_dir.glob("*.csv.txt"):
        wavelengths, intensities = data_manager.load_spectral_file(file_path)
        
        # Apply wavelength standardization if enabled  
        if cfg.enable_wavelength_standardization:
            wavelengths, intensities = data_manager.standardize_wavelength_grid(
                wavelengths, intensities,
                interpolation_method=cfg.wavelength_interpolation_method
            )
        
        if intensities.ndim == 1:
            intensities = intensities.reshape(-1, 1)
        
        clean_intensities = data_cleanser.clean_spectra(str(file_path), intensities)
        
        if clean_intensities.size > 0:
            sample_id = file_path.name.replace('.csv.txt', '')
            
            # Check if this sample exists in metadata
            if sample_id in metadata[cfg.sample_id_column].values:
                target = metadata.loc[metadata[cfg.sample_id_column] == sample_id, cfg.target_column].values[0]
                processed_validation_data.append({
                    cfg.sample_id_column: sample_id,
                    "wavelengths": wavelengths,
                    "intensities": clean_intensities,
                    cfg.target_column: target
                })
            else:
                logger.warning(f"Validation sample {sample_id} not found in metadata, skipping...")
    
    if not processed_validation_data:
        raise DataValidationError("No valid validation data found after processing.")
    
    logger.info(f"Validation data processing complete. {len(processed_validation_data)} files ready.")
    
    # Clean up temporary directories
    shutil.rmtree(val_averaged_dir, ignore_errors=True)
    shutil.rmtree(val_cleansed_dir, ignore_errors=True)
    
    return pd.DataFrame(processed_validation_data)

def run_training_pipeline(use_gpu: bool = False, use_raw_spectral: bool = False,
                          validation_dir: Optional[str] = None,
                          config_path: Optional[str] = None,
                          models: Optional[list] = None,
                          strategy: Optional[str] = None,
                          use_parallel: bool = False,
                          n_jobs: int = -1,
                          use_data_parallel: bool = False,
                          data_n_jobs: int = -1,
                          exclude_suspects_file: Optional[str] = None,
                          shap_features_file: Optional[str] = None,
                          shap_top_n: int = 30,
                          shap_min_importance: Optional[float] = None):
    """Executes the standard model training pipeline."""
    cfg = setup_pipeline_config(use_gpu, use_raw_spectral, validation_dir, config_path)

    # Override models if specified via command line
    if models:
        cfg.models_to_train = models
        logger.info(f"Using models from command line: {models}")
    else:
        logger.info(f"Using models from config: {cfg.models_to_train}")

    # Override strategy if specified via command line
    if strategy:
        cfg.feature_strategies = [strategy]
        logger.info(f"Using strategy from command line: {strategy}")
    else:
        logger.info(f"Using feature strategies from config: {cfg.feature_strategies}")

    # Configure SHAP-based feature selection if specified
    if shap_features_file:
        cfg.use_shap_feature_selection = True
        cfg.shap_importance_file = shap_features_file
        cfg.shap_top_n_features = shap_top_n
        cfg.shap_min_importance = shap_min_importance
        logger.info(f"[SHAP FEATURE SELECTION] Enabled via command line:")
        logger.info(f"  - SHAP file: {shap_features_file}")
        logger.info(f"  - Top N: {shap_top_n}")
        if shap_min_importance:
            logger.info(f"  - Min importance: {shap_min_importance}")
    
    logger.info(f"Starting standard training run: {cfg.run_timestamp}")
    
    run_data_preparation(cfg, use_data_parallel, data_n_jobs)
    reporter = Reporter(cfg)
    full_dataset, global_data_manager = load_and_clean_data(cfg, use_data_parallel, data_n_jobs, exclude_suspects_file)
    
    # Use either the passed validation_dir or the one from config
    validation_directory = validation_dir or cfg.custom_validation_dir
    
    if validation_directory:
        # Process validation data separately
        val_path = Path(validation_directory)
        if not val_path.exists():
            raise DataValidationError(f"Validation directory not found: {val_path}")
        
        validation_dataset = process_validation_data(cfg, val_path, global_data_manager)
        
        # Extract sample IDs from validation dataset
        val_sample_ids = set(validation_dataset[cfg.sample_id_column])
        
        # Split data based on validation sample IDs
        test_df = full_dataset[full_dataset[cfg.sample_id_column].isin(val_sample_ids)]
        train_df = full_dataset[~full_dataset[cfg.sample_id_column].isin(val_sample_ids)]
        
        # If some validation samples weren't in the main dataset, add them
        missing_val_samples = val_sample_ids - set(test_df[cfg.sample_id_column])
        if missing_val_samples:
            missing_df = validation_dataset[validation_dataset[cfg.sample_id_column].isin(missing_val_samples)]
            test_df = pd.concat([test_df, missing_df], ignore_index=True)
        
        logger.info(f"Using custom validation split: {len(train_df)} training, {len(test_df)} validation samples")
    else:
        # Use standard random split
        train_df, test_df = DataManager(cfg).create_reproducible_splits(full_dataset)

    # When using raw spectral data, bypass traditional feature engineering strategies
    if cfg.use_raw_spectral_data:
        logger.info("Raw spectral mode enabled - bypassing traditional feature strategies")
        model_trainer = ModelTrainer(cfg, strategy="raw-spectral", reporter=reporter, data_manager=global_data_manager, use_parallel_features=use_parallel, feature_n_jobs=n_jobs)
        model_trainer.train_and_evaluate(train_df, test_df)
    else:
        # Traditional feature engineering mode - iterate through configured strategies
        for strategy in cfg.feature_strategies:
            model_trainer = ModelTrainer(cfg, strategy=strategy, reporter=reporter, data_manager=global_data_manager, use_parallel_features=use_parallel, feature_n_jobs=n_jobs)
            model_trainer.train_and_evaluate(train_df, test_df)
    
    # Use strategies and models info for summary filename
    if cfg.use_raw_spectral_data:
        strategies_str = "raw-spectral"
        description = f"Training run with {len(cfg.models_to_train)} models using raw-spectral mode"
    else:
        strategies_str = "_".join(cfg.feature_strategies) if len(cfg.feature_strategies) <= 3 else f"{len(cfg.feature_strategies)}strategies"
        description = f"Training run with {len(cfg.models_to_train)} models and {len(cfg.feature_strategies)} strategies"
    
    models_str = "_".join(cfg.models_to_train) if len(cfg.models_to_train) <= 3 else f"{len(cfg.models_to_train)}models"
    reporter.save_summary_report(strategy=strategies_str, model_name=models_str)
    reporter.save_config()
    
    # Save configuration for future reuse
    config_name = f"training_run_{cfg.run_timestamp}"
    config_manager.save_config(cfg, config_name, description)
    
    logger.info("Standard training pipeline run completed successfully.")

def run_autogluon_pipeline(use_gpu: bool = False, use_raw_spectral: bool = False,
                           validation_dir: Optional[str] = None,
                           config_path: Optional[str] = None,
                           strategy: Optional[str] = None,
                           use_parallel: bool = False,
                           n_jobs: int = -1,
                           use_data_parallel: bool = False,
                           data_n_jobs: int = -1,
                           exclude_suspects_file: Optional[str] = None,
                           shap_features_file: Optional[str] = None,
                           shap_top_n: Optional[int] = None,
                           shap_min_importance: Optional[float] = None):
    """Executes the dedicated AutoGluon training pipeline."""
    cfg = setup_pipeline_config(use_gpu, use_raw_spectral, validation_dir, config_path)
    logger.info(f"Starting AutoGluon training run: {cfg.run_timestamp}")
    reporter = Reporter(cfg)
    run_data_preparation(cfg, use_data_parallel, data_n_jobs)
    full_dataset, global_data_manager = load_and_clean_data(cfg, use_data_parallel, data_n_jobs, exclude_suspects_file)
    
    # Use either the passed validation_dir or the one from config
    validation_directory = validation_dir or cfg.custom_validation_dir
    
    if validation_directory:
        # Process validation data separately
        val_path = Path(validation_directory)
        if not val_path.exists():
            raise DataValidationError(f"Validation directory not found: {val_path}")
        
        validation_dataset = process_validation_data(cfg, val_path, global_data_manager)
        
        # Extract sample IDs from validation dataset
        val_sample_ids = set(validation_dataset[cfg.sample_id_column])
        
        # Split data based on validation sample IDs
        test_df = full_dataset[full_dataset[cfg.sample_id_column].isin(val_sample_ids)]
        train_df = full_dataset[~full_dataset[cfg.sample_id_column].isin(val_sample_ids)]
        
        # If some validation samples weren't in the main dataset, add them
        missing_val_samples = val_sample_ids - set(test_df[cfg.sample_id_column])
        if missing_val_samples:
            missing_df = validation_dataset[validation_dataset[cfg.sample_id_column].isin(missing_val_samples)]
            test_df = pd.concat([test_df, missing_df], ignore_index=True)
        
        logger.info(f"Using custom validation split: {len(train_df)} training, {len(test_df)} validation samples")
    else:
        # Create train/test splits like the standard pipeline
        train_df, test_df = DataManager(cfg).create_reproducible_splits(full_dataset)
    
    # Use raw-spectral mode if enabled, otherwise use configured or command line strategies
    if cfg.use_raw_spectral_data:
        strategy_for_autogluon = "raw-spectral"
        display_strategy = get_display_strategy_name(strategy_for_autogluon, cfg.use_raw_spectral_data)
        logger.info(f"Using raw-spectral mode for AutoGluon (bypassing traditional strategies).")
    elif strategy:
        # Use strategy from command line parameter
        strategy_for_autogluon = strategy
        display_strategy = get_display_strategy_name(strategy_for_autogluon, cfg.use_raw_spectral_data)
        logger.info(f"Using feature strategy '{display_strategy}' from command line for AutoGluon.")
    elif cfg.feature_strategies:
        strategy_for_autogluon = cfg.feature_strategies[0]
        display_strategy = get_display_strategy_name(strategy_for_autogluon, cfg.use_raw_spectral_data)
        logger.info(f"Using feature strategy '{display_strategy}' from config for AutoGluon.")
    else:
        strategy_for_autogluon = "simple_only"
        display_strategy = get_display_strategy_name(strategy_for_autogluon, cfg.use_raw_spectral_data)
        logger.info(f"No feature strategies in config, defaulting to '{display_strategy}' for AutoGluon.")
    
    # Use command-line SHAP parameters if provided, otherwise use config values
    effective_shap_file = shap_features_file or (cfg.shap_importance_file if cfg.use_shap_feature_selection else None)
    effective_shap_top_n = shap_top_n if shap_top_n is not None else cfg.shap_top_n_features
    effective_shap_min_importance = shap_min_importance if shap_min_importance is not None else cfg.shap_min_importance

    if effective_shap_file:
        # Check if file exists
        shap_file_path = Path(effective_shap_file)
        if not shap_file_path.exists():
            logger.warning(f"⚠️  SHAP file specified in config does not exist: {effective_shap_file}")
            logger.warning(f"   SHAP feature selection will be DISABLED for this run")
            logger.warning(f"   To fix: Update shap_importance_file in pipeline_config.py or run SHAP analysis first")
            effective_shap_file = None
        else:
            logger.info("="*80)
            logger.info("SHAP FEATURE SELECTION ENABLED (from config)")
            logger.info("="*80)
            logger.info(f"  File: {effective_shap_file}")
            logger.info(f"  Top N: {effective_shap_top_n}")
            logger.info(f"  Min importance: {effective_shap_min_importance}")
            logger.info("="*80)

    trainer = AutoGluonTrainer(
        config=cfg,
        strategy=strategy_for_autogluon,
        reporter=reporter,
        use_concentration_features=cfg.use_concentration_features,
        use_parallel_features=use_parallel,
        feature_n_jobs=n_jobs,
        shap_features_file=effective_shap_file,
        shap_top_n=effective_shap_top_n,
        shap_min_importance=effective_shap_min_importance
    )
    trainer.train_and_evaluate(train_df, test_df)
    
    reporter.save_summary_report(strategy=strategy_for_autogluon, model_name="autogluon")
    reporter.save_config()
    
    # Save configuration for future reuse
    config_name = f"autogluon_run_{cfg.run_timestamp}"
    description = f"AutoGluon run with {strategy_for_autogluon} strategy"
    config_manager.save_config(cfg, config_name, description)
    
    logger.info("AutoGluon pipeline run completed successfully.")

def run_tuning_pipeline(use_gpu: bool = False, use_raw_spectral: bool = False,
                        validation_dir: Optional[str] = None,
                        config_path: Optional[str] = None,
                        models: Optional[list] = None,
                        exclude_suspects_file: Optional[str] = None):
    """Executes the hyperparameter tuning pipeline."""
    cfg = setup_pipeline_config(use_gpu, use_raw_spectral, validation_dir, config_path)
    
    # Override models if specified via command line
    if models:
        cfg.tuner.models_to_tune = models
        logger.info(f"Using models from command line for tuning: {models}")
    else:
        logger.info(f"Using models from config for tuning: {cfg.tuner.models_to_tune}")
    
    logger.info(f"Starting tuning run: {cfg.run_timestamp}")
    
    run_data_preparation(cfg)
    reporter = Reporter(cfg)
    full_dataset, global_data_manager = load_and_clean_data(cfg, exclude_suspects_file=exclude_suspects_file)
    
    # Use either the passed validation_dir or the one from config
    validation_directory = validation_dir or cfg.custom_validation_dir
    
    if validation_directory:
        # Process validation data separately
        val_path = Path(validation_directory)
        if not val_path.exists():
            raise DataValidationError(f"Validation directory not found: {val_path}")
        
        validation_dataset = process_validation_data(cfg, val_path, global_data_manager)
        
        # Extract sample IDs from validation dataset
        val_sample_ids = set(validation_dataset[cfg.sample_id_column])
        
        # Split data based on validation sample IDs
        test_df = full_dataset[full_dataset[cfg.sample_id_column].isin(val_sample_ids)]
        train_df = full_dataset[~full_dataset[cfg.sample_id_column].isin(val_sample_ids)]
        
        # If some validation samples weren't in the main dataset, add them
        missing_val_samples = val_sample_ids - set(test_df[cfg.sample_id_column])
        if missing_val_samples:
            missing_df = validation_dataset[validation_dataset[cfg.sample_id_column].isin(missing_val_samples)]
            test_df = pd.concat([test_df, missing_df], ignore_index=True)
        
        logger.info(f"Using custom validation split: {len(train_df)} training, {len(test_df)} validation samples")
    else:
        train_df, test_df = DataManager(cfg).create_reproducible_splits(full_dataset)
    
    # When using raw spectral data, bypass traditional feature engineering strategies
    if cfg.use_raw_spectral_data:
        logger.info("Raw spectral mode enabled - using raw-spectral strategy for tuning")
        tuner = ModelTuner(cfg, reporter=reporter, strategy="raw-spectral")
        tuner.tune(train_df, test_df)
    else:
        # Traditional feature engineering mode - iterate through configured strategies
        for strategy in cfg.feature_strategies:
            tuner = ModelTuner(cfg, reporter=reporter, strategy=strategy)
            tuner.tune(train_df, test_df)
    
    reporter.save_config()
    
    # Save configuration for future reuse
    config_name = f"tuning_run_{cfg.run_timestamp}"
    description = f"Tuning run with {len(cfg.tuner.models_to_tune)} models"
    config_manager.save_config(cfg, config_name, description)
    
    logger.info("Hyperparameter tuning pipeline run completed successfully.")

def run_xgboost_optimization_pipeline(use_gpu: bool = False, validation_dir: Optional[str] = None,
                                     config_path: Optional[str] = None, strategy: Optional[str] = None,
                                     n_trials: Optional[int] = None, timeout: Optional[int] = None,
                                     use_parallel: bool = False, n_jobs: int = -1,
                                     exclude_suspects_file: Optional[str] = None):
    """Executes dedicated XGBoost optimization pipeline."""
    cfg = setup_pipeline_config(use_gpu, validation_dir, config_path)
    
    # Use config values when command line values are not provided
    if n_trials is None:
        n_trials = cfg.tuner.n_trials
        logger.info(f"Using n_trials from config: {n_trials}")
    else:
        logger.info(f"Using n_trials from command line: {n_trials}")
    
    if timeout is None:
        timeout = cfg.tuner.timeout
        logger.info(f"Using timeout from config: {timeout}")
    else:
        logger.info(f"Using timeout from command line: {timeout}")
    
    if strategy is None:
        strategy = cfg.feature_strategies[0] if cfg.feature_strategies else "simple_only"
        logger.info(f"Using strategy from config: {strategy}")
    else:
        logger.info(f"Using strategy from command line: {strategy}")
        
    logger.info(f"Starting XGBoost optimization run: {cfg.run_timestamp}")
    
    run_data_preparation(cfg)
    full_dataset, global_data_manager = load_and_clean_data(cfg, exclude_suspects_file=exclude_suspects_file)
    
    # Use either the passed validation_dir or the one from config
    validation_directory = validation_dir or cfg.custom_validation_dir
    
    if validation_directory:
        # Process validation data separately
        val_path = Path(validation_directory)
        if not val_path.exists():
            raise DataValidationError(f"Validation directory not found: {val_path}")
        
        validation_dataset = process_validation_data(cfg, val_path, global_data_manager)
        
        # Extract sample IDs from validation dataset
        val_sample_ids = set(validation_dataset[cfg.sample_id_column])
        
        # Split data based on validation sample IDs
        test_df = full_dataset[full_dataset[cfg.sample_id_column].isin(val_sample_ids)]
        train_df = full_dataset[~full_dataset[cfg.sample_id_column].isin(val_sample_ids)]
        
        # If some validation samples weren't in the main dataset, add them
        missing_val_samples = val_sample_ids - set(test_df[cfg.sample_id_column])
        if missing_val_samples:
            missing_df = validation_dataset[validation_dataset[cfg.sample_id_column].isin(missing_val_samples)]
            test_df = pd.concat([test_df, missing_df], ignore_index=True)
        
        logger.info(f"Using custom validation split: {len(train_df)} training, {len(test_df)} validation samples")
    else:
        # Use standard random split
        train_df, test_df = DataManager(cfg).create_reproducible_splits(full_dataset)
    
    # Extract features and targets - following training pattern
    X_train_raw = train_df.drop(columns=[cfg.target_column])
    y_train = train_df[cfg.target_column].values
    X_test_raw = test_df.drop(columns=[cfg.target_column])  
    y_test = test_df[cfg.target_column].values
    
    # Extract sample IDs for detailed prediction saving (following training pattern)
    test_sample_ids = X_test_raw[cfg.sample_id_column]
    
    # Prepare features for optimization (drop sample_id)
    X_train = X_train_raw.drop(columns=[cfg.sample_id_column])
    X_test = X_test_raw.drop(columns=[cfg.sample_id_column])
    
    # Initialize optimizer
    optimizer = XGBoostOptimizer(cfg, strategy, use_parallel_features=use_parallel, feature_n_jobs=n_jobs)
    
    # Run optimization
    logger.info(f"Starting XGBoost optimization with {n_trials} trials, timeout: {timeout}s")
    study = optimizer.optimize(X_train, y_train, n_trials, timeout)
    
    # Train final model with best parameters
    final_pipeline, train_metrics, test_metrics = optimizer.train_final_model(
        X_train, y_train, X_test, y_test
    )
    
    # Make predictions for detailed reporting (following training pattern)
    y_pred = final_pipeline.predict(X_test)
    
    # Initialize reporter and save results (following training pattern)
    from src.reporting.reporter import Reporter
    reporter = Reporter(cfg)
    
    # Add run results to reporter
    model_name = f"optimized_xgboost"
    reporter.add_run_results(strategy, model_name, test_metrics, optimizer.best_params)
    
    # Save detailed predictions and generate calibration plot
    predictions_df = reporter.save_prediction_results(
        y_test, y_pred, test_sample_ids, strategy, model_name
    )
    reporter.generate_calibration_plot(predictions_df, strategy, model_name)
    
    # Save model
    timestamp = cfg.run_timestamp
    model_path = cfg.model_dir / f"optimized_xgboost_{strategy}_{timestamp}.pkl"
    import joblib
    joblib.dump(final_pipeline, model_path)
    
    # Save optimized configuration
    optimized_config = cfg.model_copy(deep=True)
    optimized_config.model_params.xgboost = optimizer.best_params
    config_name = f"optimized_xgboost_{strategy}_{timestamp}"
    description = f"Optimized XGBoost - R²: {test_metrics['r2']:.4f}, Strategy: {strategy}"
    config_manager.save_config(optimized_config, config_name, description)
    
    # Log results
    logger.info(f"XGBoost optimization completed successfully")
    logger.info(f"Best optimization score: {optimizer.best_score:.4f}")
    logger.info(f"Final test R²: {test_metrics['r2']:.4f}")
    logger.info(f"Final test RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"Final test RRMSE: {test_metrics.get('rrmse', 0):.2f}%")
    logger.info(f"Final test MAPE: {test_metrics.get('mape', 0):.2f}%")
    logger.info(f"Final test Within 20.5%: {test_metrics.get('within_20.5%', 0):.2f}%")
    logger.info(f"Model saved to: {model_path}")
    
    # Print results
    print(f"\n--- XGBOOST OPTIMIZATION RESULTS ---")
    display_strategy = get_display_strategy_name(strategy, cfg.use_raw_spectral_data)
    print(f"Strategy: {display_strategy}")
    print(f"Optimization Score: {optimizer.best_score:.4f}")
    print(f"\nTraining Metrics:")
    print(f"  R²: {train_metrics['r2']:.4f}")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  RRMSE: {train_metrics.get('rrmse', 0):.2f}%")
    print(f"\nTest Metrics:")
    print(f"  R²: {test_metrics['r2']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  RRMSE: {test_metrics.get('rrmse', 0):.2f}%")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  MAPE: {test_metrics.get('mape', 0):.2f}%")
    print(f"  Within 20.5%: {test_metrics.get('within_20.5%', 0):.2f}%")
    print(f"\nModel saved to: {model_path}")
    print(f"Config saved as: {config_name}")
    print("-----------------------------------\n")
    
    return final_pipeline, train_metrics, test_metrics, optimizer.best_params

def run_autogluon_optimization_pipeline(use_gpu: bool = False, validation_dir: Optional[str] = None,
                                       config_path: Optional[str] = None, strategy: Optional[str] = None,
                                       n_trials: Optional[int] = None, timeout: Optional[int] = None,
                                       use_parallel: bool = False, n_jobs: int = -1,
                                       exclude_suspects_file: Optional[str] = None):
    """Executes dedicated AutoGluon optimization pipeline."""
    from src.models.optimize_autogluon import AutoGluonOptimizer
    
    cfg = setup_pipeline_config(use_gpu, validation_dir, config_path)
    
    # Use config values when command line values are not provided  
    if n_trials is None:
        # For AutoGluon, use the specific AutoGluon num_trials config
        n_trials = cfg.autogluon.num_trials
        logger.info(f"Using n_trials from AutoGluon config: {n_trials}")
    else:
        logger.info(f"Using n_trials from command line: {n_trials}")
        
    if timeout is None:
        timeout = cfg.tuner.timeout
        logger.info(f"Using timeout from config: {timeout}")
    else:
        logger.info(f"Using timeout from command line: {timeout}")
    
    if strategy is None:
        strategy = cfg.feature_strategies[0] if cfg.feature_strategies else "simple_only"
        logger.info(f"Using strategy from config: {strategy}")
    else:
        logger.info(f"Using strategy from command line: {strategy}")
        
    logger.info(f"Starting AutoGluon optimization run: {cfg.run_timestamp}")
    
    run_data_preparation(cfg)
    full_dataset, global_data_manager = load_and_clean_data(cfg, exclude_suspects_file=exclude_suspects_file)
    
    # Use either the passed validation_dir or the one from config
    validation_directory = validation_dir or cfg.custom_validation_dir
    
    if validation_directory:
        # Process validation data separately
        val_path = Path(validation_directory)
        if not val_path.exists():
            raise DataValidationError(f"Validation directory not found: {val_path}")
        
        validation_dataset = process_validation_data(cfg, val_path, global_data_manager)
        
        # Extract sample IDs from validation dataset
        val_sample_ids = set(validation_dataset[cfg.sample_id_column])
        
        # Split data based on validation sample IDs
        test_df = full_dataset[full_dataset[cfg.sample_id_column].isin(val_sample_ids)]
        train_df = full_dataset[~full_dataset[cfg.sample_id_column].isin(val_sample_ids)]
        
        # If some validation samples weren't in the main dataset, add them
        missing_val_samples = val_sample_ids - set(test_df[cfg.sample_id_column])
        if missing_val_samples:
            missing_df = validation_dataset[validation_dataset[cfg.sample_id_column].isin(missing_val_samples)]
            test_df = pd.concat([test_df, missing_df], ignore_index=True)
            logger.info(f"Added {len(missing_val_samples)} validation samples not found in main dataset")
    else:
        # Use standard data splits
        data_manager = DataManager(cfg)
        train_df, test_df = data_manager.create_reproducible_splits(full_dataset)
    
    logger.info(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Extract features and targets - following training pattern
    X_train_raw = train_df.drop(columns=[cfg.target_column])
    y_train = train_df[cfg.target_column]
    X_test_raw = test_df.drop(columns=[cfg.target_column])
    y_test = test_df[cfg.target_column]
    
    # Extract sample IDs for detailed prediction saving (following training pattern)
    test_sample_ids = X_test_raw[cfg.sample_id_column]
    
    # Prepare features for optimization (drop sample_id)
    X_train = X_train_raw.drop(columns=[cfg.sample_id_column])
    X_test = X_test_raw.drop(columns=[cfg.sample_id_column])
    
    logger.info(f"Starting AutoGluon optimization with {n_trials} trials and {timeout} second timeout...")
    
    # Run optimization
    optimizer = AutoGluonOptimizer(cfg, strategy, use_parallel_features=use_parallel, feature_n_jobs=n_jobs)
    study = optimizer.optimize(X_train, y_train, n_trials, timeout)
    
    # Train final model with best parameters
    logger.info("Training final AutoGluon model with best parameters...")
    final_regressor, train_metrics, test_metrics = optimizer.train_final_model(
        X_train, y_train, X_test, y_test
    )
    
    # Make predictions for detailed reporting (following training pattern)
    y_pred = final_regressor.predict(
        optimizer._fitted_feature_pipeline.transform(X_test_raw.drop(columns=[cfg.sample_id_column]))
    )
    
    # Initialize reporter and save results (following training pattern)
    from src.reporting.reporter import Reporter
    reporter = Reporter(cfg)
    
    # Add run results to reporter
    model_name = f"optimized_autogluon"
    reporter.add_run_results(strategy, model_name, test_metrics, optimizer.best_params)
    
    # Save detailed predictions and generate calibration plot
    predictions_df = reporter.save_prediction_results(
        y_test, y_pred, test_sample_ids, strategy, model_name
    )
    reporter.generate_calibration_plot(predictions_df, strategy, model_name)
    
    # Display results
    print("\n" + "="*50)
    print("AUTOGLUON OPTIMIZATION RESULTS")
    print("="*50)
    display_strategy = get_display_strategy_name(strategy, cfg.use_raw_spectral_data)
    print(f"Strategy: {display_strategy}")
    print(f"Best Optimization Score: {optimizer.best_score:.6f}")
    print("\nBest Parameters:")
    for param, value in optimizer.best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\nFinal Model Performance:")
    print(f"Training Metrics:")
    print(f"  R²: {train_metrics['r2']:.6f}")
    print(f"  RMSE: {train_metrics['rmse']:.6f}")
    print(f"  RRMSE: {train_metrics.get('rrmse', 0):.2f}%")
    print(f"\nTest Metrics:")
    print(f"  R²: {test_metrics['r2']:.6f}")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  RRMSE: {test_metrics.get('rrmse', 0):.2f}%")
    print(f"  MAE: {test_metrics['mae']:.6f}")
    print(f"  MAPE: {test_metrics.get('mape', 0):.2f}%")
    print(f"  Within 20.5%: {test_metrics.get('within_20.5%', 0):.2f}%")
    
    # Save summary report with strategy and model name
    reporter.save_summary_report(strategy=strategy, model_name=model_name)
    reporter.save_config()
    
    # Save optimized configuration
    optimized_config = cfg.model_copy(deep=True)
    
    # Create a simplified version of best parameters for config storage
    config_name = f"optimized_autogluon_{strategy}_{cfg.run_timestamp}"
    description = f"Optimized AutoGluon config - Test R²: {test_metrics['r2']:.4f}, Strategy: {strategy}"
    
    config_manager.save_config(optimized_config, config_name, description)
    
    print("-----------------------------------")
    print(f"Config saved as: {config_name}")
    print("-----------------------------------\n")
    
    return final_regressor, train_metrics, test_metrics, optimizer.best_params

def run_model_optimization_pipeline(model_names: list, use_gpu: bool = False, use_raw_spectral: bool = False,
                                   validation_dir: Optional[str] = None,
                                   config_path: Optional[str] = None, strategy: Optional[str] = None,
                                   n_trials: Optional[int] = None, timeout: Optional[int] = None,
                                   use_parallel: bool = False, n_jobs: int = -1,
                                   use_data_parallel: bool = False, data_n_jobs: int = -1,
                                   exclude_suspects_file: Optional[str] = None,
                                   shap_features_file: Optional[str] = None,
                                   shap_top_n: int = 30,
                                   shap_min_importance: Optional[float] = None):
    """Executes optimization for specified models."""

    # Determine the strategy first to override raw spectral mode if needed
    temp_cfg = setup_pipeline_config(use_gpu, use_raw_spectral, validation_dir, config_path)
    if strategy is None:
        strategy = temp_cfg.feature_strategies[0] if temp_cfg.feature_strategies else "simple_only"
        logger.info(f"Using strategy from config: {strategy}")
    else:
        logger.info(f"Using strategy from command line: {strategy}")

    # Override raw spectral mode for traditional feature engineering strategies
    if strategy in ["simple_only", "full_context", "K_only"]:
        logger.info(f"Strategy '{strategy}' specified - disabling raw spectral mode to use traditional feature engineering")
        use_raw_spectral_override = False
    else:
        use_raw_spectral_override = use_raw_spectral

    cfg = setup_pipeline_config(use_gpu, use_raw_spectral_override, validation_dir, config_path)

    # Configure SHAP-based feature selection if specified
    if shap_features_file:
        cfg.use_shap_feature_selection = True
        cfg.shap_importance_file = shap_features_file
        cfg.shap_top_n_features = shap_top_n
        cfg.shap_min_importance = shap_min_importance
        logger.info(f"[SHAP FEATURE SELECTION] Enabled via command line:")
        logger.info(f"  - SHAP file: {shap_features_file}")
        logger.info(f"  - Top N: {shap_top_n}")
        if shap_min_importance:
            logger.info(f"  - Min importance: {shap_min_importance}")

    # Use config values when command line values are not provided
    if n_trials is None:
        n_trials = cfg.tuner.n_trials
        logger.info(f"Using n_trials from config: {n_trials}")
    else:
        logger.info(f"Using n_trials from command line: {n_trials}")

    if timeout is None:
        timeout = cfg.tuner.timeout
        logger.info(f"Using timeout from config: {timeout}")
    else:
        logger.info(f"Using timeout from command line: {timeout}")

    logger.info(f"Starting multi-model optimization run: {cfg.run_timestamp}")
    
    run_data_preparation(cfg, use_data_parallel, data_n_jobs)
    full_dataset, global_data_manager = load_and_clean_data(cfg, use_data_parallel, data_n_jobs, exclude_suspects_file)
    
    # Use either the passed validation_dir or the one from config
    validation_directory = validation_dir or cfg.custom_validation_dir
    
    if validation_directory:
        # Process validation data separately
        val_path = Path(validation_directory)
        if not val_path.exists():
            raise DataValidationError(f"Validation directory not found: {val_path}")
        
        validation_dataset = process_validation_data(cfg, val_path, global_data_manager)
        
        # Extract sample IDs from validation dataset
        val_sample_ids = set(validation_dataset[cfg.sample_id_column])
        
        # Split data based on validation sample IDs
        test_df = full_dataset[full_dataset[cfg.sample_id_column].isin(val_sample_ids)]
        train_df = full_dataset[~full_dataset[cfg.sample_id_column].isin(val_sample_ids)]
        
        # If some validation samples weren't in the main dataset, add them
        missing_val_samples = val_sample_ids - set(test_df[cfg.sample_id_column])
        if missing_val_samples:
            missing_df = validation_dataset[validation_dataset[cfg.sample_id_column].isin(missing_val_samples)]
            test_df = pd.concat([test_df, missing_df], ignore_index=True)
        
        logger.info(f"Using custom validation split: {len(train_df)} training, {len(test_df)} validation samples")
    else:
        # Use standard random split
        train_df, test_df = DataManager(cfg).create_reproducible_splits(full_dataset)
    
    # Extract features and targets - following training pattern
    X_train_raw = train_df.drop(columns=[cfg.target_column])
    y_train = train_df[cfg.target_column].values
    X_test_raw = test_df.drop(columns=[cfg.target_column])  
    y_test = test_df[cfg.target_column].values
    
    # Extract sample IDs for detailed prediction saving (following training pattern)
    test_sample_ids = X_test_raw[cfg.sample_id_column]
    
    # Prepare features for optimization (drop sample_id)
    X_train = X_train_raw.drop(columns=[cfg.sample_id_column])
    X_test = X_test_raw.drop(columns=[cfg.sample_id_column])
    
    # Initialize unified optimizer
    unified_optimizer = UnifiedModelOptimizer()
    
    # Run optimization for all specified models
    logger.info(f"Starting optimization for models: {model_names}")
    logger.info(f"Training data shape: {X_train.shape}, columns: {list(X_train.columns[:5])}...")
    results = unified_optimizer.optimize_multiple_models(
        model_names, cfg, strategy, X_train, y_train, X_test, y_test, n_trials, timeout,
        use_parallel_features=use_parallel, feature_n_jobs=n_jobs
    )
    
    # Initialize reporter for detailed predictions and plots (following training pattern)
    from src.reporting.reporter import Reporter
    reporter = Reporter(cfg)
    
    # Save detailed results for each optimized model (following training pattern)
    for result in results:
        model_name = result['model_name']
        # Make predictions for detailed reporting
        y_pred = result['pipeline'].predict(X_test)
        
        # Add run results to reporter
        optimized_model_name = f"optimized_{model_name}"
        reporter.add_run_results(strategy, optimized_model_name, result['test_metrics'], result['best_params'])
        
        # Save detailed predictions and generate calibration plot
        predictions_df = reporter.save_prediction_results(
            y_test, y_pred, test_sample_ids, strategy, optimized_model_name
        )
        reporter.generate_calibration_plot(predictions_df, strategy, optimized_model_name)
    
    # Get best model
    best_model_result = unified_optimizer.get_best_model()
    
    if best_model_result is not None:
        best_model_name, best_result = best_model_result
        logger.info(f"Best performing model: {best_model_name} with R²: {best_result['test_metrics']['r2']:.4f}")
        print(f"\n🏆 BEST MODEL: {best_model_name.upper()} 🏆")
        print(f"Test Metrics:")
        print(f"  R²: {best_result['test_metrics']['r2']:.4f}")
        print(f"  RMSE: {best_result['test_metrics']['rmse']:.4f}")
        print(f"  RRMSE: {best_result['test_metrics'].get('rrmse', 0):.2f}%")
        print(f"  MAE: {best_result['test_metrics']['mae']:.4f}")
        print(f"  MAPE: {best_result['test_metrics'].get('mape', 0):.2f}%")
        print(f"  Within 20.5%: {best_result['test_metrics'].get('within_20.5%', 0):.2f}%")
        print(f"Optimization Score: {best_result['best_score']:.4f}")
        
        # Save summary report and configuration
        models_str = "_".join(model_names) if len(model_names) <= 3 else f"{len(model_names)}models"
        reporter.save_summary_report(strategy=strategy, model_name=models_str)
        reporter.save_config()
        
        # Save configuration for future reuse with best model's metrics
        config_name = f"optimization_run_{cfg.run_timestamp}"
        description = f"Model optimization run - Best: {best_model_name} with R²: {best_result['test_metrics']['r2']:.4f}, Strategy: {strategy}"
        config_manager.save_config(cfg, config_name, description)
        
        logger.info(f"Configuration saved as: {config_name}")
    else:
        logger.warning("No models were successfully optimized!")
        print("❌ No models were successfully optimized. Check logs for errors.")
        
        # Still save configuration even if optimization failed
        models_str = "_".join(model_names) if len(model_names) <= 3 else f"{len(model_names)}models" 
        reporter.save_summary_report(strategy=strategy, model_name=f"failed_{models_str}")
        reporter.save_config()
        
        config_name = f"optimization_run_failed_{cfg.run_timestamp}"
        description = f"Model optimization run (failed) - Models attempted: {', '.join(model_names)}, Strategy: {strategy}"
        config_manager.save_config(cfg, config_name, description)
    
    return results

def run_range_specialist_pipeline(use_gpu: bool = False, validation_dir: Optional[str] = None,
                                 config_path: Optional[str] = None, strategy: str = "simple_only",
                                 n_trials: Optional[int] = None, timeout: int = 7200, use_pca: bool = False,
                                 exclude_suspects_file: Optional[str] = None):
    """Executes the Range Specialist Neural Network optimization for 0.2-0.5%% magnesium range."""
    cfg = setup_pipeline_config(use_gpu, validation_dir, config_path)
    
    # Use config values when command line values are not provided
    if n_trials is None:
        n_trials = cfg.tuner.n_trials
        logger.info(f"Using n_trials from config: {n_trials}")
    else:
        logger.info(f"Using n_trials from command line: {n_trials}")
    logger.info(f"🎯 Starting Range Specialist Neural Network optimization: {cfg.run_timestamp}")
    logger.info(f"Target: R² > 0.5 in magnesium range 0.2-0.5%")
    if use_pca:
        logger.info(f"🔬 PCA enabled - will reduce dimensionality after feature engineering")
    
    # Prepare data using the same process as other optimizers
    run_data_preparation(cfg)
    full_dataset, global_data_manager = load_and_clean_data(cfg, exclude_suspects_file=exclude_suspects_file)
    
    # Use either the passed validation_dir or the one from config
    validation_directory = validation_dir or cfg.custom_validation_dir
    
    if validation_directory:
        # Process validation data separately
        val_path = Path(validation_directory)
        if not val_path.exists():
            raise DataValidationError(f"Validation directory not found: {val_path}")
        
        validation_dataset = process_validation_data(cfg, val_path, global_data_manager)
        
        # Extract sample IDs from validation dataset
        val_sample_ids = set(validation_dataset[cfg.sample_id_column])
        
        # Filter training data to exclude validation samples
        train_df = full_dataset[~full_dataset[cfg.sample_id_column].isin(val_sample_ids)]
        test_df = validation_dataset
        
        logger.info(f"Using custom validation split: {len(train_df)} training, {len(test_df)} validation samples")
    else:
        # Create train/test splits like the standard pipeline
        train_df, test_df = DataManager(cfg).create_reproducible_splits(full_dataset)
        logger.info(f"Using standard split: {len(train_df)} training, {len(test_df)} test samples")
    
    # Initialize the range specialist optimizer
    display_strategy = get_display_strategy_name(strategy, cfg.use_raw_spectral_data)
    logger.info(f"Initializing Range Specialist Optimizer with strategy: {display_strategy}")
    optimizer = RangeSpecialistOptimizer(cfg, strategy, use_pca=use_pca, use_parallel_features=use_parallel, feature_n_jobs=n_jobs)
    
    # Run optimization
    logger.info(f"Starting optimization with {n_trials} trials (timeout: {timeout}s)")
    study = optimizer.optimize(train_df, n_trials=n_trials)
    
    # Train and evaluate the best model
    logger.info("Training and evaluating best range specialist model...")
    best_model, metrics = optimizer.evaluate_best_model(train_df, test_df)
    
    # Save the model
    model_path = optimizer.save_model(best_model, metrics)
    
    # Initialize reporter for saving configuration
    from src.reporting.reporter import Reporter
    reporter = Reporter(cfg)
    
    # Save summary report and configuration
    reporter.save_summary_report(strategy=strategy, model_name="xgboost")
    reporter.save_config()
    
    # Determine success status for configuration name
    range_r2 = metrics.get('test_range', {}).get('r2', 0)
    if range_r2 > 0.5:
        status = "success"
        status_desc = f"SUCCESS R²={range_r2:.4f}"
    else:
        status = "progress"
        status_desc = f"R²={range_r2:.4f}"
    
    # Save configuration for future reuse
    config_name = f"range_specialist_{status}_{cfg.run_timestamp}"
    description = f"Range Specialist NN optimization - {status_desc} in 0.2-0.5% range, Strategy: {strategy}, PCA: {use_pca}"
    config_manager.save_config(cfg, config_name, description)
    
    # Log results
    logger.info("=" * 60)
    logger.info("🎯 RANGE SPECIALIST OPTIMIZATION COMPLETE!")
    logger.info("=" * 60)
    
    if 'test_range' in metrics and metrics['test_range']['r2'] > 0.5:
        logger.info(f"🎉 SUCCESS! Achieved R² = {metrics['test_range']['r2']:.4f} > 0.5 in target range!")
    elif 'test_range' in metrics:
        logger.info(f"⚠️  Close! R² = {metrics['test_range']['r2']:.4f} (need {0.5 - metrics['test_range']['r2']:.3f} more)")
    
    logger.info(f"✅ Best model type: {optimizer.best_params['model_type']}")
    logger.info(f"✅ Learning rate: {optimizer.best_params['learning_rate']}")
    logger.info(f"✅ Model saved: {model_path}")
    logger.info(f"✅ Configuration saved as: {config_name}")
    
    return best_model, metrics, optimizer.best_params

# --- Prediction Pipeline Functions ---

def run_single_prediction_pipeline(input_file: str, model_path: str):
    """Executes a prediction on a single, non-averaged input file."""
    cfg = setup_pipeline_config()
    logger.info(f"Starting single-file prediction for: {input_file}")

    # Save configuration for this prediction run
    reporter = Reporter(cfg)
    reporter.save_config()

    predictor = Predictor(config=cfg)
    prediction = predictor.make_prediction(
        input_file=Path(input_file),
        model_path=Path(model_path)
    )
    
    print("\n--- SINGLE PREDICTION RESULT ---")
    print(f"  Input File: {Path(input_file).name}")
    print(f"  Model Used: {Path(model_path).name}")
    print(f"  Predicted Magnesium Concentration: {prediction:.4f} %")
    print("--------------------------------\n")

def run_batch_prediction_pipeline(input_dir: str, model_path: str, output_file: str, reference_file: Optional[str] = None, use_data_parallel: Optional[bool] = None, data_n_jobs: Optional[int] = None, max_samples: Optional[int] = None):
    """Executes batch predictions on a directory, including the averaging step.

    Args:
        max_samples: Maximum number of sample IDs to process (default: None for all)
    """
    cfg = setup_pipeline_config()

    # Override parallel settings if provided (for batch predictions only)
    if use_data_parallel is not None:
        cfg.parallel.use_data_parallel = use_data_parallel
        logger.info(f"Overriding data parallel setting for batch prediction: {use_data_parallel}")
    if data_n_jobs is not None:
        cfg.parallel.data_n_jobs = data_n_jobs
        logger.info(f"Overriding data n_jobs setting for batch prediction: {data_n_jobs}")

    logger.info(f"Starting batch prediction run from dir: {input_dir}")
    logger.info(f"Batch prediction using data_parallel={cfg.parallel.use_data_parallel}, data_n_jobs={cfg.parallel.data_n_jobs}")

    # Save configuration for this prediction run
    reporter = Reporter(cfg)
    reporter.save_config()

    predictor = Predictor(config=cfg)
    results_df = predictor.make_batch_predictions(
        input_dir=Path(input_dir),
        model_path=Path(model_path),
        max_samples=max_samples
    )
    
    # Handle output file path - check if it's already a full path or relative to reports_dir
    output_path = Path(output_file)
    if output_path.is_absolute():
        output_file_path = output_path
    elif str(output_path).startswith('reports/'):
        # Remove the reports prefix and use reports_dir
        relative_path = Path(str(output_path)[8:])  # Remove 'reports/' prefix
        output_file_path = cfg.reports_dir / relative_path
    else:
        # Assume it's relative to reports_dir
        output_file_path = cfg.reports_dir / output_file
    
    # Ensure the parent directory exists
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file_path, index=False)
    
    logger.info(f"Batch prediction complete. Results saved to {output_file_path}")
    print("\n--- BATCH PREDICTION SUMMARY ---")
    print(results_df.to_string())
    print("--------------------------------\n")
    
    if reference_file:
        logger.info(f"Reference file provided. Calculating performance metrics...")
        try:
            ref_df = pd.read_excel(reference_file)
            merged_df = pd.merge(
                results_df, ref_df, left_on='sampleId',
                right_on=cfg.sample_id_column, how='inner'
            )
            merged_df.dropna(subset=['PredictedValue', cfg.target_column], inplace=True)

            if merged_df.empty:
                logger.warning("No matching samples found between predictions and reference file.")
                return

            # --- METRICS CALCULATION FOR FULL DATASET ---
            full_metrics = calculate_regression_metrics(
                merged_df[cfg.target_column].values,
                merged_df['PredictedValue'].values
            )
            print("\n--- PREDICTION METRICS (Full Dataset) ---")
            print(f"  Compared {len(merged_df)} samples against reference file.")
            for name, value in full_metrics.items():
                print(f"  - {name.upper()}: {value:.4f}")
            print("-------------------------------------------\n")

            # --- FILTER DATA TO TRAINING RANGE AND RECALCULATE METRICS ---
            filtered_df = merged_df.copy()
            if cfg.target_value_min is not None:
                filtered_df = filtered_df[filtered_df[cfg.target_column] >= cfg.target_value_min]
            if cfg.target_value_max is not None:
                filtered_df = filtered_df[filtered_df[cfg.target_column] <= cfg.target_value_max]
            
            if filtered_df.empty:
                logger.warning("No samples found within the specified training range for filtered metrics.")
                filtered_metrics = {}
            else:
                filtered_metrics = calculate_regression_metrics(
                    filtered_df[cfg.target_column].values,
                    filtered_df['PredictedValue'].values
                )
                print("\n--- PREDICTION METRICS (Filtered to Training Range) ---")
                print(f"  Compared {len(filtered_df)} samples within range [{cfg.target_value_min}-{cfg.target_value_max}].")
                for name, value in filtered_metrics.items():
                    print(f"  - {name.upper()}: {value:.4f}")
                print("-----------------------------------------------------------\n")

            # --- SAVE METRICS REPORT TO JSON ---
            report_data = {
                'model_path': str(model_path),
                'prediction_file': str(output_file_path),
                'full_dataset_metrics': {
                    'sample_count': len(merged_df),
                    'metrics': full_metrics
                },
                'filtered_dataset_metrics': {
                    'sample_count': len(filtered_df),
                    'metrics': filtered_metrics
                }
            }
            filename_parts = ["prediction_metrics_report"]
            if cfg.use_raw_spectral_data:
                filename_parts.append("raw-spectral")
            filename_parts.append(cfg.run_timestamp)
            report_filename = "_".join(filename_parts) + ".json"
            report_path = cfg.reports_dir / report_filename
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=4)
            logger.info(f"Prediction metrics report saved to: {report_path}")

        except FileNotFoundError:
            logger.error(f"Reference file not found at: {reference_file}")
        except Exception as e:
            logger.error(f"Could not calculate metrics due to an error: {e}", exc_info=True)


# --- Classification Prediction Pipeline Functions ---

def run_single_classification_pipeline(input_file: str, model_path: str):
    """Executes a binary classification prediction on a single, non-averaged input file."""
    from src.models.classification_predictor import ClassificationPredictor
    
    cfg = setup_pipeline_config()
    logger.info(f"Starting single-file classification for: {input_file}")

    # Save configuration for this prediction run
    reporter = Reporter(cfg)
    reporter.save_config()

    predictor = ClassificationPredictor(config=cfg)
    result = predictor.make_prediction(
        input_file=Path(input_file),
        model_path=Path(model_path)
    )
    
    # Display results
    print("\n--- CLASSIFICATION RESULT ---")
    print(f"Sample: {result['sample_file']}")
    print(f"Prediction: {result['label']} ({result['prediction']})")
    if result['probability'] is not None:
        print(f"Probability: {result['probability']:.4f}")
    print(f"Threshold: {result['threshold']}")
    print("-------------------------------")
    
    logger.info(f"Classification complete for {input_file}")


def run_batch_classification_pipeline(input_dir: str, model_path: str, output_file: str, use_data_parallel: Optional[bool] = None, data_n_jobs: Optional[int] = None):
    """Executes batch binary classification predictions on a directory of raw files."""
    from src.models.classification_predictor import ClassificationPredictor

    cfg = setup_pipeline_config()

    # Override parallel settings if provided (for batch classifications only)
    if use_data_parallel is not None:
        cfg.parallel.use_data_parallel = use_data_parallel
        logger.info(f"Overriding data parallel setting for batch classification: {use_data_parallel}")
    if data_n_jobs is not None:
        cfg.parallel.data_n_jobs = data_n_jobs
        logger.info(f"Overriding data n_jobs setting for batch classification: {data_n_jobs}")

    logger.info(f"Starting batch classification for directory: {input_dir}")
    logger.info(f"Batch classification using data_parallel={cfg.parallel.use_data_parallel}, data_n_jobs={cfg.parallel.data_n_jobs}")

    # Save configuration for this prediction run
    reporter = Reporter(cfg)
    reporter.save_config()

    predictor = ClassificationPredictor(config=cfg)
    results_df = predictor.make_batch_predictions(
        input_dir=Path(input_dir),
        model_path=Path(model_path)
    )
    
    # Handle output file path - check if it's already a full path or relative to reports_dir
    output_path = Path(output_file)
    if output_path.is_absolute():
        output_file_path = output_path
    elif str(output_path).startswith('reports/'):
        # Remove the reports prefix and use reports_dir
        relative_path = str(output_path)[8:]  # Remove 'reports/' prefix
        output_file_path = cfg.reports_dir / relative_path
    else:
        # Assume it's a filename to be placed in reports_dir
        output_file_path = cfg.reports_dir / output_file
    
    # Ensure output directory exists
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_df.to_csv(output_file_path, index=False)
    logger.info(f"Classification results saved to: {output_file_path}")
    
    # Display summary
    total_samples = len(results_df)
    successful_predictions = len(results_df[results_df['prediction'].notna()])
    high_predictions = len(results_df[results_df['prediction'] == 1])
    low_predictions = len(results_df[results_df['prediction'] == 0])
    
    print("\n--- BATCH CLASSIFICATION SUMMARY ---")
    print(f"  Total samples processed: {total_samples}")
    print(f"  Successful predictions: {successful_predictions}")
    print(f"  High class predictions: {high_predictions}")
    print(f"  Low class predictions: {low_predictions}")
    print(f"  Failed predictions: {total_samples - successful_predictions}")
    if len(results_df) > 0:
        print(f"  Classification threshold: {results_df.iloc[0]['threshold']}")
    print(f"  Results saved to: {output_file_path}")
    print("--------------------------------------")


# --- Config Management Functions ---

def save_current_config(name: str, description: str = ""):
    """Save the current default configuration to a file."""
    cfg = setup_pipeline_config()
    config_path = config_manager.save_config(cfg, name, description)
    print(f"Configuration saved to: {config_path}")
    logger.info(f"Configuration '{name}' saved successfully")

def list_saved_configs():
    """List all saved configurations."""
    configs = config_manager.list_configs()
    if not configs:
        print("No saved configurations found.")
        return
    
    print("\n--- SAVED CONFIGURATIONS ---")
    for i, config_meta in enumerate(configs, 1):
        print(f"{i}. {config_meta.get('name', 'Unnamed')}")
        print(f"   Description: {config_meta.get('description', 'No description')}")
        print(f"   Created: {config_meta.get('created_at', 'Unknown')}")
        print(f"   File: {config_meta.get('file_path', 'Unknown')}")
        print()

def create_training_config(name: str, models: list, strategies: list, use_gpu: bool, 
                          use_sample_weights: bool, description: str = ""):
    """Create and save a training configuration."""
    cfg = setup_pipeline_config()
    training_config = config_manager.create_training_config(
        cfg, models, strategies, use_gpu, use_sample_weights
    )
    config_path = config_manager.save_config(training_config, name, description)
    print(f"Training configuration '{name}' created and saved to: {config_path}")
    logger.info(f"Training configuration '{name}' created successfully")


def run_classification_pipeline(threshold=0.3, strategy="simple_only", models=None,
                               use_gpu=False, config_path=None, use_parallel=None,
                               n_jobs=None, use_data_parallel=None, data_n_jobs=None,
                               exclude_suspects_file: Optional[str] = None):
    """
    Run the classification pipeline for magnesium level prediction.
    
    Args:
        threshold: Classification threshold for magnesium levels (default: 0.3)
        strategy: Feature engineering strategy
        models: List of models to train
        use_gpu: Whether to use GPU acceleration
        config_path: Path to custom configuration file
        use_parallel: Whether to use parallel feature processing
        n_jobs: Number of jobs for feature processing
        use_data_parallel: Whether to use parallel data processing
        data_n_jobs: Number of jobs for data processing
    """
    logger.info("Starting CLASSIFICATION Pipeline")
    logger.info(f"Classification threshold: {threshold}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Models: {models}")
    
    # Setup
    cfg = setup_pipeline_config(use_gpu=use_gpu, config_path=config_path)
    run_data_preparation(cfg, use_data_parallel=use_data_parallel, data_n_jobs=data_n_jobs)
    
    # Load and prepare data
    full_dataset, data_manager = load_and_clean_data(cfg)
    train_df, test_df = data_manager.create_reproducible_splits(full_dataset)
    
    logger.info(f"Classification dataset prepared:")
    logger.info(f"  Training samples: {len(train_df)}")
    logger.info(f"  Test samples: {len(test_df)}")
    
    # Initialize classification trainer
    from src.models.classification_trainer import ClassificationTrainer
    trainer = ClassificationTrainer(cfg, strategy, threshold, 
                                  use_parallel_features=use_parallel, 
                                  feature_n_jobs=n_jobs)
    
    # Filter models if specified
    if models:
        logger.info(f"Training selected models: {models}")
    
    # Train models
    logger.info("Starting classification model training...")
    results = trainer.train_models(train_df, test_df)
    
    # Create comparison report
    comparison_df = trainer.create_comparison_report()
    logger.info("\n" + "="*80)
    logger.info("CLASSIFICATION MODEL COMPARISON")
    logger.info("="*80)
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Get best model
    try:
        best_model_name, best_result = trainer.get_best_model(metric='f1')
        logger.info(f"\nBest model by F1 score: {best_model_name}")
        logger.info(f"Best F1 score: {best_result['test_metrics']['f1']:.4f}")
        logger.info(f"Best model path: {best_result['model_path']}")
    except Exception as e:
        logger.warning(f"Could not determine best model: {e}")
    
    # Save configuration for future reuse
    config_name = f"classification_run_{cfg.run_timestamp}"
    description = f"Classification run - Threshold: {threshold}, Strategy: {strategy}"
    config_manager.save_config(cfg, config_name, description)
    
    logger.info("\nClassification pipeline completed successfully!")
    logger.info(f"Configuration saved as: {config_name}")


def run_mislabel_detection(focus_range_min: float = 0.0, focus_range_max: float = 0.5,
                          use_features: bool = True,
                          clustering_methods: str = "kmeans,dbscan,hierarchical",
                          outlier_methods: str = "lof,isolation_forest",
                          min_confidence: int = 2, export_results: bool = True,
                          config_path: Optional[str] = None,
                          use_feature_parallel: bool = False, feature_n_jobs: int = 1,
                          use_data_parallel: bool = False, data_n_jobs: int = -1,
                          exclude_suspects_file: Optional[str] = None,
                          strategy_override: Optional[str] = None,
                          no_raw_spectral: bool = False):
    """
    Run mislabel detection analysis to identify potentially mislabeled samples.

    Note: Raw spectral analysis is automatically enabled/disabled based on
          config.use_raw_spectral_data setting.

    Args:
        focus_range_min: Minimum concentration to focus analysis on
        focus_range_max: Maximum concentration to focus analysis on
        use_features: Whether to use engineered features for clustering
        clustering_methods: Comma-separated list of clustering methods
        outlier_methods: Comma-separated list of outlier detection methods
        min_confidence: Minimum confidence level for suspect export
        export_results: Whether to export suspect lists
        config_path: Path to configuration file
        use_feature_parallel: Enable parallel processing for feature engineering
        feature_n_jobs: Number of parallel jobs for feature processing
        use_data_parallel: Enable parallel processing for data operations
        data_n_jobs: Number of parallel jobs for data operations
    """
    cfg = setup_pipeline_config(config_path=config_path)
    logger.info(f"Starting mislabel detection analysis...")
    logger.info(f"Focus range: {focus_range_min:.2f} - {focus_range_max:.2f}")

    # Load and prepare the dataset
    full_dataset, data_manager = load_and_clean_data(cfg)
    logger.info(f"Loaded dataset with {len(full_dataset)} samples")

    # Initialize reporter for saving configuration and reports
    reporter = Reporter(cfg)

    # Initialize mislabel detector with parallel processing configuration
    detector = MislabelDetector(
        cfg,
        focus_range=(focus_range_min, focus_range_max),
        use_feature_parallel=use_feature_parallel,
        feature_n_jobs=feature_n_jobs,
        use_data_parallel=use_data_parallel,
        data_n_jobs=data_n_jobs,
        strategy_override=strategy_override
    )

    # Parse method lists
    cluster_methods = [m.strip() for m in clustering_methods.split(",")]
    outlier_methods_list = [m.strip() for m in outlier_methods.split(",")]

    # Override raw spectral setting if requested
    use_raw_spectral_final = cfg.use_raw_spectral_data and not no_raw_spectral
    if no_raw_spectral and cfg.use_raw_spectral_data:
        logger.info("Raw spectral analysis disabled by --no-raw-spectral flag")

    # Run detection analysis
    results = detector.detect_mislabels(
        dataset=full_dataset,
        use_features=use_features,
        clustering_methods=cluster_methods,
        outlier_methods=outlier_methods_list,
        min_cluster_size=5,
        use_raw_spectral_override=use_raw_spectral_final
    )

    # Print summary
    print("\n" + "="*80)
    print("MISLABEL DETECTION RESULTS")
    print("="*80)

    if "error" in results:
        print(f"❌ Analysis failed: {results['error']}")
        return

    n_suspicious = len(results["suspicious_samples"])
    total_samples = results["dataset_info"]["focus_samples"]

    print(f"📊 Dataset Information:")
    print(f"   • Total samples: {results['dataset_info']['total_samples']}")
    print(f"   • Focus range samples: {total_samples}")
    print(f"   • Suspicious samples found: {n_suspicious}")
    print(f"   • Suspicion rate: {n_suspicious/total_samples:.1%}")

    # Show high-confidence suspects
    high_confidence = [s for s in results["suspicious_samples"].values() if s["suspicion_count"] >= min_confidence]
    print(f"\n🔍 High-confidence suspects (flagged by ≥{min_confidence} methods): {len(high_confidence)}")

    if high_confidence:
        print("   Sample ID | Confidence | Methods")
        print("   " + "-"*35)
        for suspect in sorted(high_confidence, key=lambda x: x["suspicion_count"], reverse=True)[:10]:
            methods = ", ".join(set([r.split('_')[0] for r in suspect["reasons"]]))
            print(f"   {suspect['sample_id']:>9} | {suspect['suspicion_count']:>10} | {methods}")

        if len(high_confidence) > 10:
            print(f"   ... and {len(high_confidence)-10} more")

    print(f"\n💡 Recommendations:")
    for rec in results["recommendations"]:
        print(f"   {rec}")

    # Generate visualizations
    try:
        print(f"\n📈 Generating visualizations...")
        detector.visualize_results()
        print(f"   ✅ Visualizations saved to reports/mislabel_analysis/")
    except Exception as e:
        print(f"   ❌ Visualization failed: {e}")

    # Export results if requested
    if export_results and n_suspicious > 0:
        try:
            print(f"\n📤 Exporting suspicious samples...")
            export_path = detector.export_suspect_list(min_suspicion_count=min_confidence)
            if export_path:
                print(f"   ✅ Suspicious samples exported to: {export_path}")

                # Also export lower confidence for review
                if min_confidence > 1:
                    review_path = detector.export_suspect_list(min_suspicion_count=1)
                    if review_path:
                        print(f"   ✅ All suspects (for review) exported to: {review_path}")
        except Exception as e:
            print(f"   ❌ Export failed: {e}")

    print("\n🎯 Next Steps:")
    print("   1. Review the visualizations in reports/mislabel_analysis/")
    print("   2. Manually inspect high-confidence suspects")
    print("   3. Use exported sample lists to exclude from next training run")
    print("   4. Re-run training without suspicious samples")

    # Save configuration and summary report (following training pipeline pattern)
    try:
        # Save reporter configuration
        reporter.save_config()

        # Determine strategy info for config description
        if strategy_override:
            strategy_info = f"Strategy override: {strategy_override}"
        elif cfg.use_raw_spectral_data:
            strategy_info = "Raw spectral data mode"
        else:
            strategy_info = f"Strategy: {cfg.feature_strategies[0] if cfg.feature_strategies else 'simple_only'}"

        # Save configuration for future reuse
        config_name = f"mislabel_detection_{cfg.run_timestamp}"
        description = f"Mislabel detection run - {strategy_info}, Focus: {focus_range_min:.1f}-{focus_range_max:.1f}, Min confidence: {min_confidence}, Found: {n_suspicious} suspects"
        config_manager.save_config(cfg, config_name, description)

        print(f"\n💾 Configuration saved as: {config_name}")
        logger.info(f"Configuration saved as: {config_name}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        print(f"   ❌ Failed to save configuration: {e}")

    logger.info("Mislabel detection analysis completed")


def main():
    """Main entry point for the Magnesium Prediction ML Pipeline."""
    parser = argparse.ArgumentParser(description="Magnesium Prediction ML Pipeline")
    subparsers = parser.add_subparsers(dest="stage", required=True, help="Pipeline stage to run")

    # Add global options to all subcommands
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration for training (requires CUDA)")
    parent_parser.add_argument("--raw-spectral", action="store_true", help="Use raw spectral data (filtered by PeakRegions) instead of engineered features")
    parent_parser.add_argument("--validation-dir", type=str, help="Path to directory containing raw validation spectral files (.csv.txt)")
    parent_parser.add_argument("--config", type=str, help="Path to saved configuration file (.yaml)")
    parent_parser.add_argument("--config-from-env", action="store_true", help="Load configuration from environment variables (CONFIG_*)")
    parent_parser.add_argument("--config-json", type=str, help="Configuration as JSON string")
    parent_parser.add_argument("--config-base64", type=str, help="Configuration as base64-encoded JSON string")
    parent_parser.add_argument("--feature-parallel", action="store_true", help="Enable parallel processing for feature generation (overrides config)")
    parent_parser.add_argument("--no-feature-parallel", action="store_true", help="Disable parallel processing for feature generation (overrides config)")
    parent_parser.add_argument("--feature-n-jobs", type=int, help="Number of parallel jobs for feature generation (-1 uses all cores, overrides config)")
    parent_parser.add_argument("--data-parallel", action="store_true", help="Enable parallel processing for data averaging and cleansing (overrides config)")
    parent_parser.add_argument("--no-data-parallel", action="store_true", help="Disable parallel processing for data operations (overrides config)")
    parent_parser.add_argument("--data-n-jobs", type=int, help="Number of parallel jobs for data operations (-1 uses all cores, overrides config)")
    parent_parser.add_argument("--exclude-suspects", type=str, help="Path to CSV file containing sample IDs to exclude from training (e.g., reports/mislabel_analysis/suspicious_samples_min_confidence_2.csv)")
    parent_parser.add_argument("--shap-features", type=str, help="Path to SHAP importance CSV file for feature selection (e.g., models/model_name_shap_importance.csv)")
    parent_parser.add_argument("--shap-top-n", type=int, default=30, help="Number of top SHAP features to select (default: 30)")
    parent_parser.add_argument("--shap-min-importance", type=float, help="Minimum SHAP importance threshold (optional)")

    # Train subparser with models selection
    parser_train = subparsers.add_parser("train", parents=[parent_parser], help="Run the standard model training pipeline.")
    parser_train.add_argument("--models", nargs="+", 
                             choices=["ridge", "lasso", "random_forest", "xgboost", "lightgbm", "catboost", "extratrees", "neural_network", "neural_network_light", "svr"],
                             help="Models to train (default: use models from config)")
    parser_train.add_argument("--strategy", type=str, 
                             choices=["full_context", "simple_only", "K_only"],
                             help="Feature strategy to use (default: use strategies from config)")
    parser_autogluon = subparsers.add_parser("autogluon", parents=[parent_parser], help="Run the AutoGluon training pipeline.")
    parser_autogluon.add_argument("--strategy", type=str, 
                                 choices=["full_context", "simple_only", "K_only"],
                                 help="Feature strategy to use (default: use strategies from config)")
    # Tune subparser with models selection
    parser_tune = subparsers.add_parser("tune", parents=[parent_parser], help="Run hyperparameter tuning for standard models.")
    parser_tune.add_argument("--models", nargs="+", required=True,
                            choices=["ridge", "lasso", "random_forest", "xgboost", "lightgbm", "catboost", "extratrees", "neural_network", "neural_network_light", "svr"],
                            help="Models to tune")
    
    # XGBoost Optimization subparser
    parser_optimize_xgboost = subparsers.add_parser("optimize-xgboost", parents=[parent_parser], help="Run dedicated XGBoost optimization")
    parser_optimize_xgboost.add_argument("--strategy", type=str, default=None, 
                                       choices=["full_context", "simple_only", "K_only"],
                                       help="Feature strategy to use")
    parser_optimize_xgboost.add_argument("--trials", type=int, default=None, help="Number of optimization trials (default from config)")
    parser_optimize_xgboost.add_argument("--timeout", type=int, default=None, help="Timeout in seconds (default from config)")
    
    # AutoGluon Optimization subparser
    parser_optimize_autogluon = subparsers.add_parser("optimize-autogluon", parents=[parent_parser], help="Run dedicated AutoGluon optimization")
    parser_optimize_autogluon.add_argument("--strategy", type=str, default=None, 
                                         choices=["full_context", "simple_only", "K_only"],
                                         help="Feature strategy to use")
    parser_optimize_autogluon.add_argument("--trials", type=int, default=None, help="Number of optimization trials (default from config)")
    parser_optimize_autogluon.add_argument("--timeout", type=int, default=None, help="Timeout in seconds (default from config)")
    
    # Model Optimization subparser
    parser_optimize_models = subparsers.add_parser("optimize-models", parents=[parent_parser], help="Run optimization for multiple models")
    parser_optimize_models.add_argument("--models", nargs="+", required=True,
                                      choices=["xgboost", "lightgbm", "catboost", "random_forest", "extratrees", "neural_network", "neural_network_light", "autogluon"],
                                      help="Models to optimize")
    parser_optimize_models.add_argument("--strategy", type=str, default=None, 
                                      choices=["full_context", "simple_only", "K_only"],
                                      help="Feature strategy to use")
    parser_optimize_models.add_argument("--trials", type=int, default=None, help="Number of optimization trials per model (default from config)")
    parser_optimize_models.add_argument("--timeout", type=int, default=None, help="Timeout in seconds per model (default from config)")

    # Classification subparser
    parser_classify = subparsers.add_parser("classify", parents=[parent_parser], help="Train classification models for magnesium threshold prediction")
    parser_classify.add_argument("--threshold", type=float, default=0.3, help="Classification threshold for magnesium levels (default: 0.3)")
    parser_classify.add_argument("--strategy", type=str, default="simple_only", 
                               choices=["full_context", "simple_only", "K_only"],
                               help="Feature strategy to use")
    parser_classify.add_argument("--models", nargs="+", 
                               choices=["logistic", "random_forest", "extratrees", "svm", "naive_bayes", "knn", "xgboost", "lightgbm", "catboost"],
                               default=["logistic", "random_forest", "xgboost"],
                               help="Classification models to train")

    # Range Specialist Neural Network Optimization subparser
    parser_range_specialist = subparsers.add_parser("optimize-range-specialist", parents=[parent_parser], help="Optimize neural network for 0.2-0.5%% magnesium range (target R² > 0.5)")
    parser_range_specialist.add_argument("--strategy", type=str, default="simple_only", 
                                       choices=["full_context", "simple_only", "K_only"],
                                       help="Feature strategy to use")
    parser_range_specialist.add_argument("--trials", type=int, default=None, help="Number of optimization trials (default from config)")
    parser_range_specialist.add_argument("--timeout", type=int, default=7200, help="Timeout in seconds")
    parser_range_specialist.add_argument("--pca", action="store_true", help="Apply PCA after feature engineering")

    # Single Prediction subparser
    parser_predict_single = subparsers.add_parser("predict-single", help="Make a prediction on a single raw file.")
    parser_predict_single.add_argument("--input-file", type=str, required=True, help="Path to the raw spectral .csv.txt file.")
    parser_predict_single.add_argument("--model-path", type=str, required=True, help="Path to the trained model (.pkl) or AutoGluon directory.")

    # Batch Prediction subparser
    parser_predict_batch = subparsers.add_parser("predict-batch", parents=[parent_parser], help="Make batch predictions on a directory of raw files.")
    parser_predict_batch.add_argument("--input-dir", type=str, required=True, help="Path to the directory with raw spectral files.")
    parser_predict_batch.add_argument("--model-path", type=str, required=True, help="Path to the trained model (.pkl) or AutoGluon directory.")
    parser_predict_batch.add_argument("--output-file", type=str, default="batch_predictions.csv", help="Name for the output CSV file in the reports directory.")
    parser_predict_batch.add_argument("--reference-file", type=str, required=False, help="Optional: Path to an Excel file with true values to calculate metrics.")
    parser_predict_batch.add_argument("--max-samples", type=int, default=None, help="Maximum number of sample IDs to process (default: all)")

    # Classification Prediction subparsers
    parser_classify_single = subparsers.add_parser("classify-single", help="Make a binary classification prediction on a single raw file.")
    parser_classify_single.add_argument("--input-file", type=str, required=True, help="Path to the raw spectral .csv.txt file.")
    parser_classify_single.add_argument("--model-path", type=str, required=True, help="Path to the trained classification model (.pkl).")

    parser_classify_batch = subparsers.add_parser("classify-batch", help="Make batch binary classification predictions on a directory of raw files.")
    parser_classify_batch.add_argument("--input-dir", type=str, required=True, help="Path to the directory with raw spectral files.")
    parser_classify_batch.add_argument("--model-path", type=str, required=True, help="Path to the trained classification model (.pkl).")
    parser_classify_batch.add_argument("--output-file", type=str, default="batch_classifications.csv", help="Name for the output CSV file in the reports directory.")

    # Config Management subparsers
    parser_save_config = subparsers.add_parser("save-config", help="Save current configuration to file")
    parser_save_config.add_argument("--name", type=str, required=True, help="Name for the configuration")
    parser_save_config.add_argument("--description", type=str, default="", help="Description of the configuration")

    parser_list_configs = subparsers.add_parser("list-configs", help="List all saved configurations")

    parser_create_training_config = subparsers.add_parser("create-training-config", help="Create a training configuration")
    parser_create_training_config.add_argument("--name", type=str, required=True, help="Name for the configuration")
    parser_create_training_config.add_argument("--models", nargs="+", default=["xgboost", "lightgbm", "extratrees"], help="Models to train")
    parser_create_training_config.add_argument("--strategies", nargs="+", default=["simple_only"], help="Feature strategies")
    parser_create_training_config.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser_create_training_config.add_argument("--sample-weights", action="store_true", default=True, help="Use sample weights")
    parser_create_training_config.add_argument("--description", type=str, default="", help="Configuration description")

    # Mislabel Detection subparser
    parser_mislabel = subparsers.add_parser("detect-mislabels", parents=[parent_parser], help="Detect potentially mislabeled samples using clustering")
    parser_mislabel.add_argument("--focus-min", type=float, default=0.0, help="Minimum concentration to focus on (default: 0.0)")
    parser_mislabel.add_argument("--focus-max", type=float, default=0.5, help="Maximum concentration to focus on (default: 0.5)")
    parser_mislabel.add_argument("--no-features", action="store_true", help="Skip engineered features analysis")
    parser_mislabel.add_argument("--clustering-methods", type=str, default="kmeans,dbscan,hierarchical", help="Clustering methods to use (comma-separated)")
    parser_mislabel.add_argument("--outlier-methods", type=str, default="lof,isolation_forest", help="Outlier detection methods to use (comma-separated)")
    parser_mislabel.add_argument("--min-confidence", type=int, default=2, help="Minimum confidence level for suspect export (default: 2)")
    parser_mislabel.add_argument("--no-export", action="store_true", help="Skip exporting results")
    parser_mislabel.add_argument("--strategy", type=str,
                               choices=["full_context", "simple_only", "K_only"],
                               help="Feature strategy to use (overrides config setting)")
    parser_mislabel.add_argument("--no-raw-spectral", action="store_true", help="Skip raw spectral analysis (force only feature-based analysis)")

    args = parser.parse_args()
    
    # # For now, keep the hardcoded fallback for testing
    # args = argparse.Namespace()
    # time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # args.stage = "train"
    # args.gpu = True  # Set to True to test GPU mode
    # #args.input_dir = "/home/payanico/magnesium_pipeline/data/raw/data_5278_Phase3"
    # args.input_dir = "/home/payanico/magnesium_pipeline/data/raw/combo_6_8"
    # args.model_path = "/home/payanico/magnesium_pipeline/models/optimized_simple_only_xgboost_20250724_003832.pkl"
    # args.output_file = f"batch_predictions_{time_stamp}.csv"
    # args.reference_file = "/home/payanico/magnesium_pipeline/data/reference_data/Final_Lab_Data_Nico_New.xlsx"
    
    try:
        use_gpu = getattr(args, 'gpu', False)
        # Pass None if flag not provided, True if provided (action="store_true")
        use_raw_spectral = True if (hasattr(args, 'raw_spectral') and args.raw_spectral) else None
        validation_dir = getattr(args, 'validation_dir', None)
        config_path = getattr(args, 'config', None)
        config_from_env = getattr(args, 'config_from_env', False)
        config_json = getattr(args, 'config_json', None)
        config_base64 = getattr(args, 'config_base64', None)
        
        # If config-from-env is specified, create a temporary config file from environment variables
        if config_from_env:
            config_path = create_config_from_env()
        # If config-json is specified, create a temporary config file from JSON
        elif config_json:
            config_path = create_config_from_json(config_json)
        # If config-base64 is specified, decode and create a temporary config file from JSON
        elif config_base64:
            config_path = create_config_from_base64(config_base64)
        # Load config to get default parallel processing settings
        if config_path:
            temp_config = config_manager.load_config(config_path)
            default_feature_parallel = temp_config.parallel.use_feature_parallel
            default_data_parallel = temp_config.parallel.use_data_parallel
            default_feature_n_jobs = temp_config.parallel.feature_n_jobs
            default_data_n_jobs = temp_config.parallel.data_n_jobs
        else:
            default_feature_parallel = config.parallel.use_feature_parallel
            default_data_parallel = config.parallel.use_data_parallel
            default_feature_n_jobs = config.parallel.feature_n_jobs
            default_data_n_jobs = config.parallel.data_n_jobs
        
        # Use command line arguments if explicitly provided, otherwise use config defaults
        # Handle boolean flags with explicit enable/disable options
        if getattr(args, 'feature_parallel', False):
            use_parallel = True
        elif getattr(args, 'no_feature_parallel', False):
            use_parallel = False
        else:
            use_parallel = default_feature_parallel
            
        if getattr(args, 'data_parallel', False):
            use_data_parallel = True
        elif getattr(args, 'no_data_parallel', False):
            use_data_parallel = False
        else:
            use_data_parallel = default_data_parallel
        
        # For numeric arguments, use provided value or config default
        n_jobs = getattr(args, 'feature_n_jobs', None)
        if n_jobs is None:
            n_jobs = default_feature_n_jobs
            
        data_n_jobs = getattr(args, 'data_n_jobs', None)
        if data_n_jobs is None:
            data_n_jobs = default_data_n_jobs

        # Get exclude_suspects parameter
        exclude_suspects_file = getattr(args, 'exclude_suspects', None)

        # Get SHAP feature selection parameters
        shap_features_file = getattr(args, 'shap_features', None)
        shap_top_n = getattr(args, 'shap_top_n', 30)
        shap_min_importance = getattr(args, 'shap_min_importance', None)

        if args.stage == "train":
            models = getattr(args, 'models', None)
            strategy = getattr(args, 'strategy', None)
            run_training_pipeline(use_gpu=use_gpu, use_raw_spectral=use_raw_spectral, validation_dir=validation_dir, config_path=config_path, models=models, strategy=strategy, use_parallel=use_parallel, n_jobs=n_jobs, use_data_parallel=use_data_parallel, data_n_jobs=data_n_jobs, exclude_suspects_file=exclude_suspects_file, shap_features_file=shap_features_file, shap_top_n=shap_top_n, shap_min_importance=shap_min_importance)
        elif args.stage == "autogluon":
            strategy = getattr(args, 'strategy', None)
            run_autogluon_pipeline(use_gpu=use_gpu, use_raw_spectral=use_raw_spectral, validation_dir=validation_dir, config_path=config_path, strategy=strategy, use_parallel=use_parallel, n_jobs=n_jobs, use_data_parallel=use_data_parallel, data_n_jobs=data_n_jobs, exclude_suspects_file=exclude_suspects_file, shap_features_file=shap_features_file, shap_top_n=shap_top_n, shap_min_importance=shap_min_importance)
        elif args.stage == "tune":
            models = getattr(args, 'models', None)
            run_tuning_pipeline(use_gpu=use_gpu, use_raw_spectral=use_raw_spectral, validation_dir=validation_dir, config_path=config_path, models=models, exclude_suspects_file=exclude_suspects_file)
        elif args.stage == "optimize-xgboost":
            run_xgboost_optimization_pipeline(
                use_gpu=use_gpu,
                validation_dir=validation_dir,
                config_path=config_path,
                strategy=args.strategy,
                n_trials=args.trials,
                timeout=args.timeout,
                use_parallel=use_parallel,
                n_jobs=n_jobs,
                exclude_suspects_file=exclude_suspects_file
            )
        elif args.stage == "optimize-autogluon":
            run_autogluon_optimization_pipeline(
                use_gpu=use_gpu,
                validation_dir=validation_dir,
                config_path=config_path,
                strategy=args.strategy,
                n_trials=args.trials,
                timeout=args.timeout,
                use_parallel=use_parallel,
                n_jobs=n_jobs,
                exclude_suspects_file=exclude_suspects_file
            )
        elif args.stage == "optimize-models":
            run_model_optimization_pipeline(
                model_names=args.models,
                use_gpu=use_gpu,
                use_raw_spectral=use_raw_spectral,
                validation_dir=validation_dir,
                config_path=config_path,
                strategy=args.strategy,
                n_trials=args.trials,
                timeout=args.timeout,
                use_parallel=use_parallel,
                n_jobs=n_jobs,
                use_data_parallel=use_data_parallel,
                data_n_jobs=data_n_jobs,
                exclude_suspects_file=exclude_suspects_file,
                shap_features_file=shap_features_file,
                shap_top_n=shap_top_n,
                shap_min_importance=shap_min_importance
            )
        elif args.stage == "optimize-range-specialist":
            run_range_specialist_pipeline(
                use_gpu=use_gpu,
                validation_dir=validation_dir,
                config_path=config_path,
                strategy=args.strategy,
                n_trials=args.trials,
                timeout=args.timeout,
                use_pca=args.pca,
                exclude_suspects_file=exclude_suspects_file
            )
        elif args.stage == "classify":
            run_classification_pipeline(
                threshold=args.threshold,
                strategy=args.strategy,
                models=args.models,
                use_gpu=use_gpu,
                config_path=config_path,
                use_parallel=use_parallel,
                n_jobs=n_jobs,
                use_data_parallel=use_data_parallel,
                data_n_jobs=data_n_jobs,
                exclude_suspects_file=exclude_suspects_file
            )
        elif args.stage == "predict-single":
            run_single_prediction_pipeline(input_file=args.input_file, model_path=args.model_path)
        elif args.stage == "predict-batch":
            run_batch_prediction_pipeline(
                input_dir=args.input_dir,
                model_path=args.model_path,
                output_file=args.output_file,
                reference_file=args.reference_file,
                use_data_parallel=use_data_parallel,
                data_n_jobs=data_n_jobs,
                max_samples=args.max_samples
            )
        elif args.stage == "classify-single":
            run_single_classification_pipeline(input_file=args.input_file, model_path=args.model_path)
        elif args.stage == "classify-batch":
            run_batch_classification_pipeline(
                input_dir=args.input_dir,
                model_path=args.model_path,
                output_file=args.output_file,
                use_data_parallel=use_data_parallel,
                data_n_jobs=data_n_jobs
            )
        elif args.stage == "save-config":
            save_current_config(name=args.name, description=args.description)
        elif args.stage == "list-configs":
            list_saved_configs()
        elif args.stage == "create-training-config":
            create_training_config(
                name=args.name,
                models=args.models,
                strategies=args.strategies,
                use_gpu=args.gpu,
                use_sample_weights=args.sample_weights,
                description=args.description
            )
        elif args.stage == "detect-mislabels":
            run_mislabel_detection(
                focus_range_min=args.focus_min,
                focus_range_max=args.focus_max,
                use_features=not args.no_features,
                clustering_methods=args.clustering_methods,
                outlier_methods=args.outlier_methods,
                min_confidence=args.min_confidence,
                export_results=not args.no_export,
                config_path=config_path,
                use_feature_parallel=use_parallel,
                feature_n_jobs=n_jobs,
                use_data_parallel=use_data_parallel,
                data_n_jobs=data_n_jobs,
                exclude_suspects_file=exclude_suspects_file,
                strategy_override=getattr(args, 'strategy', None),
                no_raw_spectral=getattr(args, 'no_raw_spectral', False)
            )          
    except (DataValidationError, PipelineError, FileNotFoundError) as e:
        logger.error(f"Pipeline stopped due to a known error: {e}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred in pipeline stage '{args.stage}': {e}", exc_info=True)


if __name__ == "__main__":
    main()