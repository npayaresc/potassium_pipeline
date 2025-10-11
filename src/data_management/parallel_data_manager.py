"""
Parallel Data Management Module: Parallelized versions of file averaging and cleansing operations.
"""
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import shutil

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _average_single_group(args: Tuple) -> Dict[str, Any]:
    """
    Average a single group of files. This function is designed to be pickle-able
    for multiprocessing.
    
    Args:
        args: Tuple containing (prefix, file_paths, output_path)
    
    Returns:
        Dictionary with 'prefix', 'success', and 'error' keys
    """
    prefix, file_paths, output_path = args
    
    try:
        # Import here to avoid pickling issues
        from src.data_management.data_manager import DataManager
        from src.config.pipeline_config import config
        
        data_manager = DataManager(config)
        
        all_data, wavelengths = [], None
        for file_path in file_paths:
            try:
                df = data_manager._read_raw_intensity_data(file_path)
                if wavelengths is None: 
                    wavelengths = df['Wavelength'].values
                
                intensity_cols = [col for col in df.columns if col != 'Wavelength']
                all_data.append(df[intensity_cols].values)
            except Exception as e:
                logger.error(f"  Error reading {file_path.name}: {e}")
                continue
        
        if not all_data:
            return {'prefix': prefix, 'success': False, 'error': 'No valid data found'}
        
        # Find minimum dimensions across all data arrays
        min_shots = min(data.shape[1] for data in all_data)
        trimmed_data = [data[:, :min_shots] for data in all_data]
        
        stacked_data = np.stack(trimmed_data, axis=0)
        averaged_data = np.mean(stacked_data, axis=0)
        
        # Create output DataFrame
        output_df = pd.DataFrame()
        output_df['Wavelength'] = wavelengths
        for i in range(min_shots):
            output_df[f'Intensity{i+1}'] = averaged_data[:, i]
        
        # Write to temporary file first, then move to final location (atomic write)
        temp_path = output_path.with_suffix(output_path.suffix + '.tmp')
        output_df.to_csv(temp_path, index=False)
        shutil.move(str(temp_path), str(output_path))
        
        return {'prefix': prefix, 'success': True, 'files_processed': len(file_paths)}
        
    except Exception as e:
        return {'prefix': prefix, 'success': False, 'error': str(e)}


def parallel_average_raw_files(data_manager, n_jobs: int = -1):
    """
    Parallel version of average_raw_files that processes multiple groups simultaneously.
    
    Args:
        data_manager: DataManager instance
        n_jobs: Number of parallel jobs. -1 uses all CPU cores, -2 uses all but one
    """
    from collections import defaultdict
    
    logger.info(f"Starting PARALLEL raw file averaging from: {data_manager.config.raw_data_dir}")
    data_manager.config.averaged_files_dir.mkdir(parents=True, exist_ok=True)

    # Group files by sample ID
    file_groups = defaultdict(list)
    for file_path in data_manager.config.raw_data_dir.glob('**/*.csv.txt'):
        prefix = data_manager._extract_file_prefix(file_path.name)
        file_groups[prefix].append(file_path)
    
    # Sort files within each group to ensure consistent averaging order
    for prefix in file_groups:
        file_groups[prefix].sort()
    
    logger.info(f"Found {len(file_groups)} unique sample groups to average.")
    
    # Determine number of workers
    if n_jobs == -1:
        n_workers = mp.cpu_count()
    elif n_jobs == -2:
        n_workers = max(1, mp.cpu_count() - 1)
    else:
        n_workers = max(1, n_jobs)
    
    logger.info(f"Using {n_workers} parallel workers for file averaging")
    
    # Prepare arguments for parallel processing
    args_list = [
        (prefix, file_paths, data_manager.config.averaged_files_dir / f"{prefix}.csv.txt")
        for prefix, file_paths in file_groups.items()
    ]
    
    # Process in parallel
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_args = {
            executor.submit(_average_single_group, args): args[0]
            for args in args_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_args):
            prefix = future_to_args[future]
            try:
                result = future.result()
                if result['success']:
                    successful += 1
                    logger.debug(f"Successfully averaged group: {prefix} ({result.get('files_processed', 0)} files)")
                else:
                    failed += 1
                    logger.error(f"Failed to average group {prefix}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                failed += 1
                logger.error(f"Exception while processing group {prefix}: {e}")
    
    logger.info(f"Parallel averaging complete: {successful} successful, {failed} failed")


def _clean_single_file(args: Tuple) -> Dict[str, Any]:
    """
    Clean a single spectral file. This function is designed to be pickle-able
    for multiprocessing.
    
    Args:
        args: Tuple containing (file_path, config_dict, global_wavelength_range, file_name)
    
    Returns:
        Dictionary with cleaning results
    """
    file_path, config_dict, global_wavelength_range, file_name = args
    
    try:
        # Import here to avoid pickling issues
        from src.config.pipeline_config import Config
        from src.data_management.data_manager import DataManager
        from src.cleansing.data_cleanser import DataCleanser
        
        # Reconstruct config from dict
        config = Config(**config_dict)
        data_manager = DataManager(config)
        data_cleanser = DataCleanser(config)
        
        # Set global wavelength range if provided
        if global_wavelength_range:
            data_manager._global_wavelength_range = global_wavelength_range
        
        # Load spectral data
        wavelengths, intensities = data_manager.load_spectral_file(file_path)
        
        # Apply wavelength standardization if enabled
        if config.enable_wavelength_standardization:
            wavelengths, intensities = data_manager.standardize_wavelength_grid(
                wavelengths, intensities,
                interpolation_method=config.wavelength_interpolation_method
            )
        
        # Ensure intensity is 2D for the cleanser
        if intensities.ndim == 1:
            intensities = intensities.reshape(-1, 1)
        
        # Clean the spectra
        clean_intensities = data_cleanser.clean_spectra(str(file_path), intensities)
        
        if clean_intensities.size > 0:
            # Prepare cleansed data
            cleansed_path = config.cleansed_files_dir / file_name
            cleansed_df = pd.DataFrame({'Wavelength': wavelengths})
            
            for i in range(clean_intensities.shape[1]):
                cleansed_df[f'Intensity{i+1}'] = clean_intensities[:, i]
            
            # Write to temporary file first, then move to final location (atomic write)
            temp_path = cleansed_path.with_suffix(cleansed_path.suffix + '.tmp')
            cleansed_df.to_csv(temp_path, index=False)
            shutil.move(str(temp_path), str(cleansed_path))
            
            # Get sample ID for metadata
            sample_id = file_name.replace('.csv.txt', '')
            
            return {
                'file_path': str(file_path),
                'success': True,
                'sample_id': sample_id,
                'wavelengths': wavelengths,
                'intensities': clean_intensities,
                'cleansed_path': str(cleansed_path)
            }
        else:
            return {
                'file_path': str(file_path),
                'success': False,
                'reason': 'excessive_outliers',
                'sample_id': file_name.replace('.csv.txt', '')
            }
            
    except Exception as e:
        return {
            'file_path': str(file_path),
            'success': False,
            'error': str(e),
            'sample_id': file_name.replace('.csv.txt', '') if file_name else 'unknown'
        }


def parallel_load_and_clean_data(cfg, training_files: List[Path], metadata: pd.DataFrame, 
                                 global_wavelength_range: Tuple[float, float] = None,
                                 n_jobs: int = -1) -> List[Dict[str, Any]]:
    """
    Parallel version of the cleansing loop that processes multiple files simultaneously.
    
    Args:
        cfg: Pipeline configuration
        training_files: List of files to process
        metadata: Sample metadata DataFrame
        global_wavelength_range: Global wavelength range for standardization
        n_jobs: Number of parallel jobs. -1 uses all CPU cores, -2 uses all but one
        
    Returns:
        List of processed data dictionaries
    """
    import shutil
    
    logger.info(f"Starting PARALLEL cleansing of {len(training_files)} averaged files...")
    
    # Determine number of workers
    if n_jobs == -1:
        n_workers = mp.cpu_count()
    elif n_jobs == -2:
        n_workers = max(1, mp.cpu_count() - 1)
    else:
        n_workers = max(1, n_jobs)
    
    logger.info(f"Using {n_workers} parallel workers for file cleansing")
    
    # Convert config to dict for pickling
    config_dict = cfg.model_dump()
    
    # Prepare arguments for parallel processing
    args_list = [
        (file_path, config_dict, global_wavelength_range, file_path.name)
        for file_path in training_files
    ]
    
    # Process in parallel
    processed_data_for_training = []
    bad_files = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(_clean_single_file, args): args[0]
            for args in args_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                
                if result['success']:
                    # Get metadata for this sample
                    sample_id = result['sample_id']
                    sample_metadata = metadata[metadata[cfg.sample_id_column] == sample_id]
                    
                    if not sample_metadata.empty:
                        target_value = sample_metadata[cfg.target_column].values[0]
                        processed_data_for_training.append({
                            cfg.sample_id_column: sample_id,
                            'wavelengths': result['wavelengths'],
                            'intensities': result['intensities'],
                            cfg.target_column: target_value
                        })
                        logger.debug(f"Successfully processed: {sample_id}")
                    else:
                        logger.warning(f"No metadata found for sample: {sample_id}")
                else:
                    if result.get('reason') == 'excessive_outliers':
                        bad_files.append(file_path)
                        logger.debug(f"Excessive outliers in: {file_path.name}")
                    else:
                        logger.error(f"Failed to process {file_path.name}: {result.get('error', 'Unknown error')}")
                        
            except Exception as e:
                logger.error(f"Exception while processing {file_path}: {e}")
    
    # Move bad files to bad_files directory
    if bad_files:
        logger.info(f"Moving {len(bad_files)} bad files to {cfg.bad_files_dir}")
        for file_path in bad_files:
            try:
                destination = cfg.bad_files_dir / file_path.name
                shutil.move(str(file_path), str(destination))
                logger.debug(f"Moved bad averaged file {file_path.name} to {cfg.bad_files_dir}")
            except Exception as e:
                logger.error(f"Failed to move bad file {file_path.name}: {e}")
    
    logger.info(f"Parallel cleansing complete: {len(processed_data_for_training)} files processed successfully")
    
    # Sort data by sample ID to ensure consistent ordering regardless of parallel processing order
    processed_data_for_training.sort(key=lambda x: x[cfg.sample_id_column])
    
    return processed_data_for_training