"""
Data Management Module: Handles loading, validating, splitting, and saving of datasets.
"""
from collections import defaultdict
import logging
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d

from src.config.pipeline_config import Config
from src.utils.custom_exceptions import DataValidationError

logger = logging.getLogger(__name__)

class DataManager:
    """Manages the data loading and preparation stages."""
    def __init__(self, config: Config):
        self.config = config
        self.sample_metadata: Optional[pd.DataFrame] = None
        self._global_wavelength_range: Optional[Tuple[float, float]] = None
        # Initialize parallel settings from config
        self.use_parallel_data_ops: bool = getattr(config.parallel, 'use_data_parallel', False)
        self.data_ops_n_jobs: int = getattr(config.parallel, 'data_n_jobs', -1)

    def load_and_prepare_metadata(self) -> pd.DataFrame:
        """Loads, filters, and validates reference potassium data."""
        logger.info(f"Loading reference data from: {self.config.reference_data_path}")
        try:
            df = pd.read_excel(self.config.reference_data_path)
            df.columns = df.columns.str.strip()
        except FileNotFoundError:
            raise DataValidationError(f"Reference file not found: {self.config.reference_data_path}")

        required_cols = {self.config.sample_id_column, self.config.target_column}
        if not required_cols.issubset(df.columns):
            raise DataValidationError(f"Reference Excel must contain columns: {required_cols}")

        df = df.dropna(subset=[self.config.target_column])
        df = df[df[self.config.target_column] > 0]
        logger.info(f"Found {len(df)} samples with positive target values.")

        if self.config.exclude_pot_samples:
            original_count = len(df)
            df = df[~df[self.config.sample_id_column].str.startswith('POT', na=False)]
            logger.info(f"Filtered out {original_count - len(df)} POT samples.")
            
        # --- ADDED: TARGET VALUE RANGE FILTER ---
        original_count = len(df)
        if self.config.target_value_min is not None:
            df = df[df[self.config.target_column] >= self.config.target_value_min]
        if self.config.target_value_max is not None:
            df = df[df[self.config.target_column] <= self.config.target_value_max]
        
        if len(df) < original_count:
            logger.info(
                f"Filtered samples by target range "
                f"[{self.config.target_value_min} - {self.config.target_value_max}]. "
                f"Removed {original_count - len(df)} samples."
            )
        # --- END OF ADDED LOGIC ---
        
        
        if df.empty: raise DataValidationError("No valid samples found after filtering.")

        self.sample_metadata = df
        logger.info(f"Metadata prepared for {len(self.sample_metadata)} samples.")
        return self.sample_metadata
    
    # --- Start of logic adapted from average_intensity_files.py ---

    def _extract_file_prefix(self, filename: str) -> str:
        """
        Extracts the prefix from a filename based on the pattern:
        everything before the last underscore.
        Example: '1053789KENP1S001_G_2025_01_04_1.csv.txt' -> '1053789KENP1S001_G_2025_01_04'
        """
        name_without_ext = filename.replace('.csv.txt', '')
        last_underscore_index = name_without_ext.rfind('_')
        if last_underscore_index == -1:
            return name_without_ext
        return name_without_ext[:last_underscore_index]
    
    def _read_raw_intensity_data(self, file_path: Path) -> pd.DataFrame:
        """
        Reads intensity data from a raw CSV file, skipping metadata headers.
        Handles exponential notation and mixed data types by forcing numeric conversion.
        """
        data_start_line = 0
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if line.strip().startswith('Wavelength'):
                    data_start_line = i
                    break
        if data_start_line is None:
            raise ValueError(f"Could not find 'Wavelength' header in {file_path}")
        
        df = pd.read_csv(file_path, skiprows=data_start_line)
        df.columns = [col.strip() for col in df.columns]
        
        # Convert all columns except Wavelength to numeric, handling exponential notation
        # This forces conversion of strings like "1.23E-4" to proper floats
        for col in df.columns:
            if col != 'Wavelength':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle Wavelength column separately to ensure it's numeric
        df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')
        
        # Remove any rows with NaN values that couldn't be converted
        df = df.dropna()
        
        return df
    
    def average_files_in_memory(self, file_paths: List[Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Core logic to average a group of spectral files in memory.

        Args:
            file_paths: A list of file paths belonging to a single sample.

        Returns:
            A tuple of (wavelengths, averaged_intensity). Returns (None, None) on failure.
        """
        all_data, wavelengths = [], None
        for file_path in file_paths:
            try:
                df = self._read_raw_intensity_data(file_path)
                if wavelengths is None:
                    wavelengths = df['Wavelength'].values
                
                intensity_cols = [col for col in df.columns if col != 'Wavelength']
                all_data.append(df[intensity_cols].values)
            except Exception as e:
                logger.error(f"  Error reading {file_path.name} for averaging: {e}")
                continue
        
        if not all_data:
            return None, None

        # Find minimum dimensions across all data arrays
        min_wavelengths = min(data.shape[0] for data in all_data)
        min_shots = min(data.shape[1] for data in all_data)
        
        # Trim all data to the same dimensions
        trimmed_data = [data[:min_wavelengths, :min_shots] for data in all_data]
        
        # Also trim wavelengths array to match
        wavelengths = wavelengths[:min_wavelengths]
        
        stacked_data = np.stack(trimmed_data, axis=0)
        averaged_data = np.mean(stacked_data, axis=0)
        
        return wavelengths, averaged_data
    
    def average_raw_files(self, use_parallel: bool = None, n_jobs: int = None):
        """
        Groups raw files by sample ID, averages their intensities, and saves
        a single file per sample to the averaged_files_dir.
        This logic is a direct port from the average_files.py script.
        
        Args:
            use_parallel: Whether to use parallel processing. If None, uses self.use_parallel_data_ops
            n_jobs: Number of parallel jobs. If None, uses self.data_ops_n_jobs
        """
        use_parallel = use_parallel if use_parallel is not None else self.use_parallel_data_ops
        n_jobs = n_jobs if n_jobs is not None else self.data_ops_n_jobs
        
        if use_parallel:
            from src.data_management.parallel_data_manager import parallel_average_raw_files
            return parallel_average_raw_files(self, n_jobs=n_jobs)
            
        logger.info(f"Starting raw file averaging from: {self.config.raw_data_dir}")
        self.config.averaged_files_dir.mkdir(parents=True, exist_ok=True)

        file_groups = defaultdict(list)
        for file_path in self.config.raw_data_dir.glob('**/*.csv.txt'):
            prefix = self._extract_file_prefix(file_path.name)
            file_groups[prefix].append(file_path)

        logger.info(f"Found {len(file_groups)} unique sample groups to average.")

        for prefix, file_paths in file_groups.items():
            logger.debug(f"Processing group: {prefix} ({len(file_paths)} files)")
            output_path = self.config.averaged_files_dir / f"{prefix}.csv.txt"

            all_data, wavelengths = [], None
            for file_path in file_paths:
                try:
                    df = self._read_raw_intensity_data(file_path)
                    if wavelengths is None: wavelengths = df['Wavelength'].values
                    
                    intensity_cols = [col for col in df.columns if col != 'Wavelength']
                    all_data.append(df[intensity_cols].values)
                except Exception as e:
                    logger.error(f"  Error reading {file_path.name}: {e}")
                    continue
            
            if not all_data:
                logger.error(f"  No valid data found for group {prefix}")
                continue

            min_shots = min(data.shape[1] for data in all_data)
            trimmed_data = [data[:, :min_shots] for data in all_data]
            
            stacked_data = np.stack(trimmed_data, axis=0)
            averaged_data = np.mean(stacked_data, axis=0)
            
            output_df = pd.DataFrame()
            output_df['Wavelength'] = wavelengths
            for i in range(min_shots):
                output_df[f'Intensity{i+1}'] = averaged_data[:, i]
            
            output_df.to_csv(output_path, index=False)
        
    

    def get_training_data_paths(self) -> List[Path]:
        """
        Gets a list of SPECTRAL files that have corresponding metadata.
        NOW READS FROM THE AVERAGED FILES DIRECTORY.
        """
        if self.sample_metadata is None: self.load_and_prepare_metadata()

        # This method now points to the averaged files directory
        all_files = list(self.config.averaged_files_dir.glob("*.csv.txt"))
        valid_sample_ids = set(self.sample_metadata[self.config.sample_id_column])
        
        # Match the prefix of the averaged file to the sample ID
        training_files = [f for f in all_files if f.name.replace('.csv.txt', '') in valid_sample_ids]

        logger.info(f"Found {len(training_files)} averaged files that match metadata.")
        
        if self.config.max_samples:
            logger.warning(f"Subsetting to a maximum of {self.config.max_samples} samples.")
            return training_files[:self.config.max_samples]
            
        return training_files

    @staticmethod
    def load_spectral_file(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads a spectral data file (now expects averaged files).
        This method is updated to correctly parse the averaged CSVs and handle numeric conversion.
        """
        df = pd.read_csv(file_path)
        
        # Ensure Wavelength column is numeric
        df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')
        
        # Find intensity columns and ensure they're numeric
        intensity_cols = [col for col in df.columns if 'Intensity' in col]
        for col in intensity_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        wavelengths = df['Wavelength'].values
        intensities = df[intensity_cols].values
        return wavelengths, intensities

    def standardize_wavelength_grid(self, wavelengths: np.ndarray, intensities: np.ndarray, 
                                   target_wavelengths: Optional[np.ndarray] = None,
                                   interpolation_method: str = 'linear',
                                   use_global_range: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standardizes spectral data to a common wavelength grid using interpolation.
        
        This function addresses wavelength calibration drift and resolution differences
        by resampling all spectra onto a consistent wavelength grid.
        
        Args:
            wavelengths: Original wavelength array (shape: [n_wavelengths])
            intensities: Original intensity array (shape: [n_wavelengths, n_shots])
            target_wavelengths: Target wavelength grid. If None, creates from data range
            interpolation_method: Interpolation method ('linear', 'cubic', 'nearest')
            use_global_range: If True, uses global range from all files for consistency
            
        Returns:
            Tuple of (standardized_wavelengths, standardized_intensities)
            
        Raises:
            DataValidationError: If wavelength ranges don't overlap sufficiently
        """
        
        # Generate target wavelength grid if not provided
        if target_wavelengths is None:
            if use_global_range and self._global_wavelength_range is not None:
                # Use the pre-determined global wavelength range for consistency
                min_wl, max_wl = self._global_wavelength_range
                logger.debug(f"Using global wavelength range: {min_wl:.2f}-{max_wl:.2f}nm")
            else:
                # Use the actual wavelength range from the current input data
                min_wl = float(wavelengths.min())
                max_wl = float(wavelengths.max())
                logger.debug(f"Using local wavelength range: {min_wl:.2f}-{max_wl:.2f}nm")
            
            # Use configured resolution (default 0.1nm for LIBS applications)
            resolution = self.config.wavelength_resolution
            target_wavelengths = np.arange(min_wl, max_wl + resolution, resolution)
            logger.debug(f"Generated target wavelength grid: {len(target_wavelengths)} points at {resolution}nm resolution")
        
        # Validate wavelength overlap
        overlap_min = max(wavelengths.min(), target_wavelengths.min())
        overlap_max = min(wavelengths.max(), target_wavelengths.max())
        overlap_fraction = (overlap_max - overlap_min) / (target_wavelengths.max() - target_wavelengths.min())
        
        if overlap_fraction < 0.8:  # Require 80% overlap
            raise DataValidationError(
                f"Insufficient wavelength overlap for interpolation. "
                f"Original range: [{wavelengths.min():.1f}, {wavelengths.max():.1f}]nm, "
                f"Target range: [{target_wavelengths.min():.1f}, {target_wavelengths.max():.1f}]nm, "
                f"Overlap: {overlap_fraction:.1%}"
            )
        
        # Ensure intensities is 2D
        if intensities.ndim == 1:
            intensities = intensities.reshape(-1, 1)
        
        n_wavelengths, n_shots = intensities.shape
        standardized_intensities = np.zeros((len(target_wavelengths), n_shots))
        
        # Interpolate each shot separately
        for shot_idx in range(n_shots):
            try:
                # Create interpolation function
                interpolator = interp1d(
                    wavelengths, 
                    intensities[:, shot_idx],
                    kind=interpolation_method,
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                
                # Interpolate to target grid
                standardized_intensities[:, shot_idx] = interpolator(target_wavelengths)
                
            except Exception as e:
                logger.error(f"Failed to interpolate shot {shot_idx}: {e}")
                # Fill with NaN values if interpolation fails
                standardized_intensities[:, shot_idx] = np.nan
        
        # Check for interpolation artifacts
        nan_count = np.isnan(standardized_intensities).sum()
        if nan_count > 0:
            logger.warning(f"Interpolation produced {nan_count} NaN values, replacing with zeros")
            standardized_intensities = np.nan_to_num(standardized_intensities, nan=0.0)
        
        # Validate output ranges are reasonable
        if np.any(standardized_intensities < 0):
            neg_count = (standardized_intensities < 0).sum()
            logger.warning(f"Interpolation produced {neg_count} negative intensity values")
            
        logger.info(
            f"Wavelength standardization complete: "
            f"{len(wavelengths)} -> {len(target_wavelengths)} wavelength points, "
            f"{n_shots} shots, method={interpolation_method}"
        )
        
        return target_wavelengths, standardized_intensities

    def create_reproducible_splits(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits data into training and testing sets and saves them."""
        logger.info("Creating reproducible train/test splits.")
        train_df, test_df = train_test_split(
            data, test_size=self.config.test_split_size, random_state=self.config.random_state
        )
        train_path = self.config.processed_data_dir / f"train_{self.config.run_timestamp}.csv"
        test_path = self.config.processed_data_dir / f"test_{self.config.run_timestamp}.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info(f"Train data ({len(train_df)} samples) saved to {train_path}")
        logger.info(f"Test data ({len(test_df)} samples) saved to {test_path}")
        return train_df, test_df
    
    def determine_global_wavelength_range_from_raw_files(self) -> Tuple[float, float]:
        """
        Determines the global wavelength range by examining raw files directly.
        
        Returns:
            Tuple of (min_wavelength, max_wavelength) across all raw files
        """
        min_wavelengths = []
        max_wavelengths = []

        # Get raw files from the raw data directory (including subdirectories)
        raw_files = list(self.config.raw_data_dir.glob("**/*.csv.txt"))

        logger.info(f"Determining global wavelength range from {len(raw_files)} raw files...")
        
        # Sample first 10 raw files for efficiency
        for file_path in raw_files[:10]:
            try:
                df = self._read_raw_intensity_data(file_path)
                wavelengths = df['Wavelength'].values
                min_wavelengths.append(wavelengths.min())
                max_wavelengths.append(wavelengths.max())
                logger.debug(f"File {file_path.name}: {wavelengths.min():.2f} - {wavelengths.max():.2f} nm")
            except Exception as e:
                logger.warning(f"Could not read wavelengths from {file_path.name}: {e}")
                continue
        
        if not min_wavelengths:
            raise DataValidationError("Could not determine wavelength range from any raw files")
        
        global_min = min(min_wavelengths)
        global_max = max(max_wavelengths)
        
        logger.info(f"Global wavelength range determined from raw files: {global_min:.2f} - {global_max:.2f} nm")
        return global_min, global_max