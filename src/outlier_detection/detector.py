"""
Spectral Outlier Detection Module

This module provides standalone outlier detection functionality for spectral data,
extracted from the LIBSsa project. It implements two algorithms:
- SAM (Spectral Angle Mapper)
- MAD (Median Absolute Deviation)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class OutlierResult:
	"""Container for outlier detection results."""
	cleaned_spectra: List[np.ndarray]
	removed_report: np.ndarray
	outlier_indices: List[List[int]]


class SpectralOutlierDetector:
	"""
	A class for detecting and removing outliers in spectral data.
	
	This class implements two outlier detection algorithms:
	1. SAM (Spectral Angle Mapper): Uses cosine similarity between spectra
	2. MAD (Median Absolute Deviation): Statistical method based on median deviation
	
	Attributes:
		method (str): The outlier detection method ('SAM' or 'MAD')
		threshold (float): The threshold for outlier detection
	"""
	
	def __init__(self, method: str = 'SAM', threshold: float = None):
		"""
		Initialize the outlier detector.
		
		Args:
			method: Detection method ('SAM' or 'MAD')
			threshold: Detection threshold. If None, uses default values:
					  - SAM: 0.95 (cosine similarity threshold)
					  - MAD: 3.0 (number of MADs from median)
		
		Raises:
			ValueError: If method is not 'SAM' or 'MAD'
		"""
		self.method = method.upper()
		if self.method not in ['SAM', 'MAD']:
			raise ValueError(f"Invalid method '{method}'. Must be 'SAM' or 'MAD'")
		
		if threshold is None:
			self.threshold = 0.95 if self.method == 'SAM' else 3.0
		else:
			self.threshold = threshold
			
		# Validate threshold ranges
		if self.method == 'SAM' and not (0 <= self.threshold <= 1):
			raise ValueError("SAM threshold must be between 0 and 1")
		elif self.method == 'MAD' and not (1 <= self.threshold <= 5):
			raise ValueError("MAD threshold should typically be between 1 and 5")
	
	def detect_outliers(self, spectra_list: List[np.ndarray], 
					   progress_callback: Optional[Callable[[int], None]] = None) -> OutlierResult:
		"""
		Detect and remove outliers from a list of spectral datasets.
		
		Args:
			spectra_list: List of 2D numpy arrays, where each array has shape 
						 (n_wavelengths, n_spectra) representing multiple spectra 
						 from a single sample
			progress_callback: Optional callback function that receives progress 
							  updates (current sample index)
		
		Returns:
			OutlierResult containing:
				- cleaned_spectra: List of arrays with outliers removed
				- removed_report: Array of shape (n_samples, 2) with [removed, total] counts
				- outlier_indices: List of lists containing indices of removed spectra
		
		Raises:
			ValueError: If too many outliers are removed (all spectra from a sample)
		"""
		if self.method == 'SAM':
			return self._detect_outliers_sam(spectra_list, progress_callback)
		else:
			return self._detect_outliers_mad(spectra_list, progress_callback)
	
	def _detect_outliers_sam(self, spectra_list: List[np.ndarray], 
						   progress_callback: Optional[Callable[[int], None]] = None) -> OutlierResult:
		"""
		Detect outliers using Spectral Angle Mapper (SAM) algorithm.
		
		SAM calculates the cosine of the angle between the average spectrum 
		and each individual spectrum. Spectra with cosine values below the 
		threshold are marked as outliers.
		"""
		cleaned_spectra = []
		removed_report = []
		outlier_indices = []
		
		for i, spectra in enumerate(spectra_list):
			# Calculate average spectrum
			avg_spectrum = np.mean(spectra, axis=1)
			
			# Initialize empty output array  
			clean_spectra = None
			outliers = []
			removed = [0, spectra.shape[1]]
			
			# Check each spectrum
			for j in range(spectra.shape[1]):
				spectrum = spectra[:, j]
				
				# Calculate cosine similarity
				cos_theta = np.dot(avg_spectrum, spectrum) / (
					np.linalg.norm(avg_spectrum) * np.linalg.norm(spectrum)
				)
				
				if cos_theta >= self.threshold:
					# Keep this spectrum
					if clean_spectra is None:
						clean_spectra = spectrum.reshape(-1, 1)
					else:
						clean_spectra = np.column_stack((clean_spectra, spectrum))
				else:
					# Mark as outlier
					removed[0] += 1
					outliers.append(j)
			
			# Check if any spectra remain
			if clean_spectra is None:
				raise ValueError(f"Sample {i}: Too many outliers removed. "
							   f"No spectra remain after outlier removal.")
			
			cleaned_spectra.append(clean_spectra)
			removed_report.append(removed)
			outlier_indices.append(outliers)
			
			if progress_callback:
				progress_callback(i)
		
		return OutlierResult(
			cleaned_spectra=cleaned_spectra,
			removed_report=np.array(removed_report),
			outlier_indices=outlier_indices
		)
	
	def _detect_outliers_mad(self, spectra_list: List[np.ndarray], 
						   progress_callback: Optional[Callable[[int], None]] = None) -> OutlierResult:
		"""
		Detect outliers using Median Absolute Deviation (MAD) algorithm.
		
		MAD is a robust statistical method that identifies outliers based on 
		their deviation from the median. A spectrum is considered an outlier 
		if more than 5% of its wavelengths deviate from the median by more 
		than the threshold number of MADs.
		"""
		cleaned_spectra = []
		removed_report = []
		outlier_indices = []
		
		# Scaling factor for MAD (makes it consistent with standard deviation)
		b = 1.4826
		
		for i, spectra in enumerate(spectra_list):
			# Calculate median spectrum and MAD for each wavelength
			median_spectrum = np.median(spectra, axis=1)
			deviations = np.abs(spectra.T - median_spectrum).T
			mad_vector = b * np.median(deviations, axis=1)
			
			# Initialize empty output array
			clean_spectra = None
			outliers = []
			removed = [0, spectra.shape[1]]
			
			# Check each spectrum
			for j in range(spectra.shape[1]):
				spectrum = spectra[:, j]
				
				# Calculate normalized deviations
				# Avoid division by zero
				mad_vector_safe = np.where(mad_vector == 0, 1e-10, mad_vector)
				normalized_deviations = np.abs((spectrum - median_spectrum) / mad_vector_safe)
				
				# Check if 95% of wavelengths are within threshold MADs
				within_threshold = normalized_deviations < self.threshold
				if np.sum(within_threshold) / len(spectrum) >= 0.95:
					# Keep this spectrum
					if clean_spectra is None:
						clean_spectra = spectrum.reshape(-1, 1)
					else:
						clean_spectra = np.column_stack((clean_spectra, spectrum))
				else:
					# Mark as outlier
					removed[0] += 1
					outliers.append(j)
			
			# Check if any spectra remain
			if clean_spectra is None:
				raise ValueError(f"Sample {i}: Too many outliers removed. "
							   f"No spectra remain after outlier removal.")
			
			cleaned_spectra.append(clean_spectra)
			removed_report.append(removed)
			outlier_indices.append(outliers)
			
			if progress_callback:
				progress_callback(i)
		
		return OutlierResult(
			cleaned_spectra=cleaned_spectra,
			removed_report=np.array(removed_report),
			outlier_indices=outlier_indices
		)
	
	def detect_outliers_single(self, spectra: np.ndarray) -> Tuple[np.ndarray, List[int]]:
		"""
		Detect outliers in a single spectral dataset.
		
		Args:
			spectra: 2D array of shape (n_wavelengths, n_spectra)
		
		Returns:
			Tuple of (cleaned_spectra, outlier_indices)
		"""
		result = self.detect_outliers([spectra])
		return result.cleaned_spectra[0], result.outlier_indices[0]