"""
Peak shape functions for spectral fitting.

This module contains mathematical functions for fitting various peak shapes
commonly found in spectroscopy data, extracted from the LIBSsa project.
"""

import numpy as np
from scipy.special import wofz
from typing import List, Dict, Any


class PeakShapes:
    """Collection of peak shape functions for spectral fitting."""
    
    @staticmethod
    def lorentzian(x: np.ndarray, height: float, width: float, center: float) -> np.ndarray:
        """
        Lorentzian peak function.
        
        Args:
            x: Wavelength array
            height: Peak height
            width: Full width at half maximum (FWHM)
            center: Peak center position
            
        Returns:
            Intensity values for Lorentzian peak
        """
        return height / (1 + 4 * ((x - center) / width) ** 2)
    
    @staticmethod
    def lorentzian_multi(x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Multiple Lorentzian peaks function.
        
        Args:
            x: Wavelength array
            *args: Parameters (height, width, center) for each peak
            **kwargs: Additional parameters (not used in basic version)
            
        Returns:
            Combined intensity values for all peaks
        """
        n_params = 3
        n_peaks = len(args) // n_params
        params = np.array_split(np.abs(args), n_peaks)
        
        result = np.zeros_like(x)
        for p in params:
            height, width, center = p
            result += PeakShapes.lorentzian(x, height, width, center)
        
        return result
    
    @staticmethod
    def lorentzian_fixed_center(x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Lorentzian with fixed center positions.
        
        Args:
            x: Wavelength array
            *args: Parameters (height, width) for each peak
            **kwargs: Must contain 'centers' - list of fixed center positions
            
        Returns:
            Combined intensity values for all peaks
        """
        centers = kwargs.get('centers', [])
        n_params = 2  # height, width (center is fixed)
        n_peaks = len(args) // n_params
        params = np.array_split(np.abs(args), n_peaks)
        
        result = np.zeros_like(x)
        for i, p in enumerate(params):
            if i < len(centers):
                height, width = p
                center = centers[i]
                result += PeakShapes.lorentzian(x, height, width, center)
        
        return result
    
    @staticmethod
    def gaussian(x: np.ndarray, height: float, width: float, center: float) -> np.ndarray:
        """
        Gaussian peak function.
        
        Args:
            x: Wavelength array
            height: Peak height
            width: Full width at half maximum (FWHM)
            center: Peak center position
            
        Returns:
            Intensity values for Gaussian peak
        """
        sigma = width / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
        return height * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    
    @staticmethod
    def gaussian_multi(x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Multiple Gaussian peaks function.
        
        Args:
            x: Wavelength array
            *args: Parameters (height, width, center) for each peak
            **kwargs: Additional parameters
            
        Returns:
            Combined intensity values for all peaks
        """
        n_params = 3
        n_peaks = len(args) // n_params
        params = np.array_split(np.abs(args), n_peaks)
        
        result = np.zeros_like(x)
        for p in params:
            height, width, center = p
            result += PeakShapes.gaussian(x, height, width, center)
        
        return result
    
    @staticmethod
    def gaussian_fixed_center(x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Gaussian with fixed center positions.
        
        Args:
            x: Wavelength array
            *args: Parameters (height, width) for each peak
            **kwargs: Must contain 'centers' - list of fixed center positions
            
        Returns:
            Combined intensity values for all peaks
        """
        centers = kwargs.get('centers', [])
        n_params = 2  # height, width (center is fixed)
        n_peaks = len(args) // n_params
        params = np.array_split(np.abs(args), n_peaks)
        
        result = np.zeros_like(x)
        for i, p in enumerate(params):
            if i < len(centers):
                height, width = p
                center = centers[i]
                result += PeakShapes.gaussian(x, height, width, center)
        
        return result
    
    @staticmethod
    def voigt(x: np.ndarray, area: float, width_l: float, width_g: float, center: float) -> np.ndarray:
        """
        Voigt profile (convolution of Lorentzian and Gaussian).
        
        Args:
            x: Wavelength array
            area: Peak area
            width_l: Lorentzian width component
            width_g: Gaussian width component  
            center: Peak center position
            
        Returns:
            Intensity values for Voigt profile
        """
        sigma = width_g / (2 * np.sqrt(2 * np.log(2)))
        gamma = width_l / 2
        
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        w = wofz(z)
        
        return area * np.real(w) / (sigma * np.sqrt(2 * np.pi))
    
    @staticmethod
    def voigt_multi(x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Multiple Voigt profiles function.
        
        Args:
            x: Wavelength array
            *args: Parameters (area, width_l, width_g, center) for each peak
            **kwargs: Additional parameters
            
        Returns:
            Combined intensity values for all peaks
        """
        n_params = 4
        n_peaks = len(args) // n_params
        
        result = np.zeros_like(x)
        for i in range(n_peaks):
            p = args[i*n_params : (i+1)*n_params]
            area, width_l, width_g = np.abs(p[:3])
            center = p[3]
            result += PeakShapes.voigt(x, area, width_l, width_g, center)
        
        return result
    
    @staticmethod
    def asymmetric_lorentzian(x: np.ndarray, height: float, width: float, 
                            center: float, asymmetry: float) -> np.ndarray:
        """
        Asymmetric Lorentzian peak function.
        
        Args:
            x: Wavelength array
            height: Peak height
            width: Peak width
            center: Peak center position
            asymmetry: Asymmetry parameter
            
        Returns:
            Intensity values for asymmetric Lorentzian peak
        """
        # Simple asymmetric Lorentzian implementation
        dx = x - center
        width_left = width * (1 + asymmetry)
        width_right = width * (1 - asymmetry)
        
        result = np.zeros_like(x)
        left_mask = dx <= 0
        right_mask = dx > 0
        
        # Left side
        result[left_mask] = height / (1 + 4 * (dx[left_mask] / width_left) ** 2)
        
        # Right side  
        result[right_mask] = height / (1 + 4 * (dx[right_mask] / width_right) ** 2)
        
        return result
    
    @staticmethod
    def trapezoidal_integration(x: np.ndarray, y: np.ndarray) -> float:
        """
        Simple trapezoidal rule integration.
        
        Args:
            x: Wavelength array
            y: Intensity array
            
        Returns:
            Integrated area under the curve
        """
        return np.trapezoid(y, x)
    
    @classmethod
    def get_shape_function(cls, shape_name: str):
        """
        Get peak shape function by name.
        
        Args:
            shape_name: Name of the peak shape
            
        Returns:
            Corresponding peak shape function
            
        Raises:
            ValueError: If shape name is not recognized
        """
        shape_map = {
            'lorentzian': cls.lorentzian_multi,
            'lorentzian_fixed_center': cls.lorentzian_fixed_center,
            'gaussian': cls.gaussian_multi,
            'gaussian_fixed_center': cls.gaussian_fixed_center,
            'voigt': cls.voigt_multi,
            'asymmetric_lorentzian': cls.asymmetric_lorentzian,
            'trapezoidal': cls.trapezoidal_integration
        }
        
        shape_lower = shape_name.lower()
        if shape_lower in shape_map:
            return shape_map[shape_lower]
        else:
            raise ValueError(f"Unknown peak shape: {shape_name}")
    
    @staticmethod
    def calculate_peak_area(height: float, width: float, shape: str) -> float:
        """
        Calculate analytical peak area based on shape.
        
        Args:
            height: Peak height
            width: Peak width (FWHM)
            shape: Peak shape name
            
        Returns:
            Calculated peak area
        """
        shape_lower = shape.lower()
        
        if 'lorentzian' in shape_lower:
            return (height * width * np.pi) / 2
        elif 'gaussian' in shape_lower:
            return (2 * height * width) * np.sqrt(np.pi / 2)
        elif 'voigt' in shape_lower:
            # For Voigt, area is the first parameter
            return height  # Assuming height is actually area for Voigt
        else:
            # Default to rectangular approximation
            return height * width
    
    @staticmethod
    def calculate_fwhm(width_l: float, width_g: float, shape: str) -> float:
        """
        Calculate Full Width at Half Maximum for combined profiles.
        
        Args:
            width_l: Lorentzian width component
            width_g: Gaussian width component
            shape: Peak shape name
            
        Returns:
            Combined FWHM
        """
        if 'voigt' in shape.lower():
            # Voigt FWHM approximation
            return 0.5346 * width_l + np.sqrt(0.2166 * width_l**2 + width_g**2)
        else:
            # For other shapes, return the primary width
            return width_l