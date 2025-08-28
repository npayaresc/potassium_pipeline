#!/usr/bin/env python3
"""
Usage example for enhanced features in the nitrogen prediction pipeline.
Shows how to configure and use the new features.
"""

from src.config.pipeline_config import config

def configure_enhanced_features():
    """Example of how to configure enhanced features."""
    
    print("Enhanced Features Configuration Guide")
    print("=" * 50)
    
    print("\n1. Molecular Band Detection:")
    print("   - Detects CN, NH, NO molecular bands")
    print("   - Calculates molecular/atomic intensity ratios")
    print("   - Usage: config.enable_molecular_bands = True")
    
    print("\n2. Advanced Nutrient Ratios:")
    print("   - N/P, N/K, N/S ratios")
    print("   - Complex balance indicators (NP/KCa)")
    print("   - Log-transformed ratios")
    print("   - Usage: config.enable_advanced_ratios = True")
    
    print("\n3. Spectral Pattern Analysis:")
    print("   - FWHM (Full Width at Half Maximum)")
    print("   - Peak asymmetry indicators")
    print("   - Continuum background levels")
    print("   - Usage: config.enable_spectral_patterns = True")
    
    print("\n4. Interference Correction:")
    print("   - Fe interference on N lines")
    print("   - Self-absorption indicators")
    print("   - Spectral overlap detection")
    print("   - Usage: config.enable_interference_correction = True")
    
    print("\n5. Plasma State Indicators:")
    print("   - Ionic/neutral line ratios")
    print("   - Plasma temperature indicators")
    print("   - Excitation state features")
    print("   - Usage: config.enable_plasma_indicators = True")
    
    print("\n" + "=" * 50)
    print("Configuration Examples:")
    print("=" * 50)
    
    print("\n# Enable all enhanced features (default):")
    print("config.enable_molecular_bands = True")
    print("config.enable_macro_elements = True")
    print("config.enable_micro_elements = True")
    print("config.enable_oxygen_hydrogen = True")
    print("config.enable_advanced_ratios = True")
    print("config.enable_spectral_patterns = True")
    print("config.enable_interference_correction = True")
    print("config.enable_plasma_indicators = True")
    
    print("\n# Minimal configuration (only core improvements):")
    print("config.enable_molecular_bands = True")
    print("config.enable_advanced_ratios = True")
    print("config.enable_macro_elements = True")
    print("# Other features disabled for faster processing")
    
    print("\n# Memory-efficient configuration:")
    print("config.enable_molecular_bands = False")
    print("config.enable_spectral_patterns = False")
    print("config.enable_advanced_ratios = True")
    print("# Reduces feature count while keeping key ratios")
    
    print("\n" + "=" * 50)
    print("Wavelength Coverage Requirements:")
    print("=" * 50)
    
    print("\nFor optimal results, ensure your spectrometer covers:")
    print("- UV region (200-400 nm): Molecular bands, B, Cu, Mo lines")
    print("- Visible (400-700 nm): N, Fe, Mn, Zn, H-alpha lines")
    print("- Near-IR (700-900 nm): N, C, K, O, S lines")
    
    print("\nNote: Features will be set to NaN for unavailable wavelength regions")
    
    print("\n" + "=" * 50)
    print("Performance Considerations:")
    print("=" * 50)
    
    print("\n- Molecular bands: +15-20 features, moderate computation")
    print("- Advanced ratios: +20-30 features, fast computation")
    print("- Spectral patterns: +10-15 features, slow computation (FWHM)")
    print("- Interference correction: +5-10 features, moderate computation")
    print("- Plasma indicators: +5-10 features, fast computation")
    print("\nTotal potential new features: 55-85 additional features")

def show_current_configuration():
    """Display current configuration status."""
    
    print("\n" + "=" * 50)
    print("Current Configuration Status:")
    print("=" * 50)
    
    features = [
        ("Molecular Bands", config.enable_molecular_bands),
        ("Macro Elements", config.enable_macro_elements),
        ("Micro Elements", config.enable_micro_elements), 
        ("Oxygen/Hydrogen", config.enable_oxygen_hydrogen),
        ("Advanced Ratios", config.enable_advanced_ratios),
        ("Spectral Patterns", config.enable_spectral_patterns),
        ("Interference Correction", config.enable_interference_correction),
        ("Plasma Indicators", config.enable_plasma_indicators),
    ]
    
    for name, enabled in features:
        status = "✓ ENABLED" if enabled else "✗ DISABLED"
        print(f"{name:<25}: {status}")
    
    # Count enabled regions
    total_regions = len(config.all_regions)
    base_regions = len(config.context_regions) + 1  # +1 for nitrogen_region
    enhanced_regions = total_regions - base_regions
    
    print(f"\nSpectral Regions:")
    print(f"- Base regions: {base_regions}")
    print(f"- Enhanced regions: {enhanced_regions}")
    print(f"- Total regions: {total_regions}")

if __name__ == "__main__":
    configure_enhanced_features()
    show_current_configuration()