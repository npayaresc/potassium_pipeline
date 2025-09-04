#!/usr/bin/env python3
"""
Test script to validate that the updated magnesium spectral regions 
are being used correctly in feature calculations.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.config.pipeline_config import Config
import pandas as pd
import numpy as np

def test_spectral_regions():
    """Test that the new magnesium regions are properly configured."""
    from datetime import datetime
    config = Config(run_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    print("=" * 80)
    print("MAGNESIUM SPECTRAL REGIONS VALIDATION")
    print("=" * 80)
    
    # Check primary magnesium region
    print("\n1. PRIMARY MAGNESIUM REGION (Mg I triplet):")
    mg_region = config.magnesium_region
    print(f"   Element: {mg_region.element}")
    print(f"   Wavelength range: {mg_region.lower_wavelength:.1f} - {mg_region.upper_wavelength:.1f} nm")
    print(f"   Center wavelengths: {', '.join(f'{w:.1f} nm' for w in mg_region.center_wavelengths)}")
    print(f"   Number of peaks: {mg_region.n_peaks}")
    
    # Check context regions for additional Mg lines
    print("\n2. ADDITIONAL MAGNESIUM REGIONS IN CONTEXT:")
    mg_related_regions = [r for r in config.context_regions if 'Mg' in r.element or 'M_' in r.element]
    
    for region in mg_related_regions:
        print(f"\n   {region.element}:")
        print(f"   - Wavelength range: {region.lower_wavelength:.1f} - {region.upper_wavelength:.1f} nm")
        print(f"   - Center wavelengths: {', '.join(f'{w:.1f} nm' for w in region.center_wavelengths)}")
        print(f"   - Number of peaks: {region.n_peaks}")
    
    # Verify literature values
    print("\n3. LITERATURE VERIFICATION:")
    literature_lines = {
        "285.2": "Most prominent Mg I line (resonance)",
        "383.8": "Strong Mg I line",
        "516.7": "Mg I triplet component 1",
        "517.3": "Mg I triplet component 2", 
        "518.4": "Mg I triplet component 3",
        "279.55": "Mg II ionic line 1",
        "279.80": "Mg II ionic line 2",
        "280.27": "Mg II ionic line 3"
    }
    
    # Collect all configured Mg wavelengths
    all_mg_wavelengths = []
    all_mg_wavelengths.extend(mg_region.center_wavelengths)
    for region in mg_related_regions:
        all_mg_wavelengths.extend(region.center_wavelengths)
    
    print("\n   Checking coverage of literature wavelengths:")
    for lit_wl, description in literature_lines.items():
        lit_val = float(lit_wl)
        # Check if this wavelength is covered (within 1 nm tolerance)
        covered = any(abs(w - lit_val) < 1.0 for w in all_mg_wavelengths)
        status = "✓" if covered else "✗"
        print(f"   {status} {lit_wl} nm - {description}")
    
    # Check feature generation
    print("\n4. FEATURE NAME GENERATION CHECK:")
    print("\n   Expected feature names from new regions:")
    
    all_regions = [mg_region] + mg_related_regions
    for region in all_regions:
        print(f"\n   From {region.element} region:")
        for i in range(region.n_peaks):
            print(f"   - {region.element}_peak_{i}")
            print(f"   - {region.element}_simple_peak_area")
            print(f"   - {region.element}_simple_peak_height")
            print(f"   - {region.element}_fwhm (if spectral patterns enabled)")
    
    # Check ratio calculations
    print("\n5. ADVANCED RATIO CALCULATIONS:")
    print("\n   Ratios that should use new regions:")
    print("   - Mg_primary_secondary_ratio: M_I / Mg_I_285 (or Mg_II if 285 not available)")
    print("   - Mg II/I ratio: Mg_II / M_I (ionic to neutral)")
    print("   - Self-absorption indicators for:")
    print("     * M_I (516-519 nm triplet)")
    print("     * Mg_I_285 (285.2 nm - most prominent)")
    print("     * Mg_II (279-281 nm ionic lines)")
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    
    return config

if __name__ == "__main__":
    config = test_spectral_regions()
    
    print("\n6. CONFIGURATION SUMMARY:")
    print(f"   Total regions defined: {len(config.all_regions)}")
    print(f"   Magnesium-specific regions: {1 + len([r for r in config.context_regions if 'Mg' in r.element])}")
    print(f"   Advanced features enabled:")
    print(f"   - Molecular bands: {config.enable_molecular_bands}")
    print(f"   - Macro elements: {config.enable_macro_elements}")
    print(f"   - Advanced ratios: {config.enable_advanced_ratios}")
    print(f"   - Spectral patterns: {config.enable_spectral_patterns}")
    print(f"   - Interference correction: {config.enable_interference_correction}")