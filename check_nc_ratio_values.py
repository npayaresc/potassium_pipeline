"""Check what typical NC_ratio values look like"""
import pandas as pd
import numpy as np
from pathlib import Path

# Check what typical NC_ratio values look like in the training data
cleansed_dir = Path('data/cleansed_files_per_sample')
cleansed_files = list(cleansed_dir.glob('*.csv.txt'))

print(f'Found {len(cleansed_files)} cleansed files')

if cleansed_files:
    # Load a few samples to check NC_ratio values
    sample_file = cleansed_files[0]
    print(f'Testing with: {sample_file.name}')
    
    df = pd.read_csv(sample_file)
    wavelengths = df['Wavelength'].values
    intensity_cols = [col for col in df.columns if col != 'Wavelength']
    intensities = df[intensity_cols].values
    
    # Simulate nitrogen region extraction
    n_mask = (wavelengths >= 741) & (wavelengths <= 743)
    c_mask = (wavelengths >= 193.0) & (wavelengths <= 250.0)  # Approximate carbon region
    
    n_intensities = intensities[n_mask].mean()
    c_intensities = intensities[c_mask].mean()
    
    print(f'N region avg intensity: {n_intensities:.2f}')
    print(f'C region avg intensity: {c_intensities:.2f}')
    
    # Calculate rough NC_ratio
    c_safe = max(abs(c_intensities), 1e-6)
    nc_ratio = n_intensities / c_safe
    
    print(f'NC_ratio: {nc_ratio:.6f}')
    print(f'NC_ratio_squared: {nc_ratio**2:.2e}')
    print(f'NC_ratio_cubic: {nc_ratio**3:.2e}')
    
    print(f'\nIf NC_ratio were 1000 (extreme case):')
    print(f'  NC_ratio_squared: {1000**2:.2e}')
    print(f'  NC_ratio_cubic: {1000**3:.2e}')
    
    # Check multiple samples
    print(f'\nChecking first 10 samples for NC_ratio range:')
    nc_ratios = []
    
    for i, sample_file in enumerate(cleansed_files[:10]):
        try:
            df = pd.read_csv(sample_file)
            wavelengths = df['Wavelength'].values
            intensity_cols = [col for col in df.columns if col != 'Wavelength']
            intensities = df[intensity_cols].values
            
            n_mask = (wavelengths >= 741) & (wavelengths <= 743)
            c_mask = (wavelengths >= 193.0) & (wavelengths <= 250.0)
            
            n_avg = intensities[n_mask].mean()
            c_avg = intensities[c_mask].mean()
            c_safe = max(abs(c_avg), 1e-6)
            nc_ratio = n_avg / c_safe
            
            nc_ratios.append(nc_ratio)
            if abs(nc_ratio) > 100:  # Flag extreme values
                print(f'  {sample_file.name}: NC_ratio = {nc_ratio:.2f} (EXTREME)')
            
        except Exception as e:
            print(f'  Error with {sample_file.name}: {e}')
    
    if nc_ratios:
        nc_array = np.array(nc_ratios)
        print(f'\nNC_ratio statistics across {len(nc_ratios)} samples:')
        print(f'  Min: {nc_array.min():.2f}')
        print(f'  Max: {nc_array.max():.2f}')
        print(f'  Mean: {nc_array.mean():.2f}')
        print(f'  Std: {nc_array.std():.2f}')
        print(f'  Values > 100: {(np.abs(nc_array) > 100).sum()}')
        print(f'  Values > 1000: {(np.abs(nc_array) > 1000).sum()}')
        
        # Show what the extreme values would create when squared/cubed
        max_ratio = np.abs(nc_array).max()
        print(f'\nWith max ratio {max_ratio:.2f}:')
        print(f'  Squared: {max_ratio**2:.2e}')
        print(f'  Cubed: {max_ratio**3:.2e}')

else:
    print('No cleansed files found!')