# Wavelength Standardization Integration Guide

## Overview

The `standardize_wavelength_grid` function has been implemented in `src/data_management/data_manager.py` to address wavelength calibration drift and resolution differences in LIBS spectral data. This guide explains how to integrate it into your pipeline.

## Function Features

- **Automatic Grid Generation**: Creates a standardized wavelength grid covering all configured spectral regions
- **Flexible Interpolation**: Supports linear, cubic, and nearest-neighbor interpolation
- **Validation**: Checks for sufficient wavelength overlap (≥80%) before interpolation  
- **Error Handling**: Gracefully handles interpolation failures and artifacts
- **Configurable**: Resolution and interpolation method controlled via config

## Configuration Options

New configuration parameters in `pipeline_config.py`:

```python
# Wavelength standardization configuration
enable_wavelength_standardization: bool = False  # Enable/disable standardization
wavelength_interpolation_method: Literal['linear', 'cubic', 'nearest'] = 'linear'
wavelength_resolution: float = 0.1  # nm resolution for standardized grid
```

## Integration Points

### 1. Training Pipeline Integration

**Location**: `main.py` in `load_and_clean_data()` function

**Before** (around line 106):
```python
wavelengths, intensities = data_manager.load_spectral_file(file_path)
```

**After**:
```python
wavelengths, intensities = data_manager.load_spectral_file(file_path)

# Apply wavelength standardization if enabled
if cfg.enable_wavelength_standardization:
    wavelengths, intensities = data_manager.standardize_wavelength_grid(
        wavelengths, intensities, 
        interpolation_method=cfg.wavelength_interpolation_method
    )
```

### 2. Validation Pipeline Integration  

**Location**: `main.py` in `process_validation_data()` function

**Before** (around line 186):
```python
wavelengths, intensities = data_manager.load_spectral_file(file_path)
```

**After**:
```python
wavelengths, intensities = data_manager.load_spectral_file(file_path)

# Apply wavelength standardization if enabled  
if cfg.enable_wavelength_standardization:
    wavelengths, intensities = data_manager.standardize_wavelength_grid(
        wavelengths, intensities,
        interpolation_method=cfg.wavelength_interpolation_method
    )
```

### 3. Prediction Pipeline Integration

**Location**: `src/models/predictor.py` in `make_prediction()` method

**Before** (around line 137):
```python
clean_intensities = self.data_cleanser.clean_spectra(str(input_file), intensities)
```

**After**:
```python
clean_intensities = self.data_cleanser.clean_spectra(str(input_file), intensities)

# Apply wavelength standardization if enabled
if self.config.enable_wavelength_standardization:
    wavelengths, clean_intensities = self.data_manager.standardize_wavelength_grid(
        wavelengths, clean_intensities,
        interpolation_method=self.config.wavelength_interpolation_method
    )
```

**Also in** `make_batch_predictions()` method (around line 191):
```python
clean_intensities = self.data_cleanser.clean_spectra(sample_id, averaged_intensities)

# Apply wavelength standardization if enabled
if self.config.enable_wavelength_standardization:
    wavelengths, clean_intensities = self.data_manager.standardize_wavelength_grid(
        wavelengths, clean_intensities,
        interpolation_method=self.config.wavelength_interpolation_method  
    )
```

## Usage Examples

### Enable Wavelength Standardization

1. **Edit config** (`src/config/pipeline_config.py`):
```python
enable_wavelength_standardization: bool = True
wavelength_interpolation_method: Literal['linear', 'cubic', 'nearest'] = 'linear'
wavelength_resolution: float = 0.1  # 0.1nm resolution
```

2. **Run training**:
```bash
python main.py train
```

3. **Run prediction** (will automatically use same standardization):
```bash
python main.py predict-single --input-file path/to/file.csv.txt --model-path path/to/model.pkl
```

### Custom Wavelength Grid

For advanced users, you can provide a custom target wavelength grid:

```python
# Example: Create custom grid for specific spectral range
custom_grid = np.arange(740.0, 750.0, 0.05)  # 740-750nm at 0.05nm resolution

# Apply standardization with custom grid
wavelengths, intensities = data_manager.standardize_wavelength_grid(
    wavelengths, intensities,
    target_wavelengths=custom_grid,
    interpolation_method='cubic'
)
```

## Benefits

1. **Consistent Feature Dimensions**: All spectra have identical wavelength points
2. **Calibration Drift Compensation**: Corrects for small wavelength shifts between instruments
3. **Resolution Standardization**: Ensures consistent spectral resolution across datasets
4. **Improved Model Robustness**: Reduces variation from instrumental differences

## Performance Considerations

- **Memory**: Standardized grids may have different sizes than original data
- **Computation**: Interpolation adds processing time (~10-50ms per spectrum)
- **Storage**: Standardized data size depends on target resolution

## Validation and Testing

After integration, validate the standardization:

1. **Check wavelength ranges** in logs for overlap warnings
2. **Compare feature counts** before/after standardization  
3. **Monitor interpolation artifacts** (negative values, NaN warnings)
4. **Test prediction consistency** with validation data

## Troubleshooting

- **"Insufficient wavelength overlap"**: Original and target ranges don't overlap enough
- **Negative intensity values**: Interpolation artifacts, consider different method
- **Performance issues**: Reduce wavelength resolution or use 'linear' interpolation
- **Memory errors**: Process data in smaller batches

## Rollback

To disable standardization:
```python
enable_wavelength_standardization: bool = False
```

Models trained with standardization require standardization during prediction.