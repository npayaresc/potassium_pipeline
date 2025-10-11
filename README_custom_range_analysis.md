# Custom Range Analysis Tool

## Overview

`analyze_custom_range.py` is a flexible Python script that analyzes magnesium concentration predictions within any user-defined range. It calculates comprehensive performance metrics and provides detailed insights into model accuracy across different concentration zones.

## Features

- **Flexible Range Selection**: Analyze any concentration range using `--low-limit` and `--high-limit` parameters
- **Comprehensive Metrics**: R², RMSE, MAE, RRMSE, MAPE, and accuracy percentages (within 20.5%, 15%, 10%)
- **Sub-range Analysis**: Automatically breaks down large ranges into meaningful sub-ranges
- **Sample Display**: Shows representative sample predictions with error analysis
- **Comparison Analysis**: Compares range performance against overall dataset
- **Smart Formatting**: Adaptive display based on range size and sample count

## Usage

### Basic Usage
```bash
python analyze_custom_range.py --low-limit 0.2 --high-limit 1.5
```

### Advanced Usage
```bash
# Use custom predictions file
python analyze_custom_range.py --low-limit 0.15 --high-limit 2.5 --predictions-file path/to/predictions.csv

# Skip sample display for quick analysis
python analyze_custom_range.py --low-limit 0.3 --high-limit 1.0 --no-samples

# Control sample display count
python analyze_custom_range.py --low-limit 0.2 --high-limit 2.0 --max-samples 30
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--low-limit` | Yes | Lower limit of concentration range (% Mg) |
| `--high-limit` | Yes | Upper limit of concentration range (% Mg) |
| `--predictions-file` | No | Path to predictions CSV file (defaults to AutoGluon predictions) |
| `--no-samples` | No | Skip displaying sample predictions |
| `--max-samples` | No | Maximum sample predictions to display (default: 20) |

## Output Sections

### 1. Range Characteristics
- Range width and sample count
- Coverage percentage of total dataset
- Actual concentration range found in data

### 2. Performance Metrics
- **R²**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error  
- **RRMSE**: Relative Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error

### 3. Accuracy Metrics
- **Within 20.5%**: Percentage of predictions within 20.5% relative error
- **Within 15%**: Percentage of predictions within 15% relative error
- **Within 10%**: Percentage of predictions within 10% relative error

### 4. Comparison Analysis
- Side-by-side comparison with overall dataset performance
- Status indicators (Better/Worse/Same) for each metric

### 5. Sub-range Breakdown
- Automatic division into meaningful sub-ranges
- Performance metrics for each sub-range
- Identifies optimal concentration zones

### 6. Sample Predictions
- Representative predictions showing actual vs predicted values
- Error percentages and accuracy indicators
- Sorted by concentration for easy analysis

## Input File Format

The script expects a CSV file with the following columns:
- `sampleId`: Unique sample identifier
- `ElementValue`: True magnesium concentration (% Mg)
- `PredictedValue`: Predicted magnesium concentration (% Mg)

## Example Output

```
================================================================================
ANALYSIS FOR CUSTOM RANGE [0.3, 1.6] % Mg
================================================================================

RANGE CHARACTERISTICS:
==================================================
Range: [0.3, 1.6] % Mg
Range Width: 1.300 % Mg
Number of Samples: 182 (72.2% of total)
Mean True Value: 0.7231 % Mg

PERFORMANCE METRICS:
==================================================
R²: 0.632330
MAE: 0.155158 % Mg
RMSE: 0.223776 % Mg

ACCURACY METRICS:
==================================================
Within 20.5%: 59.89%
Within 15%: 53.85%
Within 10%: 41.76%

SUMMARY:
==================================================
📊 Range Coverage: 182/252 samples (72.2%)
🎯 Accuracy: 59.89% within 20.5% error
📈 vs Overall: +10.68 percentage points
✅ This range shows SIGNIFICANTLY better accuracy than the overall dataset
```

## Use Cases

1. **Optimal Range Identification**: Find concentration ranges where your model performs best
2. **Model Validation**: Assess prediction quality across different concentration zones
3. **Application Planning**: Determine reliable operating ranges for your magnesium prediction system
4. **Performance Optimization**: Identify where model improvements are most needed
5. **Quality Control**: Set concentration-based quality thresholds for predictions

## Tips

- **Wide ranges** (> 1.5% Mg) are good for general coverage analysis
- **Narrow ranges** (< 0.8% Mg) are better for precision-focused applications
- Use `--no-samples` for quick performance comparisons
- Check sub-range breakdowns to identify optimal concentration zones
- Compare multiple ranges to find the best balance of coverage vs accuracy