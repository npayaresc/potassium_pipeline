# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning pipeline for predicting potassium concentration from LIBS (Laser-Induced Breakdown Spectroscopy) spectral data. The pipeline processes spectral intensity measurements at specific wavelengths to predict potassium percentage in samples.

## Key Commands

### Development Setup
```bash
# Install dependencies using uv (the project's package manager)
uv sync

# Or install in a virtual environment
uv venv
source .venv/bin/activate  # Linux/Mac
uv pip install -e .
```

### Running the Pipeline

```bash
# Train standard models (CatBoost, LightGBM, XGBoost, etc.)
python main.py train

# Train with GPU acceleration (requires CUDA)
python main.py train --gpu

# Train while excluding suspicious samples identified by mislabel detection
python main.py train --exclude-suspects reports/mislabel_analysis/suspicious_samples_min_confidence_2.csv

# Combine GPU training with sample exclusion
python main.py train --gpu --exclude-suspects reports/mislabel_analysis/suspicious_samples_min_confidence_2.csv

# Run AutoGluon automated training
python main.py autogluon

# Run AutoGluon with GPU acceleration
python main.py autogluon --gpu

# Run AutoGluon while excluding suspicious samples
python main.py autogluon --exclude-suspects reports/mislabel_analysis/suspicious_samples_min_confidence_2.csv

# Run hyperparameter tuning with Optuna
python main.py tune

# Run hyperparameter tuning with GPU acceleration
python main.py tune --gpu

# Run tuning while excluding suspicious samples
python main.py tune --exclude-suspects reports/mislabel_analysis/suspicious_samples_min_confidence_2.csv

# Run dedicated XGBoost optimization
python main.py optimize-xgboost --strategy full_context --trials 300 --gpu

# Run XGBoost optimization while excluding suspicious samples
python main.py optimize-xgboost --strategy full_context --trials 300 --gpu --exclude-suspects reports/mislabel_analysis/suspicious_samples_min_confidence_2.csv

# Run optimization for multiple models
python main.py optimize-models --models xgboost lightgbm catboost --strategy full_context --trials 200 --gpu

# Run optimization while excluding suspicious samples
python main.py optimize-models --models xgboost lightgbm catboost --strategy full_context --trials 200 --gpu --exclude-suspects reports/mislabel_analysis/suspicious_samples_min_confidence_2.csv

# Run optimization for tree-based models
python main.py optimize-models --models random_forest extratrees --strategy simple_only --trials 150

# Run neural network optimization
python main.py optimize-models --models neural_network neural_network_light --strategy full_context --trials 100 --gpu

# Make predictions on a single file
python main.py predict-single --input-file path/to/file.csv.txt --model-path path/to/model.pkl

# Make batch predictions
python main.py predict-batch --input-dir path/to/directory --model-path path/to/model.pkl --output-file predictions.csv

# Make batch predictions with limited sample IDs (useful for testing)
python main.py predict-batch --input-dir path/to/directory --model-path path/to/model.pkl --output-file predictions.csv --max-samples 50

# Configuration Management
python main.py save-config --name "my_config" --description "Configuration description"
python main.py list-configs
python main.py create-training-config --name "quick_test" --models xgboost lightgbm --strategies simple_only --gpu

# Use saved configuration
python main.py train --config configs/my_config_20250131_143052.yaml --gpu

# Detect potentially mislabeled samples
python main.py detect-mislabels --focus-min 0.0 --focus-max 0.5 --min-confidence 2

# Detect mislabels with custom settings
python main.py detect-mislabels --focus-min 0.0 --focus-max 0.3 --clustering-methods "kmeans,dbscan" --outlier-methods "lof" --min-confidence 1

# Use parallel processing for faster analysis
python main.py detect-mislabels --feature-parallel --data-parallel
python main.py detect-mislabels --feature-parallel --feature-n-jobs 8 --data-parallel --data-n-jobs 4
```

### GPU Support

The pipeline supports GPU acceleration for compatible models when the `--gpu` flag is used:

- **XGBoost**: Uses `tree_method='hist'` with `device='cuda'` (modern syntax)
- **LightGBM**: Uses `device='gpu'` with platform and device ID 0
- **CatBoost**: Uses `task_type='GPU'` with device 0
- **AutoGluon**: Uses `num_gpus=1` for neural network models and GPU-enabled base models

Requirements for GPU support:
- NVIDIA GPU with CUDA support
- CUDA drivers installed
- GPU-enabled versions of the ML libraries (XGBoost, LightGBM, CatBoost, AutoGluon)

### Clean Generated Files
```bash
rm -rf bad_files/* bad_prediction_files/* logs/* reports/* catboost_info/
```

### SHAP Feature Importance Analysis

Analyze feature importance for any trained model using SHAP (SHapley Additive exPlanations):

```bash
# Analyze latest model by type
./run_shap_analysis.sh --latest ridge
./run_shap_analysis.sh --latest xgboost
./run_shap_analysis.sh --latest catboost

# Analyze specific model
./run_shap_analysis.sh models/simple_only_ridge_20251006_024858.pkl

# Show help
./run_shap_analysis.sh --help
```

**Key Features:**
- Works with ANY model type (Ridge, XGBoost, LightGBM, CatBoost, Random Forest, etc.)
- Automatically loads feature names from `.feature_names.json` files
- Respects feature selection (only analyzes features actually used by the model)
- Auto-detects model strategy and extracts corresponding training data
- Generates importance rankings and visualizations

**Output:**
- `models/<model_name>_shap_importance.csv` - Feature importance table
- `models/shap_analysis/<model_name>_shap_*.png` - Visualizations (summary, bar, custom plots)

**See:** `SHAP_ANALYSIS_GUIDE.md` for detailed documentation and best practices

### SHAP-Based Feature Selection

Train models using only the top N most important features identified by SHAP analysis:

```bash
# Complete workflow
# 1. Train initial model
python main.py train --models lightgbm --strategy full_context --gpu

# 2. Run SHAP analysis
./run_shap_analysis.sh --latest lightgbm

# 3. Train with top 30 SHAP features
python main.py train \
    --models xgboost lightgbm \
    --strategy full_context \
    --shap-features models/full_context_lightgbm_*_shap_importance.csv \
    --shap-top-n 30 \
    --gpu
```

**Key Features:**
- Reduces features from 495 → 30 (or your chosen N)
- Often improves performance and interpretability
- 3-5x faster training and inference
- Works with training and optimization

**Arguments:**
- `--shap-features <path>`: Path to SHAP importance CSV file (required)
- `--shap-top-n <N>`: Number of top features to select (default: 30)
- `--shap-min-importance <threshold>`: Minimum importance threshold (optional)

**See:** `SHAP_FEATURE_SELECTION_GUIDE.md` for complete documentation and examples

### Sample Exclusion

The pipeline supports excluding suspicious samples identified by mislabel detection from training and optimization:

```bash
# Use --exclude-suspects flag with any training or optimization command
python main.py train --exclude-suspects path/to/suspicious_samples.csv

# The CSV file should have a 'sample_id' column containing the IDs to exclude
# Typically use the output from mislabel detection:
python main.py train --exclude-suspects reports/mislabel_analysis/suspicious_samples_min_confidence_2.csv
```

**How it works:**
- Samples are excluded AFTER data cleaning but BEFORE training
- The exclusion applies to all models and strategies in the run
- Excluded samples are logged in the console output
- Works with all training, tuning, and optimization commands

### Mislabel Detection

The pipeline includes a sophisticated mislabel detection system to identify potentially mislabeled samples, especially in the lower concentration range where mislabeling is more common:

```bash
# Basic mislabel detection (focuses on 0.0-0.5% range)
python main.py detect-mislabels

# Focus on specific concentration range
python main.py detect-mislabels --focus-min 0.0 --focus-max 0.3

# Custom clustering and outlier detection methods
python main.py detect-mislabels --clustering-methods "kmeans,dbscan" --outlier-methods "lof,isolation_forest"

# High confidence suspects only (flagged by 2+ methods)
python main.py detect-mislabels --min-confidence 2

# Skip certain analysis types
python main.py detect-mislabels --no-features  # Skip engineered features
python main.py detect-mislabels --no-raw-spectral  # Skip raw spectral analysis
```

**Detection Methods:**
- **Clustering-based**: K-means, DBSCAN, Hierarchical clustering on both raw spectral and engineered features
- **Outlier detection**: Local Outlier Factor (LOF) and Isolation Forest
- **Concentration consistency**: Identifies samples that don't fit their cluster's concentration pattern
- **Parallel processing**: Supports `--feature-parallel` and `--data-parallel` for faster analysis

**Output:**
- Interactive visualizations showing cluster patterns and suspect locations
- Detailed suspect lists with confidence scores
- CSV files with sample IDs for exclusion from training
- Analysis reports with recommendations

**Complete Workflow Example:**
1. Run detection on your dataset:
   ```bash
   python main.py detect-mislabels --focus-min 0.0 --focus-max 0.5 --min-confidence 2
   ```

2. Review visualizations in `reports/mislabel_analysis/`

3. Manually inspect high-confidence suspects in the generated CSV files

4. Use exported sample ID lists to exclude suspects from training:
   ```bash
   # Train without suspicious samples
   python main.py train --gpu --exclude-suspects reports/mislabel_analysis/suspicious_samples_min_confidence_2.csv

   # Or optimize models without suspicious samples
   python main.py optimize-models --models xgboost lightgbm --strategy full_context --trials 200 --gpu \
     --exclude-suspects reports/mislabel_analysis/suspicious_samples_min_confidence_2.csv
   ```

5. Compare model performance with and without suspicious samples to validate the improvement

## Architecture Overview

### Core Pipeline Flow
1. **Raw Data Input** → Spectral files in `.csv.txt` format
2. **Data Averaging** → Groups and averages multiple measurements per sample
3. **Outlier Detection** → Uses SAM and MAD algorithms to identify bad data
4. **Feature Engineering** → Extracts peak features from spectral regions (213-215nm and 253-256nm for magnesium)
5. **Model Training** → Multiple algorithms including ensemble methods
6. **Prediction & Reporting** → Generates predictions with calibration plots and metrics

### Key Components

- **Configuration** (`src/config/pipeline_config.py`): Centralized Pydantic-based configuration defining spectral regions, model parameters, and processing options
- **Data Management** (`src/data_management/`): Handles file I/O, averaging, splitting, and reference data integration
- **Feature Engineering** (`src/features/`): Three strategies - K_only (potassium peaks), simple_only (basic features), full_context (all spectral regions)
- **Model Training** (`src/models/`): Supports standard ML models and AutoGluon ensemble learning with GPU support
- **Spectral Processing** (`src/spectral_extraction/`): Peak extraction, Lorentzian fitting, and baseline correction

### Feature Engineering Strategies
- **K_only**: Focus on potassium spectral regions (766-770nm and 404nm)
- **simple_only**: Basic spectral features without complex transformations
- **full_context**: All spectral regions including C, H, O, N, P, Mg, and molecular bands

### Model Types
- Standard models: Ridge, Lasso, Random Forest, XGBoost, LightGBM, CatBoost, SVR, ExtraTrees
- Neural Networks: Custom PyTorch models with spectral-specific architecture
- AutoGluon: Automated ensemble with neural networks (when GPU available)
- All models support hyperparameter tuning via Optuna

### Neural Network Features
- **Architecture**: Two variants - full (256→128→64→32→16→1) and light (64→32→16→1)
- **Custom Loss**: MagnesiumLoss with 2x weighting for extreme concentrations (< 0.25% or > 0.40%)
- **Regularization**: Dropout, batch normalization, L2 weight decay, early stopping
- **Sample Weighting**: Configurable via `use_sample_weights` parameter in model config
  - `False` (default): Use only custom loss weighting (recommended to avoid double-weighting)
  - `True`: Use both sample weights and custom loss weighting (for experimentation)
- **GPU Support**: Automatic CUDA detection and usage when `--gpu` flag is specified

### Data Directory Structure
```
data/
├── raw/data_5278_Phase3/      # Raw spectral files
├── averaged_files_per_sample/   # Averaged measurements
├── cleansed_files_per_sample/   # Outlier-removed data
└── reference files (Excel)      # Ground truth magnesium values
```

### Output Artifacts
- `models/`: Saved trained models (.pkl files)
- `models/autogluon/`: AutoGluon ensemble models
- `reports/`: Training summaries, predictions, and calibration plots
- `logs/`: Execution logs with timestamps
- `bad_files/`: Files rejected during cleansing
- `bad_prediction_files/`: Files that failed during prediction
- please always use a timeout of at least 5 minutes when executing python commands