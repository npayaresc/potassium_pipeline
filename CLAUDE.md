# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning pipeline for predicting magnesium concentration from LIBS (Laser-Induced Breakdown Spectroscopy) spectral data. The pipeline processes spectral intensity measurements at specific wavelengths to predict magnesium percentage in samples.

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

# Run AutoGluon automated training
python main.py autogluon

# Run AutoGluon with GPU acceleration
python main.py autogluon --gpu

# Run hyperparameter tuning with Optuna
python main.py tune

# Run hyperparameter tuning with GPU acceleration
python main.py tune --gpu

# Run dedicated XGBoost optimization
python main.py optimize-xgboost --strategy full_context --trials 300 --gpu

# Run optimization for multiple models
python main.py optimize-models --models xgboost lightgbm catboost --strategy full_context --trials 200 --gpu

# Run optimization for tree-based models
python main.py optimize-models --models random_forest extratrees --strategy simple_only --trials 150

# Run neural network optimization
python main.py optimize-models --models neural_network neural_network_light --strategy full_context --trials 100 --gpu

# Make predictions on a single file
python main.py predict-single --input-file path/to/file.csv.txt --model-path path/to/model.pkl

# Make batch predictions
python main.py predict-batch --input-dir path/to/directory --model-path path/to/model.pkl --output-file predictions.csv

# Configuration Management
python main.py save-config --name "my_config" --description "Configuration description"
python main.py list-configs
python main.py create-training-config --name "quick_test" --models xgboost lightgbm --strategies simple_only --gpu

# Use saved configuration
python main.py train --config configs/my_config_20250131_143052.yaml --gpu
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
- **Feature Engineering** (`src/features/`): Three strategies - Mg_only (magnesium peaks), simple_only (basic features), full_context (all spectral regions)
- **Model Training** (`src/models/`): Supports standard ML models and AutoGluon ensemble learning with GPU support
- **Spectral Processing** (`src/spectral_extraction/`): Peak extraction, Lorentzian fitting, and baseline correction

### Feature Engineering Strategies
- **Mg_only**: Focus on magnesium spectral regions (213-215nm and 253-256nm)
- **simple_only**: Basic spectral features without complex transformations
- **full_context**: All spectral regions including C, H, O, N, P, and molecular bands

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