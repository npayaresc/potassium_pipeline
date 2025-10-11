# Potassium Pipeline Migration Notes

This project was created from the Phosphorous Pipeline and adapted for Potassium detection.

## Changes Made

### 1. Automated Updates
- All references to "phosphorous"/"phosphorus" changed to "potassium"
- Element symbol changed from "P" to "K"
- Project name and metadata updated
- Spectral regions updated for K detection

### 2. Spectral Regions
Updated from P spectral lines to K spectral lines:
- **Original P regions**: 213-215nm, 253-256nm
- **New K regions**: 766-770nm (K I doublet), 404nm (K I)

Common K LIBS lines for reference:
- K I: 766.49 nm, 769.90 nm (doublet), 404.41 nm, 404.72 nm

## Manual Tasks Required

### 1. Data Preparation
- [ ] Update reference Excel files with K concentration values
- [ ] Ensure spectral data covers K wavelength regions
- [ ] Verify data file naming conventions

### 2. Configuration Updates
- [ ] Review `src/config/pipeline_config.py` for K-specific settings
- [ ] Adjust concentration ranges (typical K ranges may differ from P)
- [ ] Update outlier detection thresholds if needed

### 3. Model Tuning
- [ ] Adjust hyperparameter ranges for K concentration prediction
- [ ] Update sample weighting if K distribution differs from P
- [ ] Consider different model architectures if needed

### 4. Validation
- [ ] Verify spectral peak extraction works for K lines
- [ ] Test with sample K data
- [ ] Validate model performance metrics

### 5. Cloud Deployment
- [ ] Update GCP project settings if using different project
- [ ] Modify bucket names in cloud configurations
- [ ] Update Docker image names

## Quick Start

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Prepare your K data**:
   - Place spectral files in `data/raw/`
   - Add reference Excel with K concentrations

3. **Update spectral regions** (if needed):
   Edit `src/config/pipeline_config.py` to match your LIBS setup

4. **Train models**:
   ```bash
   python main.py train --gpu
   python main.py autogluon --gpu
   ```

5. **Deploy**:
   ```bash
   ./deploy/local-deploy.sh build
   ./deploy/gcp_deploy.sh all
   ```

## Important Files to Review

1. `src/config/pipeline_config.py` - Main configuration
2. `src/features/feature_engineering.py` - Feature extraction
3. `src/spectral_extraction/peak_extraction.py` - Peak detection
4. `config/cloud_config.yml` - Cloud deployment settings

## Support

For questions about the original phosphorous pipeline, refer to the source project.
For K-specific adaptations, document changes in this file.
