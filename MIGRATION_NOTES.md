# Magnesium Pipeline Migration Notes

This project was created from the Phosphorous Pipeline and adapted for Magnesium detection.

## Changes Made

### 1. Automated Updates
- All references to "phosphorous"/"phosphorus" changed to "magnesium"
- Element symbol changed from "P" to "Mg"
- Project name and metadata updated
- Spectral regions updated for Mg detection

### 2. Spectral Regions
Updated from P spectral lines to Mg spectral lines:
- **Original P regions**: 213-215nm, 253-256nm  
- **New Mg regions**: 279-281nm (Mg II), 382-385nm (Mg I triplet)

Common Mg LIBS lines for reference:
- Mg I: 285.21 nm, 382.94 nm, 383.23 nm, 383.83 nm, 518.36 nm
- Mg II: 279.55 nm, 280.27 nm, 448.11 nm

## Manual Tasks Required

### 1. Data Preparation
- [ ] Update reference Excel files with Mg concentration values
- [ ] Ensure spectral data covers Mg wavelength regions
- [ ] Verify data file naming conventions

### 2. Configuration Updates
- [ ] Review `src/config/pipeline_config.py` for Mg-specific settings
- [ ] Adjust concentration ranges (typical Mg ranges may differ from P)
- [ ] Update outlier detection thresholds if needed

### 3. Model Tuning
- [ ] Adjust hyperparameter ranges for Mg concentration prediction
- [ ] Update sample weighting if Mg distribution differs from P
- [ ] Consider different model architectures if needed

### 4. Validation
- [ ] Verify spectral peak extraction works for Mg lines
- [ ] Test with sample Mg data
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

2. **Prepare your Mg data**:
   - Place spectral files in `data/raw/`
   - Add reference Excel with Mg concentrations

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
For Mg-specific adaptations, document changes in this file.
