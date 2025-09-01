# Comprehensive Experiment Plan for Magnesium Pipeline
## Target Metrics: R² ≥ 0.8, MAPE < 10%, MAE < 0.04

## Phase 1: Baseline & Feature Engineering (Days 1-3)

### 1.1 Feature Strategy Experiments
Test all three feature engineering strategies with standard models:

```bash
# Experiment 1A: Mg_only features (focused on magnesium peaks)
python main.py train --config configs/exp1a_mg_only.yaml --gpu

# Experiment 1B: Simple_only features (basic spectral features)  
python main.py train --config configs/exp1b_simple_only.yaml --gpu

# Experiment 1C: Full_context features (all spectral regions)
python main.py train --config configs/exp1c_full_context.yaml --gpu
```

### 1.2 Enhanced Feature Combinations
Test combinations of spectral regions for optimal feature set:

```bash
# Experiment 2A: Mg + Macro elements (S, Ca, K interactions)
python main.py train --config configs/exp2a_mg_macro.yaml --gpu

# Experiment 2B: Mg + Micro elements (Fe, Mn, B, Zn competition)
python main.py train --config configs/exp2b_mg_micro.yaml --gpu

# Experiment 2C: Mg + Molecular bands (CN, NH, NO organic indicators)
python main.py train --config configs/exp2c_mg_molecular.yaml --gpu

# Experiment 2D: All features enabled
python main.py train --config configs/exp2d_all_features.yaml --gpu
```

### 1.3 Raw Spectral Data Mode
Test raw intensity features vs engineered features:

```bash
# Experiment 3A: Raw spectral intensities
python main.py train --config configs/exp3a_raw_spectral.yaml --gpu

# Experiment 3B: Raw + PCA dimensionality reduction
python main.py train --config configs/exp3b_raw_pca.yaml --gpu

# Experiment 3C: Raw + PLS dimensionality reduction  
python main.py train --config configs/exp3c_raw_pls.yaml --gpu
```

## Phase 2: Advanced Dimensionality Reduction (Days 3-5)

### 2.1 Linear Methods
```bash
# Experiment 4A: PCA with 95% variance
python main.py train --config configs/exp4a_pca_95.yaml --gpu

# Experiment 4B: PCA with optimal components (10-50)
python main.py train --config configs/exp4b_pca_optimal.yaml --gpu

# Experiment 4C: PLS with target-aware reduction (10-30 components)
python main.py train --config configs/exp4c_pls.yaml --gpu
```

### 2.2 Neural Dimensionality Reduction
```bash
# Experiment 5A: Standard Autoencoder
python main.py train --config configs/exp5a_autoencoder.yaml --gpu

# Experiment 5B: Variational Autoencoder (VAE)
python main.py train --config configs/exp5b_vae.yaml --gpu

# Experiment 5C: Denoising Autoencoder
python main.py train --config configs/exp5c_denoising_ae.yaml --gpu

# Experiment 5D: Sparse Autoencoder
python main.py train --config configs/exp5d_sparse_ae.yaml --gpu
```

### 2.3 Feature Clustering
```bash
# Experiment 6: Feature clustering with automatic component selection
python main.py train --config configs/exp6_feature_clustering.yaml --gpu
```

## Phase 3: Hyperparameter Optimization (Days 5-8)

### 3.1 Tree-based Models
```bash
# Experiment 7A: XGBoost optimization (500 trials)
python main.py optimize-xgboost --strategy full_context --trials 500 --gpu

# Experiment 7B: LightGBM optimization
python main.py optimize-models --models lightgbm --strategy full_context --trials 400 --gpu

# Experiment 7C: CatBoost optimization
python main.py optimize-models --models catboost --strategy full_context --trials 400 --gpu

# Experiment 7D: Random Forest & ExtraTrees
python main.py optimize-models --models random_forest extratrees --strategy simple_only --trials 300
```

### 3.2 Different Objective Functions
Test various objective functions for optimization:

```bash
# Experiment 8A: Distribution-based objective (handles imbalanced ranges)
python main.py tune --config configs/exp8a_dist_objective.yaml --gpu

# Experiment 8B: MAPE-focused objective (optimize for percentage error)
python main.py tune --config configs/exp8b_mape_objective.yaml --gpu

# Experiment 8C: Robust V2 objective (stability + generalization)
python main.py tune --config configs/exp8c_robust_objective.yaml --gpu

# Experiment 8D: Quantile-weighted objective
python main.py tune --config configs/exp8d_quantile_objective.yaml --gpu
```

### 3.3 Sample Weighting Strategies
```bash
# Experiment 9A: Legacy weighting method
python main.py train --config configs/exp9a_legacy_weights.yaml --gpu

# Experiment 9B: Improved percentile-based weighting
python main.py train --config configs/exp9b_improved_weights.yaml --gpu

# Experiment 9C: Distribution-based weighting
python main.py train --config configs/exp9c_dist_weights.yaml --gpu

# Experiment 9D: Hybrid weighting
python main.py train --config configs/exp9d_hybrid_weights.yaml --gpu
```

## Phase 4: Neural Network Architectures (Days 8-10)

### 4.1 Architecture Variations
```bash
# Experiment 10A: Full architecture (256→128→64→32→16→1)
python main.py optimize-models --models neural_network --strategy full_context --trials 200 --gpu

# Experiment 10B: Light architecture (64→32→16→1)
python main.py optimize-models --models neural_network_light --strategy simple_only --trials 200 --gpu

# Experiment 10C: AutoGluon-optimized architecture
python main.py train --config configs/exp10c_nn_autogluon.yaml --gpu
```

### 4.2 Neural Network Hyperparameters
```bash
# Experiment 11A: High dropout (0.5) + heavy regularization
python main.py train --config configs/exp11a_nn_high_reg.yaml --gpu

# Experiment 11B: Low dropout (0.2) + light regularization
python main.py train --config configs/exp11b_nn_low_reg.yaml --gpu

# Experiment 11C: Custom loss with extreme concentration weighting
python main.py train --config configs/exp11c_nn_custom_loss.yaml --gpu
```

## Phase 5: AutoGluon Ensemble Learning (Days 10-14)

### 5.1 Stacking Configurations
```bash
# Experiment 12A: No stacking (bag only)
python main.py autogluon --config configs/exp12a_ag_no_stack.yaml --gpu

# Experiment 12B: 1-level stacking
python main.py autogluon --config configs/exp12b_ag_stack1.yaml --gpu

# Experiment 12C: 2-level stacking (recommended)
python main.py autogluon --config configs/exp12c_ag_stack2.yaml --gpu

# Experiment 12D: 3-level stacking (maximum)
python main.py autogluon --config configs/exp12d_ag_stack3.yaml --gpu
```

### 5.2 AutoGluon Presets
```bash
# Experiment 13A: Good quality preset (faster)
python main.py autogluon --config configs/exp13a_ag_good.yaml --gpu

# Experiment 13B: Best quality preset (slower, better performance)
python main.py autogluon --config configs/exp13b_ag_best.yaml --gpu

# Experiment 13C: High quality preset
python main.py autogluon --config configs/exp13c_ag_high.yaml --gpu
```

### 5.3 AutoGluon Model Selection
```bash
# Experiment 14A: All models enabled
python main.py autogluon --config configs/exp14a_ag_all_models.yaml --gpu

# Experiment 14B: Tree-based models only (RF, XT, GBM, XGB, CAT)
python main.py autogluon --config configs/exp14b_ag_trees_only.yaml --gpu

# Experiment 14C: Neural + tree models (exclude KNN, FASTAI)
python main.py autogluon --config configs/exp14c_ag_nn_trees.yaml --gpu
```

## Phase 6: Advanced Techniques (Days 14-16)

### 6.1 Data Augmentation & Preprocessing
```bash
# Experiment 15A: Wavelength standardization
python main.py train --config configs/exp15a_wavelength_std.yaml --gpu

# Experiment 15B: Aggressive outlier removal (SAM 0.9)
python main.py train --config configs/exp15b_strict_outliers.yaml --gpu

# Experiment 15C: Lenient outlier removal (SAM 0.7)
python main.py train --config configs/exp15c_lenient_outliers.yaml --gpu
```

### 6.2 Target Range Focusing
```bash
# Experiment 16A: Focus on 0.15-0.45% range
python main.py train --config configs/exp16a_target_focus.yaml --gpu

# Experiment 16B: Full range training
python main.py train --config configs/exp16b_full_range.yaml --gpu
```

### 6.3 Cross-validation Strategies
```bash
# Experiment 17A: 3-fold CV (faster)
python main.py train --config configs/exp17a_cv3.yaml --gpu

# Experiment 17B: 5-fold CV (standard)
python main.py train --config configs/exp17b_cv5.yaml --gpu

# Experiment 17C: 10-fold CV (more robust)
python main.py train --config configs/exp17c_cv10.yaml --gpu
```

## Phase 7: Best Model Combinations (Days 16-18)

### 7.1 Top Performer Ensembles
Based on preliminary results, combine best approaches:

```bash
# Experiment 18A: Best feature set + XGBoost optimized
python main.py train --config configs/exp18a_best_xgb.yaml --gpu

# Experiment 18B: Best feature set + AutoGluon with 2-level stacking
python main.py autogluon --config configs/exp18b_best_autogluon.yaml --gpu

# Experiment 18C: PLS reduction + Neural Network
python main.py train --config configs/exp18c_pls_nn.yaml --gpu
```

### 7.2 Final Optimization Runs
```bash
# Experiment 19A: Extended AutoGluon training (6 hours)
python main.py autogluon --config configs/exp19a_ag_extended.yaml --gpu

# Experiment 19B: Massive hyperparameter search (1000 trials)
python main.py tune --config configs/exp19b_massive_tune.yaml --gpu
```

## Parallel Execution Strategy

To maximize cloud resources, run experiments in parallel batches:

### Batch 1 (Feature Engineering - can run all simultaneously)
```bash
# Terminal 1
python main.py train --config configs/exp1a_mg_only.yaml --gpu &
# Terminal 2  
python main.py train --config configs/exp1b_simple_only.yaml --gpu &
# Terminal 3
python main.py train --config configs/exp1c_full_context.yaml --gpu &
```

### Batch 2 (Dimension Reduction - run after identifying best features)
```bash
# Use best feature strategy from Batch 1
python main.py train --config configs/exp4c_pls.yaml --gpu &
python main.py train --config configs/exp5b_vae.yaml --gpu &
```

### Batch 3 (Model Optimization - intensive, run fewer in parallel)
```bash
# Terminal 1
python main.py optimize-xgboost --strategy full_context --trials 500 --gpu &
# Terminal 2
python main.py autogluon --config configs/exp12c_ag_stack2.yaml --gpu &
```

## Monitoring & Analysis

### Track Progress
```bash
# Monitor training logs
tail -f logs/pipeline_*.log

# Check model performance
ls -la reports/training_summary_*.csv

# Compare results
python analyze_results.py --compare-all
```

### Key Metrics to Track
1. **R² Score**: Target ≥ 0.8
2. **MAPE**: Target < 10%
3. **MAE**: Target < 0.04
4. **Cross-validation stability**: CV std < 0.05
5. **Train-test gap**: < 0.1 R² difference

## Expected Timeline

- **Days 1-3**: Feature engineering experiments (20 runs)
- **Days 3-5**: Dimensionality reduction (10 runs)
- **Days 5-8**: Hyperparameter optimization (15 runs)
- **Days 8-10**: Neural network experiments (10 runs)
- **Days 10-14**: AutoGluon ensemble learning (12 runs)
- **Days 14-16**: Advanced techniques (10 runs)
- **Days 16-18**: Best combinations & final runs (6 runs)

**Total**: ~83 experiments over 18 days

## Success Criteria

An experiment is considered successful if it achieves:
- R² ≥ 0.8 on test set
- MAPE < 10% across all concentration ranges
- MAE < 0.04
- Stable cross-validation (std < 0.05)
- Good generalization (train-test gap < 0.1)

## Next Steps After Experiments

1. **Analysis**: Compare all results, identify top 5 models
2. **Ensemble**: Create weighted ensemble of best models
3. **Validation**: Test on held-out validation set
4. **Production**: Deploy best model(s) with confidence intervals
5. **Documentation**: Create detailed report with insights

## Quick Start Commands

For immediate testing with highest potential:

```bash
# 1. Best AutoGluon configuration
python main.py autogluon --gpu

# 2. Optimized XGBoost
python main.py optimize-xgboost --strategy full_context --trials 500 --gpu

# 3. Neural network with custom loss
python main.py optimize-models --models neural_network --strategy simple_only --trials 200 --gpu
```