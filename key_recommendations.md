1. Spectral Preprocessing ⭐⭐⭐

  Missing entirely! LIBS data is noisy - preprocessing could give +5-10% R² improvement.
  - Savitzky-Golay smoothing (reduces noise)
  - Standard Normal Variate (SNV) (normalizes instrument variations)
  - Easy to add to feature pipeline

  2. Uncertainty Quantification ⭐⭐⭐

  Currently: point predictions only
  Add: Prediction intervals via quantile regression
  - Critical for production: "Predicted 2.3% K ± 0.3%"
  - XGBoost/LightGBM support this natively
  - Helps identify when to re-measure

  3. Weighted Ensemble ⭐⭐

  You train many models but pick one winner
  Better: Combine top 3 models
  - Expected: +2-5% R² over best single model
  - Simple implementation after optimization

  4. SHAP Interpretation ⭐⭐

  Why: Validate model uses K emission lines (not noise)
  - Critical for scientific trust
  - Easy to add for tree models
  - Shows which wavelengths matter

  📊 What You're Already Doing Well

  ✅ K_only strategy (60 features) - perfect for 720 samples✅ 5-fold stratified CV - now optimized✅ Mislabel detection - unique advantage✅ Multiple model types with optimization✅ GPU
  acceleration

  🔬 LIBS-Specific Opportunities

  - Physics-informed features: Peak asymmetry, FWHM, Stark broadening
  - Range specialists: Different models for low/medium/high K (you have code for this!)
  - Peak shape descriptors: Area, kurtosis, asymmetry

  ⚡ Quick Win Experiments

  # 1. Test if spectral preprocessing helps
  Add SNV → retrain → compare R²

  # 2. Build 3-model ensemble
  weights = [0.4*xgb + 0.4*lgbm + 0.2*nn] → compare to best single

  # 3. Add SHAP to top model
  Verify K_766, K_770 lines have high importance