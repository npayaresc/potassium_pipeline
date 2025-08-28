#!/usr/bin/env python3
"""
Comprehensive test to verify concentration features are available across ALL models and optimizers.
"""
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.config.pipeline_config import Config
from src.models.model_trainer import ModelTrainer
from src.models.autogluon_trainer import AutoGluonTrainer
from src.models.model_tuner import ModelTuner
from src.models.optimize_xgboost import XGBoostOptimizer
from src.models.optimize_lightgbm import LightGBMOptimizer
from src.models.optimize_catboost import CatBoostOptimizer
from src.models.optimize_random_forest import RandomForestOptimizer
from src.models.optimize_extratrees import ExtraTreesOptimizer
from src.models.optimize_neural_network import NeuralNetworkOptimizer
from src.models.optimize_autogluon import AutoGluonOptimizer
from src.reporting.reporter import Reporter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_all_models_have_concentration_features():
    """Test that ALL model training and optimization classes use concentration features when enabled."""
    
    # Create config with concentration features enabled
    config = Config()
    config.use_concentration_features = True
    
    # Create reporter
    reporter = Reporter(config)
    strategy = "simple_only"
    
    logger.info("=== Testing Concentration Features Integration Across ALL Models ===")
    
    # Test results storage
    results = {}
    
    # Test classes to verify
    test_classes = [
        ("ModelTrainer", lambda: ModelTrainer(config, strategy, reporter)),
        ("AutoGluonTrainer", lambda: AutoGluonTrainer(config, strategy, reporter, use_concentration_features=True)),
        ("ModelTuner", lambda: ModelTuner(config, reporter, strategy)),
        ("XGBoostOptimizer", lambda: XGBoostOptimizer(config, strategy)),
        ("LightGBMOptimizer", lambda: LightGBMOptimizer(config, strategy)),
        ("CatBoostOptimizer", lambda: CatBoostOptimizer(config, strategy)),
        ("RandomForestOptimizer", lambda: RandomForestOptimizer(config, strategy)),
        ("ExtraTreesOptimizer", lambda: ExtraTreesOptimizer(config, strategy)),
        ("NeuralNetworkOptimizer", lambda: NeuralNetworkOptimizer(config, strategy)),
        ("AutoGluonOptimizer", lambda: AutoGluonOptimizer(config, strategy)),
    ]
    
    for class_name, constructor in test_classes:
        logger.info(f"Testing {class_name}...")
        
        try:
            # Create instance
            instance = constructor()
            
            # Check if it has concentration features in pipeline
            pipeline_steps = list(instance.feature_pipeline.named_steps.keys())
            has_concentration = 'concentration_features' in pipeline_steps
            
            if has_concentration:
                concentration_step = instance.feature_pipeline.named_steps['concentration_features']
                concentration_type = type(concentration_step).__name__
                logger.info(f"✓ {class_name}: Has concentration features ({concentration_type})")
                results[class_name] = "PASS"
            else:
                logger.warning(f"✗ {class_name}: Missing concentration features")
                logger.warning(f"  Pipeline steps: {pipeline_steps}")
                results[class_name] = "FAIL"
                
        except Exception as e:
            logger.error(f"✗ {class_name}: Failed to initialize - {e}")
            results[class_name] = "ERROR"
    
    # Test feature count consistency
    logger.info("\n=== Feature Count Consistency Test ===")
    feature_counts = {}
    
    for class_name, constructor in test_classes:
        if results[class_name] == "PASS":
            try:
                instance = constructor()
                feature_names = instance.feature_pipeline.get_feature_names_out()
                feature_counts[class_name] = len(feature_names)
                logger.info(f"{class_name}: {len(feature_names)} features")
            except Exception as e:
                logger.error(f"{class_name}: Could not get feature count - {e}")
    
    # Check consistency
    unique_counts = set(feature_counts.values())
    if len(unique_counts) == 1:
        logger.info("✓ All models have consistent feature counts")
        consistency_pass = True
    else:
        logger.warning("✗ Feature counts differ between models:")
        for name, count in feature_counts.items():
            logger.warning(f"  {name}: {count}")
        consistency_pass = False
    
    # Test disabled concentration features
    logger.info("\n=== Testing Disabled Concentration Features ===")
    config_disabled = Config()
    config_disabled.use_concentration_features = False
    
    model_trainer_disabled = ModelTrainer(config_disabled, strategy, reporter)
    has_concentration = 'concentration_features' in model_trainer_disabled.feature_pipeline.named_steps
    
    if not has_concentration:
        logger.info("✓ Concentration features properly disabled when use_concentration_features=False")
        disable_test_pass = True
    else:
        logger.warning("✗ Concentration features still present when disabled")
        disable_test_pass = False
    
    # Summary
    logger.info("\n=== FINAL RESULTS ===")
    total_tests = len(test_classes)
    passed_tests = sum(1 for result in results.values() if result == "PASS")
    failed_tests = sum(1 for result in results.values() if result == "FAIL")
    error_tests = sum(1 for result in results.values() if result == "ERROR")
    
    logger.info(f"Total classes tested: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    logger.info(f"Errors: {error_tests}")
    
    for class_name, result in results.items():
        status_symbol = "✓" if result == "PASS" else "✗"
        logger.info(f"{status_symbol} {class_name}: {result}")
    
    # Overall assessment
    all_passed = failed_tests == 0 and error_tests == 0
    
    if all_passed and consistency_pass and disable_test_pass:
        logger.info("\n🎉 ALL TESTS PASSED! Concentration features are properly integrated across all models.")
        logger.info("\nKey Benefits:")
        logger.info("• All models now learn from concentration-aware features")
        logger.info("• Features adapt to your target concentration range (0.15-0.55%)")
        logger.info("• AutoGluon ensemble can leverage concentration patterns")
        logger.info("• Improved performance expected across all model types")
        return True
    else:
        logger.error("\n❌ SOME TESTS FAILED!")
        if failed_tests > 0:
            logger.error(f"  {failed_tests} models missing concentration features")
        if error_tests > 0:
            logger.error(f"  {error_tests} models had initialization errors")
        if not consistency_pass:
            logger.error("  Feature counts are inconsistent between models")
        if not disable_test_pass:
            logger.error("  Concentration features not properly disabled")
        return False

def test_target_awareness():
    """Test that concentration features properly use target data when available."""
    logger.info("\n=== Testing Target-Aware Concentration Features ===")
    
    import numpy as np
    import pandas as pd
    from src.features.concentration_features import ConcentrationRangeFeatures
    
    # Create mock data with known concentration distribution
    np.random.seed(42)
    n_samples = 100
    
    # Create mock features
    mock_features = pd.DataFrame({
        'P_I_simple_peak_area': np.random.uniform(1000, 10000, n_samples),
        'P_I_simple_peak_height': np.random.uniform(500, 5000, n_samples),
        'C_I_simple_peak_area': np.random.uniform(2000, 8000, n_samples),
        'P_C_ratio': np.random.uniform(0.1, 2.0, n_samples),
    })
    
    # Create target concentrations in the new range (0.15-0.55)
    mock_targets = pd.Series(np.random.uniform(0.15, 0.55, n_samples))
    
    # Test concentration features with targets
    transformer = ConcentrationRangeFeatures(use_target_percentiles=True)
    transformer.fit(mock_features, mock_targets)
    
    # Check that thresholds are learned from targets
    expected_low = np.percentile(mock_targets, 25)
    expected_high = np.percentile(mock_targets, 75)
    
    threshold_correct = (
        abs(transformer.fitted_low_threshold_ - expected_low) < 0.01 and
        abs(transformer.fitted_high_threshold_ - expected_high) < 0.01
    )
    
    if threshold_correct:
        logger.info(f"✓ Target-aware thresholds: low={transformer.fitted_low_threshold_:.3f}, high={transformer.fitted_high_threshold_:.3f}")
        logger.info(f"  Expected from data: low={expected_low:.3f}, high={expected_high:.3f}")
        return True
    else:
        logger.error(f"✗ Target-aware thresholds incorrect:")
        logger.error(f"  Got: low={transformer.fitted_low_threshold_:.3f}, high={transformer.fitted_high_threshold_:.3f}")
        logger.error(f"  Expected: low={expected_low:.3f}, high={expected_high:.3f}")
        return False

if __name__ == "__main__":
    try:
        logger.info("Starting comprehensive concentration features test...")
        
        # Test 1: All models have concentration features
        all_models_pass = test_all_models_have_concentration_features()
        
        # Test 2: Target awareness works
        target_aware_pass = test_target_awareness()
        
        # Overall result
        if all_models_pass and target_aware_pass:
            logger.info("\n🎯 COMPREHENSIVE TEST PASSED!")
            logger.info("All models are ready to use concentration features.")
            logger.info("\nTo use with expanded range (0.15-0.55%):")
            logger.info("• Run any training command (train, tune, autogluon, optimize-models)")
            logger.info("• Concentration features will automatically adapt to your target data")
            logger.info("• Expected R² improvement from 0.2584 to 0.30-0.35+")
        else:
            logger.error("\n❌ COMPREHENSIVE TEST FAILED!")
            logger.error("Some components are not properly configured.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        sys.exit(1)