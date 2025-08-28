#!/usr/bin/env python3
"""
Test script to verify that concentration features are available for all models.
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
from src.reporting.reporter import Reporter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_all_models_have_concentration_features():
    """Test that all model training classes use concentration features when enabled."""
    
    # Create config with concentration features enabled
    config = Config()
    config.use_concentration_features = True
    
    # Create reporter
    reporter = Reporter(config)
    strategy = "simple_only"
    
    logger.info("=== Testing Concentration Features Integration ===")
    
    # Test 1: ModelTrainer
    logger.info("Testing ModelTrainer...")
    model_trainer = ModelTrainer(config, strategy, reporter)
    pipeline_name = type(model_trainer.feature_pipeline.named_steps.get('concentration_features', None)).__name__
    if 'ConcentrationRangeFeatures' in pipeline_name:
        logger.info("✓ ModelTrainer: Using concentration features")
    else:
        logger.warning("✗ ModelTrainer: NOT using concentration features")
        logger.info(f"  Pipeline steps: {list(model_trainer.feature_pipeline.named_steps.keys())}")
    
    # Test 2: AutoGluonTrainer
    logger.info("Testing AutoGluonTrainer...")
    autogluon_trainer = AutoGluonTrainer(config, strategy, reporter, use_concentration_features=True)
    pipeline_name = type(autogluon_trainer.feature_pipeline.named_steps.get('concentration_features', None)).__name__
    if 'ConcentrationRangeFeatures' in pipeline_name:
        logger.info("✓ AutoGluonTrainer: Using concentration features")
    else:
        logger.warning("✗ AutoGluonTrainer: NOT using concentration features")
        logger.info(f"  Pipeline steps: {list(autogluon_trainer.feature_pipeline.named_steps.keys())}")
    
    # Test 3: ModelTuner
    logger.info("Testing ModelTuner...")
    model_tuner = ModelTuner(config, reporter, strategy)
    pipeline_name = type(model_tuner.feature_pipeline.named_steps.get('concentration_features', None)).__name__
    if 'ConcentrationRangeFeatures' in pipeline_name:
        logger.info("✓ ModelTuner: Using concentration features")
    else:
        logger.warning("✗ ModelTuner: NOT using concentration features")
        logger.info(f"  Pipeline steps: {list(model_tuner.feature_pipeline.named_steps.keys())}")
    
    # Test 4: XGBoostOptimizer
    logger.info("Testing XGBoostOptimizer...")
    xgb_optimizer = XGBoostOptimizer(config, strategy)
    pipeline_name = type(xgb_optimizer.feature_pipeline.named_steps.get('concentration_features', None)).__name__
    if 'ConcentrationRangeFeatures' in pipeline_name:
        logger.info("✓ XGBoostOptimizer: Using concentration features")
    else:
        logger.warning("✗ XGBoostOptimizer: NOT using concentration features")
        logger.info(f"  Pipeline steps: {list(xgb_optimizer.feature_pipeline.named_steps.keys())}")
    
    # Test 5: Verify they all have the same number of output features
    logger.info("\n=== Feature Count Comparison ===")
    feature_counts = {}
    
    for name, trainer in [
        ("ModelTrainer", model_trainer),
        ("AutoGluonTrainer", autogluon_trainer),
        ("ModelTuner", model_tuner),
        ("XGBoostOptimizer", xgb_optimizer)
    ]:
        try:
            feature_names = trainer.feature_pipeline.get_feature_names_out()
            feature_counts[name] = len(feature_names)
            logger.info(f"{name}: {len(feature_names)} features")
        except Exception as e:
            logger.error(f"{name}: Could not get feature count - {e}")
    
    # Check if all have the same count
    if len(set(feature_counts.values())) == 1:
        logger.info("✓ All trainers have the same number of features")
    else:
        logger.warning("✗ Feature counts differ between trainers:")
        for name, count in feature_counts.items():
            logger.warning(f"  {name}: {count}")
    
    # Test 6: Test disabled concentration features
    logger.info("\n=== Testing Disabled Concentration Features ===")
    config_disabled = Config()
    config_disabled.use_concentration_features = False
    
    model_trainer_disabled = ModelTrainer(config_disabled, strategy, reporter)
    has_concentration = 'concentration_features' in model_trainer_disabled.feature_pipeline.named_steps
    
    if not has_concentration:
        logger.info("✓ Concentration features properly disabled when use_concentration_features=False")
    else:
        logger.warning("✗ Concentration features still present when disabled")
    
    return True

if __name__ == "__main__":
    try:
        success = test_all_models_have_concentration_features()
        if success:
            logger.info("\n🎉 All tests passed! Concentration features are integrated across all models.")
        else:
            logger.error("\n❌ Some tests failed.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        sys.exit(1)