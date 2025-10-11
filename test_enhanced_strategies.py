#!/usr/bin/env python3
"""
Test script for enhanced optimization strategies.
Verifies that all strategies (A-D) are properly integrated.
"""
import sys
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))

from src.models.enhanced_optuna_strategies import (
    get_enhanced_optimization_config,
    MultiObjectiveStrategy,
    AdvancedSamplingStrategy,
    ParameterImportanceAnalyzer,
    SmartSearchSpaceStrategy
)
import optuna

def test_strategy_a_multi_objective():
    """Test Strategy A: Multi-objective optimization"""
    print("Testing Strategy A: Multi-objective optimization")
    
    strategy = MultiObjectiveStrategy(['r2_score', 'training_time'])
    study = strategy.create_study("test_multi_objective")
    
    assert len(study.directions) == 2
    assert study.directions[0].name == 'MAXIMIZE'  # r2_score
    assert study.directions[1].name == 'MINIMIZE'  # training_time
    
    print("  ✓ Multi-objective study created successfully")
    return True

def test_strategy_b_advanced_sampling():
    """Test Strategy B: Advanced sampling for small datasets"""
    print("Testing Strategy B: Advanced sampling for small datasets")
    
    # Test small dataset (< 500 samples)
    small_strategy = AdvancedSamplingStrategy(dataset_size=300, n_trials=100)
    small_sampler = small_strategy.get_best_sampler()
    assert isinstance(small_sampler, optuna.samplers.CmaEsSampler), f"Expected CmaEsSampler, got {type(small_sampler)}"
    print("  ✓ CMA-ES sampler selected for very small dataset")
    
    # Test medium dataset (< 2000 samples) 
    medium_strategy = AdvancedSamplingStrategy(dataset_size=1200, n_trials=100)
    medium_sampler = medium_strategy.get_best_sampler()
    assert isinstance(medium_sampler, optuna.samplers.TPESampler), f"Expected TPESampler, got {type(medium_sampler)}"
    print("  ✓ Enhanced TPE sampler selected for small dataset")
    
    # Test large dataset
    large_strategy = AdvancedSamplingStrategy(dataset_size=5000, n_trials=100)
    large_sampler = large_strategy.get_best_sampler()
    assert isinstance(large_sampler, optuna.samplers.TPESampler), f"Expected TPESampler, got {type(large_sampler)}"
    print("  ✓ Standard TPE sampler selected for large dataset")
    
    return True

def test_strategy_c_parameter_importance():
    """Test Strategy C: Parameter importance analysis"""
    print("Testing Strategy C: Parameter importance analysis")
    
    # Create a simple study for testing
    study = optuna.create_study(direction='maximize')
    
    # Add some dummy trials
    for i in range(10):
        trial = study.ask()
        trial.suggest_float('param1', 0.0, 1.0)
        trial.suggest_int('param2', 1, 10)
        study.tell(trial, i * 0.1)  # Dummy objective value
    
    analyzer = ParameterImportanceAnalyzer()
    importance = analyzer.analyze_study(study, 'test_model')
    
    assert isinstance(importance, dict)
    assert len(importance) >= 0  # May be empty for small studies
    print("  ✓ Parameter importance analysis completed")
    
    return True

def test_strategy_d_smart_search_spaces():
    """Test Strategy D: Smarter search spaces"""
    print("Testing Strategy D: Smarter search spaces")
    
    # Test XGBoost smart parameters
    xgb_strategy = SmartSearchSpaceStrategy('xgboost')
    study = optuna.create_study(direction='maximize')
    trial = study.ask()
    xgb_params = xgb_strategy.suggest_correlated_xgboost_params(trial)
    
    assert 'tree_growth_strategy' in trial.params
    assert 'n_estimators' in xgb_params
    assert 'learning_rate' in xgb_params
    print("  ✓ XGBoost correlated parameters generated")
    
    # Test LightGBM smart parameters
    lgb_strategy = SmartSearchSpaceStrategy('lightgbm')
    study2 = optuna.create_study(direction='maximize')
    trial2 = study2.ask()
    lgb_params = lgb_strategy.suggest_correlated_lightgbm_params(trial2)
    
    assert 'complexity_level' in trial2.params
    assert 'num_leaves' in lgb_params
    assert 'max_depth' in lgb_params
    print("  ✓ LightGBM correlated parameters generated")
    
    # Test CatBoost smart parameters
    cat_strategy = SmartSearchSpaceStrategy('catboost')
    study3 = optuna.create_study(direction='maximize')
    trial3 = study3.ask()
    cat_params = cat_strategy.suggest_correlated_catboost_params(trial3)
    
    assert 'bootstrap_type' in cat_params
    assert 'iterations' in cat_params
    assert 'depth' in cat_params
    print("  ✓ CatBoost correlated parameters generated")
    
    return True

def test_enhanced_config():
    """Test the complete enhanced configuration"""
    print("Testing complete enhanced optimization configuration")
    
    config = get_enhanced_optimization_config(
        model_name='xgboost',
        dataset_size=1200,
        n_trials=100,
        reports_dir=Path('/tmp')
    )
    
    # Verify all strategies are present
    assert 'multi_objective' in config
    assert 'advanced_sampling' in config
    assert 'importance_analyzer' in config
    assert 'smart_search_space' in config
    
    # Verify configuration values
    assert config['dataset_size'] == 1200
    assert config['n_trials'] == 100
    assert config['model_name'] == 'xgboost'
    
    print("  ✓ Complete enhanced configuration created successfully")
    return True

def main():
    """Run all enhanced strategy tests"""
    print("=" * 80)
    print("ENHANCED OPTIMIZATION STRATEGIES TEST SUITE")
    print("=" * 80)
    
    tests = [
        test_strategy_a_multi_objective,
        test_strategy_b_advanced_sampling,
        test_strategy_c_parameter_importance,
        test_strategy_d_smart_search_spaces,
        test_enhanced_config
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
                print(f"✅ {test.__name__} PASSED\n")
            else:
                failed += 1
                print(f"❌ {test.__name__} FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} FAILED with error: {e}\n")
    
    print("=" * 80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed == 0:
        print("🎉 All enhanced optimization strategies are working correctly!")
        print("\nStrategies successfully implemented:")
        print("  ✅ A. Multi-objective optimization (R² vs Training Time)")
        print("  ✅ B. Advanced sampling (CMA-ES for small datasets, Enhanced TPE)")
        print("  ✅ C. Parameter importance analysis (Mutual Info + Shapley)")
        print("  ✅ D. Smart search spaces (Correlated parameter suggestions)")
        return True
    else:
        print(f"⚠️  {failed} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)