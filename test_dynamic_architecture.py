#!/usr/bin/env python3
"""
Test script for dynamic architecture generation.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))

from src.models.model_tuner import ModelTuner
from src.models.optimize_neural_network import NeuralNetworkOptimizer
from src.config.pipeline_config import config
from src.reporting.reporter import Reporter
import optuna

def test_model_tuner_dynamic_architecture():
    """Test ModelTuner dynamic architecture generation."""
    print("=== Testing ModelTuner Dynamic Architecture ===")
    
    # Create a mock tuner
    reporter = Reporter(config)
    tuner = ModelTuner(config, reporter=reporter, strategy='full_context')
    
    # Test different input feature counts
    for input_count in [10, 15, 20, 25, 30]:
        print(f"\nTesting with {input_count} input features:")
        
        # Mock the input feature count
        tuner._cached_features = type('MockFeatures', (), {'shape': (100, input_count)})()
        
        # Test full model - create separate study for each test
        study_full = optuna.create_study()
        def objective_full(trial):
            arch = tuner._generate_dynamic_architecture(trial, input_count, 'full')
            return tuner._calculate_parameter_count(input_count, arch)
        
        study_full.optimize(objective_full, n_trials=1)
        arch_full = tuner._generate_dynamic_architecture(study_full.best_trial, input_count, 'full')
        param_count_full = tuner._calculate_parameter_count(input_count, arch_full)
        print(f"  Full model: {arch_full} ({param_count_full} parameters)")
        
        # Test light model - create separate study
        study_light = optuna.create_study()
        def objective_light(trial):
            arch = tuner._generate_dynamic_architecture(trial, input_count, 'light')
            return tuner._calculate_parameter_count(input_count, arch)
        
        study_light.optimize(objective_light, n_trials=1)
        arch_light = tuner._generate_dynamic_architecture(study_light.best_trial, input_count, 'light')
        param_count_light = tuner._calculate_parameter_count(input_count, arch_light)
        print(f"  Light model: {arch_light} ({param_count_light} parameters)")
        
        # Validate architectures
        assert all(layer >= 2 for layer in arch_full), f"Full model has layer < 2: {arch_full}"
        assert all(layer >= 2 for layer in arch_light), f"Light model has layer < 2: {arch_light}"
        assert param_count_full < 10000, f"Full model too many params: {param_count_full}"
        assert param_count_light < 5000, f"Light model too many params: {param_count_light}"
    
    print("✓ ModelTuner dynamic architecture tests passed!")

def test_neural_network_optimizer_dynamic_architecture():
    """Test NeuralNetworkOptimizer dynamic architecture generation."""
    print("\n=== Testing NeuralNetworkOptimizer Dynamic Architecture ===")
    
    # Test both model types
    for model_type in ['full', 'light']:
        print(f"\nTesting {model_type} model type:")
        
        # Create optimizer
        optimizer = NeuralNetworkOptimizer(config, strategy='full_context', model_type=model_type)
        
        # Create a mock trial
        study = optuna.create_study()
        
        def objective(trial):
            # Test different input feature counts
            for input_count in [8, 12, 18, 25, 35]:
                print(f"  {input_count} inputs:", end=' ')
                
                # Mock the input feature count
                optimizer._cached_features = type('MockFeatures', (), {'shape': (100, input_count)})()
                
                # Generate architecture
                arch = optimizer._generate_dynamic_architecture(trial, input_count, model_type)
                param_count = optimizer._calculate_parameter_count(input_count, arch)
                print(f"{arch} ({param_count} params)")
                
                # Validate architecture
                assert all(layer >= 2 for layer in arch), f"Architecture has layer < 2: {arch}"
                assert len(arch) <= 4, f"Too many layers: {len(arch)}"
                assert param_count < 8000, f"Too many parameters: {param_count}"
            
            return 0.5  # Dummy value
        
        # Run one trial to test
        study.optimize(objective, n_trials=1)
    
    print("✓ NeuralNetworkOptimizer dynamic architecture tests passed!")

def test_parameter_budgeting():
    """Test parameter budgeting works correctly."""
    print("\n=== Testing Parameter Budgeting ===")
    
    reporter = Reporter(config)
    tuner = ModelTuner(config, reporter=reporter, strategy='full_context')
    
    study = optuna.create_study()
    
    def objective(trial):
        # Test with very small budget
        max_params_ratio = 1.0  # Very restrictive
        max_total_params = int(700 * max_params_ratio)  # 700 params max
        
        input_count = 20
        tuner._cached_features = type('MockFeatures', (), {'shape': (100, input_count)})()
        
        # Mock the parameter ratio in trial
        trial.suggest_float = lambda name, low, high, **kwargs: max_params_ratio if 'ratio' in name else 2.0
        trial.suggest_categorical = lambda name, choices: choices[0]
        trial.suggest_int = lambda name, low, high: (low + high) // 2
        
        # Generate architecture
        arch = tuner._generate_dynamic_architecture(trial, input_count, 'full')
        param_count = tuner._calculate_parameter_count(input_count, arch)
        
        print(f"Input: {input_count}, Architecture: {arch}, Params: {param_count}, Limit: {max_total_params}")
        
        # Should respect budget
        assert param_count <= max_total_params * 1.1, f"Exceeded budget: {param_count} > {max_total_params}"
        
        return 0.5
    
    study.optimize(objective, n_trials=1)
    print("✓ Parameter budgeting test passed!")

if __name__ == "__main__":
    test_model_tuner_dynamic_architecture()
    test_neural_network_optimizer_dynamic_architecture()
    test_parameter_budgeting()
    print("\n🎉 All dynamic architecture tests passed!")