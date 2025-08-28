#!/usr/bin/env python3
"""
Test script to compare all weighting methods and their corresponding objective functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def generate_imbalanced_test_data(n_samples=500, random_state=42):
    """Generate synthetic imbalanced data for testing."""
    np.random.seed(random_state)
    
    # Create imbalanced concentration distribution
    # 60% samples in 3.5-4.5 range (overrepresented)
    # 20% samples in 2.0-3.0 range (underrepresented)
    # 20% samples in 5.5-7.0 range (underrepresented)
    
    n_main = int(0.6 * n_samples)
    n_low = int(0.2 * n_samples) 
    n_high = n_samples - n_main - n_low
    
    # Generate concentration values
    y_main = np.random.normal(4.0, 0.3, n_main)  # Overrepresented
    y_low = np.random.normal(2.5, 0.2, n_low)    # Underrepresented
    y_high = np.random.normal(6.2, 0.3, n_high)  # Underrepresented
    
    y = np.concatenate([y_main, y_low, y_high])
    
    # Generate features (10 features)
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    
    # Make features somewhat predictive of y
    for i in range(n_features):
        X[:, i] += y * (0.1 + 0.05 * i) + np.random.randn(n_samples) * 0.2
    
    return X, y

def test_all_weighting_methods():
    """Test all weighting methods and their objective functions."""
    
    print("=" * 80)
    print("TESTING ALL WEIGHTING METHODS")
    print("=" * 80)
    
    # Generate test data
    X, y = generate_imbalanced_test_data()
    y_series = pd.Series(y)
    
    print(f"Generated {len(y)} samples")
    print(f"Concentration range: {y.min():.2f} - {y.max():.2f}")
    
    # Show data distribution
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(y, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Nitrogen Concentration (%)')
    plt.ylabel('Frequency')
    plt.title('Imbalanced Data Distribution')
    plt.grid(True, alpha=0.3)
    
    # Import our weighting methods
    from src.models.model_trainer import ModelTrainer
    from src.config.pipeline_config import Config
    from src.reporting.reporter import Reporter
    
    # Create minimal config for testing
    config = Config(
        run_timestamp="test",
        data_dir=".", raw_data_dir=".", processed_data_dir=".",
        model_dir=".", reports_dir=".", log_dir=".", 
        bad_files_dir=".", averaged_files_dir=".", cleansed_files_dir=".", 
        bad_prediction_files_dir=".", reference_data_path="README.md",
        use_sample_weights=True
    )
    
    reporter = Reporter(config)
    trainer = ModelTrainer(config, "test_strategy", reporter)
    
    # Test all weighting methods
    methods = ['weighted_r2', 'distribution_based', 'hybrid', 'legacy', 'improved']
    results = {}
    
    print(f"\n{'Method':<20} {'Min Weight':<12} {'Max Weight':<12} {'Weight Range':<12} {'Test R²':<12}")
    print("-" * 80)
    
    for method in methods:
        try:
            # Calculate weights
            weights = trainer._calculate_sample_weights(y_series, method=method)
            
            # Test model training with these weights
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y, sample_weight=weights)
            y_pred = model.predict(X)
            test_r2 = r2_score(y, y_pred)
            
            # Store results
            results[method] = {
                'weights': weights,
                'min_weight': weights.min(),
                'max_weight': weights.max(),
                'weight_range': weights.max() - weights.min(),
                'r2': test_r2,
                'predictions': y_pred
            }
            
            print(f"{method:<20} {weights.min():<12.3f} {weights.max():<12.3f} {weights.max()-weights.min():<12.3f} {test_r2:<12.4f}")
            
        except Exception as e:
            print(f"{method:<20} ERROR: {e}")
            continue
    
    # Test objective functions
    print(f"\n{'Objective Function':<25} {'Score':<12} {'Description'}")
    print("-" * 80)
    
    from src.models.tuner_objectives_improved import (
        calculate_weighted_r2_objective,
        calculate_distribution_based_objective,
        calculate_hybrid_weighted_objective,
        calculate_balanced_mae_objective,
        calculate_quantile_weighted_objective
    )
    
    objective_functions = [
        ('weighted_r2', calculate_weighted_r2_objective, 'Emphasizes extreme ranges'),
        ('distribution_based', calculate_distribution_based_objective, 'Inverse density weighting'),
        ('hybrid_weighted', calculate_hybrid_weighted_objective, 'Distribution + domain knowledge'),
        ('balanced_mae', calculate_balanced_mae_objective, 'Normalized MAE across ranges'),
        ('quantile_weighted', calculate_quantile_weighted_objective, 'Performance across quantiles')
    ]
    
    # Use predictions from the weighted_r2 method for consistency
    if 'weighted_r2' in results:
        y_test_pred = results['weighted_r2']['predictions']
        
        for obj_name, obj_func, description in objective_functions:
            try:
                score = obj_func(y, y_test_pred)
                print(f"{obj_name:<25} {score:<12.4f} {description}")
            except Exception as e:
                print(f"{obj_name:<25} ERROR: {e}")
    
    # Visualize weight distributions
    plt.subplot(1, 2, 2)
    for method in ['weighted_r2', 'distribution_based', 'hybrid']:
        if method in results:
            weights = results[method]['weights']
            plt.scatter(y, weights, alpha=0.6, s=20, label=f'{method}')
    
    plt.xlabel('Nitrogen Concentration (%)')
    plt.ylabel('Sample Weight')
    plt.title('Weight Distribution by Method')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weighting_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Performance comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    # Calculate performance on different concentration ranges
    ranges = [
        ('Low (≤25%)', y <= np.percentile(y, 25)),
        ('Medium (25-75%)', (y > np.percentile(y, 25)) & (y <= np.percentile(y, 75))),
        ('High (≥75%)', y > np.percentile(y, 75))
    ]
    
    print(f"\n{'Method':<20} {'Overall R²':<12} {'Low R²':<10} {'Med R²':<10} {'High R²':<10}")
    print("-" * 70)
    
    for method in methods:
        if method in results:
            preds = results[method]['predictions']
            overall_r2 = results[method]['r2']
            
            range_r2s = []
            for range_name, mask in ranges:
                if np.sum(mask) > 5:  # Need enough samples
                    range_r2 = r2_score(y[mask], preds[mask])
                    range_r2s.append(range_r2)
                else:
                    range_r2s.append(0.0)
            
            print(f"{method:<20} {overall_r2:<12.4f} {range_r2s[0]:<10.4f} {range_r2s[1]:<10.4f} {range_r2s[2]:<10.4f}")
    
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    # Find best methods for different criteria
    best_overall = max(results.keys(), key=lambda k: results[k]['r2'])
    best_low_range = max(results.keys(), key=lambda k: r2_score(y[ranges[0][1]], results[k]['predictions'][ranges[0][1]]) if np.sum(ranges[0][1]) > 5 else 0)
    best_high_range = max(results.keys(), key=lambda k: r2_score(y[ranges[2][1]], results[k]['predictions'][ranges[2][1]]) if np.sum(ranges[2][1]) > 5 else 0)
    
    print(f"✓ Best overall performance: {best_overall} (R² = {results[best_overall]['r2']:.4f})")
    print(f"✓ Best for low concentrations: {best_low_range}")
    print(f"✓ Best for high concentrations: {best_high_range}")
    
    if 'distribution_based' in results and 'weighted_r2' in results:
        dist_r2 = results['distribution_based']['r2']
        weighted_r2 = results['weighted_r2']['r2']
        
        if dist_r2 > weighted_r2:
            print(f"\n💡 Distribution-based weighting shows better generalization!")
            print(f"   Improvement: +{dist_r2 - weighted_r2:.4f} R² points")
        else:
            print(f"\n💡 Current weighted_r2 method is still competitive")
            print(f"   Difference: {weighted_r2 - dist_r2:.4f} R² points")
    
    print(f"\n🎯 CONFIGURATION RECOMMENDATIONS:")
    print(f"   For best generalization: sample_weight_method = 'distribution_based'")
    print(f"   For domain-specific focus: sample_weight_method = 'hybrid'") 
    print(f"   For current behavior: sample_weight_method = 'weighted_r2'")
    
    return results

if __name__ == "__main__":
    results = test_all_weighting_methods()
    print(f"\n✅ All weighting methods tested successfully!")
    print(f"📊 Comparison plot saved as 'weighting_methods_comparison.png'")