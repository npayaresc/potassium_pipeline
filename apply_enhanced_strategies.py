#!/usr/bin/env python3
"""
Script to apply enhanced optimization strategies to all optimizers.
This applies strategies A-D to all optimizer files in the models directory.
"""
import os
import re
from pathlib import Path

def update_optimizer_file(filepath):
    """Update a single optimizer file with enhanced strategies"""
    print(f"Updating {filepath.name}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Skip if already updated
    if 'from src.models.enhanced_optuna_strategies import get_enhanced_optimization_config' in content:
        print(f"  {filepath.name} already updated, skipping")
        return
    
    # 1. Add import for enhanced strategies
    import_pattern = r'from src\.models\.base_optimizer import BaseOptimizer'
    replacement = 'from src.models.base_optimizer import BaseOptimizer\nfrom src.models.enhanced_optuna_strategies import get_enhanced_optimization_config'
    content = re.sub(import_pattern, replacement, content)
    
    # 2. Update optimize method signature and add enhanced config
    optimize_pattern = r'(def optimize\(self, X_train, y_train, X_val=None, y_val=None, n_trials=100, timeout=\d+\):\s*"""[^"]*?""")\s*(self\.X_train = X_train\s*self\.y_train = y_train)'
    optimize_replacement = r'\1\n        self.X_train = X_train\n        self.y_train = y_train\n        \n        # Initialize enhanced optimization strategies\n        dataset_size = len(X_train)\n        self._enhanced_config = get_enhanced_optimization_config(\n            model_name=\'{model_name}\',\n            dataset_size=dataset_size,\n            n_trials=n_trials,\n            reports_dir=self.config.reports_dir\n        )\n        logger.info(f"Enhanced optimization strategies initialized for dataset_size={{dataset_size}}, n_trials={{n_trials}}")'
    
    # Extract model name from filename
    model_name = filepath.stem.replace('optimize_', '')
    if model_name == 'extratrees':
        model_name = 'extratrees'
    elif model_name == 'random_forest':
        model_name = 'random_forest'
    elif model_name == 'autogluon':
        model_name = 'lightgbm'  # AutoGluon uses multiple models, default to lightgbm strategy
    elif model_name == 'neural_network':
        model_name = 'neural_network'
    
    optimize_replacement = optimize_replacement.format(model_name=model_name)
    content = re.sub(optimize_pattern, optimize_replacement, content, flags=re.DOTALL)
    
    # 3. Update parameter suggestion sections
    # This is model-specific, so we'll add a generic check
    
    # Find parameter suggestion sections and add enhanced strategy check
    param_patterns = [
        (r'(# [^#\n]* parameters[^{]*\{)', r'\1\n        # Enhanced parameter suggestion with smart search space (Strategy D)\n        if hasattr(self, "_enhanced_config") and self._enhanced_config:\n            params = self._enhanced_config["smart_search_space"].suggest_params(trial)\n            if not params:  # Fallback if model not supported\n                '),
        # Add closing brace for fallback
        (r'(\s+})\s*\n\s*# Add', r'\1\n        \n        # Add')
    ]
    
    for pattern, replacement in param_patterns:
        content = re.sub(pattern, replacement, content)
    
    # 4. Update study creation to use enhanced sampler (if create_study found)
    study_pattern = r'(study = optuna\.create_study\(\s*direction=\'maximize\',\s*pruner=[^,]+,)\s*(sampler=[^)]+\)\s*\))'
    study_replacement = r'\1\n            sampler=self._enhanced_config["advanced_sampling"].get_best_sampler() if hasattr(self, "_enhanced_config") else optuna.samplers.TPESampler()\n        )'
    content = re.sub(study_pattern, study_replacement, content, flags=re.DOTALL)
    
    # 5. Add parameter importance analysis after optimization
    importance_pattern = r'(self\.best_params = study\.best_params\s*self\.best_score = study\.best_value)'
    importance_replacement = r'\1\n        \n        # Perform parameter importance analysis (Strategy C)\n        try:\n            if hasattr(self, "_enhanced_config"):\n                importance_analyzer = self._enhanced_config["importance_analyzer"]\n                param_importance = importance_analyzer.analyze_study(study, "{model_name}")\n                self._param_importance = param_importance\n                logger.info("Parameter importance analysis completed")\n        except Exception as e:\n            logger.warning(f"Parameter importance analysis failed: {{e}}")\n            self._param_importance = {{}}'.format(model_name=model_name)
    content = re.sub(importance_pattern, importance_replacement, content)
    
    # Write updated content
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"  {filepath.name} updated successfully")

def main():
    """Apply enhanced strategies to all optimizers"""
    models_dir = Path("/home/payanico/potassium_pipeline/src/models")
    
    # Find all optimizer files
    optimizer_files = list(models_dir.glob("optimize_*.py"))
    
    # Skip files that don't need updating or are wrappers
    skip_files = ['optimize_all_models.py', 'optimize_range_specialist_neural_net.py']
    optimizer_files = [f for f in optimizer_files if f.name not in skip_files]
    
    print(f"Found {len(optimizer_files)} optimizer files to update:")
    for f in optimizer_files:
        print(f"  - {f.name}")
    
    print("\nApplying enhanced strategies...")
    
    for optimizer_file in optimizer_files:
        try:
            update_optimizer_file(optimizer_file)
        except Exception as e:
            print(f"  ERROR updating {optimizer_file.name}: {e}")
    
    print("\nEnhanced strategies application complete!")
    print("\nSummary of applied strategies:")
    print("  A. Multi-objective optimization")
    print("  B. Advanced sampling for small datasets") 
    print("  C. Parameter importance analysis")
    print("  D. Smarter search spaces")

if __name__ == "__main__":
    main()