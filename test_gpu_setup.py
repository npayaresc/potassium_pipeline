#!/usr/bin/env python3
"""
Test GPU setup for AutoGluon training.

This script helps diagnose GPU-related issues and tests the GPU-safe configurations.
"""
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_gpu_libraries():
    """Test GPU library availability and versions."""
    print("🔍 TESTING GPU LIBRARY COMPATIBILITY")
    print("="*50)
    
    # Test PyTorch CUDA
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU device: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available - GPU acceleration disabled")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Test LightGBM
    try:
        import lightgbm as lgb
        print(f"✅ LightGBM version: {lgb.__version__}")
    except ImportError:
        print("⚠️  LightGBM not installed")
    
    # Test CatBoost
    try:
        import catboost as cb
        print(f"✅ CatBoost version: {cb.__version__}")
    except ImportError:
        print("⚠️  CatBoost not installed")
    
    # Test XGBoost
    try:
        import xgboost as xgb
        print(f"✅ XGBoost version: {xgb.__version__}")
    except ImportError:
        print("⚠️  XGBoost not installed")
    
    return True

def test_autogluon_config():
    """Test AutoGluon configuration."""
    print("\n🎛️  TESTING AUTOGLUON CONFIGURATION")
    print("="*50)
    
    try:
        from src.config.pipeline_config import config
        
        print("Default Configuration:")
        print(f"  Preset: {config.autogluon.presets}")
        print(f"  GPU Safe Preset: {config.autogluon.gpu_safe_preset}")
        print(f"  Excluded Models: {config.autogluon.excluded_model_types}")
        
        # Test GPU-safe features
        if hasattr(config.autogluon, 'gpu_safe_hyperparameters'):
            print(f"✅ GPU-safe hyperparameters available")
            safe_models = list(config.autogluon.gpu_safe_hyperparameters.keys())
            print(f"  Safe models: {safe_models}")
        else:
            print("❌ GPU-safe hyperparameters not found")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_dimension_reduction():
    """Test dimension reduction with GPU support."""
    print("\n🔄 TESTING DIMENSION REDUCTION")
    print("="*50)
    
    try:
        from src.features.dimension_reduction import DimensionReductionFactory
        import numpy as np
        
        # Test data
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        
        # Test PLS (current default)
        pls_reducer = DimensionReductionFactory.create_reducer('pls', {
            'n_components': 8, 'scale': True
        })
        X_pls = pls_reducer.fit_transform(X, y)
        print(f"✅ PLS reduction: {X.shape[1]} → {X_pls.shape[1]}")
        
        # Test autoencoder with CPU
        ae_reducer = DimensionReductionFactory.create_reducer('autoencoder', {
            'n_components': 8, 'epochs': 5, 'device': 'cpu'
        })
        X_ae = ae_reducer.fit_transform(X)
        print(f"✅ Autoencoder reduction: {X.shape[1]} → {X_ae.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dimension reduction test failed: {e}")
        return False

def test_small_autogluon_training():
    """Test small AutoGluon training to verify GPU setup."""
    print("\n🤏 TESTING SMALL AUTOGLUON TRAINING")
    print("="*50)
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.datasets import make_regression
        from src.models.autogluon_trainer import AutoGluonTrainer
        from src.config.pipeline_config import config
        from src.reporting.reporter import Reporter
        import tempfile
        
        # Create small synthetic spectral dataset in the correct format
        n_samples = 100
        n_wavelengths = 1000  # Many more wavelengths for precise coverage
        
        # Create synthetic spectral data covering all required spectral regions
        # Get actual wavelength range from config
        all_regions = config.all_regions
        min_wl = min(r.lower_wavelength for r in all_regions) - 10  # 279 - 10 = 269
        max_wl = max(r.upper_wavelength for r in all_regions) + 10   # 870 + 10 = 880
        
        wavelengths = np.linspace(min_wl, max_wl, n_wavelengths)  # Dense coverage for all regions
        spectra_data = []
        targets = []
        
        for i in range(n_samples):
            # Generate random spectrum with some peaks
            intensities = np.random.randn(n_wavelengths) * 0.1 + 1.0
            # Add some synthetic peaks
            for _ in range(3):
                peak_center = np.random.randint(10, n_wavelengths-10)
                peak_width = np.random.randint(3, 8)
                peak_height = np.random.uniform(0.5, 2.0)
                for j in range(max(0, peak_center-peak_width), min(n_wavelengths, peak_center+peak_width)):
                    intensities[j] += peak_height * np.exp(-0.5 * ((j - peak_center) / (peak_width/3))**2)
            
            # Create target based on spectrum
            target = np.sum(intensities[20:30]) * 0.01 + np.random.normal(0, 0.02)
            
            spectra_data.append({
                config.sample_id_column: f'sample_{i}',
                'wavelengths': wavelengths.tolist(),
                'intensities': intensities.tolist(),
                config.target_column: target
            })
            
        # Create DataFrame
        df = pd.DataFrame(spectra_data)
        
        print("✅ Created synthetic spectral dataset")
        print(f"  Shape: {df.shape}")
        print(f"  Samples: {n_samples}")
        print(f"  Wavelengths per spectrum: {n_wavelengths}")
        print(f"  Columns: {list(df.columns)}")
        
        # Test with CPU first
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config = config.model_copy(deep=True)
            test_config.reports_dir = Path(temp_dir)
            test_config.model_dir = Path(temp_dir)
            test_config.log_dir = Path(temp_dir)
            test_config.use_gpu = False  # CPU test
            
            # Reduce time limit for quick test
            test_config.autogluon.time_limit = 60  # 1 minute
            
            reporter = Reporter(test_config)
            trainer = AutoGluonTrainer(test_config, 'simple_only', reporter)
            
            print("🚀 Starting small CPU training test...")
            trainer.train(df)
            print("✅ CPU training completed successfully")
            
        # Test with GPU if CUDA is available
        import torch
        if torch.cuda.is_available():
            print("🎮 Starting GPU training test...")
            with tempfile.TemporaryDirectory() as temp_dir:
                gpu_config = config.model_copy(deep=True)
                gpu_config.reports_dir = Path(temp_dir)
                gpu_config.model_dir = Path(temp_dir)
                gpu_config.log_dir = Path(temp_dir)
                gpu_config.use_gpu = True  # GPU test
                
                # Reduce time limit for quick test
                gpu_config.autogluon.time_limit = 60  # 1 minute
                
                reporter = Reporter(gpu_config)
                trainer = AutoGluonTrainer(gpu_config, 'simple_only', reporter)
                
                print("🚀 Starting small GPU training test...")
                trainer.train(df)
                print("✅ GPU training completed successfully")
        else:
            print("⚠️  CUDA not available - skipping GPU test")
            
        return True
        
    except Exception as e:
        print(f"❌ Small training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all GPU setup tests."""
    print("🧪 GPU SETUP DIAGNOSTIC TOOL")
    print("="*50)
    
    tests = [
        ("GPU Libraries", test_gpu_libraries),
        ("AutoGluon Config", test_autogluon_config), 
        ("Dimension Reduction", test_dimension_reduction),
        ("Small Training", test_small_autogluon_training),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n📋 Running: {test_name}")
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your GPU setup should work with AutoGluon.")
        print("\nYou can now run:")
        print("  python main.py autogluon --gpu")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("\nTroubleshooting:")
        print("  1. Check GPU drivers and CUDA installation")
        print("  2. Verify library versions (see docs/gpu_troubleshooting.md)")
        print("  3. Try CPU mode first: python main.py autogluon")
    
    return 0 if passed == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())