#!/usr/bin/env python3
"""
GPU Support Diagnostic Script for Nitrogen Pipeline

This script checks if the ML libraries in your virtual environment 
have GPU support enabled and can detect your NVIDIA GPU.
"""

import sys
import subprocess
import importlib.util

def check_command_exists(command):
    """Check if a command exists in the system."""
    try:
        subprocess.run([command, '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_cuda_availability():
    """Check CUDA installation and GPU availability."""
    print("=" * 60)
    print("CHECKING CUDA AVAILABILITY")
    print("=" * 60)
    
    # Check nvidia-smi
    if check_command_exists('nvidia-smi'):
        print("✓ nvidia-smi found")
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
            print("GPU Information:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("✗ nvidia-smi failed:", e)
    else:
        print("✗ nvidia-smi not found - NVIDIA drivers may not be installed")
        return False
    
    # Check nvcc (CUDA compiler)
    if check_command_exists('nvcc'):
        print("✓ nvcc found")
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
            print("CUDA Compiler Version:")
            print(result.stdout)
        except subprocess.CalledProcessError:
            print("✗ nvcc failed")
    else:
        print("⚠ nvcc not found - CUDA toolkit may not be installed (some libraries may still work)")
    
    return True

def test_xgboost_gpu():
    """Test XGBoost GPU support."""
    print("\n" + "=" * 60)
    print("TESTING XGBOOST GPU SUPPORT")
    print("=" * 60)
    
    try:
        import xgboost as xgb
        print(f"✓ XGBoost version: {xgb.__version__}")
        
        # Test GPU training
        import numpy as np
        from sklearn.datasets import make_regression
        
        print("Testing GPU training capability...")
        X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
        
        try:
            # Try to create a GPU-enabled model
            model = xgb.XGBRegressor(
                tree_method='gpu_hist',
                gpu_id=0,
                n_estimators=10,
                verbosity=1
            )
            model.fit(X, y)
            print("✓ XGBoost GPU training successful!")
            return True
            
        except Exception as e:
            print(f"✗ XGBoost GPU training failed: {e}")
            print("  This usually means CUDA support is not available")
            return False
            
    except ImportError as e:
        print(f"✗ XGBoost not installed: {e}")
        return False

def test_lightgbm_gpu():
    """Test LightGBM GPU support."""
    print("\n" + "=" * 60)
    print("TESTING LIGHTGBM GPU SUPPORT")
    print("=" * 60)
    
    try:
        import lightgbm as lgb
        print(f"✓ LightGBM version: {lgb.__version__}")
        
        # Test GPU training
        import numpy as np
        from sklearn.datasets import make_regression
        
        print("Testing GPU training capability...")
        X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
        
        try:
            # Try to create a GPU-enabled model
            model = lgb.LGBMRegressor(
                device='gpu',
                gpu_platform_id=0,
                gpu_device_id=0,
                n_estimators=10,
                verbosity=-1
            )
            model.fit(X, y)
            print("✓ LightGBM GPU training successful!")
            return True
            
        except Exception as e:
            print(f"✗ LightGBM GPU training failed: {e}")
            print("  This usually means OpenCL/CUDA support is not available")
            # Try to get more specific error info
            try:
                import lightgbm as lgb
                print("  Available LightGBM devices:")
                # This will show what devices are available
                train_data = lgb.Dataset(X, label=y)
                params = {'device': 'gpu', 'objective': 'regression', 'verbose': -1}
                lgb.train(params, train_data, num_boost_round=1)
            except Exception as detailed_error:
                print(f"  Detailed error: {detailed_error}")
            return False
            
    except ImportError as e:
        print(f"✗ LightGBM not installed: {e}")
        return False

def test_catboost_gpu():
    """Test CatBoost GPU support."""
    print("\n" + "=" * 60)
    print("TESTING CATBOOST GPU SUPPORT")
    print("=" * 60)
    
    try:
        import catboost as cb
        print(f"✓ CatBoost version: {cb.__version__}")
        
        # Test GPU training
        import numpy as np
        from sklearn.datasets import make_regression
        
        print("Testing GPU training capability...")
        X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
        
        try:
            # Try to create a GPU-enabled model
            model = cb.CatBoostRegressor(
                task_type='GPU',
                devices='0',
                iterations=10,
                verbose=False
            )
            model.fit(X, y)
            print("✓ CatBoost GPU training successful!")
            return True
            
        except Exception as e:
            print(f"✗ CatBoost GPU training failed: {e}")
            print("  This usually means CUDA support is not available")
            return False
            
    except ImportError as e:
        print(f"✗ CatBoost not installed: {e}")
        return False

def test_autogluon_gpu():
    """Test AutoGluon GPU support."""
    print("\n" + "=" * 60)
    print("TESTING AUTOGLUON GPU SUPPORT")
    print("=" * 60)
    
    try:
        from autogluon.tabular import TabularPredictor
        print("✓ AutoGluon installed")
        
        # Check if torch is available with CUDA
        try:
            import torch
            print(f"✓ PyTorch version: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"✓ CUDA available in PyTorch - {torch.cuda.device_count()} GPU(s)")
                print(f"  Current device: {torch.cuda.get_device_name()}")
            else:
                print("⚠ CUDA not available in PyTorch")
        except ImportError:
            print("⚠ PyTorch not installed - AutoGluon may have limited GPU support")
        
        return torch.cuda.is_available() if 'torch' in locals() else False
        
    except ImportError as e:
        print(f"✗ AutoGluon not installed: {e}")
        return False

def main():
    """Run all GPU diagnostic tests."""
    print("GPU Support Diagnostic for Nitrogen Pipeline")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check CUDA
    cuda_available = check_cuda_availability()
    
    # Test each library
    results = {
        'CUDA': cuda_available,
        'XGBoost': test_xgboost_gpu(),
        'LightGBM': test_lightgbm_gpu(),
        'CatBoost': test_catboost_gpu(),
        'AutoGluon': test_autogluon_gpu()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for library, status in results.items():
        status_symbol = "✓" if status else "✗"
        print(f"{status_symbol} {library}: {'GPU Ready' if status else 'GPU Not Available'}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if not results['CUDA']:
        print("⚠ CUDA not properly detected. Ensure:")
        print("  - NVIDIA GPU drivers are installed")
        print("  - CUDA toolkit is installed")
        print("  - nvidia-smi command works")
    
    failed_libraries = [lib for lib, status in results.items() if not status and lib != 'CUDA']
    
    if failed_libraries:
        print(f"\n⚠ The following libraries don't have GPU support: {', '.join(failed_libraries)}")
        print("\nTo enable GPU support, you may need to:")
        print("  - Install GPU-enabled versions: pip install xgboost[gpu] lightgbm catboost")
        print("  - For LightGBM: May need OpenCL or custom build")
        print("  - For AutoGluon: Ensure PyTorch with CUDA is installed")
    
    if all(results.values()):
        print("🎉 All libraries have GPU support! You can use --gpu flag safely.")
    
if __name__ == "__main__":
    main()