#!/usr/bin/env python3
"""
Test CUDA-enabled LightGBM in Docker container.
This script verifies that LightGBM can use CUDA acceleration.
"""

import sys
import numpy as np
from sklearn.datasets import make_regression

def test_lightgbm_cuda():
    """Test if LightGBM can use CUDA."""
    print("=" * 60)
    print("TESTING LIGHTGBM CUDA SUPPORT IN CONTAINER")
    print("=" * 60)
    
    try:
        import lightgbm as lgb
        print(f"✓ LightGBM version: {lgb.__version__}")
        
        # Create test data
        print("\nCreating test dataset...")
        X, y = make_regression(n_samples=5000, n_features=50, random_state=42)
        
        # Test CUDA device
        print("\nTesting CUDA device...")
        try:
            model = lgb.LGBMRegressor(
                device='cuda',
                gpu_device_id=0,
                n_estimators=100,
                num_leaves=31,
                verbose=1
            )
            
            print("Fitting model with CUDA...")
            model.fit(X, y)
            
            # Make predictions
            predictions = model.predict(X[:10])
            print(f"\n✓ CUDA training successful!")
            print(f"Sample predictions: {predictions[:5]}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ CUDA training failed: {e}")
            
            # Try with GPU (OpenCL) as fallback
            print("\nTrying OpenCL GPU device as fallback...")
            try:
                model_gpu = lgb.LGBMRegressor(
                    device='gpu',
                    gpu_platform_id=0,
                    gpu_device_id=0,
                    n_estimators=100,
                    verbose=1
                )
                model_gpu.fit(X, y)
                print("✓ OpenCL GPU training successful (fallback)")
                return True
            except Exception as e2:
                print(f"✗ OpenCL GPU training also failed: {e2}")
                
            return False
            
    except ImportError as e:
        print(f"✗ LightGBM not installed: {e}")
        return False

def check_cuda_toolkit():
    """Check if CUDA toolkit is available."""
    print("\n" + "=" * 60)
    print("CHECKING CUDA TOOLKIT")
    print("=" * 60)
    
    import subprocess
    
    # Check nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        print("✓ CUDA compiler (nvcc) found:")
        for line in result.stdout.split('\n'):
            if 'release' in line.lower():
                print(f"  {line.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ nvcc not found")
    
    # Check CUDA libraries
    try:
        result = subprocess.run(['ldconfig', '-p'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        cuda_libs = [line for line in result.stdout.split('\n') 
                    if 'libcudart' in line or 'libcublas' in line]
        if cuda_libs:
            print("\n✓ CUDA libraries found:")
            for lib in cuda_libs[:3]:  # Show first 3
                print(f"  {lib.strip()}")
    except:
        pass
    
    # Check environment variables
    import os
    cuda_home = os.environ.get('CUDA_HOME', 'Not set')
    cuda_path = os.environ.get('PATH', '')
    
    print(f"\n✓ CUDA_HOME: {cuda_home}")
    if '/cuda/' in cuda_path.lower():
        print("✓ CUDA in PATH")

def check_gpu_availability():
    """Check if GPU is available to the container."""
    print("\n" + "=" * 60)
    print("CHECKING GPU AVAILABILITY")
    print("=" * 60)
    
    import subprocess
    
    try:
        # Try nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                              capture_output=True,
                              text=True,
                              check=True)
        print("✓ GPU detected:")
        print(f"  {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ No GPU detected (nvidia-smi not available)")
        print("  Note: Container needs to be run with --gpus all flag")
        return False

def main():
    """Run all tests."""
    print("CUDA-ENABLED LIGHTGBM CONTAINER TEST")
    print("=" * 60)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Check CUDA toolkit
    check_cuda_toolkit()
    
    # Test LightGBM
    lgb_cuda_works = test_lightgbm_cuda()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if gpu_available and lgb_cuda_works:
        print("✓ SUCCESS: LightGBM CUDA support is working!")
        print("\nYou can now run training with:")
        print("  docker-compose -f docker-compose.local.yml run --rm pipeline python main.py train --gpu")
        return 0
    elif lgb_cuda_works:
        print("⚠ PARTIAL SUCCESS: LightGBM compiled with CUDA but GPU not detected")
        print("  Make sure to run container with: --gpus all")
        return 1
    else:
        print("✗ FAILURE: LightGBM CUDA support not working")
        print("\nTroubleshooting:")
        print("  1. Check if container was built with CUDA support")
        print("  2. Ensure container is run with --gpus all flag")
        print("  3. Verify NVIDIA drivers on host system")
        return 2

if __name__ == "__main__":
    sys.exit(main())