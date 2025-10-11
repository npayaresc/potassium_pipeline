#!/usr/bin/env python
"""
Download all TabPFNv2 regressors to AutoGluon's cache.

This script pre-downloads all TabPFNv2 model variants to the cache that AutoGluon uses,
avoiding the need to download them during model training.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import json
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    missing_deps = []
    
    try:
        import autogluon
        try:
            logger.info(f"AutoGluon version: {autogluon.__version__}")
        except AttributeError:
            # Some AutoGluon versions don't have __version__
            logger.info("AutoGluon is installed (version unknown)")
    except ImportError:
        missing_deps.append("autogluon")
    
    try:
        import tabpfn
        try:
            logger.info(f"TabPFN version: {tabpfn.__version__}")
        except AttributeError:
            # Some TabPFN versions don't have __version__
            logger.info("TabPFN is installed (version unknown)")
    except ImportError:
        missing_deps.append("tabpfn")
    
    if missing_deps:
        logger.error(f"Missing required dependencies: {', '.join(missing_deps)}")
        logger.info("Please install them with: pip install autogluon tabpfn")
        return False
    
    return True

def get_cache_directories():
    """Get the cache directories used by AutoGluon and TabPFN."""
    cache_dirs = {}
    
    # AutoGluon cache
    ag_cache = os.environ.get('AUTOGLUON_CACHE_DIR')
    if not ag_cache:
        ag_cache = Path.home() / '.autogluon' / 'cache'
    else:
        ag_cache = Path(ag_cache)
    cache_dirs['autogluon'] = ag_cache
    
    # TabPFN cache (typically uses HuggingFace cache)
    hf_cache = os.environ.get('HF_HOME')
    if not hf_cache:
        hf_cache = Path.home() / '.cache' / 'huggingface'
    else:
        hf_cache = Path(hf_cache)
    cache_dirs['huggingface'] = hf_cache
    
    # TabPFN specific cache
    tabpfn_cache = os.environ.get('TABPFN_CACHE_DIR')
    if not tabpfn_cache:
        tabpfn_cache = Path.home() / '.cache' / 'tabpfn'
    else:
        tabpfn_cache = Path(tabpfn_cache)
    cache_dirs['tabpfn'] = tabpfn_cache
    
    return cache_dirs

def ensure_cache_dirs_exist(cache_dirs):
    """Ensure all cache directories exist."""
    for name, path in cache_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"{name} cache directory: {path}")

def download_tabpfnv2_models():
    """Download all TabPFNv2 model variants."""
    try:
        from tabpfn import TabPFNRegressor
        
        # List of TabPFNv2 model configurations to download
        # These are the typical ensemble sizes used by AutoGluon
        model_configs = [
            {'n_estimators': 1, 'device': 'cpu'},
            {'n_estimators': 4, 'device': 'cpu'},
            {'n_estimators': 8, 'device': 'cpu'},
            {'n_estimators': 16, 'device': 'cpu'},
            {'n_estimators': 32, 'device': 'cpu'},
        ]
        
        logger.info("Starting TabPFNv2 model downloads...")
        logger.info(f"Will download {len(model_configs)} model configurations")
        
        downloaded_models = []
        failed_models = []
        
        for i, config in enumerate(model_configs, 1):
            try:
                logger.info(f"\n[{i}/{len(model_configs)}] Downloading TabPFNv2 with config: {config}")
                
                # Create a dummy dataset for initialization
                import numpy as np
                X_dummy = np.random.randn(10, 5)
                y_dummy = np.random.randn(10)
                
                # Initialize the model (this triggers download if not cached)
                start_time = time.time()
                model = TabPFNRegressor(**config)
                
                # Fit on dummy data to ensure model is fully initialized
                model.fit(X_dummy, y_dummy)
                
                elapsed_time = time.time() - start_time
                logger.info(f"✓ Successfully initialized TabPFNv2 with {config['n_estimators']} estimators (took {elapsed_time:.2f}s)")
                
                downloaded_models.append(config)
                
            except Exception as e:
                logger.error(f"✗ Failed to download TabPFNv2 with config {config}: {str(e)}")
                failed_models.append((config, str(e)))
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("="*60)
        logger.info(f"Successfully downloaded: {len(downloaded_models)}/{len(model_configs)} models")
        
        if downloaded_models:
            logger.info("\nSuccessfully downloaded models:")
            for config in downloaded_models:
                logger.info(f"  - n_estimators={config['n_estimators']}, device={config['device']}")
        
        if failed_models:
            logger.warning("\nFailed downloads:")
            for config, error in failed_models:
                logger.warning(f"  - {config}: {error}")
        
        return len(downloaded_models), len(failed_models)
        
    except ImportError as e:
        logger.error(f"Failed to import TabPFN: {e}")
        logger.info("Please install TabPFN: pip install tabpfn")
        return 0, len(model_configs)

def download_via_autogluon():
    """Alternative method: Download TabPFNv2 models via AutoGluon."""
    try:
        from autogluon.tabular import TabularPredictor
        import pandas as pd
        import numpy as np
        import tempfile
        import shutil
        
        logger.info("\nAttempting to download TabPFNv2 via AutoGluon...")
        
        # Create a temporary directory for the model
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy data
            n_samples = 100
            n_features = 10
            X = np.random.randn(n_samples, n_features)
            y = np.random.randn(n_samples)
            
            # Create DataFrame
            df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
            df['target'] = y
            
            # Train with TabPFNv2 only
            logger.info("Training AutoGluon with TabPFNv2 model only...")
            predictor = TabularPredictor(
                label='target',
                path=tmpdir,
                verbosity=2
            )
            
            # Fit with only TabPFNv2
            predictor.fit(
                df,
                hyperparameters={'TABPFNV2': {}},
                time_limit=60,  # Quick training just to trigger download
                excluded_model_types=['GBM', 'CAT', 'XGB', 'RF', 'XT', 'KNN', 'NN_TORCH', 'LR', 'FASTAI']
            )
            
            logger.info("✓ Successfully triggered TabPFNv2 download via AutoGluon")
            return True
            
    except Exception as e:
        logger.error(f"Failed to download TabPFNv2 via AutoGluon: {e}")
        return False

def verify_downloads(cache_dirs):
    """Verify that models have been downloaded to cache."""
    logger.info("\n" + "="*60)
    logger.info("VERIFYING CACHE CONTENTS")
    logger.info("="*60)
    
    total_size = 0
    model_files = []
    
    for name, path in cache_dirs.items():
        if path.exists():
            logger.info(f"\n{name} cache ({path}):")
            
            # Look for model files
            patterns = ['*.pt', '*.pth', '*.bin', '*.safetensors', '*.pkl', '*.ckpt']
            for pattern in patterns:
                for file in path.rglob(pattern):
                    size_mb = file.stat().st_size / (1024 * 1024)
                    total_size += size_mb
                    model_files.append(file)
                    # Only show files > 1MB (likely model files)
                    if size_mb > 1:
                        logger.info(f"  - {file.relative_to(path)}: {size_mb:.2f} MB")
        else:
            logger.warning(f"{name} cache directory does not exist: {path}")
    
    logger.info(f"\nTotal cache size: {total_size:.2f} MB")
    logger.info(f"Total model files found: {len(model_files)}")
    
    return len(model_files) > 0

def main():
    """Main function to orchestrate the download process."""
    parser = argparse.ArgumentParser(
        description="Download all TabPFNv2 regressors to AutoGluon's cache"
    )
    parser.add_argument(
        '--method',
        choices=['direct', 'autogluon', 'both'],
        default='both',
        help='Method to use for downloading models'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing cache contents without downloading'
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("TabPFNv2 Model Download Script")
    logger.info("="*60)
    
    # Check dependencies
    if not args.verify_only and not check_dependencies():
        sys.exit(1)
    
    # Get and create cache directories
    cache_dirs = get_cache_directories()
    ensure_cache_dirs_exist(cache_dirs)
    
    if args.verify_only:
        # Just verify existing downloads
        success = verify_downloads(cache_dirs)
        if success:
            logger.info("\n✓ Models found in cache")
        else:
            logger.warning("\n✗ No models found in cache")
        sys.exit(0 if success else 1)
    
    # Perform downloads
    total_success = 0
    total_failed = 0
    
    if args.method in ['direct', 'both']:
        logger.info("\n" + "="*60)
        logger.info("DIRECT DOWNLOAD METHOD")
        logger.info("="*60)
        success, failed = download_tabpfnv2_models()
        total_success += success
        total_failed += failed
    
    if args.method in ['autogluon', 'both']:
        logger.info("\n" + "="*60)
        logger.info("AUTOGLUON DOWNLOAD METHOD")
        logger.info("="*60)
        if download_via_autogluon():
            total_success += 1
        else:
            total_failed += 1
    
    # Verify downloads
    verify_downloads(cache_dirs)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    
    if total_failed == 0:
        logger.info("✓ All downloads completed successfully!")
        logger.info(f"  Total successful: {total_success}")
    else:
        logger.warning(f"⚠ Some downloads failed!")
        logger.warning(f"  Successful: {total_success}")
        logger.warning(f"  Failed: {total_failed}")
    
    # Set environment variable hint for future runs
    logger.info("\nTo use custom cache directories, set these environment variables:")
    logger.info("  export AUTOGLUON_CACHE_DIR=/path/to/autogluon/cache")
    logger.info("  export HF_HOME=/path/to/huggingface/cache")
    logger.info("  export TABPFN_CACHE_DIR=/path/to/tabpfn/cache")
    
    sys.exit(0 if total_failed == 0 else 1)

if __name__ == "__main__":
    main()
