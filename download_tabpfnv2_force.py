#!/usr/bin/env python
"""
Force download all TabPFNv2 models for AutoGluon.

This script ensures all TabPFNv2 model weights are downloaded,
even if some are already cached.
"""

import os
import sys
import logging
import shutil
import tempfile
from pathlib import Path
import urllib.request
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TabPFNv2 model URLs (these are the actual model locations)
TABPFN_MODEL_URLS = {
    'base': 'https://huggingface.co/PriorLabs/TabPFNv2/resolve/main/model.ckpt',
    'ensemble_1': 'https://github.com/PriorLabs/TabPFNv2/releases/download/v2.0/tabpfn-v2-regressor.ckpt',
    # Add more model URLs as they become available
}

# Alternative: HuggingFace hub downloads
HUGGINGFACE_MODELS = [
    'PriorLabs/TabPFNv2',
    'PriorLabs/TabPFN',
]

def get_cache_dir():
    """Get the TabPFN cache directory."""
    cache_dir = os.environ.get('TABPFN_CACHE_DIR')
    if not cache_dir:
        cache_dir = Path.home() / '.cache' / 'tabpfn'
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def download_file(url, destination):
    """Download a file from URL to destination with progress reporting."""
    try:
        logger.info(f"Downloading from: {url}")
        logger.info(f"Destination: {destination}")
        
        # Create a temporary file first
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
            # Download with progress
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024) if total_size > 0 else 0
                
                # Print progress every 10%
                if int(percent) % 10 == 0 or percent >= 100:
                    logger.info(f"Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            
            urllib.request.urlretrieve(url, tmp_path, reporthook=download_progress)
            
            # Move to final destination
            shutil.move(tmp_path, destination)
            logger.info(f"✓ Successfully downloaded to {destination}")
            return True
            
    except Exception as e:
        logger.error(f"✗ Failed to download {url}: {e}")
        return False

def download_via_huggingface_cli():
    """Use huggingface-cli to download models."""
    try:
        import subprocess
        
        logger.info("Attempting to download via huggingface-cli...")
        
        for model_name in HUGGINGFACE_MODELS:
            logger.info(f"Downloading {model_name}...")
            cmd = [
                'huggingface-cli', 'download',
                model_name,
                '--cache-dir', str(get_cache_dir().parent / 'huggingface'),
                '--local-dir', str(get_cache_dir() / model_name.replace('/', '_'))
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✓ Successfully downloaded {model_name}")
            else:
                logger.error(f"✗ Failed to download {model_name}: {result.stderr}")
                
    except Exception as e:
        logger.error(f"huggingface-cli not available: {e}")
        logger.info("Install with: pip install huggingface-hub")

def download_via_torch_hub():
    """Download TabPFNv2 models using torch.hub."""
    try:
        import torch
        
        logger.info("Attempting to download via torch.hub...")
        
        # Set torch hub directory
        torch_cache = get_cache_dir().parent / 'torch' / 'hub'
        torch_cache.mkdir(parents=True, exist_ok=True)
        os.environ['TORCH_HOME'] = str(torch_cache.parent)
        
        # Try to load TabPFNv2 (this should trigger download)
        try:
            model = torch.hub.load('PriorLabs/TabPFNv2', 'tabpfn_v2_regressor', trust_repo=True)
            logger.info("✓ Successfully downloaded TabPFNv2 via torch.hub")
        except Exception as e:
            logger.warning(f"Could not load via torch.hub: {e}")
            
    except ImportError:
        logger.warning("PyTorch not available for torch.hub download")

def download_autogluon_models():
    """Download models that AutoGluon uses with TabPFNv2."""
    try:
        from autogluon.core.models import AbstractModel
        from autogluon.tabular.models.tabpfnv2 import TabPFNv2Model
        
        logger.info("Attempting to download AutoGluon TabPFNv2 models...")
        
        # Create a temporary directory for the model
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Initialize TabPFNv2 model to trigger downloads
                model = TabPFNv2Model(
                    path=tmpdir,
                    name='tabpfnv2_temp',
                    problem_type='regression',
                    eval_metric='rmse'
                )
                
                # Create dummy data
                import pandas as pd
                import numpy as np
                X = pd.DataFrame(np.random.randn(100, 10))
                y = pd.Series(np.random.randn(100))
                
                # Fit model (this should download if needed)
                model.fit(X_train=X, y_train=y, time_limit=30)
                logger.info("✓ Successfully triggered AutoGluon TabPFNv2 download")
                
            except Exception as e:
                logger.error(f"Failed to initialize TabPFNv2Model: {e}")
                
    except ImportError as e:
        logger.error(f"AutoGluon TabPFNv2 model not available: {e}")

def clear_cache(cache_dir):
    """Clear existing cache to force re-download."""
    if cache_dir.exists():
        logger.warning(f"Clearing existing cache at {cache_dir}")
        response = input("Are you sure you want to clear the cache? (y/N): ")
        if response.lower() == 'y':
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache cleared")
            return True
    return False

def verify_tabpfn_installation():
    """Verify and attempt to properly install TabPFN."""
    try:
        import tabpfn
        logger.info(f"TabPFN is installed (version: {getattr(tabpfn, '__version__', 'unknown')})")
        
        # Try to import the actual model
        try:
            from tabpfn import TabPFNRegressor
            logger.info("TabPFNRegressor is available")
            
            # Try to get the model URLs or download function
            if hasattr(tabpfn, 'download_model'):
                logger.info("Found tabpfn.download_model function")
                tabpfn.download_model()
            elif hasattr(TabPFNRegressor, 'download_weights'):
                logger.info("Found TabPFNRegressor.download_weights method")
                TabPFNRegressor.download_weights()
            else:
                logger.warning("No explicit download function found in TabPFN")
                
        except ImportError as e:
            logger.error(f"Cannot import TabPFNRegressor: {e}")
            
    except ImportError:
        logger.error("TabPFN not installed!")
        logger.info("Installing TabPFN...")
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'tabpfn'], check=True)

def download_with_wget():
    """Download models using wget (system command)."""
    import subprocess
    
    cache_dir = get_cache_dir()
    
    # Known TabPFNv2 model URLs
    models = [
        {
            'url': 'https://github.com/PriorLabs/TabPFNv2/releases/download/v2.0.0/tabpfn-v2-regressor.ckpt',
            'name': 'tabpfn-v2-regressor.ckpt'
        },
        {
            'url': 'https://github.com/automl/TabPFN/releases/download/v2.0.0/tabpfn_models.tar.gz',
            'name': 'tabpfn_models.tar.gz'
        }
    ]
    
    for model in models:
        output_path = cache_dir / model['name']
        if output_path.exists():
            logger.info(f"Model already exists: {output_path}")
            continue
            
        logger.info(f"Downloading {model['name']} with wget...")
        cmd = ['wget', '-O', str(output_path), model['url']]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logger.info(f"✓ Downloaded {model['name']}")
                
                # Extract if it's a tar.gz
                if model['name'].endswith('.tar.gz'):
                    logger.info(f"Extracting {model['name']}...")
                    subprocess.run(['tar', '-xzf', str(output_path), '-C', str(cache_dir)], check=True)
                    logger.info(f"✓ Extracted {model['name']}")
            else:
                logger.error(f"wget failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.error(f"Download timed out for {model['name']}")
        except FileNotFoundError:
            logger.error("wget not found. Install with: sudo apt-get install wget")
            break

def main():
    """Main function to orchestrate downloads."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Force download TabPFNv2 models for AutoGluon"
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear existing cache before downloading'
    )
    parser.add_argument(
        '--method',
        choices=['all', 'direct', 'huggingface', 'torch', 'autogluon', 'wget'],
        default='all',
        help='Download method to use'
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("TabPFNv2 Model Force Download Script")
    logger.info("="*60)
    
    cache_dir = get_cache_dir()
    logger.info(f"Cache directory: {cache_dir}")
    
    # Clear cache if requested
    if args.clear_cache:
        clear_cache(cache_dir)
    
    # Check current cache
    logger.info("\nChecking current cache...")
    existing_files = list(cache_dir.glob('*.ckpt')) + list(cache_dir.glob('*.pt')) + list(cache_dir.glob('*.pth'))
    if existing_files:
        logger.info(f"Found {len(existing_files)} model files:")
        for f in existing_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"  - {f.name}: {size_mb:.2f} MB")
    else:
        logger.warning("No model files found in cache")
    
    # Verify TabPFN installation
    logger.info("\nVerifying TabPFN installation...")
    verify_tabpfn_installation()
    
    # Perform downloads based on method
    logger.info(f"\nStarting downloads with method: {args.method}")
    
    if args.method in ['all', 'direct']:
        logger.info("\n" + "="*40)
        logger.info("DIRECT DOWNLOAD")
        logger.info("="*40)
        for name, url in TABPFN_MODEL_URLS.items():
            dest = cache_dir / f"tabpfn-{name}.ckpt"
            if not dest.exists():
                download_file(url, dest)
            else:
                logger.info(f"Already exists: {dest}")
    
    if args.method in ['all', 'wget']:
        logger.info("\n" + "="*40)
        logger.info("WGET DOWNLOAD")
        logger.info("="*40)
        download_with_wget()
    
    if args.method in ['all', 'huggingface']:
        logger.info("\n" + "="*40)
        logger.info("HUGGINGFACE DOWNLOAD")
        logger.info("="*40)
        download_via_huggingface_cli()
    
    if args.method in ['all', 'torch']:
        logger.info("\n" + "="*40)
        logger.info("TORCH HUB DOWNLOAD")
        logger.info("="*40)
        download_via_torch_hub()
    
    if args.method in ['all', 'autogluon']:
        logger.info("\n" + "="*40)
        logger.info("AUTOGLUON DOWNLOAD")
        logger.info("="*40)
        download_autogluon_models()
    
    # Final check
    logger.info("\n" + "="*60)
    logger.info("FINAL CACHE STATUS")
    logger.info("="*60)
    
    final_files = list(cache_dir.glob('*.ckpt')) + list(cache_dir.glob('*.pt')) + list(cache_dir.glob('*.pth'))
    if final_files:
        total_size = sum(f.stat().st_size for f in final_files) / (1024 * 1024)
        logger.info(f"✓ Found {len(final_files)} model files (Total: {total_size:.2f} MB)")
        for f in final_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"  - {f.name}: {size_mb:.2f} MB")
    else:
        logger.warning("✗ No model files found after download attempts")
        logger.info("\nTroubleshooting:")
        logger.info("1. Check internet connection")
        logger.info("2. Try manual download from: https://github.com/PriorLabs/TabPFNv2")
        logger.info("3. Install with: pip install tabpfn --upgrade")
        logger.info("4. Check if models are in a different location:")
        logger.info("   find ~/.cache -name '*.ckpt' 2>/dev/null")

if __name__ == "__main__":
    main()
