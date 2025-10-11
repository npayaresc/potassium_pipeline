# TabPFNv2 Model Download Script

## Overview
This script pre-downloads all TabPFNv2 regressor models to the cache that AutoGluon uses, avoiding the need to download them during model training. This is particularly useful for:
- Offline environments
- CI/CD pipelines
- Docker containers
- Cloud deployments where you want to pre-cache models

## Features
- Downloads all TabPFNv2 model variants (1, 4, 8, 16, 32 estimators)
- Two download methods: Direct TabPFN and via AutoGluon
- Verifies cached models
- Reports cache sizes and locations
- Handles multiple cache directories (AutoGluon, HuggingFace, TabPFN)

## Requirements
```bash
pip install autogluon tabpfn
```

## Usage

### Download all models (recommended)
```bash
python download_tabpfnv2_models.py --method both
```

### Download using specific method
```bash
# Direct TabPFN method only
python download_tabpfnv2_models.py --method direct

# AutoGluon method only
python download_tabpfnv2_models.py --method autogluon
```

### Verify existing cache
```bash
python download_tabpfnv2_models.py --verify-only
```

## Cache Locations
By default, models are cached in:
- AutoGluon: `~/.autogluon/cache`
- HuggingFace: `~/.cache/huggingface`
- TabPFN: `~/.cache/tabpfn`

### Custom Cache Directories
You can specify custom cache directories using environment variables:
```bash
export AUTOGLUON_CACHE_DIR=/path/to/autogluon/cache
export HF_HOME=/path/to/huggingface/cache
export TABPFN_CACHE_DIR=/path/to/tabpfn/cache
```

## Docker Integration
Add this to your Dockerfile to pre-cache models:
```dockerfile
# Install dependencies
RUN pip install autogluon tabpfn

# Copy download script
COPY download_tabpfnv2_models.py /tmp/

# Download models to cache
RUN python /tmp/download_tabpfnv2_models.py --method both

# Optional: Set cache directories
ENV AUTOGLUON_CACHE_DIR=/app/.cache/autogluon
ENV HF_HOME=/app/.cache/huggingface
ENV TABPFN_CACHE_DIR=/app/.cache/tabpfn
```

## What Gets Downloaded
The script downloads TabPFNv2 models with the following configurations:
- **n_estimators=1**: Single model (fastest, least accurate)
- **n_estimators=4**: Small ensemble
- **n_estimators=8**: Medium ensemble
- **n_estimators=16**: Large ensemble
- **n_estimators=32**: Extra large ensemble (slowest, most accurate)

Total cache size: ~1.2 GB (including related models)

## Integration with AutoGluon
When using AutoGluon with TabPFNv2, the models will be loaded from cache:
```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label='target')
predictor.fit(
    train_data,
    hyperparameters={'TABPFNV2': {}},  # Will use cached models
    presets='extreme',  # Or any preset that includes TabPFNv2
)
```

## Troubleshooting

### Models not downloading
- Check internet connection
- Verify pip packages are installed: `pip install autogluon tabpfn`
- Check disk space (need ~1.2 GB free)

### Cache not found
- Verify cache directories exist and have write permissions
- Check environment variables are set correctly
- Use `--verify-only` to check current cache status

### AutoGluon not finding models
- Ensure you're using the correct model name: `TABPFNV2` (not `TABPFN`)
- Check AutoGluon version supports TabPFNv2: `autogluon>=1.4.0`

## Output Example
```
============================================================
TabPFNv2 Model Download Script
============================================================
AutoGluon version: 1.4.0
TabPFN version: 2.1.3
autogluon cache directory: /home/user/.autogluon/cache
huggingface cache directory: /home/user/.cache/huggingface
tabpfn cache directory: /home/user/.cache/tabpfn

============================================================
DIRECT DOWNLOAD METHOD
============================================================
Starting TabPFNv2 model downloads...
Will download 5 model configurations

[1/5] Downloading TabPFNv2 with config: {'n_estimators': 1, 'device': 'cpu'}
✓ Successfully initialized TabPFNv2 with 1 estimators (took 0.07s)
...

============================================================
FINAL SUMMARY
============================================================
✓ All downloads completed successfully!
  Total successful: 6

Total cache size: 1148.01 MB
Total model files found: 4
```

## Notes
- The script uses CPU device by default for compatibility
- GPU models will be downloaded separately if needed during training
- Models are shared across all AutoGluon predictors on the same machine
- Cache persists across Python sessions and restarts

## Citation
If you use TabPFNv2 in your research, please cite:
```bibtex
@article{tabpfnv2,
  title={TabPFNv2: A Foundation Model for Tabular Data},
  author={PriorLabs Team},
  year={2025}
}
```
