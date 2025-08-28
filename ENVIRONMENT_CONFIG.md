# Environment Configuration

The pipeline automatically detects whether it's running locally or in a container environment (like GCP) and adjusts paths accordingly.

## Automatic Detection

The configuration system uses the following logic:

1. **Container Environment (GCP)**: If `/app` directory exists and is writable → uses `/app` as base path
2. **Environment Override**: If `PIPELINE_ROOT` environment variable is set → uses that path
3. **Local Development**: Otherwise → uses the project root directory

## Usage Examples

### Local Development (Default)
```bash
# Runs with project root as base path
python main.py train --feature-parallel
```

### Container Environment (GCP)
```bash
# Automatically detects /app and uses container paths
python main.py train --feature-parallel
```

### Custom Path Override
```bash
# Use custom base path
export PIPELINE_ROOT=/custom/path
python main.py train --parallel
```

## Directory Structure

The pipeline creates the following structure relative to the base path:

```
<base_path>/
├── data/
│   ├── raw/data_5278_Phase3/
│   ├── averaged_files_per_sample/
│   ├── cleansed_files_per_sample/
│   ├── processed/
│   └── reference_data/
├── models/
├── reports/
├── logs/
├── bad_files/
└── bad_prediction_files/
```

## Benefits

- **No code changes** needed when deploying to different environments
- **Backwards compatible** with existing configurations
- **Flexible** - can be overridden if needed
- **Automatic** - works out of the box

## Troubleshooting

If you see path-related errors:

1. Check the detected base path in the logs: `[CONFIG] Detected ... using base path: ...`
2. For custom deployments, set `PIPELINE_ROOT` environment variable
3. Ensure the base directory has write permissions
4. For containers, make sure `/app` exists and is writable