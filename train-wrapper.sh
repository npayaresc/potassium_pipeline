#!/bin/bash
# Training wrapper script that ensures data is downloaded from GCS before training
set -e

echo "[INFO] Training wrapper starting..."
echo "[INFO] Environment:"
echo "  STORAGE_TYPE: ${STORAGE_TYPE}"
echo "  STORAGE_BUCKET_NAME: ${STORAGE_BUCKET_NAME}"
echo "  CLOUD_STORAGE_PREFIX: ${CLOUD_STORAGE_PREFIX:-magnesium-pipeline}"

# Download data from GCS if in cloud mode
if [[ "${STORAGE_TYPE}" == "gcs" && -n "${STORAGE_BUCKET_NAME}" ]]; then
    echo "[INFO] Cloud mode detected, downloading training data..."
    
    # Source the download function from entrypoint
    source /app/docker-entrypoint.sh
    download_training_data
else
    echo "[INFO] Local mode or no bucket specified, using existing data"
fi

# Execute the training command
echo "[INFO] Starting training: $@"
exec "$@"