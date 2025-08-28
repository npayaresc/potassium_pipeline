#!/bin/bash
set -e

echo "[INFO] Docker entrypoint starting..."
echo "[INFO] Environment variables:"
echo "  STORAGE_TYPE: ${STORAGE_TYPE}"
echo "  STORAGE_BUCKET_NAME: ${STORAGE_BUCKET_NAME}"
echo "  CLOUD_STORAGE_PREFIX: ${CLOUD_STORAGE_PREFIX}"
echo "[INFO] Command to execute: $@"

# Cloud data download function
download_training_data() {
    if [[ "${STORAGE_TYPE}" == "gcs" && -n "${STORAGE_BUCKET_NAME}" ]]; then
        echo "[INFO] Downloading training data from GCS bucket: ${STORAGE_BUCKET_NAME}"
        
        # Install gsutil if not available
        if ! command -v gsutil &> /dev/null; then
            echo "[INFO] Installing Google Cloud SDK..."
            curl https://sdk.cloud.google.com | bash
            source $HOME/google-cloud-sdk/path.bash.inc
        fi
        
        # Set up service account authentication if available
        if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS}" && -f "${GOOGLE_APPLICATION_CREDENTIALS}" ]]; then
            echo "[INFO] Authenticating with service account..."
            gcloud auth activate-service-account --key-file="${GOOGLE_APPLICATION_CREDENTIALS}"
        fi
        
        # Create local data directories (ensure they exist)
        mkdir -p /app/data/{raw/data_5278_Phase3,processed,averaged_files_per_sample,cleansed_files_per_sample,reference_data}
        mkdir -p /app/{models/autogluon,reports,logs,bad_files,bad_prediction_files,catboost_info,configs}
        
        # Download data from GCS bucket
        local bucket_prefix="${CLOUD_STORAGE_PREFIX:-magnesium-pipeline}"
        
        echo "[INFO] Downloading data from gs://${STORAGE_BUCKET_NAME}/${bucket_prefix}/"
        
        # Download reference files (Excel) - first check for the specific file, then fall back to any Excel files
        if gsutil ls "gs://${STORAGE_BUCKET_NAME}/${bucket_prefix}/Final_Lab_Data_Nico_New.xlsx" &>/dev/null; then
            echo "[INFO] Downloading specific reference file: Final_Lab_Data_Nico_New.xlsx"
            gsutil cp "gs://${STORAGE_BUCKET_NAME}/${bucket_prefix}/Final_Lab_Data_Nico_New.xlsx" /app/data/reference_data/ || echo "[WARN] Failed to download Final_Lab_Data_Nico_New.xlsx"
        else
            echo "[INFO] Specific reference file not found, downloading all Excel files"
            gsutil -m cp "gs://${STORAGE_BUCKET_NAME}/${bucket_prefix}/*.xlsx" /app/data/reference_data/ 2>/dev/null || echo "[WARN] No .xlsx files found"
            gsutil -m cp "gs://${STORAGE_BUCKET_NAME}/${bucket_prefix}/*.xls" /app/data/reference_data/ 2>/dev/null || echo "[WARN] No .xls files found"
        fi
        
        # Also check in the reference_data subdirectory if it exists in GCS
        if gsutil ls "gs://${STORAGE_BUCKET_NAME}/${bucket_prefix}/reference_data/" &>/dev/null; then
            echo "[INFO] Found reference_data directory in GCS, downloading contents"
            gsutil -m rsync -r "gs://${STORAGE_BUCKET_NAME}/${bucket_prefix}/reference_data/" /app/data/reference_data/ 2>/dev/null || echo "[WARN] Failed to sync reference_data directory"
        fi
        
        # Download only raw data (averaged and cleansed are generated during training)
        gsutil -m rsync -r "gs://${STORAGE_BUCKET_NAME}/${bucket_prefix}/data_5278_Phase3/" /app/data/raw/data_5278_Phase3/ 2>/dev/null || echo "[WARN] Raw data not found"
        # Note: averaged_files_per_sample and cleansed_files_per_sample are generated during training, not downloaded
        
        echo "[INFO] Data download completed"
        
        # Verify critical files and directories
        echo "[INFO] Verifying downloaded data..."
        local verification_failed=false
        
        # Check for reference data (at least one Excel file)
        if ! ls /app/data/reference_data/*.xlsx /app/data/reference_data/*.xls 2>/dev/null | grep -q .; then
            echo "[ERROR] No Excel reference files found in /app/data/reference_data/"
            verification_failed=true
        else
            echo "[INFO] Found reference files: $(ls /app/data/reference_data/*.xlsx /app/data/reference_data/*.xls 2>/dev/null | wc -l) files"
        fi
        
        # Check for raw data directory
        if [ ! -d "/app/data/raw/data_5278_Phase3" ] || [ -z "$(ls -A /app/data/raw/data_5278_Phase3 2>/dev/null)" ]; then
            echo "[WARN] Raw data directory is empty or missing: /app/data/raw/data_5278_Phase3"
            # Not a critical failure if we have averaged or cleansed files
        fi
        
        # Check for raw data files (averaged and cleansed will be generated during training)
        local has_raw=$(ls /app/data/raw/data_5278_Phase3/*.txt 2>/dev/null | wc -l)
        
        if [ "$has_raw" -eq 0 ]; then
            echo "[ERROR] No raw data files found in /app/data/raw/data_5278_Phase3/"
            verification_failed=true
        else
            echo "[INFO] Found $has_raw raw data files"
        fi
        
        if [ "$verification_failed" = true ]; then
            echo "[ERROR] Critical data missing. Training may fail."
            echo "[INFO] Please ensure data is uploaded to GCS using: ./deploy/gcp_deploy.sh upload-data"
            # Don't exit here - let the training script handle the missing data with proper error messages
        else
            echo "[INFO] Data verification passed"
        fi
        
        # List downloaded files for verification
        echo "[INFO] Downloaded data structure:"
        find /app/data -type f | head -20
        
    else
        echo "[INFO] Not in cloud mode or no bucket specified. Using local data."
    fi
}

# Download models function for inference
download_models_for_inference() {
    if [[ "${STORAGE_TYPE}" == "gcs" && -n "${STORAGE_BUCKET_NAME}" ]]; then
        echo "[INFO] Downloading trained models from GCS for inference..."
        
        local bucket_prefix="${CLOUD_STORAGE_PREFIX:-magnesium-pipeline}"
        
        # Create models directory
        mkdir -p /app/models/autogluon
        
        # Download latest models from results directories
        echo "[INFO] Looking for trained models in gs://${STORAGE_BUCKET_NAME}/${bucket_prefix}/results/"
        
        # Find the most recent results directory
        local latest_result=$(gsutil ls "gs://${STORAGE_BUCKET_NAME}/${bucket_prefix}/results/" | sort -r | head -1)
        
        if [ -n "$latest_result" ]; then
            echo "[INFO] Found latest training results: $latest_result"
            
            # Download models from latest training run
            if gsutil ls "${latest_result}models/" &>/dev/null; then
                echo "[INFO] Downloading models from ${latest_result}models/"
                gsutil -m rsync -r "${latest_result}models/" /app/models/ || echo "[WARN] Failed to download some models"
            fi
            
            # Also try to download from the main models directory if it exists
            if gsutil ls "gs://${STORAGE_BUCKET_NAME}/${bucket_prefix}/models/" &>/dev/null; then
                echo "[INFO] Downloading additional models from main models directory"
                gsutil -m rsync -r "gs://${STORAGE_BUCKET_NAME}/${bucket_prefix}/models/" /app/models/ || echo "[WARN] Failed to download some models from main directory"
            fi
        else
            echo "[WARN] No training results found. Checking main models directory..."
            # Fallback: try main models directory
            if gsutil ls "gs://${STORAGE_BUCKET_NAME}/${bucket_prefix}/models/" &>/dev/null; then
                echo "[INFO] Downloading models from main models directory"
                gsutil -m rsync -r "gs://${STORAGE_BUCKET_NAME}/${bucket_prefix}/models/" /app/models/
            else
                echo "[ERROR] No trained models found in GCS bucket"
                echo "[ERROR] Please train a model first using: ./deploy/gcp_deploy.sh train"
                exit 1
            fi
        fi
        
        # Verify models were downloaded
        local model_count=$(find /app/models -name "*.pkl" -o -name "*.joblib" | wc -l)
        local autogluon_count=$(find /app/models -type d -name "*autogluon*" | wc -l)
        
        echo "[INFO] Downloaded models:"
        echo "  Standard models (.pkl/.joblib): $model_count"
        echo "  AutoGluon models: $autogluon_count"
        
        if [[ $model_count -eq 0 && $autogluon_count -eq 0 ]]; then
            echo "[ERROR] No models were downloaded successfully"
            exit 1
        fi
        
        # List available models for verification
        echo "[INFO] Available models for inference:"
        find /app/models -name "*.pkl" -o -name "*.joblib" -o -type d -name "*autogluon*" | head -10
        
    else
        echo "[INFO] Not in cloud mode or no bucket specified. Using local models."
        if [ ! -d "/app/models" ] || [ -z "$(ls -A /app/models 2>/dev/null)" ]; then
            echo "[WARN] No local models found in /app/models/"
        fi
    fi
}

# Upload results function
# Function to ensure logs are flushed
flush_logs() {
    echo "[INFO] Flushing logs and checking log status before upload..."
    
    # Force flush any buffered logs
    sync
    
    # Show current log directory status
    echo "[INFO] Log directory status:"
    if [[ -d "/app/logs" ]]; then
        ls -la /app/logs/
        echo "[INFO] Log file sizes:"
        find /app/logs -type f -exec wc -l {} \; 2>/dev/null | head -5
    else
        echo "[WARN] /app/logs directory not found"
    fi
    
    # Also check if there are any logs in the current directory
    echo "[INFO] Checking for logs in current working directory:"
    find /app -name "*.log" -type f -exec ls -la {} \; 2>/dev/null | head -5
}

upload_results() {
    # Flush logs before upload
    flush_logs
    
    if [[ "${STORAGE_TYPE}" == "gcs" && -n "${STORAGE_BUCKET_NAME}" ]]; then
        echo "[INFO] Uploading results to GCS bucket..."
        
        local bucket_prefix="${CLOUD_STORAGE_PREFIX:-magnesium-pipeline}"
        local timestamp=$(date +%Y%m%d_%H%M%S)
        local base_path="gs://${STORAGE_BUCKET_NAME}/${bucket_prefix}/results/${timestamp}"
        
        # Upload all output directories with content checks
        local uploaded_something=false
        
        # Core output directories
        if [[ -d "/app/models" ]] && [[ $(find /app/models -type f | wc -l) -gt 0 ]]; then
            echo "[INFO] Uploading models..."
            gsutil -q -m rsync -r /app/models/ "${base_path}/models/" 2>&1 | grep -v "Copying file" | grep -v "Done" || true
            uploaded_something=true
        fi
        
        if [[ -d "/app/reports" ]] && [[ $(find /app/reports -type f | wc -l) -gt 0 ]]; then
            echo "[INFO] Uploading reports..."
            gsutil -q -m rsync -r /app/reports/ "${base_path}/reports/" 2>&1 | grep -v "Copying file" | grep -v "Done" || true
            uploaded_something=true
        fi
        
        if [[ -d "/app/logs" ]]; then
            local log_count=$(find /app/logs -type f 2>/dev/null | wc -l)
            echo "[INFO] Found logs directory with ${log_count} files"
            if [[ $log_count -gt 0 ]]; then
                echo "[INFO] Uploading logs..."
                echo "[INFO] Log files found:"
                find /app/logs -type f -exec ls -la {} \; | head -10
                gsutil -q -m rsync -r /app/logs/ "${base_path}/logs/" 2>&1 | grep -v "Copying file" | grep -v "Done" || true
                if [[ $? -eq 0 ]]; then
                    echo "[SUCCESS] Logs uploaded to ${base_path}/logs/"
                    uploaded_something=true
                else
                    echo "[ERROR] Failed to upload logs"
                fi
            else
                echo "[WARN] Logs directory exists but no log files found"
                echo "[INFO] Contents of /app/logs:"
                ls -la /app/logs/ || echo "Directory is empty or inaccessible"
            fi
        else
            echo "[WARN] Logs directory /app/logs does not exist"
        fi
        
        # Additional output directories (only if they have content)
        if [[ -d "/app/bad_files" ]] && [[ $(find /app/bad_files -type f | wc -l) -gt 0 ]]; then
            echo "[INFO] Uploading rejected files..."
            gsutil -q -m rsync -r /app/bad_files/ "${base_path}/bad_files/" 2>&1 | grep -v "Copying file" | grep -v "Done" || true
            uploaded_something=true
        fi
        
        if [[ -d "/app/bad_prediction_files" ]] && [[ $(find /app/bad_prediction_files -type f | wc -l) -gt 0 ]]; then
            echo "[INFO] Uploading failed prediction files..."
            gsutil -q -m rsync -r /app/bad_prediction_files/ "${base_path}/bad_prediction_files/" 2>&1 | grep -v "Copying file" | grep -v "Done" || true
            uploaded_something=true
        fi
        
        if [[ -d "/app/catboost_info" ]] && [[ $(find /app/catboost_info -type f | wc -l) -gt 0 ]]; then
            echo "[INFO] Uploading CatBoost training info..."
            gsutil -q -m rsync -r /app/catboost_info/ "${base_path}/catboost_info/" 2>&1 | grep -v "Copying file" | grep -v "Done" || true
            uploaded_something=true
        fi
        
        if [[ -d "/app/configs" ]] && [[ $(find /app/configs -type f | wc -l) -gt 0 ]]; then
            echo "[INFO] Uploading saved configurations..."
            gsutil -q -m rsync -r /app/configs/ "${base_path}/configs/" 2>&1 | grep -v "Copying file" | grep -v "Done" || true
            uploaded_something=true
        fi
        
        # Upload processed data (intermediate results that might be useful)
        if [[ -d "/app/data/processed" ]] && [[ $(find /app/data/processed -type f | wc -l) -gt 0 ]]; then
            echo "[INFO] Uploading processed data splits..."
            gsutil -q -m rsync -r /app/data/processed/ "${base_path}/data/processed/" 2>&1 | grep -v "Copying file" | grep -v "Done" || true
            uploaded_something=true
        fi
        
        if [[ -d "/app/data/averaged_files_per_sample" ]] && [[ $(find /app/data/averaged_files_per_sample -type f | wc -l) -gt 0 ]]; then
            echo "[INFO] Uploading averaged files..."
            gsutil -q -m rsync -r /app/data/averaged_files_per_sample/ "${base_path}/data/averaged_files_per_sample/" 2>&1 | grep -v "Copying file" | grep -v "Done" || true
            uploaded_something=true
        fi
        
        if [[ -d "/app/data/cleansed_files_per_sample" ]] && [[ $(find /app/data/cleansed_files_per_sample -type f | wc -l) -gt 0 ]]; then
            echo "[INFO] Uploading cleansed files..."
            gsutil -q -m rsync -r /app/data/cleansed_files_per_sample/ "${base_path}/data/cleansed_files_per_sample/" 2>&1 | grep -v "Copying file" | grep -v "Done" || true
            uploaded_something=true
        fi
        
        if [[ "$uploaded_something" == "true" ]]; then
            echo "[SUCCESS] Results uploaded to ${base_path}/"
            echo "[INFO] Summary of uploaded directories:"
            gsutil ls "${base_path}/" 2>/dev/null || echo "[WARN] Could not list uploaded directories"
            echo "[INFO] Checking if logs were uploaded:"
            gsutil ls "${base_path}/logs/" 2>/dev/null && echo "[SUCCESS] Logs found in GCS" || echo "[WARN] No logs found in GCS"
            echo "[INFO] Full structure:"
            gsutil ls -r "${base_path}/" | head -30
        else
            echo "[WARN] No output files found to upload"
            echo "[INFO] This might be because:"
            echo "  - Training failed before generating outputs"
            echo "  - Output directories were not created"
            echo "  - Files were created in unexpected locations"
        fi
    fi
}

# Trap to upload results on exit
trap upload_results EXIT

# Main execution
echo "[INFO] Starting Magnesium Pipeline Container"
echo "[INFO] Storage Type: ${STORAGE_TYPE:-local}"
echo "[INFO] Bucket: ${STORAGE_BUCKET_NAME:-none}"
echo "[INFO] Logs will be written to: /app/logs/"
echo "[INFO] Current directory structure:"
ls -la /app/ | head -10
echo "[INFO] Environment: ${ENVIRONMENT:-development}"

# Download training data if in cloud mode
download_training_data

# Execute the passed command
echo "[INFO] Executing command: $@"
exec "$@"