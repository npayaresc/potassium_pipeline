@echo off
setlocal enabledelayedexpansion

REM Google Cloud Platform Deployment Script for Magnesium Pipeline (Windows)
REM This script provides multiple deployment options for GCP

REM Configuration
if not defined PROJECT_ID set "PROJECT_ID=mapana-ai-models"
if not defined REGION set "REGION=us-central1"
if not defined SERVICE_NAME set "SERVICE_NAME=magnesium-pipeline"
if not defined REPO_NAME set "REPO_NAME=magnesium-repo"
if not defined IMAGE_NAME set "IMAGE_NAME=magnesium-pipeline"
if not defined BUCKET_NAME set "BUCKET_NAME=%PROJECT_ID%-magnesium-data"

REM Cloud Configuration
if not defined CLOUD_CONFIG_FILE set "CLOUD_CONFIG_FILE=config/cloud_config.yml"
if not defined ENVIRONMENT set "ENVIRONMENT=production"
if not defined USE_CLOUD_CONFIG set "USE_CLOUD_CONFIG=true"

REM Training Configuration
if not defined TRAINING_MODE set "TRAINING_MODE=autogluon"
if not defined USE_GPU set "USE_GPU=true"
if not defined USE_RAW_SPECTRAL set "USE_RAW_SPECTRAL=false"
if not defined MODELS set "MODELS="
if not defined STRATEGY set "STRATEGY=full_context"
if not defined TRIALS set "TRIALS="
if not defined TIMEOUT set "TIMEOUT="
if not defined MACHINE_TYPE set "MACHINE_TYPE=n1-standard-4"
if not defined ACCELERATOR_TYPE set "ACCELERATOR_TYPE=NVIDIA_TESLA_T4"
if not defined ACCELERATOR_COUNT set "ACCELERATOR_COUNT=1"

echo [INFO] Google Cloud Platform Deployment Script (Windows)
echo.

REM Parse command
if "%1"=="" goto :show_help
if "%1"=="help" goto :show_help
if "%1"=="--help" goto :show_help
if "%1"=="-h" goto :show_help
if "%1"=="/?" goto :show_help

REM Route to appropriate function
if "%1"=="setup" goto :setup_project
if "%1"=="build" goto :build_image
if "%1"=="build-local" goto :build_image_local
if "%1"=="cloud-run" goto :deploy_cloud_run
if "%1"=="gke" goto :deploy_gke
if "%1"=="vertex-ai" goto :deploy_vertex_ai
if "%1"=="train" goto :train_vertex
if "%1"=="tune" goto :tune_vertex
if "%1"=="autogluon" goto :autogluon_vertex
if "%1"=="optimize" goto :optimize_vertex
if "%1"=="upload-data" goto :upload_data
if "%1"=="check-data" goto :check_data
if "%1"=="list-projects" goto :list_projects
if "%1"=="cleanup" goto :cleanup
if "%1"=="all" goto :deploy_all

echo [ERROR] Unknown command: %1
goto :show_help

:check_prerequisites
echo [INFO] Checking prerequisites...

REM Check if gcloud is installed
gcloud --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] gcloud CLI is not installed
    echo Please install Google Cloud SDK from https://cloud.google.com/sdk/docs/install
    exit /b 1
)

REM Check if logged in
gcloud auth list --filter=status:ACTIVE --format="value(account)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Not logged into gcloud. Run: gcloud auth login
    exit /b 1
)

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    exit /b 1
)

echo [INFO] Prerequisites check passed!
goto :eof

:setup_project
call :check_prerequisites
if errorlevel 1 exit /b 1

echo [INFO] Setting up GCP project...

REM Set project
gcloud config set project %PROJECT_ID%

REM Enable required APIs
echo [INFO] Enabling required APIs...
call gcloud services enable cloudbuild.googleapis.com
call gcloud services enable run.googleapis.com
call gcloud services enable container.googleapis.com
call gcloud services enable storage-api.googleapis.com
call gcloud services enable aiplatform.googleapis.com
call gcloud services enable artifactregistry.googleapis.com

REM Create storage bucket
echo [INFO] Creating storage bucket...
gsutil ls gs://%BUCKET_NAME% >nul 2>&1
if errorlevel 1 (
    gsutil mb -l %REGION% gs://%BUCKET_NAME%
    echo [INFO] Created bucket: gs://%BUCKET_NAME%
) else (
    echo [WARN] Bucket gs://%BUCKET_NAME% already exists
)

REM Create Artifact Registry repository
echo [INFO] Creating Artifact Registry repository...
gcloud artifacts repositories describe %REPO_NAME% --location=%REGION% >nul 2>&1
if errorlevel 1 (
    gcloud artifacts repositories create %REPO_NAME% ^
        --repository-format=docker ^
        --location=%REGION% ^
        --description="Docker repository for Magnesium Pipeline"
    echo [INFO] Created Artifact Registry repo: %REPO_NAME%
) else (
    echo [WARN] Artifact Registry repo %REPO_NAME% already exists
)

echo [INFO] Project setup completed!
goto :end

:build_image
call :check_prerequisites
if errorlevel 1 exit /b 1

echo [INFO] Building and pushing container image with Cloud Build...
set "IMAGE_URI=%REGION%-docker.pkg.dev/%PROJECT_ID%/%REPO_NAME%/%IMAGE_NAME%:latest"

REM Configure Docker authentication
gcloud auth configure-docker %REGION%-docker.pkg.dev

REM Build with Cloud Build
gcloud builds submit --tag %IMAGE_URI% .
if errorlevel 1 (
    echo [ERROR] Failed to build image
    exit /b 1
)

echo [INFO] Image built and pushed to %IMAGE_URI%
goto :end

:build_image_local
call :check_prerequisites
if errorlevel 1 exit /b 1

echo [INFO] Building container image locally with Docker...
set "IMAGE_URI=%REGION%-docker.pkg.dev/%PROJECT_ID%/%REPO_NAME%/%IMAGE_NAME%:latest"

REM Configure Docker authentication
gcloud auth configure-docker %REGION%-docker.pkg.dev

REM Build locally
docker build -t %IMAGE_URI% .
if errorlevel 1 (
    echo [ERROR] Failed to build image locally
    exit /b 1
)

REM Push to registry
echo [INFO] Pushing image to Artifact Registry...
docker push %IMAGE_URI%
if errorlevel 1 (
    echo [ERROR] Failed to push image
    exit /b 1
)

echo [INFO] Image built locally and pushed to %IMAGE_URI%
goto :end

:deploy_cloud_run
call :check_prerequisites
if errorlevel 1 exit /b 1

echo [INFO] Deploying to Cloud Run...
set "IMAGE_URI=%REGION%-docker.pkg.dev/%PROJECT_ID%/%REPO_NAME%/%IMAGE_NAME%:latest"

gcloud run deploy %SERVICE_NAME% ^
    --image %IMAGE_URI% ^
    --platform managed ^
    --region %REGION% ^
    --memory 8Gi ^
    --cpu 4 ^
    --timeout 3600 ^
    --concurrency 10 ^
    --max-instances 5 ^
    --allow-unauthenticated ^
    --set-env-vars="STORAGE_TYPE=gcs,STORAGE_BUCKET_NAME=%BUCKET_NAME%,GPU_ENABLED=false"

REM Get service URL
for /f "tokens=*" %%i in ('gcloud run services describe %SERVICE_NAME% --region=%REGION% --format="value(status.url)"') do set SERVICE_URL=%%i
echo [INFO] Service deployed at: %SERVICE_URL%
goto :end

:deploy_gke
call :check_prerequisites
if errorlevel 1 exit /b 1

echo [INFO] Deploying to GKE...
set "IMAGE_URI=%REGION%-docker.pkg.dev/%PROJECT_ID%/%REPO_NAME%/%IMAGE_NAME%:latest"
set "CLUSTER_NAME=%SERVICE_NAME%-cluster"

REM Create GKE cluster if it doesn't exist
gcloud container clusters describe %CLUSTER_NAME% --region=%REGION% >nul 2>&1
if errorlevel 1 (
    echo [INFO] Creating GKE cluster...
    gcloud container clusters create %CLUSTER_NAME% ^
        --region=%REGION% ^
        --num-nodes=1 ^
        --machine-type=n1-standard-4 ^
        --enable-autoscaling ^
        --min-nodes=0 ^
        --max-nodes=3 ^
        --disk-size=100GB
)

REM Get cluster credentials
gcloud container clusters get-credentials %CLUSTER_NAME% --region=%REGION%

REM Apply Kubernetes manifests (would need to create these)
echo [INFO] GKE deployment requires kubectl and manifest files
echo [INFO] Cluster created: %CLUSTER_NAME%
goto :end

:deploy_vertex_ai
call :check_prerequisites
if errorlevel 1 exit /b 1

echo [INFO] Deploying training job to Vertex AI...
set "IMAGE_URI=%REGION%-docker.pkg.dev/%PROJECT_ID%/%REPO_NAME%/%IMAGE_NAME%:latest"

REM Build training command
set "training_cmd=python main.py %TRAINING_MODE%"
if "%USE_GPU%"=="true" set "training_cmd=%training_cmd% --gpu"
if "%USE_RAW_SPECTRAL%"=="true" set "training_cmd=%training_cmd% --raw-spectral"
if not "%MODELS%"=="" set "training_cmd=%training_cmd% --models %MODELS%"
if not "%STRATEGY%"=="" set "training_cmd=%training_cmd% --strategy %STRATEGY%"
if not "%TRIALS%"=="" set "training_cmd=%training_cmd% --trials %TRIALS%"

echo [INFO] Training command: %training_cmd%

REM Create a temporary JSON config file for the job
set "CONFIG_FILE=%TEMP%\vertex-ai-job-%RANDOM%.json"
(
echo {
echo   "workerPoolSpecs": [
echo     {
echo       "machineSpec": {
echo         "machineType": "%MACHINE_TYPE%",
echo         "acceleratorType": "%ACCELERATOR_TYPE%",
echo         "acceleratorCount": %ACCELERATOR_COUNT%
echo       },
echo       "replicaCount": 1,
echo       "containerSpec": {
echo         "imageUri": "%IMAGE_URI%",
echo         "command": ["bash", "-c", "source /app/docker-entrypoint.sh && download_training_data && %training_cmd%"],
echo         "env": [
echo           {"name": "STORAGE_TYPE", "value": "gcs"},
echo           {"name": "STORAGE_BUCKET_NAME", "value": "%BUCKET_NAME%"},
echo           {"name": "CLOUD_STORAGE_PREFIX", "value": "magnesium-pipeline"},
echo           {"name": "ENVIRONMENT", "value": "%ENVIRONMENT%"}
echo         ]
echo       }
echo     }
echo   ]
echo }
) > "%CONFIG_FILE%"

REM Create training job using the config file
gcloud ai custom-jobs create ^
    --region=%REGION% ^
    --display-name="%SERVICE_NAME%-%TRAINING_MODE%-%date:~-4%%date:~4,2%%date:~7,2%-%time:~0,2%%time:~3,2%%time:~6,2%" ^
    --config="%CONFIG_FILE%"

REM Clean up temp file
del "%CONFIG_FILE%" 2>nul

echo [INFO] Vertex AI training job submitted with data download from GCS
echo [INFO] Monitor progress: gcloud ai custom-jobs list --region=%REGION%
goto :end

:train_vertex
set "TRAINING_MODE=train"
goto :deploy_vertex_ai

:tune_vertex
set "TRAINING_MODE=tune"
goto :deploy_vertex_ai

:autogluon_vertex
set "TRAINING_MODE=autogluon"
goto :deploy_vertex_ai

:optimize_vertex
set "TRAINING_MODE=optimize-models"
goto :deploy_vertex_ai

:upload_data
call :check_prerequisites
if errorlevel 1 exit /b 1

echo [INFO] Uploading training data to GCS bucket...
set "bucket_prefix=magnesium-pipeline"

REM Upload data directories
if exist "data\raw\data_5278_Phase3" (
    echo [INFO] Uploading raw spectral data...
    gsutil -m rsync -r -d "data\raw\data_5278_Phase3" "gs://%BUCKET_NAME%/%bucket_prefix%/data_5278_Phase3/"
)

if exist "data\averaged_files_per_sample" (
    echo [INFO] Uploading averaged files...
    gsutil -m rsync -r -d "data\averaged_files_per_sample" "gs://%BUCKET_NAME%/%bucket_prefix%/averaged_files_per_sample/"
)

if exist "data\cleansed_files_per_sample" (
    echo [INFO] Uploading cleansed files...
    gsutil -m rsync -r -d "data\cleansed_files_per_sample" "gs://%BUCKET_NAME%/%bucket_prefix%/cleansed_files_per_sample/"
)

REM Upload Excel files
for %%f in (data\*.xlsx data\*.xls) do (
    if exist "%%f" (
        echo [INFO] Uploading %%f...
        gsutil cp "%%f" "gs://%BUCKET_NAME%/%bucket_prefix%/"
    )
)

echo [INFO] Data upload completed
goto :end

:check_data
call :check_prerequisites
if errorlevel 1 exit /b 1

echo [INFO] Checking training data in GCS bucket...
set "bucket_prefix=magnesium-pipeline"

gsutil ls "gs://%BUCKET_NAME%/%bucket_prefix%/" >nul 2>&1
if errorlevel 1 (
    echo [WARN] No training data found in bucket
) else (
    echo [INFO] Training data found in bucket:
    gsutil ls -r "gs://%BUCKET_NAME%/%bucket_prefix%/" | more
)
goto :end

:list_projects
echo [INFO] Listing available GCP projects...
gcloud projects list --format="table(projectId,name,projectNumber)"
echo.
echo To use a project, set: set PROJECT_ID=your-project-id
goto :end

:cleanup
echo [WARN] This will delete all created resources
set /p confirm="Are you sure? (y/N): "
if /i not "%confirm%"=="y" (
    echo [INFO] Cleanup cancelled
    goto :end
)

echo [INFO] Cleaning up resources...

REM Delete Cloud Run service
gcloud run services delete %SERVICE_NAME% --region=%REGION% --quiet 2>nul

REM Delete GKE cluster
gcloud container clusters delete %SERVICE_NAME%-cluster --region=%REGION% --quiet 2>nul

REM Optional: Delete storage bucket
set /p delete_bucket="Delete storage bucket? (y/N): "
if /i "%delete_bucket%"=="y" (
    gsutil rm -r gs://%BUCKET_NAME% 2>nul
)

echo [INFO] Cleanup completed
goto :end

:deploy_all
call :setup_project
if errorlevel 1 exit /b 1
call :build_image
if errorlevel 1 exit /b 1
call :deploy_cloud_run
if errorlevel 1 exit /b 1
goto :end

:show_help
echo Google Cloud Platform Deployment Script for Magnesium Pipeline (Windows)
echo.
echo Usage: gcp-deploy.bat [COMMAND] [OPTIONS]
echo.
echo Infrastructure Commands:
echo   setup           Setup GCP project and enable APIs
echo   build           Build container image with Cloud Build
echo   build-local     Build container image locally with Docker
echo   cloud-run       Deploy to Cloud Run (serverless)
echo   gke             Deploy to Google Kubernetes Engine
echo   cleanup         Clean up all resources
echo   all             Run setup, build, and cloud-run
echo.
echo Training Commands (Vertex AI):
echo   vertex-ai       Submit training job to Vertex AI
echo   train           Train standard models on Vertex AI
echo   tune            Hyperparameter tuning on Vertex AI
echo   autogluon       AutoGluon training on Vertex AI
echo   optimize        Multi-model optimization on Vertex AI
echo.
echo Data Management:
echo   upload-data     Upload training data to GCS
echo   check-data      Check if training data exists in GCS
echo.
echo Utility Commands:
echo   list-projects   List available GCP projects
echo   help            Show this help message
echo.
echo Environment Variables:
echo   set PROJECT_ID=your-project-id
echo   set REGION=us-central1
echo   set CLOUD_CONFIG_FILE=config/staging_config.yml
echo   set ENVIRONMENT=staging
echo   set TRAINING_MODE=train
echo   set USE_GPU=true
echo   set USE_RAW_SPECTRAL=false
echo   set MODELS=xgboost,lightgbm
echo   set STRATEGY=full_context
echo   set TRIALS=100
echo.
echo Examples:
echo   REM Setup and deploy to Cloud Run
echo   gcp-deploy.bat all
echo.
echo   REM Train XGBoost with staging config
echo   set CLOUD_CONFIG_FILE=config/staging_config.yml
echo   set ENVIRONMENT=staging
echo   set MODELS=xgboost
echo   set USE_RAW_SPECTRAL=false
echo   gcp-deploy.bat train
echo.
echo   REM AutoGluon training
echo   gcp-deploy.bat autogluon
echo.
echo   REM Build locally and deploy
echo   gcp-deploy.bat build-local
echo   gcp-deploy.bat cloud-run
echo.
goto :end

:end
endlocal