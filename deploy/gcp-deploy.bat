@echo off
setlocal enabledelayedexpansion

REM Google Cloud Platform Deployment Script for Magnesium Pipeline
REM This script provides multiple deployment options for GCP
REM
REM IMPORTANT: AutoGluon uses Ray for distributed machine learning.
REM Ray is fully compatible with Vertex AI custom training jobs.
REM For optimal performance in production, consider multi-node Ray clusters.

REM Configuration
if not defined PROJECT_ID set "PROJECT_ID=mapana-ai-models"
if not defined REGION set "REGION=us-central1"
if not defined SERVICE_NAME set "SERVICE_NAME=magnesium-pipeline"
if not defined REPO_NAME set "REPO_NAME=magnesium-repo"
if not defined IMAGE_NAME set "IMAGE_NAME=magnesium-pipeline"
if not defined BUCKET_NAME set "BUCKET_NAME=%PROJECT_ID%-magnesium-data"
if not defined AUTO_COMMIT_DEPLOYMENT set "AUTO_COMMIT_DEPLOYMENT=true"
if not defined AUTO_PUSH_DEPLOYMENT set "AUTO_PUSH_DEPLOYMENT=false"

REM Cloud Configuration
if not defined CLOUD_CONFIG_FILE set "CLOUD_CONFIG_FILE=config/cloud_config.yml"
if not defined ENVIRONMENT set "ENVIRONMENT=production"
if not defined USE_CLOUD_CONFIG set "USE_CLOUD_CONFIG=true"

REM Training Configuration - will be initialized from Python config if not set
if not defined TRAINING_MODE set "TRAINING_MODE="
if not defined USE_GPU set "USE_GPU="
if not defined USE_RAW_SPECTRAL set "USE_RAW_SPECTRAL="
if not defined MODELS set "MODELS="
if not defined STRATEGY set "STRATEGY="
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
if "%1"=="inference" goto :deploy_inference_service
if "%1"=="gke" goto :deploy_gke
if "%1"=="vertex-ai" goto :deploy_vertex_ai
if "%1"=="train" goto :train_vertex
if "%1"=="tune" goto :tune_vertex
if "%1"=="autogluon" goto :autogluon_vertex
if "%1"=="optimize" goto :optimize_vertex
if "%1"=="config-help" goto :show_config_help
if "%1"=="generate-config" goto :generate_cloud_config
if "%1"=="list-projects" goto :list_gcp_projects
if "%1"=="test-config" goto :test_config
if "%1"=="upload-data" goto :upload_training_data
if "%1"=="check-data" goto :check_data_exists
if "%1"=="list-data" goto :list_and_verify_gcs_data
if "%1"=="upload-logs" goto :upload_logs
if "%1"=="cleanup" goto :cleanup
if "%1"=="all" goto :deploy_all

echo [ERROR] Unknown command: %1
goto :show_help

:get_python_config_defaults
REM Get default values from Python pipeline_config.py
python -c "import sys; sys.exit(0)" >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=1,2 delims==" %%a in ('python -c "
import sys
import os
sys.path.insert(0, '.')
try:
    from src.config.pipeline_config import config
    
    print(f'PYTHON_USE_GPU={str(config.use_gpu).lower()}')
    print(f'PYTHON_USE_RAW_SPECTRAL={str(config.use_raw_spectral_data).lower()}')
    print(f'PYTHON_DEFAULT_STRATEGY=full_context')
    print(f'PYTHON_TRAINING_MODE=autogluon')
except Exception as e:
    pass
" 2^>nul') do (
        set "%%a=%%b"
    )
    
    if defined PYTHON_USE_GPU (
        echo [INFO] Loaded defaults from Python config:
        echo   USE_GPU default: !PYTHON_USE_GPU!
        echo   USE_RAW_SPECTRAL default: !PYTHON_USE_RAW_SPECTRAL!
        echo   STRATEGY default: !PYTHON_DEFAULT_STRATEGY!
        echo   TRAINING_MODE default: !PYTHON_TRAINING_MODE!
    )
)
goto :eof

:load_cloud_config
REM Save command-line values if they were explicitly set
set "cmd_use_gpu=!USE_GPU!"
set "cmd_strategy=!STRATEGY!"
set "cmd_use_raw_spectral=!USE_RAW_SPECTRAL!"

REM First, get defaults from Python config (lowest priority)
call :get_python_config_defaults

REM Apply Python defaults if values are still empty
if not defined USE_GPU set "USE_GPU=!PYTHON_USE_GPU!"
if not defined USE_GPU set "USE_GPU=true"
if not defined USE_RAW_SPECTRAL set "USE_RAW_SPECTRAL=!PYTHON_USE_RAW_SPECTRAL!"
if not defined USE_RAW_SPECTRAL set "USE_RAW_SPECTRAL=false"
if not defined STRATEGY set "STRATEGY=!PYTHON_DEFAULT_STRATEGY!"
if not defined STRATEGY set "STRATEGY=full_context"
if not defined TRAINING_MODE if defined PYTHON_TRAINING_MODE set "TRAINING_MODE=!PYTHON_TRAINING_MODE!"

REM Second, load from YAML config if available (medium priority)
if "%USE_CLOUD_CONFIG%"=="true" if exist "%CLOUD_CONFIG_FILE%" (
    echo [INFO] Loading cloud configuration from: %CLOUD_CONFIG_FILE%
    
    REM Try to parse YAML with Python if available
    python -c "import sys; sys.exit(0)" >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=1,2 delims==" %%a in ('python -c "
import yaml
import sys
try:
    with open('%CLOUD_CONFIG_FILE%', 'r') as f:
        config = yaml.safe_load(f)
    
    app = config.get('app', {})
    if 'environment' in app:
        print(f'CLOUD_ENVIRONMENT={app.get('environment')}')
    
    gcp = config.get('cloud_providers', {}).get('gcp', {})
    print(f'GCP_PROJECT_ID={gcp.get('project_id', '')}')
    print(f'GCP_REGION={gcp.get('region', '')}')
    
    storage = config.get('storage', {})
    print(f'CLOUD_STORAGE_BUCKET={storage.get('bucket_name', '')}')
    print(f'CLOUD_STORAGE_PREFIX={storage.get('prefix', '')}')
    
    compute = config.get('compute', {})
    if 'gpu_enabled' in compute:
        print(f'CLOUD_GPU_ENABLED={str(compute.get('gpu_enabled')).lower()}')
    
    pipeline = config.get('pipeline', {})
    if 'default_strategy' in pipeline:
        print(f'CLOUD_DEFAULT_STRATEGY={pipeline.get('default_strategy')}')
    print(f'CLOUD_TIME_LIMIT={pipeline.get('time_limit', '')}')
    if 'enable_gpu' in pipeline:
        print(f'CLOUD_ENABLE_GPU={str(pipeline.get('enable_gpu')).lower()}')
    
except Exception as e:
    pass
" 2^>nul') do (
            set "%%a=%%b"
        )
        
        REM Override defaults with cloud config values
        if defined CLOUD_ENVIRONMENT set "ENVIRONMENT=!CLOUD_ENVIRONMENT!"
        if defined GCP_PROJECT_ID set "PROJECT_ID=!GCP_PROJECT_ID!"
        if defined GCP_REGION set "REGION=!GCP_REGION!"
        if defined CLOUD_STORAGE_BUCKET set "BUCKET_NAME=!CLOUD_STORAGE_BUCKET!"
        if defined CLOUD_STORAGE_PREFIX set "CLOUD_STORAGE_PREFIX=!CLOUD_STORAGE_PREFIX!"
        
        REM Override training settings from YAML only if not set by command line
        if defined CLOUD_DEFAULT_STRATEGY if not defined cmd_strategy set "STRATEGY=!CLOUD_DEFAULT_STRATEGY!"
        if defined CLOUD_ENABLE_GPU if not defined cmd_use_gpu set "USE_GPU=!CLOUD_ENABLE_GPU!"
        
        REM Restore command-line values if they were set
        if defined cmd_strategy set "STRATEGY=!cmd_strategy!"
        if defined cmd_use_gpu set "USE_GPU=!cmd_use_gpu!"
        
        echo [INFO] Configuration loaded:
        echo   Environment: !ENVIRONMENT!
        echo   Project ID: !PROJECT_ID!
        echo   Region: !REGION!
        echo   Bucket: !BUCKET_NAME!
        echo   Strategy: !STRATEGY!
        echo   GPU Enabled: !USE_GPU!
    ) else (
        echo [WARN] Python3 not available for YAML parsing
    )
) else (
    if "%USE_CLOUD_CONFIG%"=="false" (
        echo [INFO] Cloud configuration disabled, using environment defaults
    ) else (
        echo [WARN] Cloud configuration file not found: %CLOUD_CONFIG_FILE%
    )
)
goto :eof

:create_deployment_record
set "deployment_type=%~1"
set "deployment_id=%~2"
set "deployment_status=%~3"
set "deployment_details=%~4"

set "deployment_dir=.deployments"
if not exist "%deployment_dir%" mkdir "%deployment_dir%"

REM Generate timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,8%_%dt:~8,6%"

set "deployment_file=%deployment_dir%\deployment_%timestamp%_%deployment_type%.json"

REM Get current git commit and branch
for /f "tokens=*" %%i in ('git rev-parse HEAD 2^>nul') do set "git_commit=%%i"
if not defined git_commit set "git_commit=unknown"
for /f "tokens=*" %%i in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "git_branch=%%i"
if not defined git_branch set "git_branch=unknown"

(
echo {
echo   "deployment_id": "%deployment_id%",
echo   "deployment_type": "%deployment_type%",
echo   "timestamp": "%timestamp%",
echo   "status": "%deployment_status%",
echo   "environment": "%ENVIRONMENT%",
echo   "project_id": "%PROJECT_ID%",
echo   "region": "%REGION%",
echo   "training_mode": "%TRAINING_MODE%",
echo   "strategy": "%STRATEGY%",
echo   "use_gpu": "%USE_GPU%",
echo   "git_commit": "%git_commit%",
echo   "git_branch": "%git_branch%",
echo   "details": %deployment_details%
echo }
) > "%deployment_file%"

echo %deployment_file%
goto :eof

:commit_and_tag_deployment
set "deployment_type=%~1"
set "deployment_id=%~2"
set "deployment_file=%~3"

if not "%AUTO_COMMIT_DEPLOYMENT%"=="true" (
    echo [INFO] Auto-commit disabled. Skipping git operations.
    goto :eof
)

REM Check if we're in a git repository
git rev-parse --git-dir >nul 2>&1
if errorlevel 1 (
    echo [WARN] Not in a git repository. Skipping commit and tag.
    goto :eof
)

REM Check for uncommitted changes
git diff-index --quiet HEAD -- 2>nul
if errorlevel 1 (
    echo [INFO] Committing deployment changes...
    
    REM Add all changes including the deployment record
    git add -A
    
    REM Create commit message
    git commit -m "Deploy: %deployment_type% [%deployment_id%]" -m "Environment: %ENVIRONMENT%" -m "Training Mode: %TRAINING_MODE%" -m "Strategy: %STRATEGY%"
    
    if not errorlevel 1 (
        for /f "tokens=*" %%i in ('git rev-parse --short HEAD') do set "commit_short=%%i"
        echo [INFO] Created deployment commit: !commit_short!
    )
) else (
    echo [INFO] No uncommitted changes to commit.
)

REM Create a tag for this deployment
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "tag_name=deploy-%ENVIRONMENT%-%deployment_type%-%dt:~0,8%-%dt:~8,6%"

git tag -a "%tag_name%" -m "Deployment %deployment_id%"
echo [INFO] Created deployment tag: %tag_name%

REM Push if enabled
if "%AUTO_PUSH_DEPLOYMENT%"=="true" (
    echo [INFO] Pushing deployment commit and tags...
    git push origin HEAD
    git push origin "%tag_name%"
    echo [INFO] Pushed deployment to remote repository
) else (
    echo [INFO] Auto-push disabled. To push manually, run:
    echo   git push origin HEAD
    echo   git push origin %tag_name%
)
goto :eof

:check_basic_prerequisites
echo [INFO] Checking basic prerequisites...

REM Check if gcloud is installed
gcloud --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] gcloud CLI is not installed. Please install Google Cloud SDK.
    exit /b 1
)

REM Check if logged in
gcloud auth list --filter=status:ACTIVE --format="value(account)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Not logged into gcloud. Run: gcloud auth login
    exit /b 1
)

REM Check if docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop.
    exit /b 1
)

echo [INFO] Basic prerequisites check passed!
goto :eof

:check_project_configuration
REM Check if PROJECT_ID is set to default placeholder after loading cloud config
if "%PROJECT_ID%"=="your-gcp-project-id" (
    echo [WARN] PROJECT_ID is not configured. Let's set it up...
    
    echo [INFO] Available projects:
    gcloud projects list --format="table(projectId,name)" 2>nul
    
    if not errorlevel 1 (
        echo.
        set /p selected_project_id="Enter your project ID from the list above: "
        
        if defined selected_project_id (
            set "PROJECT_ID=!selected_project_id!"
            echo [INFO] Using project ID: !PROJECT_ID!
            
            set /p save_choice="Save this project ID to %CLOUD_CONFIG_FILE%? (y/n): "
            if /i "!save_choice!"=="y" (
                call :update_config_project_id "!PROJECT_ID!"
            )
        ) else (
            echo [ERROR] No project ID entered.
            exit /b 1
        )
    ) else (
        echo [ERROR] Unable to list projects. Please check your gcloud authentication.
        echo [INFO] Set PROJECT_ID using: set PROJECT_ID=your-actual-project-id
        exit /b 1
    )
)

echo [INFO] Project configuration validated: %PROJECT_ID%
goto :eof

:update_config_project_id
set "new_project_id=%~1"
set "config_file=%CLOUD_CONFIG_FILE%"

if not exist "%config_file%" (
    echo [WARN] Config file %config_file% not found. Creating it...
    call :generate_cloud_config "cloud"
)

REM Create a backup
copy "%config_file%" "%config_file%.backup" >nul

REM Update the project_id line (Windows doesn't have sed, use PowerShell)
powershell -Command "(gc '%config_file%') -replace 'project_id: \".*\"', 'project_id: \"%new_project_id%\"' | Out-File -encoding ASCII '%config_file%'"

if not errorlevel 1 (
    echo [INFO] Updated project_id in %config_file%
    echo [INFO] Backup saved as %config_file%.backup
) else (
    echo [ERROR] Failed to update config file
    move /y "%config_file%.backup" "%config_file%" >nul
)
goto :eof

:setup_project
call :check_basic_prerequisites
if errorlevel 1 exit /b 1

call :load_cloud_config
call :check_project_configuration
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
goto :end

:check_image_exists
set "image_uri=%~1"
echo [INFO] Checking if container image exists: %image_uri%

gcloud artifacts docker images describe "%image_uri%" --quiet >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Container image found: %image_uri%
    exit /b 0
) else (
    echo [WARN] Container image not found: %image_uri%
    exit /b 1
)

:build_image
echo [INFO] Building and pushing container image with caching...
set "image_uri=%REGION%-docker.pkg.dev/%PROJECT_ID%/%REPO_NAME%/%IMAGE_NAME%:latest"

REM Configure Docker to use gcloud as credential helper
gcloud auth configure-docker %REGION%-docker.pkg.dev

REM Build with Cloud Build
gcloud builds submit --tag %image_uri% .
if errorlevel 1 (
    echo [ERROR] Failed to build image
    exit /b 1
)

echo [INFO] Image built and pushed to %image_uri%
goto :end

:build_image_local
echo [INFO] Building container image locally with Docker...
set "image_uri=%REGION%-docker.pkg.dev/%PROJECT_ID%/%REPO_NAME%/%IMAGE_NAME%:latest"
set "skip_push=%SKIP_PUSH%"
if not defined skip_push set "skip_push=false"

REM Check if Docker is available
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed. Please install Docker or use 'build' for Cloud Build.
    exit /b 1
)

REM Configure Docker to use gcloud as credential helper
if not "%skip_push%"=="true" (
    gcloud auth configure-docker %REGION%-docker.pkg.dev
)

REM Build locally with Docker
echo [INFO] Building with local Docker for faster caching...
docker build -t %image_uri% --cache-from %image_uri% --build-arg BUILDKIT_INLINE_CACHE=1 .

if "%skip_push%"=="true" (
    echo [INFO] Skipping push (SKIP_PUSH=true). Image built locally: %image_uri%
    echo [WARN] Note: You'll need to push manually before deploying to cloud
) else (
    REM Push to registry
    echo [INFO] Pushing image to Artifact Registry...
    docker push %image_uri%
    echo [INFO] Image built locally and pushed to %image_uri%
)
goto :end

:check_data_exists_in_gcs
set "bucket_prefix=magnesium-pipeline"
if defined CLOUD_STORAGE_PREFIX set "bucket_prefix=%CLOUD_STORAGE_PREFIX%"

echo [INFO] Checking if training data exists in GCS bucket...

REM Check for key data directory
gsutil ls "gs://%BUCKET_NAME%/%bucket_prefix%/data_5278_Phase3/" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Found: gs://%BUCKET_NAME%/%bucket_prefix%/data_5278_Phase3/
    exit /b 0
) else (
    echo [WARN] Missing: gs://%BUCKET_NAME%/%bucket_prefix%/data_5278_Phase3/
    exit /b 1
)

:list_and_verify_gcs_data
set "bucket_prefix=magnesium-pipeline"
if defined CLOUD_STORAGE_PREFIX set "bucket_prefix=%CLOUD_STORAGE_PREFIX%"

echo ======================================
echo GCS Data Inventory and Verification
echo ======================================
echo Bucket: gs://%BUCKET_NAME%/%bucket_prefix%/
echo.

set "raw_dir=gs://%BUCKET_NAME%/%bucket_prefix%/data_5278_Phase3/"
set "reference_dir=gs://%BUCKET_NAME%/%bucket_prefix%/reference_data/"

echo 1. RAW DATA (data_5278_Phase3):
echo --------------------------------
gsutil ls "%raw_dir%" >nul 2>&1
if not errorlevel 1 (
    echo [OK] Directory exists
    for /f %%i in ('gsutil ls "%raw_dir%*.csv.txt" 2^>nul ^| find /c /v ""') do set raw_count=%%i
    if !raw_count! gtr 0 (
        echo [OK] Contains !raw_count! CSV files
    ) else (
        echo [ERROR] No CSV files found!
    )
) else (
    echo [ERROR] Directory NOT FOUND - This is critical for training!
)
echo.

echo 2. REFERENCE DATA (Excel files):
echo --------------------------------
gsutil ls "%reference_dir%" >nul 2>&1
if not errorlevel 1 (
    echo [OK] Directory exists
    for /f %%i in ('gsutil ls "%reference_dir%*.xlsx" "%reference_dir%*.xls" 2^>nul ^| find /c /v ""') do set excel_count=%%i
    if !excel_count! gtr 0 (
        echo [OK] Contains !excel_count! Excel file(s)
    ) else (
        echo [ERROR] No Excel files found - Required for ground truth values!
    )
) else (
    echo [WARN] Reference directory not found - Will need to upload Excel files
)
goto :end

:upload_training_data
set "bucket_prefix=magnesium-pipeline"
if defined CLOUD_STORAGE_PREFIX set "bucket_prefix=%CLOUD_STORAGE_PREFIX%"

REM Check if data already exists in GCS
call :check_data_exists_in_gcs
if not errorlevel 1 (
    if not "%2"=="--force" (
        echo [INFO] Training data already exists in GCS bucket. Skipping upload.
        echo [INFO] Use 'gcp-deploy.bat upload-data --force' to re-upload data.
        goto :end
    )
)

echo [INFO] Uploading training data to GCS bucket...

REM Upload data directories with rsync for efficiency
if exist "data\raw\data_5278_Phase3" (
    echo [INFO] Uploading raw spectral data...
    gsutil -m rsync -r -d "data\raw\data_5278_Phase3" "gs://%BUCKET_NAME%/%bucket_prefix%/data_5278_Phase3/"
)

REM Upload reference files (Excel files)
for %%f in (data\*.xlsx) do (
    if exist "%%f" (
        echo [INFO] Uploading %%f...
        gsutil -m cp "%%f" "gs://%BUCKET_NAME%/%bucket_prefix%/"
    )
)

for %%f in (data\*.xls) do (
    if exist "%%f" (
        echo [INFO] Uploading %%f...
        gsutil -m cp "%%f" "gs://%BUCKET_NAME%/%bucket_prefix%/"
    )
)

REM Upload reference_data directory if it exists
if exist "data\reference_data" (
    echo [INFO] Uploading reference data directory...
    gsutil -m rsync -r -d "data\reference_data" "gs://%BUCKET_NAME%/%bucket_prefix%/reference_data/"
)

echo [INFO] Training data upload completed to gs://%BUCKET_NAME%/%bucket_prefix%/
goto :end

:deploy_cloud_run
call :check_basic_prerequisites
if errorlevel 1 exit /b 1
call :load_cloud_config
call :check_project_configuration
if errorlevel 1 exit /b 1

echo [INFO] Deploying to Cloud Run...
set "image_uri=%REGION%-docker.pkg.dev/%PROJECT_ID%/%REPO_NAME%/%IMAGE_NAME%:latest"

REM Deploy to Cloud Run
gcloud run deploy %SERVICE_NAME% ^
    --image %image_uri% ^
    --platform managed ^
    --region %REGION% ^
    --memory 8Gi ^
    --cpu 4 ^
    --timeout 3600 ^
    --concurrency 10 ^
    --max-instances 5 ^
    --allow-unauthenticated ^
    --set-env-vars="STORAGE_TYPE=gcs,STORAGE_BUCKET_NAME=%BUCKET_NAME%,GPU_ENABLED=false"

if not errorlevel 1 (
    REM Get service URL
    for /f "tokens=*" %%i in ('gcloud run services describe %SERVICE_NAME% --region=%REGION% --format="value(status.url)"') do set SERVICE_URL=%%i
    echo [INFO] Service deployed at: !SERVICE_URL!
    
    REM Test health endpoint
    echo [INFO] Testing deployment...
    timeout /t 5 /nobreak >nul
    curl -s "!SERVICE_URL!/health" 2>nul | findstr /i "healthy" >nul
    if not errorlevel 1 (
        echo [INFO] Deployment test successful!
    ) else (
        echo [WARN] Deployment test failed. Check logs with: gcloud run logs tail %SERVICE_NAME% --region %REGION%
    )
) else (
    echo [ERROR] Cloud Run deployment failed
)
goto :end

:deploy_inference_service
call :check_basic_prerequisites
if errorlevel 1 exit /b 1
call :load_cloud_config
call :check_project_configuration
if errorlevel 1 exit /b 1

echo [INFO] Deploying inference service to Cloud Run...
set "image_uri=%REGION%-docker.pkg.dev/%PROJECT_ID%/%REPO_NAME%/%IMAGE_NAME%:latest"
set "inference_service_name=%SERVICE_NAME%-inference"
set "prefix=magnesium-pipeline"
if defined CLOUD_STORAGE_PREFIX set "prefix=%CLOUD_STORAGE_PREFIX%"

gcloud run deploy %inference_service_name% ^
    --image %image_uri% ^
    --platform managed ^
    --region %REGION% ^
    --memory 4Gi ^
    --cpu 2 ^
    --timeout 900 ^
    --concurrency 100 ^
    --max-instances 10 ^
    --allow-unauthenticated ^
    --port 8000 ^
    --set-env-vars="STORAGE_TYPE=gcs,STORAGE_BUCKET_NAME=%BUCKET_NAME%,CLOUD_STORAGE_PREFIX=%prefix%,INFERENCE_MODE=true,GPU_ENABLED=false" ^
    --command="bash" ^
    --args="-c,source /app/docker-entrypoint.sh && download_models_for_inference && python api_server.py"

REM Get service URL
for /f "tokens=*" %%i in ('gcloud run services describe %inference_service_name% --region=%REGION% --format="value(status.url)"') do set INFERENCE_URL=%%i
echo [INFO] Inference service deployed at: !INFERENCE_URL!

REM Test health endpoint
echo [INFO] Testing inference service...
timeout /t 10 /nobreak >nul
curl -s "!INFERENCE_URL!/health" 2>nul | findstr /i "healthy" >nul
if not errorlevel 1 (
    echo [INFO] Inference service test successful!
    echo [INFO] API Documentation: !INFERENCE_URL!/docs
) else (
    echo [WARN] Inference service test failed. Check logs with: gcloud run logs tail %inference_service_name% --region %REGION%
)
goto :end

:deploy_gke
echo [INFO] GKE deployment requires kubectl and additional setup.
echo [INFO] Please refer to the shell script version for full GKE deployment.
goto :end

:validate_training_config
set "valid_modes=train autogluon tune optimize-models optimize-xgboost optimize-autogluon optimize-range-specialist"
echo !valid_modes! | findstr /i "%TRAINING_MODE%" >nul
if errorlevel 1 (
    echo [ERROR] Invalid TRAINING_MODE: %TRAINING_MODE%
    echo [ERROR] Valid options: !valid_modes!
    exit /b 1
)

set "valid_strategies=full_context simple_only Mg_only"
echo !valid_strategies! | findstr /i "%STRATEGY%" >nul
if errorlevel 1 (
    echo [ERROR] Invalid STRATEGY: %STRATEGY%
    echo [ERROR] Valid options: !valid_strategies!
    exit /b 1
)

echo [INFO] Training Configuration:
echo   Mode: %TRAINING_MODE%
echo   GPU: %USE_GPU%
echo   Raw Spectral: %USE_RAW_SPECTRAL%
echo   Strategy: %STRATEGY%
if defined MODELS echo   Models: %MODELS%
if defined TRIALS echo   Trials: %TRIALS%
if defined TIMEOUT echo   Timeout: %TIMEOUT%s
echo   Machine: %MACHINE_TYPE% with %ACCELERATOR_COUNT%x %ACCELERATOR_TYPE%
goto :eof

:build_training_command
set "base_cmd=python main.py %TRAINING_MODE%"

if "%USE_GPU%"=="true" set "base_cmd=!base_cmd! --gpu"
if "%USE_RAW_SPECTRAL%"=="true" set "base_cmd=!base_cmd! --raw-spectral"

if defined MODELS (
    set "models_list=!MODELS:,= !"
    set "base_cmd=!base_cmd! --models !models_list!"
)

echo %TRAINING_MODE% | findstr /i "train optimize-xgboost optimize-autogluon optimize-models optimize-range-specialist" >nul
if not errorlevel 1 (
    if defined STRATEGY set "base_cmd=!base_cmd! --strategy %STRATEGY%"
)

if defined TRIALS (
    echo %TRAINING_MODE% | findstr /i "optimize" >nul
    if not errorlevel 1 set "base_cmd=!base_cmd! --trials %TRIALS%"
)

if defined TIMEOUT (
    echo %TRAINING_MODE% | findstr /i "optimize" >nul
    if not errorlevel 1 set "base_cmd=!base_cmd! --timeout %TIMEOUT%"
)

set "training_cmd=!base_cmd!"
goto :eof

:deploy_vertex_ai
call :check_basic_prerequisites
if errorlevel 1 exit /b 1
call :load_cloud_config
call :validate_training_config
if errorlevel 1 exit /b 1

echo [INFO] Deploying to Vertex AI for %TRAINING_MODE% training...

set "image_uri=%REGION%-docker.pkg.dev/%PROJECT_ID%/%REPO_NAME%/%IMAGE_NAME%:latest"

REM Check if container image exists, build if needed
call :check_image_exists "%image_uri%"
if errorlevel 1 (
    echo [INFO] Container image not found. Building image...
    call :setup_project
    call :build_image
)

REM Upload training data to GCS bucket
call :upload_training_data

REM Build the training command
call :build_training_command
echo [INFO] Training command: !training_cmd!

REM Determine appropriate timeout based on training mode
set "job_timeout=7200"
echo %TRAINING_MODE% | findstr /i "optimize" >nul
if not errorlevel 1 set "job_timeout=86400"
if "%TRAINING_MODE%"=="autogluon" set "job_timeout=14400"

REM Create a unique job name
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "job_display_name=%SERVICE_NAME%-%TRAINING_MODE%-%dt:~0,8%-%dt:~8,6%"

REM Create deployment record
set "deployment_details={"image_uri": "%image_uri%", "machine_type": "%MACHINE_TYPE%"}"
for /f "tokens=*" %%i in ('call :create_deployment_record "vertex-ai" "pending" "started" "!deployment_details!"') do set "deployment_file=%%i"
echo [INFO] Created deployment record: %deployment_file%

REM Create a training job
echo [INFO] Submitting Vertex AI training job...

REM Create temporary JSON config file
set "CONFIG_FILE=%TEMP%\vertex-job-%RANDOM%.json"
set "prefix=magnesium-pipeline"
if defined CLOUD_STORAGE_PREFIX set "prefix=%CLOUD_STORAGE_PREFIX%"

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
echo         "imageUri": "%image_uri%",
echo         "command": ["bash", "-c", "source /app/docker-entrypoint.sh && download_training_data && !training_cmd!"],
echo         "env": [
echo           {"name": "STORAGE_TYPE", "value": "gcs"},
echo           {"name": "STORAGE_BUCKET_NAME", "value": "%BUCKET_NAME%"},
echo           {"name": "CLOUD_STORAGE_PREFIX", "value": "%prefix%"},
echo           {"name": "ENVIRONMENT", "value": "%ENVIRONMENT%"},
echo           {"name": "PYTHONPATH", "value": "/app"}
echo         ]
echo       }
echo     }
echo   ],
echo   "scheduling": {
echo     "timeout": "%job_timeout%s"
echo   }
echo }
) > "%CONFIG_FILE%"

gcloud ai custom-jobs create ^
    --region=%REGION% ^
    --display-name="%job_display_name%" ^
    --config="%CONFIG_FILE%"

if not errorlevel 1 (
    echo [INFO] Vertex AI training job created: %job_display_name%
    echo [INFO] Timeout: %job_timeout%s
    echo [INFO] Monitor job progress with:
    echo   gcloud ai custom-jobs list --region=%REGION%
    echo   gcloud ai custom-jobs stream-logs [JOB_ID] --region=%REGION%
) else (
    echo [ERROR] Failed to create Vertex AI job
)

REM Clean up temp file
del "%CONFIG_FILE%" 2>nul
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

:upload_logs
echo [INFO] Manually uploading logs from recent Vertex AI job...

REM Get the most recent job
for /f "tokens=*" %%i in ('gcloud ai custom-jobs list --region=%REGION% --format="value(name)" --limit=1 2^>nul') do set recent_job=%%i

if defined recent_job (
    echo [INFO] Found recent job: %recent_job%
    echo [INFO] Fetching logs...
    
    for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
    set "log_file=vertex-ai-logs-%dt:~0,8%_%dt:~8,6%.log"
    
    gcloud ai custom-jobs stream-logs "%recent_job%" --region=%REGION% > "%log_file%" 2>&1
    
    if exist "%log_file%" (
        echo [INFO] Logs saved to: %log_file%
        
        set "prefix=magnesium-pipeline"
        if defined CLOUD_STORAGE_PREFIX set "prefix=%CLOUD_STORAGE_PREFIX%"
        set "gcs_path=gs://%BUCKET_NAME%/%prefix%/manual-logs/%dt:~0,8%_%dt:~8,6%/"
        
        echo [INFO] Uploading logs to GCS: !gcs_path!
        gsutil cp "%log_file%" "!gcs_path!vertex-ai-training.log"
        
        if not errorlevel 1 (
            echo [INFO] Logs uploaded to: !gcs_path!vertex-ai-training.log
        ) else (
            echo [ERROR] Failed to upload logs to GCS
        )
    ) else (
        echo [WARN] No logs found for job: %recent_job%
    )
) else (
    echo [WARN] No recent Vertex AI jobs found
)
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

:test_config
call :check_basic_prerequisites
if errorlevel 1 exit /b 1
echo [INFO] Testing configuration loading...
echo [INFO] CLOUD_CONFIG_FILE: %CLOUD_CONFIG_FILE%
call :load_cloud_config
call :check_project_configuration
echo [INFO] Configuration test completed successfully!
goto :end

:deploy_all
call :setup_project
if errorlevel 1 exit /b 1
call :build_image
if errorlevel 1 exit /b 1
call :deploy_cloud_run
if errorlevel 1 exit /b 1
goto :end

:generate_cloud_config
set "env_name=%2"
if not defined env_name set "env_name=development"
set "config_file=config\%env_name%_config.yml"

echo [INFO] Generating cloud configuration template for environment: %env_name%

if not exist "config" mkdir "config"

(
echo # %env_name% Environment Configuration for Magnesium Pipeline
echo # Generated on %date% %time%
echo.
echo # General Settings
echo app:
echo   name: "magnesium-pipeline-%env_name%"
echo   version: "1.0.0"
echo   environment: "%env_name%"
echo   debug: false
echo.
echo # API Configuration
echo api:
echo   host: "0.0.0.0"
echo   port: 8000
echo   workers: 4
echo   timeout: 300
echo   max_file_size: "100MB"
echo   cors_origins: ["*"]
echo.
echo # Storage Configuration
echo storage:
echo   type: "gcs"
echo   data_path: "/app/data"
echo   models_path: "/app/models"
echo   reports_path: "/app/reports"
echo   logs_path: "/app/logs"
echo   bucket_name: "%env_name%-magnesium-data"
echo   credentials_path: null
echo   prefix: "magnesium-pipeline-%env_name%"
echo.
echo # Compute Configuration
echo compute:
echo   gpu_enabled: true
echo   gpu_memory_fraction: 0.8
echo   cpu_cores: null
echo   memory_limit: null
echo.
echo # Pipeline Configuration
echo pipeline:
echo   default_strategy: "full_context"
echo   time_limit: 3600
echo   enable_gpu: true
echo   enable_sample_weights: true
echo.
echo # Monitoring
echo monitoring:
echo   log_level: "INFO"
echo   enable_metrics: true
echo   metrics_port: 9090
echo   health_check_interval: 30
echo.
echo # Security
echo security:
echo   api_key_required: true
echo   rate_limiting:
echo     enabled: true
echo     requests_per_minute: 60
echo.
echo # Cloud Provider Specific Overrides
echo cloud_providers:
echo   gcp:
echo     project_id: "YOUR_GCP_PROJECT_ID"  # TODO: Update this
echo     region: "us-central1"
echo     storage_class: "STANDARD"
echo     compute_zone: "us-central1-a"
echo.
echo   aws:
echo     region: "us-east-1"
echo     storage_class: "STANDARD"
echo     instance_type: "ml.m5.large"
echo.
echo   azure:
echo     resource_group: "magnesium-%env_name%-rg"
echo     location: "East US"
echo     storage_tier: "Standard"
) > "%config_file%"

echo [INFO] Generated configuration file: %config_file%
echo [WARN] Remember to update YOUR_GCP_PROJECT_ID in the generated file!
echo.
echo To use this configuration:
echo   set CLOUD_CONFIG_FILE=%config_file%
echo   gcp-deploy.bat autogluon
goto :end

:list_gcp_projects
echo [INFO] Listing available GCP projects...

gcloud --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] gcloud CLI is not installed. Please install it first.
    exit /b 1
)

gcloud auth list --filter=status:ACTIVE --format="value(account)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Not logged into gcloud. Run: gcloud auth login
    exit /b 1
)

echo.
echo Available GCP Projects:
echo =======================
gcloud projects list --format="table(projectId:label='PROJECT_ID',name:label='PROJECT_NAME',projectNumber:label='PROJECT_NUMBER')" 2>nul

echo.
echo To use a project, set the PROJECT_ID environment variable:
echo   set PROJECT_ID=your-chosen-project-id
echo   gcp-deploy.bat setup
goto :end

:show_config_help
echo Cloud Configuration Help for Magnesium Pipeline
echo =================================================
echo.
echo The cloud_config.yml file provides environment-agnostic configuration
echo that can be used across different cloud providers and environments.
echo.
echo Key Benefits:
echo   - Centralized configuration management
echo   - Environment-specific settings (dev, staging, prod)
echo   - Cloud provider abstraction
echo   - Consistent deployment parameters
echo.
echo Configuration Priority (highest to lowest):
echo   1. Environment variables with _OVERRIDE suffix
echo   2. Direct environment variables
echo   3. Cloud configuration file values
echo   4. Script defaults
echo.
echo Usage Examples:
echo   REM Use default config
echo   gcp-deploy.bat autogluon
echo.
echo   REM Use staging config
echo   set CLOUD_CONFIG_FILE=config\staging_config.yml
echo   gcp-deploy.bat train
echo.
echo   REM Override specific values
echo   set PROJECT_ID_OVERRIDE=true
echo   set PROJECT_ID=my-project
echo   gcp-deploy.bat optimize
echo.
echo   REM Generate new config template
echo   gcp-deploy.bat generate-config production
echo.
echo Environment Variables that can be overridden:
echo   PROJECT_ID, REGION, BUCKET_NAME, STRATEGY, USE_GPU
echo   Add '_OVERRIDE=true' to force override cloud config values
echo.
echo For more examples, run: gcp-deploy.bat help
goto :end

:show_help
echo Google Cloud Platform Deployment Script for Magnesium Pipeline
echo.
echo Usage: %0 [COMMAND]
echo.
echo Infrastructure Commands:
echo   setup       Setup GCP project and enable APIs
echo   build       Build container image with Cloud Build (Kaniko caching)
echo   build-local Build container image locally with Docker (faster)
echo   cloud-run   Deploy to Cloud Run (serverless)
echo   inference   Deploy inference service to Cloud Run (loads models from GCS)
echo   gke         Deploy to Google Kubernetes Engine
echo   cleanup     Clean up all resources
echo   all         Run setup, build, and cloud-run
echo.
echo Training Commands (Vertex AI):
echo   vertex-ai   Submit training job to Vertex AI (uses TRAINING_MODE)
echo   train       Train standard models on Vertex AI
echo   tune        Hyperparameter tuning on Vertex AI
echo   autogluon   AutoGluon training on Vertex AI
echo   optimize    Multi-model optimization on Vertex AI
echo.
echo Configuration Commands:
echo   config-help         Show detailed cloud configuration help
echo   generate-config ENV Generate cloud config template for environment
echo   list-projects       List available GCP projects
echo   test-config         Test cloud configuration loading
echo.
echo Data Management Commands:
echo   upload-data         Upload training data to GCS (skip if exists)
echo   upload-data --force Force re-upload training data to GCS
echo   check-data          Check if training data exists in GCS
echo   list-data           List training data files in GCS bucket
echo   upload-logs         Manually fetch and upload logs from recent Vertex AI job
echo.
echo Infrastructure Environment Variables:
echo   set PROJECT_ID=your-gcp-project-id
echo   set REGION=us-central1
echo   set SERVICE_NAME=magnesium-pipeline
echo   set BUCKET_NAME=your-bucket-name
echo.
echo Cloud Configuration Variables:
echo   set CLOUD_CONFIG_FILE=config\cloud_config.yml
echo   set ENVIRONMENT=production
echo   set USE_CLOUD_CONFIG=true
echo.
echo Training Environment Variables:
echo   set TRAINING_MODE=autogluon
echo   set USE_GPU=true
echo   set USE_RAW_SPECTRAL=false
echo   set MODELS=xgboost,lightgbm,catboost
echo   set STRATEGY=full_context
echo   set TRIALS=200
echo   set TIMEOUT=3600
echo   set MACHINE_TYPE=n1-standard-4
echo   set ACCELERATOR_TYPE=NVIDIA_TESLA_T4
echo   set ACCELERATOR_COUNT=1
echo.
echo Infrastructure Examples:
echo   set PROJECT_ID=my-project
echo   gcp-deploy.bat all
echo.
echo   set REGION=europe-west1
echo   gcp-deploy.bat cloud-run
echo.
echo Training Examples:
echo   REM Train specific models with GPU
echo   set MODELS=xgboost,lightgbm,catboost
echo   gcp-deploy.bat train
echo.
echo   REM AutoGluon training
echo   gcp-deploy.bat autogluon
echo.
echo   REM Multi-model optimization with custom parameters
echo   set MODELS=xgboost,lightgbm
echo   set STRATEGY=simple_only
echo   set TRIALS=200
echo   gcp-deploy.bat optimize
echo.
echo   REM Hyperparameter tuning with raw spectral data
echo   set USE_RAW_SPECTRAL=true
echo   set MODELS=neural_network
echo   gcp-deploy.bat tune
echo.
echo   REM High-resource optimization
echo   set MACHINE_TYPE=n1-standard-8
echo   set ACCELERATOR_COUNT=2
echo   set TRIALS=500
echo   gcp-deploy.bat optimize
echo.
echo Quick Start Training Examples:
echo   REM Basic AutoGluon training (recommended)
echo   gcp-deploy.bat autogluon
echo.
echo   REM Train specific models quickly
echo   set MODELS=xgboost,lightgbm
echo   gcp-deploy.bat train
echo.
echo   REM Optimize XGBoost specifically
echo   set TRAINING_MODE=optimize-xgboost
echo   set TRIALS=300
echo   gcp-deploy.bat vertex-ai
echo.
echo   REM Full pipeline: setup + train
echo   gcp-deploy.bat setup
echo   gcp-deploy.bat build
echo   gcp-deploy.bat autogluon

:end
endlocal