@echo off
setlocal enabledelayedexpansion

REM Local Docker Deployment Script for Magnesium Pipeline
REM This script provides multiple deployment options for local Docker
REM Aligned with local-deploy.sh for consistency

REM Configuration
if not defined SERVICE_NAME set "SERVICE_NAME=magnesium-pipeline"
if not defined CONTAINER_PREFIX set "CONTAINER_PREFIX=magnesium"
if not defined COMPOSE_FILE set "COMPOSE_FILE=docker-compose.local.yml"

REM Local Configuration
if not defined LOCAL_CONFIG_FILE set "LOCAL_CONFIG_FILE=config/local.yml"
if not defined ENVIRONMENT set "ENVIRONMENT=development"
if not defined USE_LOCAL_CONFIG set "USE_LOCAL_CONFIG=true"

REM Training Configuration
if not defined TRAINING_MODE set "TRAINING_MODE=autogluon"
if not defined USE_GPU set "USE_GPU=true"
if not defined USE_RAW_SPECTRAL set "USE_RAW_SPECTRAL=false"
if not defined MODELS set "MODELS="
if not defined STRATEGY set "STRATEGY=simple_only"
if not defined TRIALS set "TRIALS="
if not defined TIMEOUT set "TIMEOUT="

REM Windows doesn't support colored output easily, so we'll use plain text
echo [INFO] Magnesium Pipeline Local Deployment (Windows)
echo.

REM Parse command
if "%1"=="" goto :show_help
if "%1"=="help" goto :show_help
if "%1"=="--help" goto :show_help
if "%1"=="-h" goto :show_help
if "%1"=="/?" goto :show_help

REM Route to appropriate function
if "%1"=="setup" goto :setup
if "%1"=="build" goto :build_images
if "%1"=="api" goto :start_api_server
if "%1"=="dev" goto :start_development
if "%1"=="run-training" goto :run_training
if "%1"=="train" goto :train
if "%1"=="tune" goto :tune
if "%1"=="autogluon" goto :autogluon
if "%1"=="optimize" goto :optimize
if "%1"=="predict" goto :run_prediction
if "%1"=="test-gpu" goto :test_gpu_support
if "%1"=="test-config" goto :test_config
if "%1"=="status" goto :show_status
if "%1"=="logs" goto :show_logs
if "%1"=="stop" goto :stop_services
if "%1"=="cleanup" goto :cleanup

echo [ERROR] Unknown command: %1
goto :show_help

:load_local_config
if "%USE_LOCAL_CONFIG%"=="true" if exist "%LOCAL_CONFIG_FILE%" (
    echo [INFO] Loading local configuration from: %LOCAL_CONFIG_FILE%
    
    REM Try to parse YAML with Python if available
    python -c "import sys; sys.exit(0)" >nul 2>&1
    if not errorlevel 1 (
        REM Use Python to parse YAML
        for /f "tokens=1,2 delims==" %%a in ('python -c "
import yaml
import sys
try:
    with open('%LOCAL_CONFIG_FILE%', 'r') as f:
        config = yaml.safe_load(f)
    
    pipeline = config.get('pipeline', {})
    print(f'LOCAL_DEFAULT_STRATEGY={pipeline.get('default_strategy', '')}')
    print(f'LOCAL_TIME_LIMIT={pipeline.get('time_limit', '')}')
    print(f'LOCAL_ENABLE_GPU={str(pipeline.get('enable_gpu', True)).lower()}')
    
    compute = config.get('compute', {})
    print(f'LOCAL_GPU_ENABLED={str(compute.get('gpu_enabled', True)).lower()}')
    
    app = config.get('app', {})
    print(f'LOCAL_ENVIRONMENT={app.get('environment', 'development')}')
    
    monitoring = config.get('monitoring', {})
    print(f'LOCAL_LOG_LEVEL={monitoring.get('log_level', 'INFO')}')
    
except Exception as e:
    pass
" 2^>nul') do (
            set "%%a=%%b"
        )
        
        REM Apply local config values if not overridden
        if defined LOCAL_DEFAULT_STRATEGY if not defined STRATEGY_OVERRIDE set "STRATEGY=!LOCAL_DEFAULT_STRATEGY!"
        if defined LOCAL_ENABLE_GPU if not defined USE_GPU_OVERRIDE set "USE_GPU=!LOCAL_ENABLE_GPU!"
        if defined LOCAL_TIME_LIMIT if not defined TIMEOUT_OVERRIDE set "TIMEOUT=!LOCAL_TIME_LIMIT!"
        if defined LOCAL_ENVIRONMENT if not defined ENVIRONMENT_OVERRIDE set "ENVIRONMENT=!LOCAL_ENVIRONMENT!"
        
        echo [INFO] Applied local configuration:
        echo   Environment: !ENVIRONMENT!
        echo   Default Strategy: !STRATEGY!
        echo   GPU Enabled: !USE_GPU!
        if defined TIMEOUT echo   Time Limit: !TIMEOUT!
        if defined LOCAL_LOG_LEVEL echo   Log Level: !LOCAL_LOG_LEVEL!
    ) else (
        echo [WARN] Python3 not available for YAML parsing, using defaults
    )
) else (
    if "%USE_LOCAL_CONFIG%"=="false" (
        echo [INFO] Local configuration disabled, using environment defaults
    ) else (
        echo [WARN] Local configuration file not found: %LOCAL_CONFIG_FILE%, using defaults
    )
)
goto :eof

:check_prerequisites
echo [INFO] Checking prerequisites...

REM Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop.
    exit /b 1
)

REM Check if Docker daemon is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker daemon is not running. Please start Docker Desktop.
    exit /b 1
)

REM Check for NVIDIA Docker support
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    docker info | findstr /i nvidia >nul 2>&1
    if not errorlevel 1 (
        echo [INFO] NVIDIA Docker support detected!
        set "GPU_SUPPORT=true"
    ) else (
        echo [WARN] NVIDIA GPU detected but Docker GPU support not available.
        set "GPU_SUPPORT=false"
    )
) else (
    echo [WARN] No NVIDIA GPU detected. GPU acceleration will be disabled.
    set "GPU_SUPPORT=false"
)

echo [INFO] Prerequisites check completed!
goto :eof

:create_directories
echo [INFO] Creating required directories...

REM Create data directories
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "data\averaged_files_per_sample" mkdir "data\averaged_files_per_sample"
if not exist "data\cleansed_files_per_sample" mkdir "data\cleansed_files_per_sample"
if not exist "data\reference_data" mkdir "data\reference_data"
if not exist "models\autogluon" mkdir "models\autogluon"
if not exist "reports" mkdir "reports"
if not exist "logs" mkdir "logs"
if not exist "bad_files" mkdir "bad_files"
if not exist "bad_prediction_files" mkdir "bad_prediction_files"
if not exist "config" mkdir "config"

REM Create .gitkeep files
echo. > "data\raw\.gitkeep"
echo. > "data\processed\.gitkeep"
echo. > "models\.gitkeep"
echo. > "reports\.gitkeep"
echo. > "logs\.gitkeep"

echo [INFO] Directories created successfully!
goto :eof

:build_images
echo [INFO] Building Docker images...

REM Build the main image
echo [INFO] Building production image...
docker compose -f %COMPOSE_FILE% build magnesium-api
if errorlevel 1 (
    echo [ERROR] Failed to build production image
    exit /b 1
)

if "%2"=="dev" (
    echo [INFO] Building development image...
    docker compose -f %COMPOSE_FILE% build magnesium-dev
    if errorlevel 1 (
        echo [ERROR] Failed to build development image
        exit /b 1
    )
)

if "%2"=="all" (
    echo [INFO] Building all images...
    docker compose -f %COMPOSE_FILE% build
    if errorlevel 1 (
        echo [ERROR] Failed to build images
        exit /b 1
    )
)

echo [INFO] Docker images built successfully!
goto :end

:test_gpu_support
echo [INFO] Testing GPU support in containers...

if "%GPU_SUPPORT%"=="true" (
    echo [INFO] Running GPU test container...
    docker run --rm --gpus all magnesium-pipeline:latest python3 -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
    if not errorlevel 1 (
        echo [INFO] GPU support verified in container!
    ) else (
        echo [WARN] GPU test failed. Models will run on CPU.
    )
) else (
    echo [INFO] Skipping GPU test - no GPU support detected.
)
goto :end

:start_api_server
echo [INFO] Starting API server...

REM Stop any existing containers
docker compose -f %COMPOSE_FILE% down >nul 2>&1

REM Start API server
docker compose -f %COMPOSE_FILE% up -d magnesium-api
if errorlevel 1 (
    echo [ERROR] Failed to start API server
    exit /b 1
)

REM Wait for health check
echo [INFO] Waiting for API server to be ready...
set /a counter=0
:wait_api
if !counter! geq 30 (
    echo [ERROR] API server failed to start within 30 seconds
    docker compose -f %COMPOSE_FILE% logs magnesium-api
    exit /b 1
)
timeout /t 1 /nobreak >nul
curl -s http://localhost:8000/health 2>nul | findstr /i "healthy" >nul
if not errorlevel 1 (
    echo [INFO] API server is ready!
    echo [INFO] API server started successfully at http://localhost:8000
    echo [INFO] API documentation available at http://localhost:8000/docs
    goto :end
)
set /a counter+=1
goto :wait_api

:start_development
echo [INFO] Starting development environment...

REM Stop any existing containers
docker compose -f %COMPOSE_FILE% down >nul 2>&1

REM Start development environment
docker compose -f %COMPOSE_FILE% --profile dev up -d
if errorlevel 1 (
    echo [ERROR] Failed to start development environment
    exit /b 1
)

echo [INFO] Development environment started!
echo [INFO] Jupyter Lab: http://localhost:8888
echo [INFO] API Server: http://localhost:8001
goto :end

:validate_training_config
REM Validate training mode
set "valid_modes=train autogluon tune optimize-models optimize-xgboost optimize-autogluon optimize-range-specialist"
echo !valid_modes! | findstr /i "%TRAINING_MODE%" >nul
if errorlevel 1 (
    echo [ERROR] Invalid TRAINING_MODE: %TRAINING_MODE%
    echo [ERROR] Valid options: !valid_modes!
    exit /b 1
)

REM Validate strategy
set "valid_strategies=full_context simple_only Mg_only"
echo !valid_strategies! | findstr /i "%STRATEGY%" >nul
if errorlevel 1 (
    echo [ERROR] Invalid STRATEGY: %STRATEGY%
    echo [ERROR] Valid options: !valid_strategies!
    exit /b 1
)

REM Log configuration summary
echo [INFO] Training Configuration:
echo   Mode: %TRAINING_MODE%
echo   GPU: %USE_GPU%
echo   Raw Spectral: %USE_RAW_SPECTRAL%
echo   Strategy: %STRATEGY%
if defined MODELS echo   Models: %MODELS%
if defined TRIALS echo   Trials: %TRIALS%
if defined TIMEOUT echo   Timeout: %TIMEOUT%s
goto :eof

:build_training_command
set "base_cmd=python main.py %TRAINING_MODE%"

REM Add GPU flag
if "%USE_GPU%"=="true" if "%GPU_SUPPORT%"=="true" (
    set "base_cmd=!base_cmd! --gpu"
)

REM Add raw spectral flag
if "%USE_RAW_SPECTRAL%"=="true" (
    set "base_cmd=!base_cmd! --raw-spectral"
)

REM Add models
if defined MODELS (
    set "models_list=!MODELS:,= !"
    set "base_cmd=!base_cmd! --models !models_list!"
)

REM Add strategy for optimization commands
echo %TRAINING_MODE% | findstr /i "optimize-xgboost optimize-autogluon optimize-models optimize-range-specialist" >nul
if not errorlevel 1 (
    set "base_cmd=!base_cmd! --strategy %STRATEGY%"
)

REM Add trials for optimization commands
if defined TRIALS (
    echo %TRAINING_MODE% | findstr /i "optimize" >nul
    if not errorlevel 1 (
        set "base_cmd=!base_cmd! --trials %TRIALS%"
    )
)

REM Add timeout for optimization commands
if defined TIMEOUT (
    echo %TRAINING_MODE% | findstr /i "optimize" >nul
    if not errorlevel 1 (
        set "base_cmd=!base_cmd! --timeout %TIMEOUT%"
    )
)

set "training_cmd=!base_cmd!"
goto :eof

:run_training
REM Load local configuration first
call :load_local_config

REM Validate training configuration
call :validate_training_config
if errorlevel 1 exit /b 1

echo [INFO] Running training pipeline: %TRAINING_MODE%

REM Build the training command
call :build_training_command
echo [INFO] Training command: !training_cmd!

REM Generate timestamp for container name
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,14%"

REM Run training in container
docker compose -f %COMPOSE_FILE% run --rm ^
    --name magnesium-training-%timestamp% ^
    magnesium-train ^
    "uv run --prerelease=allow !training_cmd!"

if errorlevel 1 (
    echo [ERROR] Training failed
    exit /b 1
)

echo [INFO] Training completed!
goto :end

:run_prediction
set "input_file=%2"
set "model_path=%3"

if "%input_file%"=="" (
    echo [ERROR] Usage: %0 predict ^<input_file^> ^<model_path^>
    exit /b 1
)
if "%model_path%"=="" (
    echo [ERROR] Usage: %0 predict ^<input_file^> ^<model_path^>
    exit /b 1
)

echo [INFO] Running prediction...

REM Copy input file to container-accessible location
copy "%input_file%" "data\" >nul
for %%f in ("%input_file%") do set "filename=%%~nxf"

REM Run prediction
curl -X POST "http://localhost:8000/predict" ^
    -H "Content-Type: multipart/form-data" ^
    -F "file=@data\%filename%" ^
    -F "model_path=%model_path%"

REM Clean up
del "data\%filename%" 2>nul

goto :end

:show_status
echo [INFO] Checking container status...
docker compose -f %COMPOSE_FILE% ps

echo [INFO] Checking API health...
curl -s http://localhost:8000/health 2>nul
if not errorlevel 1 (
    echo [INFO] API is healthy!
) else (
    echo [WARN] API is not responding
)
goto :end

:show_logs
set "service=%2"
if "%service%"=="" set "service=magnesium-api"
echo [INFO] Showing logs for %service%...
docker compose -f %COMPOSE_FILE% logs -f %service%
goto :end

:stop_services
echo [INFO] Stopping all services...
docker compose -f %COMPOSE_FILE% down
echo [INFO] All services stopped.
goto :end

:cleanup
echo [INFO] Cleaning up...

REM Stop and remove containers
docker compose -f %COMPOSE_FILE% down --volumes --rmi local

REM Ask about removing generated data
set /p remove_data="Remove generated data? (y/N): "
if /i "%remove_data%"=="y" (
    if exist "bad_files" rd /s /q "bad_files" 2>nul
    if exist "bad_prediction_files" rd /s /q "bad_prediction_files" 2>nul
    if exist "logs" (
        del /q "logs\*" 2>nul
    )
    if exist "reports" (
        del /q "reports\*" 2>nul
    )
    echo [INFO] Generated data removed
)

echo [INFO] Cleanup completed
goto :end

:setup
call :check_prerequisites
if errorlevel 1 exit /b 1
call :create_directories
call :build_images all
echo [INFO] Setup completed! Run '%0 api' to start the API server.
goto :end

:train
set "TRAINING_MODE=train"
goto :run_training

:tune
set "TRAINING_MODE=tune"
goto :run_training

:autogluon
set "TRAINING_MODE=autogluon"
goto :run_training

:optimize
set "TRAINING_MODE=optimize-models"
goto :run_training

:test_config
echo [INFO] Testing local configuration loading...
echo [INFO] LOCAL_CONFIG_FILE: %LOCAL_CONFIG_FILE%
call :load_local_config
call :validate_training_config
echo [INFO] Configuration test completed successfully!
goto :end

:show_help
echo Local Docker Deployment for Magnesium Pipeline
echo.
echo Usage: %0 [COMMAND] [OPTIONS]
echo.
echo Infrastructure Commands:
echo   setup              Setup environment and build images
echo   build [dev^|all]    Build Docker images
echo   api                Start API server
echo   dev                Start development environment with Jupyter
echo   status             Show container status
echo   logs [service]     Show logs (default: magnesium-api)
echo   stop               Stop all services
echo   cleanup            Remove containers and optionally data
echo   test-gpu           Test GPU support
echo   test-config        Test local configuration loading
echo.
echo Training Commands:
echo   run-training       Run training job (uses TRAINING_MODE)
echo   train              Train standard models locally
echo   tune               Hyperparameter tuning locally
echo   autogluon          AutoGluon training locally
echo   optimize           Multi-model optimization locally
echo.
echo Prediction Commands:
echo   predict ^<file^> ^<model^>  Make prediction on file
echo.
echo Configuration:
echo   LOCAL_CONFIG_FILE Path to local config file (default: config/local.yml)
echo   USE_LOCAL_CONFIG  Whether to use local config (default: true)
echo.
echo Environment Variables:
echo   set SERVICE_NAME=magnesium-pipeline
echo   set COMPOSE_FILE=docker-compose.local.yml
echo   set ENVIRONMENT=development
echo.
echo Training Environment Variables:
echo   set TRAINING_MODE=autogluon
echo   set USE_GPU=true
echo   set USE_RAW_SPECTRAL=false
echo   set MODELS=xgboost,lightgbm,catboost
echo   set STRATEGY=simple_only
echo   set TRIALS=200
echo   set TIMEOUT=3600
echo.
echo Infrastructure Examples:
echo   %0 setup                          # First time setup
echo   %0 api                            # Start API server
echo   %0 dev                            # Start Jupyter + API for development
echo.
echo Training Examples:
echo   REM Basic AutoGluon training (uses local.yml config)
echo   %0 autogluon
echo.
echo   REM Use custom config file
echo   set LOCAL_CONFIG_FILE=config\custom.yml
echo   %0 train
echo.
echo   REM Disable config and use only environment variables
echo   set USE_LOCAL_CONFIG=false
echo   set STRATEGY=full_context
echo   %0 train
echo.
echo   REM Train specific models with GPU
echo   set MODELS=xgboost,lightgbm,catboost
echo   %0 train
echo.
echo   REM Multi-model optimization with custom parameters
echo   set MODELS=xgboost,lightgbm
echo   set STRATEGY=simple_only
echo   set TRIALS=200
echo   %0 optimize
echo.
echo   REM Hyperparameter tuning with raw spectral data
echo   set USE_RAW_SPECTRAL=true
echo   set MODELS=neural_network
echo   %0 tune
echo.
echo   REM Optimize XGBoost specifically
echo   set TRAINING_MODE=optimize-xgboost
echo   set TRIALS=300
echo   %0 run-training
echo.
echo Prediction Example:
echo   %0 predict sample.csv.txt models\model.pkl
echo.
echo URLs after starting:
echo   API Server:        http://localhost:8000
echo   API Documentation: http://localhost:8000/docs
echo   Jupyter Lab:       http://localhost:8888 (dev mode only)

:end
endlocal