@echo off
setlocal enabledelayedexpansion

REM Local Deployment Script for Magnesium Pipeline (Windows)
REM This script provides local training and deployment options using Docker

REM Configuration
set "COMPOSE_FILE=docker-compose.local.yml"
set "IMAGE_NAME=magnesium-pipeline:latest"
set "CONTAINER_NAME=magnesium-local"
set "DATA_DIR=data"
set "MODELS_DIR=models"
set "REPORTS_DIR=reports"
set "LOGS_DIR=logs"

REM Training Configuration
set "TRAINING_MODE=train"
set "USE_GPU=false"
set "USE_RAW_SPECTRAL=false"
set "MODELS="
set "STRATEGY=full_context"
set "TRIALS="
set "TIMEOUT="

REM Colors for output (Windows doesn't support ANSI colors in batch by default)
REM Using echo with specific formatting instead

echo [INFO] Magnesium Pipeline Local Deployment (Windows)
echo.

REM Function to display help
if "%1"=="help" goto :show_help
if "%1"=="--help" goto :show_help
if "%1"=="-h" goto :show_help
if "%1"=="/?" goto :show_help

REM Check prerequisites
:check_prerequisites
echo [INFO] Checking prerequisites...

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed or not in PATH
    echo Please install Docker Desktop for Windows from https://www.docker.com/products/docker-desktop
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running
    echo Please start Docker Desktop
    exit /b 1
)

echo [INFO] Prerequisites check passed!
echo.

REM Parse command
if "%1"=="" goto :show_help

if "%1"=="build" goto :build_image
if "%1"=="build-fresh" goto :build_fresh
if "%1"=="train" goto :run_training
if "%1"=="autogluon" goto :run_autogluon
if "%1"=="tune" goto :run_tune
if "%1"=="optimize" goto :run_optimize
if "%1"=="shell" goto :run_shell
if "%1"=="api" goto :run_api
if "%1"=="clean" goto :clean_artifacts
if "%1"=="stop" goto :stop_containers
if "%1"=="logs" goto :show_logs
if "%1"=="test-gpu" goto :test_gpu

echo [ERROR] Unknown command: %1
goto :show_help

:build_image
echo [INFO] Building Docker image...
docker build -t %IMAGE_NAME% .
if errorlevel 1 (
    echo [ERROR] Failed to build Docker image
    exit /b 1
)
echo [INFO] Docker image built successfully: %IMAGE_NAME%
goto :end

:build_fresh
echo [INFO] Building Docker image without cache...
docker build --no-cache -t %IMAGE_NAME% .
if errorlevel 1 (
    echo [ERROR] Failed to build Docker image
    exit /b 1
)
echo [INFO] Docker image built successfully: %IMAGE_NAME%
goto :end

:run_training
echo [INFO] Running training with standard models...
set "training_cmd=python main.py train"

REM Parse additional arguments
:parse_train_args
shift
if "%1"=="" goto :execute_training
if "%1"=="--gpu" (
    set "training_cmd=!training_cmd! --gpu"
    set "USE_GPU=true"
    goto :parse_train_args
)
if "%1"=="--raw-spectral" (
    set "training_cmd=!training_cmd! --raw-spectral"
    goto :parse_train_args
)
if "%1"=="--models" (
    set "training_cmd=!training_cmd! --models %2 %3 %4 %5"
    shift
    shift
    shift
    shift
    goto :parse_train_args
)
if "%1"=="--strategy" (
    set "training_cmd=!training_cmd! --strategy %2"
    shift
    goto :parse_train_args
)
goto :parse_train_args

:execute_training
echo [INFO] Training command: !training_cmd!
docker run --rm ^
    -v "%cd%\data:/app/data" ^
    -v "%cd%\models:/app/models" ^
    -v "%cd%\reports:/app/reports" ^
    -v "%cd%\logs:/app/logs" ^
    --name %CONTAINER_NAME%-training ^
    %IMAGE_NAME% ^
    !training_cmd!
goto :end

:run_autogluon
echo [INFO] Running AutoGluon training...
set "training_cmd=python main.py autogluon"

REM Parse additional arguments
:parse_autogluon_args
shift
if "%1"=="" goto :execute_autogluon
if "%1"=="--gpu" (
    set "training_cmd=!training_cmd! --gpu"
    set "USE_GPU=true"
    goto :parse_autogluon_args
)
if "%1"=="--raw-spectral" (
    set "training_cmd=!training_cmd! --raw-spectral"
    goto :parse_autogluon_args
)
goto :parse_autogluon_args

:execute_autogluon
echo [INFO] AutoGluon command: !training_cmd!
docker run --rm ^
    -v "%cd%\data:/app/data" ^
    -v "%cd%\models:/app/models" ^
    -v "%cd%\reports:/app/reports" ^
    -v "%cd%\logs:/app/logs" ^
    --name %CONTAINER_NAME%-autogluon ^
    %IMAGE_NAME% ^
    !training_cmd!
goto :end

:run_tune
echo [INFO] Running hyperparameter tuning...
set "training_cmd=python main.py tune"

REM Parse additional arguments
:parse_tune_args
shift
if "%1"=="" goto :execute_tune
if "%1"=="--gpu" (
    set "training_cmd=!training_cmd! --gpu"
    goto :parse_tune_args
)
if "%1"=="--models" (
    set "training_cmd=!training_cmd! --models %2 %3 %4 %5"
    shift
    shift
    shift
    shift
    goto :parse_tune_args
)
if "%1"=="--trials" (
    set "training_cmd=!training_cmd! --trials %2"
    shift
    goto :parse_tune_args
)
goto :parse_tune_args

:execute_tune
echo [INFO] Tuning command: !training_cmd!
docker run --rm ^
    -v "%cd%\data:/app/data" ^
    -v "%cd%\models:/app/models" ^
    -v "%cd%\reports:/app/reports" ^
    -v "%cd%\logs:/app/logs" ^
    --name %CONTAINER_NAME%-tuning ^
    %IMAGE_NAME% ^
    !training_cmd!
goto :end

:run_optimize
echo [INFO] Running optimization...
set "training_cmd=python main.py optimize-models"

REM Parse additional arguments
:parse_optimize_args
shift
if "%1"=="" goto :execute_optimize
if "%1"=="--gpu" (
    set "training_cmd=!training_cmd! --gpu"
    goto :parse_optimize_args
)
if "%1"=="--models" (
    set "training_cmd=!training_cmd! --models %2 %3 %4 %5"
    shift
    shift
    shift
    shift
    goto :parse_optimize_args
)
if "%1"=="--strategy" (
    set "training_cmd=!training_cmd! --strategy %2"
    shift
    goto :parse_optimize_args
)
if "%1"=="--trials" (
    set "training_cmd=!training_cmd! --trials %2"
    shift
    goto :parse_optimize_args
)
goto :parse_optimize_args

:execute_optimize
echo [INFO] Optimization command: !training_cmd!
docker run --rm ^
    -v "%cd%\data:/app/data" ^
    -v "%cd%\models:/app/models" ^
    -v "%cd%\reports:/app/reports" ^
    -v "%cd%\logs:/app/logs" ^
    --name %CONTAINER_NAME%-optimization ^
    %IMAGE_NAME% ^
    !training_cmd!
goto :end

:run_shell
echo [INFO] Starting interactive shell...
docker run --rm -it ^
    -v "%cd%\data:/app/data" ^
    -v "%cd%\models:/app/models" ^
    -v "%cd%\reports:/app/reports" ^
    -v "%cd%\logs:/app/logs" ^
    --name %CONTAINER_NAME%-shell ^
    %IMAGE_NAME% ^
    /bin/bash
goto :end

:run_api
echo [INFO] Starting API server...
docker run --rm -d ^
    -p 8000:8000 ^
    -v "%cd%\data:/app/data" ^
    -v "%cd%\models:/app/models" ^
    -v "%cd%\reports:/app/reports" ^
    -v "%cd%\logs:/app/logs" ^
    --name %CONTAINER_NAME%-api ^
    %IMAGE_NAME% ^
    python api_server.py
echo [INFO] API server started at http://localhost:8000
echo [INFO] View logs with: docker logs -f %CONTAINER_NAME%-api
goto :end

:clean_artifacts
echo [INFO] Cleaning generated artifacts...
if exist "bad_files" rmdir /s /q "bad_files"
if exist "bad_prediction_files" rmdir /s /q "bad_prediction_files"
if exist "catboost_info" rmdir /s /q "catboost_info"
echo [INFO] Cleanup completed
goto :end

:stop_containers
echo [INFO] Stopping all magnesium containers...
for /f "tokens=*" %%i in ('docker ps -q --filter "name=%CONTAINER_NAME%"') do (
    docker stop %%i
)
echo [INFO] All containers stopped
goto :end

:show_logs
echo [INFO] Showing container logs...
set "container_suffix=%2"
if "%container_suffix%"=="" set "container_suffix=training"
docker logs -f %CONTAINER_NAME%-%container_suffix%
goto :end

:test_gpu
echo [INFO] Testing GPU support in container...
docker run --rm ^
    --gpus all ^
    %IMAGE_NAME% ^
    python check_gpu_support.py
goto :end

:show_help
echo Local Deployment Script for Magnesium Pipeline (Windows)
echo.
echo Usage: local-deploy.bat [COMMAND] [OPTIONS]
echo.
echo Commands:
echo   build           Build Docker image
echo   build-fresh     Build Docker image without cache
echo   train           Run standard model training
echo   autogluon       Run AutoGluon training
echo   tune            Run hyperparameter tuning
echo   optimize        Run model optimization
echo   shell           Start interactive shell in container
echo   api             Start API server
echo   clean           Clean generated artifacts
echo   stop            Stop all running containers
echo   logs [suffix]   Show container logs
echo   test-gpu        Test GPU support in container
echo   help            Show this help message
echo.
echo Training Options:
echo   --gpu           Enable GPU acceleration
echo   --raw-spectral  Use raw spectral data
echo   --models        Specify models (space-separated)
echo   --strategy      Feature strategy (full_context, simple_only, Mg_only)
echo   --trials        Number of optimization trials
echo   --timeout       Timeout in seconds
echo.
echo Examples:
echo   local-deploy.bat build
echo   local-deploy.bat train --gpu --models xgboost lightgbm
echo   local-deploy.bat autogluon --gpu
echo   local-deploy.bat tune --models neural_network --trials 100
echo   local-deploy.bat optimize --strategy simple_only --trials 200
echo.
echo Environment Variables:
echo   Set these before running the script to override defaults:
echo   set IMAGE_NAME=my-image:latest
echo   set CONTAINER_NAME=my-container
echo.
goto :end

:end
endlocal