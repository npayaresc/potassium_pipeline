#!/bin/bash

# Local Docker Deployment Script for Magnesium Pipeline
# This script provides multiple deployment options for local Docker
# Aligned with gcp_deploy.sh for consistency

set -e

# Configuration
SERVICE_NAME=${SERVICE_NAME:-"magnesium-pipeline"}
CONTAINER_PREFIX=${CONTAINER_PREFIX:-"magnesium"}
COMPOSE_FILE=${COMPOSE_FILE:-"docker-compose.local.yml"}

# Local Configuration
LOCAL_CONFIG_FILE=${LOCAL_CONFIG_FILE:-"config/local.yml"}
ENVIRONMENT=${ENVIRONMENT:-"development"}
USE_LOCAL_CONFIG=${USE_LOCAL_CONFIG:-"true"}

# Training Configuration
TRAINING_MODE=${TRAINING_MODE:-"autogluon"}  # train, autogluon, tune, optimize-models, optimize-xgboost, optimize-autogluon
USE_GPU=${USE_GPU:-"true"}  # Enable GPU acceleration
USE_RAW_SPECTRAL=${USE_RAW_SPECTRAL:-"false"}  # Use raw spectral data
MODELS=${MODELS:-""}  # Specific models to train (comma-separated)
STRATEGY=${STRATEGY:-"simple_only"}  # Feature strategy: full_context, simple_only, Mg_only
TRIALS=${TRIALS:-""}  # Number of optimization trials
TIMEOUT=${TIMEOUT:-""}  # Timeout in seconds for optimization

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_blue() {
    echo -e "${BLUE}[DEPLOY]${NC} $1"
}

load_local_config() {
    if [ "$USE_LOCAL_CONFIG" = "true" ] && [ -f "$LOCAL_CONFIG_FILE" ]; then
        log_info "Loading local configuration from: $LOCAL_CONFIG_FILE"
        
        # Read local configuration using Python if available
        if command -v python3 &> /dev/null; then
            # Use Python to parse YAML more reliably
            local config_values=$(python3 -c "
import yaml
import sys
try:
    with open('$LOCAL_CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    # Pipeline configuration
    pipeline = config.get('pipeline', {})
    print(f\"LOCAL_DEFAULT_STRATEGY={pipeline.get('default_strategy', '')}\")  
    print(f\"LOCAL_TIME_LIMIT={pipeline.get('time_limit', '')}\")  
    print(f\"LOCAL_ENABLE_GPU={str(pipeline.get('enable_gpu', True)).lower()}\")  
    print(f\"LOCAL_ENABLE_SAMPLE_WEIGHTS={str(pipeline.get('enable_sample_weights', True)).lower()}\")  
    
    # Compute configuration
    compute = config.get('compute', {})
    print(f\"LOCAL_GPU_ENABLED={str(compute.get('gpu_enabled', True)).lower()}\")  
    print(f\"LOCAL_GPU_MEMORY_FRACTION={compute.get('gpu_memory_fraction', 0.9)}\")  
    
    # API configuration
    api = config.get('api', {})
    print(f\"LOCAL_API_TIMEOUT={api.get('timeout', 600)}\")
    
    # Monitoring configuration
    monitoring = config.get('monitoring', {})
    print(f\"LOCAL_LOG_LEVEL={monitoring.get('log_level', 'INFO')}\")
    
    # App configuration
    app = config.get('app', {})
    print(f\"LOCAL_ENVIRONMENT={app.get('environment', 'development')}\")
    print(f\"LOCAL_DEBUG={str(app.get('debug', False)).lower()}\")
    
except Exception as e:
    print(f\"# Error parsing YAML: {e}\", file=sys.stderr)
" 2>/dev/null)
            
            if [ $? -eq 0 ] && [ -n "$config_values" ]; then
                eval "$config_values"
                
                # Override defaults with local config values
                if [ -n "$LOCAL_DEFAULT_STRATEGY" ] && [ -z "${STRATEGY_OVERRIDE:-}" ]; then
                    STRATEGY="$LOCAL_DEFAULT_STRATEGY"
                fi
                if [ -n "$LOCAL_ENABLE_GPU" ] && [ -z "${USE_GPU_OVERRIDE:-}" ]; then
                    USE_GPU="$LOCAL_ENABLE_GPU"
                fi
                if [ -n "$LOCAL_TIME_LIMIT" ] && [ -z "${TIMEOUT_OVERRIDE:-}" ]; then
                    TIMEOUT="$LOCAL_TIME_LIMIT"
                fi
                if [ -n "$LOCAL_ENVIRONMENT" ] && [ -z "${ENVIRONMENT_OVERRIDE:-}" ]; then
                    ENVIRONMENT="$LOCAL_ENVIRONMENT"
                fi
                
                log_info "Applied local configuration:"
                log_info "  Environment: $ENVIRONMENT"
                log_info "  Default Strategy: $STRATEGY"
                log_info "  GPU Enabled: $USE_GPU"
                log_info "  Time Limit: ${TIMEOUT:-default}"
                log_info "  Log Level: ${LOCAL_LOG_LEVEL:-INFO}"
            else
                log_warn "Failed to parse local configuration with Python, using defaults"
            fi
        else
            log_warn "Python3 not available for YAML parsing, using defaults"
        fi
    else
        if [ "$USE_LOCAL_CONFIG" = "false" ]; then
            log_info "Local configuration disabled, using environment defaults"
        else
            log_warn "Local configuration file not found: $LOCAL_CONFIG_FILE, using defaults"
        fi
    fi
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null && ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    # Check for NVIDIA Docker support
    if nvidia-smi &> /dev/null && docker info | grep -q nvidia; then
        log_info "NVIDIA Docker support detected!"
        GPU_SUPPORT=true
    else
        log_warn "NVIDIA Docker support not detected. GPU acceleration will be disabled."
        GPU_SUPPORT=false
    fi
    
    log_info "Prerequisites check completed!"
}

create_directories() {
    log_info "Creating required directories..."
    
    # Create data directories
    mkdir -p data/{raw,processed,averaged_files_per_sample,cleansed_files_per_sample,reference_data}
    mkdir -p models/{autogluon}
    mkdir -p reports
    mkdir -p logs  
    mkdir -p bad_files
    mkdir -p bad_prediction_files
    mkdir -p config
    
    # Create .gitkeep files to preserve directory structure
    touch data/raw/.gitkeep
    touch data/processed/.gitkeep
    touch models/.gitkeep
    touch reports/.gitkeep
    touch logs/.gitkeep
    
    log_info "Directories created successfully!"
}

build_images() {
    log_info "Building Docker images..."
    
    # Build the main image
    log_blue "Building production image..."
    docker compose -f docker-compose.local.yml build magnesium-api
    
    if [ "$1" = "dev" ] || [ "$1" = "all" ]; then
        log_blue "Building development image..."
        docker compose -f docker-compose.local.yml build magnesium-dev
    fi
    
    log_info "Docker images built successfully!"
}

test_gpu_support() {
    log_info "Testing GPU support in containers..."
    
    if [ "$GPU_SUPPORT" = true ]; then
        log_blue "Running GPU test container..."
        if docker run --rm --gpus all magnesium-pipeline:latest python3 -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"; then
            log_info "GPU support verified in container!"
        else
            log_warn "GPU test failed. Models will run on CPU."
        fi
    else
        log_info "Skipping GPU test - no GPU support detected."
    fi
}

start_api_server() {
    log_info "Starting API server..."
    
    # Stop any existing containers
    docker compose -f docker-compose.local.yml down
    
    # Start API server
    docker compose -f docker-compose.local.yml up -d magnesium-api
    
    # Wait for health check
    log_blue "Waiting for API server to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health | grep -q "healthy"; then
            log_info "API server is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "API server failed to start within 30 seconds"
            docker compose -f docker-compose.local.yml logs magnesium-api
            exit 1
        fi
        sleep 1
    done
    
    log_info "API server started successfully at http://localhost:8000"
    log_info "API documentation available at http://localhost:8000/docs"
}

start_development() {
    log_info "Starting development environment..."
    
    # Stop any existing containers
    docker compose -f docker-compose.local.yml down
    
    # Start development environment
    docker compose -f docker-compose.local.yml --profile dev up -d
    
    log_info "Development environment started!"
    log_info "Jupyter Lab: http://localhost:8888"
    log_info "API Server: http://localhost:8001"
}

validate_training_config() {
    local valid_modes=("train" "autogluon" "tune" "optimize-models" "optimize-xgboost" "optimize-autogluon" "optimize-range-specialist")
    local valid_strategies=("full_context" "simple_only" "Mg_only")
    local valid_models=("ridge" "lasso" "random_forest" "xgboost" "lightgbm" "catboost" "extratrees" "neural_network" "neural_network_light" "svr")
    
    # Validate training mode
    if [[ ! " ${valid_modes[@]} " =~ " ${TRAINING_MODE} " ]]; then
        log_error "Invalid TRAINING_MODE: ${TRAINING_MODE}"
        log_error "Valid options: ${valid_modes[*]}"
        exit 1
    fi
    
    # Validate strategy
    if [[ ! " ${valid_strategies[@]} " =~ " ${STRATEGY} " ]]; then
        log_error "Invalid STRATEGY: ${STRATEGY}"
        log_error "Valid options: ${valid_strategies[*]}"
        exit 1
    fi
    
    # Validate models if specified
    if [ -n "${MODELS}" ]; then
        IFS=',' read -ra MODEL_ARRAY <<< "${MODELS}"
        for model in "${MODEL_ARRAY[@]}"; do
            if [[ ! " ${valid_models[@]} " =~ " ${model} " ]]; then
                log_error "Invalid model: ${model}"
                log_error "Valid models: ${valid_models[*]}"
                exit 1
            fi
        done
    fi
    
    # Log configuration summary
    log_info "Training Configuration:"
    log_info "  Mode: ${TRAINING_MODE}"
    log_info "  GPU: ${USE_GPU}"
    log_info "  Raw Spectral: ${USE_RAW_SPECTRAL}"
    log_info "  Strategy: ${STRATEGY}"
    if [ -n "${MODELS}" ]; then
        log_info "  Models: ${MODELS}"
    fi
    if [ -n "${TRIALS}" ]; then
        log_info "  Trials: ${TRIALS}"
    fi
    if [ -n "${TIMEOUT}" ]; then
        log_info "  Timeout: ${TIMEOUT}s"
    fi
}

build_training_command() {
    local base_cmd="python main.py ${TRAINING_MODE}"
    
    # Add GPU flag
    if [ "${USE_GPU}" = "true" ] && [ "$GPU_SUPPORT" = true ]; then
        base_cmd="${base_cmd} --gpu"
    fi
    
    # Add raw spectral flag
    if [ "${USE_RAW_SPECTRAL}" = "true" ]; then
        base_cmd="${base_cmd} --raw-spectral"
    fi
    
    # Add models (convert comma-separated to space-separated)
    if [ -n "${MODELS}" ]; then
        local models_array=(${MODELS//,/ })
        base_cmd="${base_cmd} --models ${models_array[@]}"
    fi
    
    # Add strategy for optimization commands
    case "${TRAINING_MODE}" in
        "optimize-xgboost"|"optimize-autogluon"|"optimize-models"|"optimize-range-specialist")
            base_cmd="${base_cmd} --strategy ${STRATEGY}"
            ;;
    esac
    
    # Add trials for optimization commands
    if [ -n "${TRIALS}" ] && [[ "${TRAINING_MODE}" == *"optimize"* ]]; then
        base_cmd="${base_cmd} --trials ${TRIALS}"
    fi
    
    # Add timeout for optimization commands
    if [ -n "${TIMEOUT}" ] && [[ "${TRAINING_MODE}" == *"optimize"* ]]; then
        base_cmd="${base_cmd} --timeout ${TIMEOUT}"
    fi
    
    echo "${base_cmd}"
}

run_training() {
    # Load local configuration first
    load_local_config
    
    # Validate training configuration
    validate_training_config
    
    log_info "Running training pipeline: ${TRAINING_MODE}"
    
    # Build the training command
    local training_cmd=$(build_training_command)
    log_info "Training command: ${training_cmd}"
    
    # Run training in container
    docker compose -f ${COMPOSE_FILE} run --rm \
        --name magnesium-training-$(date +%s) \
        magnesium-train \
        "uv run --prerelease=allow ${training_cmd}"
    
    log_info "Training completed!"
}

run_prediction() {
    local input_file=$1
    local model_path=$2
    
    if [ -z "$input_file" ] || [ -z "$model_path" ]; then
        log_error "Usage: $0 predict <input_file> <model_path>"
        exit 1
    fi
    
    log_info "Running prediction..."
    
    # Copy input file to container-accessible location
    cp "$input_file" ./data/
    filename=$(basename "$input_file")
    
    # Run prediction
    curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@./data/$filename" \
        -F "model_path=$model_path" | jq '.'
    
    # Clean up
    rm -f "./data/$filename"
}

show_status() {
    log_info "Checking container status..."
    docker compose -f docker-compose.local.yml ps
    
    log_info "Checking API health..."
    if curl -s http://localhost:8000/health | jq '.' 2>/dev/null; then
        log_info "API is healthy!"
    else
        log_warn "API is not responding"
    fi
}

show_logs() {
    local service=${1:-"magnesium-api"}
    log_info "Showing logs for $service..."
    docker compose -f docker-compose.local.yml logs -f $service
}

stop_services() {
    log_info "Stopping all services..."
    docker compose -f docker-compose.local.yml down
    log_info "All services stopped."
}

cleanup() {
    log_info "Cleaning up..."
    
    # Stop and remove containers
    docker compose -f docker-compose.local.yml down --volumes --rmi local
    
    # Remove generated data (optional)
    read -p "Remove generated data? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf bad_files/* bad_prediction_files/* logs/* reports/*
        log_info "Generated data removed"
    fi
    
    log_info "Cleanup completed"
}

# Training command shortcuts
train() {
    export TRAINING_MODE="train"
    run_training
}

tune() {
    export TRAINING_MODE="tune"
    run_training
}

autogluon() {
    export TRAINING_MODE="autogluon"
    run_training
}

optimize() {
    export TRAINING_MODE="optimize-models"
    run_training
}

test_config() {
    log_info "Testing local configuration loading..."
    log_info "LOCAL_CONFIG_FILE: $LOCAL_CONFIG_FILE"
    load_local_config
    validate_training_config
    log_info "Configuration test completed successfully!"
}

show_help() {
    echo "Local Docker Deployment for Magnesium Pipeline"
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Infrastructure Commands:"
    echo "  setup              Setup environment and build images"
    echo "  build [dev|all]    Build Docker images"
    echo "  api                Start API server"
    echo "  dev                Start development environment with Jupyter"
    echo "  status             Show container status"
    echo "  logs [service]     Show logs (default: magnesium-api)"
    echo "  stop               Stop all services"
    echo "  cleanup            Remove containers and optionally data"
    echo "  test-gpu           Test GPU support"
    echo "  test-config        Test local configuration loading"
    echo
    echo "Training Commands:"
    echo "  run-training       Run training job (uses TRAINING_MODE)"
    echo "  train              Train standard models locally"
    echo "  tune               Hyperparameter tuning locally"
    echo "  autogluon          AutoGluon training locally"
    echo "  optimize           Multi-model optimization locally"
    echo
    echo "Prediction Commands:"
    echo "  predict <file> <model>  Make prediction on file"
    echo
    echo
    echo "Configuration:"
    echo "  LOCAL_CONFIG_FILE Path to local config file (default: config/local.yml)"
    echo "  USE_LOCAL_CONFIG  Whether to use local config (default: true)"
    echo
    echo "Environment Variables:"
    echo "  SERVICE_NAME      Service name (default: magnesium-pipeline)"
    echo "  COMPOSE_FILE      Docker compose file (default: docker-compose.local.yml)"
    echo "  ENVIRONMENT       Environment setting (default: from local.yml or development)"
    echo
    echo "Training Environment Variables:"
    echo "  TRAINING_MODE     Training mode (default: autogluon)"
    echo "                    Options: train, autogluon, tune, optimize-models,"
    echo "                             optimize-xgboost, optimize-autogluon"
    echo "  USE_GPU           Enable GPU acceleration (default: true)"
    echo "  USE_RAW_SPECTRAL  Use raw spectral data (default: false)"
    echo "  MODELS            Specific models to train (comma-separated)"
    echo "                    Options: ridge,lasso,random_forest,xgboost,lightgbm,"
    echo "                             catboost,extratrees,neural_network,neural_network_light,svr"
    echo "  STRATEGY          Feature strategy (default: from local.yml or simple_only)"
    echo "                    Options: full_context, simple_only, Mg_only"
    echo "  TRIALS            Number of optimization trials"
    echo "  TIMEOUT           Timeout in seconds for optimization"
    echo
    echo "Infrastructure Examples:"
    echo "  $0 setup                          # First time setup"
    echo "  $0 api                            # Start API server"
    echo "  $0 dev                            # Start Jupyter + API for development"
    echo
    echo "Training Examples:"
    echo "  # Basic AutoGluon training (uses local.yml config)"
    echo "  $0 autogluon"
    echo
    echo "  # Use custom config file"
    echo "  LOCAL_CONFIG_FILE=config/custom.yml $0 train"
    echo
    echo "  # Disable config and use only environment variables"
    echo "  USE_LOCAL_CONFIG=false STRATEGY=full_context $0 train"
    echo
    echo "  # Train specific models with GPU"
    echo "  MODELS=xgboost,lightgbm,catboost $0 train"
    echo
    echo "  # Multi-model optimization with custom parameters"
    echo "  MODELS=xgboost,lightgbm STRATEGY=simple_only TRIALS=200 $0 optimize"
    echo
    echo "  # Hyperparameter tuning with raw spectral data"
    echo "  USE_RAW_SPECTRAL=true MODELS=neural_network $0 tune"
    echo
    echo "  # Optimize XGBoost specifically"
    echo "  TRAINING_MODE=optimize-xgboost TRIALS=300 $0 run-training"
    echo
    echo "Prediction Example:"
    echo "  $0 predict sample.csv.txt /app/models/model.pkl"
    echo
    echo "URLs after starting:"
    echo "  API Server:        http://localhost:8000"
    echo "  API Documentation: http://localhost:8000/docs"
    echo "  Jupyter Lab:       http://localhost:8888 (dev mode only)"
}

# Main script logic
case "${1:-help}" in
    setup)
        check_prerequisites
        create_directories
        build_images all
        log_info "Setup completed! Run '$0 api' to start the API server."
        ;;
    build)
        check_prerequisites
        build_images $2
        ;;
    api)
        check_prerequisites
        create_directories
        start_api_server
        ;;
    dev)
        check_prerequisites  
        create_directories
        build_images dev
        start_development
        ;;
    run-training)
        check_prerequisites
        run_training
        ;;
    train)
        check_prerequisites
        train
        ;;
    tune)
        check_prerequisites
        tune
        ;;
    autogluon)
        check_prerequisites
        autogluon
        ;;
    optimize)
        check_prerequisites
        optimize
        ;;
    predict)
        run_prediction $2 $3
        ;;
    test-gpu)
        check_prerequisites
        test_gpu_support
        ;;
    test-config)
        test_config
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs $2
        ;;
    stop)
        stop_services
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac