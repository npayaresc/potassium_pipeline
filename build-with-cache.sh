#!/bin/bash

# Build script with advanced Docker caching
# Uses BuildKit for optimal layer caching

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_blue() {
    echo -e "${BLUE}[BUILD]${NC} $1"
}

# Enable Docker BuildKit
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Use new Compose with Bake integration if available
export COMPOSE_BAKE=true

# Create cache directory if it doesn't exist
mkdir -p /tmp/docker-cache

log_info "Docker BuildKit enabled for optimal caching"

# Function to build with caching
build_with_cache() {
    local service=$1
    log_blue "Building $service with layer caching..."
    
    if command -v docker-bake &> /dev/null; then
        # Use docker bake if available
        docker buildx bake -f docker-bake.hcl $service
    else
        # Fallback to docker-compose with BuildKit
        docker compose build --progress=plain $service
    fi
}

# Parse arguments
if [ $# -eq 0 ]; then
    log_info "Building all services with caching..."
    build_with_cache "magnesium-api"
    build_with_cache "magnesium-pipeline"
    build_with_cache "magnesium-dev"
else
    for service in "$@"; do
        build_with_cache "$service"
    done
fi

log_info "Build completed with caching!"

# Show cache usage
if [ -d /tmp/docker-cache ]; then
    log_info "Cache size: $(du -sh /tmp/docker-cache | cut -f1)"
fi

# Optionally prune old cache (uncomment if needed)
# docker builder prune --filter type=exec.cachemount --keep-storage=10GB