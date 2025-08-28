#!/bin/bash
# Optimized build strategies for faster Docker builds

# Strategy 1: Use BuildKit for better caching
echo "Building with BuildKit optimizations..."
DOCKER_BUILDKIT=1 docker build \
    --cache-from magnesium-pipeline:latest \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    -t magnesium-pipeline:latest \
    -f Dockerfile .

# Strategy 2: Create a base image with all dependencies
# Run this once to create a base image with all Python packages
create_base_image() {
    cat > Dockerfile.base << 'EOF'
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    build-essential cmake git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip3 install uv

# Pre-install all heavy dependencies
WORKDIR /tmp
COPY pyproject.toml uv.lock ./
RUN uv sync --prerelease=allow && \
    rm -rf /tmp/* /root/.cache/*

# This image now has all dependencies pre-installed
EOF
    
    docker build -t magnesium-base:latest -f Dockerfile.base .
    echo "Base image created. Update your Dockerfile to use:"
    echo "FROM magnesium-base:latest"
}

# Strategy 3: Use Docker layer caching with unchanged files first
echo "Tips for faster builds:"
echo "1. Keep dependency files (pyproject.toml, uv.lock) unchanged"
echo "2. Use .dockerignore to exclude unnecessary files"
echo "3. Order Dockerfile commands from least to most frequently changed"
echo "4. Use multi-stage builds to reduce final image size"

# Check if user wants to create base image
if [ "$1" == "--create-base" ]; then
    create_base_image
fi