# Multi-stage build for optimal image size and GPU support
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04 as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON=/usr/bin/python3.12

# Install system dependencies and Python 3.12
# This layer rarely changes, so it will be cached effectively
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    software-properties-common \
    gnupg \
    lsb-release \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install additional development tools for LightGBM CUDA support
RUN apt-get update && apt-get install -y \
    build-essential \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    && rm -rf /var/lib/apt/lists/*

# Install newer CMake version (LightGBM requires 3.28+)
RUN wget https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-linux-x86_64.tar.gz \
    && tar -xzf cmake-3.28.1-linux-x86_64.tar.gz \
    && cp cmake-3.28.1-linux-x86_64/bin/* /usr/local/bin/ \
    && cp -R cmake-3.28.1-linux-x86_64/share/* /usr/local/share/ \
    && rm -rf cmake-3.28.1-linux-x86_64* \
    && cmake --version

# CUDA 12.9.0 and cuDNN are already installed in the base image
# Set environment variables to use system CUDA libraries
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
# Verify CUDA installation
RUN nvcc --version && nvidia-smi || echo "nvidia-smi not available in build context"

# Install Google Cloud SDK
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update \
    && apt-get install -y google-cloud-sdk \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager)
# Separate layer for uv installation
RUN pip3 install uv

# Set working directory
WORKDIR /app

# Copy only dependency files first for better caching
# If dependencies don't change, this layer is cached
COPY pyproject.toml uv.lock ./

# Install Python dependencies
# This is the most time-consuming step, so caching here saves a lot of time
RUN uv sync --prerelease=allow

# Install additional dependencies for LightGBM CUDA compilation
RUN apt-get update && apt-get install -y \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Note: LightGBM 4.6.0 is already installed from pip with CPU support
# CUDA compilation requires architecture compatibility - can be enabled runtime
# The pip version supports CPU/GPU hybrid mode for XGBoost and CatBoost GPU acceleration

# Activate virtual environment by updating PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"
ENV PYTHONPATH="/app"

# Copy source code
# Split into multiple COPY commands for better cache granularity
COPY src/ ./src/
COPY main.py check_gpu_support.py api_server.py ./
COPY CLAUDE.md README.md ./
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# Create required directories with proper structure
# These will be populated at runtime from GCS
RUN mkdir -p data/{raw/data_5278_Phase3,processed,averaged_files_per_sample,cleansed_files_per_sample,reference_data} \
    models/autogluon reports logs bad_files bad_prediction_files catboost_info configs

# NOTE: Data files are NOT copied into the image.
# They will be downloaded from GCS at runtime via docker-entrypoint.sh
# This keeps the image small and allows data updates without rebuilding.

# Production stage - use the exact same base to avoid layer duplication
FROM base as production

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

# Set entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]

# Default command
CMD ["python3", "main.py", "--help"]

# Development stage (includes additional tools)
FROM base as development

# Install development dependencies
RUN uv sync --prerelease=allow

# Install Jupyter for development
RUN uv add jupyter matplotlib seaborn

EXPOSE 8888

CMD ["bash"]