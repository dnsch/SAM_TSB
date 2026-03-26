# =============================================================================

# Multi-stage Dockerfile with NVIDIA GPU + UV support (Blackwell Compatible)

# Optimized for mounting entire project directory at runtime

# =============================================================================

# -----------------------------------------------------------------------------

# Stage 1: Base with CUDA 12.8 (Blackwell support)

# -----------------------------------------------------------------------------

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# -----------------------------------------------------------------------------

# Stage 2: UV Installation

# -----------------------------------------------------------------------------

FROM base AS uv-installer

ENV UV_VERSION=0.5.5
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# -----------------------------------------------------------------------------

# Stage 3: Dependencies

# -----------------------------------------------------------------------------

FROM uv-installer AS dependencies

WORKDIR /app

COPY pyproject.toml ./
COPY requirements_docker.txt ./


RUN uv venv /opt/venv --python 3.11
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Configure UV for slower networks

ENV UV_HTTP_TIMEOUT=3000
ENV UV_CONCURRENT_DOWNLOADS=4

# Install PyTorch nightly with CUDA 12.8 (Blackwell support)

RUN uv pip install --upgrade pip && \
    uv pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    python -c "import torch; print('Installed PyTorch:', torch.__version__); assert 'cu128' in torch.__version__, 'Wrong CUDA version!'" && \
    uv pip install -r requirements_docker.txt && \
    python -c "import torch; print('Final PyTorch:', torch.__version__); assert 'cu128' in torch.__version__, 'requirements_docker.txt overwrote PyTorch!'"

# -----------------------------------------------------------------------------

# Stage 4: Download Datasets

# -----------------------------------------------------------------------------

FROM dependencies AS data-downloader

WORKDIR /app

RUN uv pip install gdown

COPY scripts/download_autoformer_dataset.py /app/scripts/

# Download to /data (outside /app)

ENV DATA_DIR=/data
RUN mkdir -p /data/autoformer_datasets && \
    python /app/scripts/download_autoformer_dataset.py && \
    echo "Dataset download complete. Files:" && \
    ls -la /data/autoformer_datasets/

# -----------------------------------------------------------------------------

# Stage 5: Final Application

# -----------------------------------------------------------------------------

FROM dependencies AS final

WORKDIR /app

COPY --from=data-downloader /data /data

# Clone third-party repos OUTSIDE /app

RUN git clone https://github.com/dnsch/pyhessian.git /third_party/utils/pyhessian && \
    git clone https://github.com/dnsch/loss_landscape.git /third_party/utils/loss_landscape

RUN chmod -R 777 /opt/venv

# Include third_party paths for internal imports

ENV PYTHONPATH="/app:/third_party/utils/pyhessian:/third_party/utils/loss_landscape"
ENV PYTHONHASHSEED=42
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# Support running as non-root user

ENV HOME=/tmp
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache
ENV XDG_CACHE_HOME=/tmp/.cache

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
