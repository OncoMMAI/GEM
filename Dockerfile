# Dockerfile for Glioma Embedding Project
# Base image with CUDA support for GPU operations
FROM nvidia/cuda:12.5.0-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

RUN uv venv
# Install Torch with CUDA 12.1 wheels (works with CUDA 12.5 drivers)
RUN uv pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
# Install the rest of the requirements
RUN uv pip install -r requirements.txt

# Install flash-attn v2 built against Torch 2.3.x
RUN uv pip install setuptools psutil
RUN UV_HTTP_TIMEOUT=1200 uv pip install 'flash-attn>=2.5.8' --no-build-isolation

# Copy the rest of the application
COPY . .

# Install the package in editable mode
RUN uv pip install -e .

# Create necessary directories
RUN mkdir -p data/{raw,training/mnrl,eval} \
    models \
    results \
    figures \
    embeddings/{tumor_type,mgmt,survival} \
    configs

# Set permissions
RUN chmod +x glioma.py

# Expose port for Jupyter if needed
EXPOSE 8888

# Expose port for vLLM server
EXPOSE 8000

# Default command
CMD ["uv", "run", "python", "glioma.py", "--help"]
