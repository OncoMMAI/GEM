#!/bin/bash

# Setup script for Glioma Finetuning environment
# This script installs UV and creates necessary directories

# Exit on error
set -e

echo "Setting up environment for Glioma Finetuning project..."

# Check Python version (requires 3.8+)
if ! command -v python3 &> /dev/null || [[ $(python3 -c 'import sys; print(sys.version_info.major * 10 + sys.version_info.minor < 38)') == "1" ]]; then
    echo "Error: Python 3.8 or higher is required."
    echo "Your current Python version is:"
    python3 --version || echo "Python 3 not found"
    exit 1
fi

# Install UV if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create virtual environment with UV
echo "Creating virtual environment with UV..."
uv venv

# Install dependencies
echo "Installing dependencies (ensure CUDA 12.5 drivers present on Linux servers)..."
PLATFORM=$(uname -s)
if [ "$PLATFORM" = "Linux" ]; then
    echo "Linux detected: installing PyTorch CUDA 12.1 wheels (works with CUDA 12.5 drivers)"
    uv pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
else
    echo "Non-Linux platform detected ($PLATFORM): installing CPU-only torch"
    uv pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
fi
uv pip install -e .

echo "Installing build dependencies for flash-attn (Linux only)..."
uv pip install setuptools psutil
if [ "$PLATFORM" = "Linux" ]; then
    echo "Installing flash-attn..."
    UV_HTTP_TIMEOUT=1200 uv pip install 'flash-attn>=2.5.8' --no-build-isolation || echo "flash-attn install failed; continuing"
else
    echo "Skipping flash-attn on non-Linux platform"
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/{raw,training/mnrl,eval}
mkdir -p models
mkdir -p results 
mkdir -p figures
mkdir -p embeddings/{tumor_type,mgmt,survival}

# Create config directory if it doesn't exist
mkdir -p configs

# Make CLI script executable
chmod +x glioma.py

echo -e "\nEnvironment setup complete!"
echo -e "\nTo start using the Glioma toolkit, run:"
echo -e "uv run python glioma.py --help"

echo -e "\nExample commands:"
echo -e "  uv run python glioma.py prepare-data --config configs/dataset_config.yaml"
echo -e "  uv run python glioma.py finetune --config configs/finetune_config.yaml"
echo -e "  uv run python glioma.py generate --config configs/generate_config.yaml"
echo -e "  uv run python glioma.py evaluate --config configs/evaluation_config.yaml"
echo -e "  uv run python glioma.py visualize --config configs/visualize_config.yaml" 
echo -e "  uv run python glioma.py inference --config configs/inference_config.yaml"

