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
echo "Installing dependencies..."
uv pip install -e .

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

