#!/bin/bash

set -e

echo "Setting up LLM2Vec environment (separate from base project)..."

# Check Python
if ! command -v python3 &> /dev/null; then
  echo "Python3 is required."; exit 1
fi

# Install uv if missing
if ! command -v uv &> /dev/null; then
  echo "Installing UV package manager..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi

cd "$(dirname "$0")"

echo "Creating virtual environment..."
uv venv

PLATFORM=$(uname -s)
if [ "$PLATFORM" = "Linux" ]; then
  echo "Installing PyTorch CUDA 12.1 wheels (works with CUDA 12.5 drivers)..."
  uv pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
else
  echo "Non-Linux platform detected ($PLATFORM): installing CPU-only torch..."
  uv pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
fi

echo "Installing llm2vec in editable mode..."
uv pip install -e .

if [ "$PLATFORM" = "Linux" ]; then
  echo "Installing flash-attn (optional, speeds up training)..."
  UV_HTTP_TIMEOUT=1200 uv pip install 'flash-attn>=2.5.8' --no-build-isolation || echo "flash-attn install failed; continuing with SDPA attention"
fi

echo "LLM2Vec environment setup complete."
echo "To run:"
echo "  uv run torchrun --nproc_per_node=8 experiments/run_mntp.py train_configs/mntp/MetaLlama3.1.json"

