#!/bin/bash
# This script runs SEQUENTIAL evaluation for all fine-tuned models
# RUN THIS ONLY AFTER BOTH run_group_1.sh AND run_group_2.sh HAVE COMPLETED

set -e

# --- Configuration ---
GLIOMA_SCRIPT="glioma.py"
BASE_EVAL_CONFIG="scripts/experiments/configs/evaluation_config.yaml"
# Allow overriding default locations via environment variables
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-models/results}"
EMBEDDINGS_BASE_DIR="${EMBEDDINGS_BASE_DIR:-embeddings/experiments}"

# List of all experiments to evaluate
ALL_EXPERIMENTS=(
    "baseline"
    "dataset_10k"
    "dataset_40k"
    "task_classification"
    "task_separation"
    "batch_sampler_default"
    "lora_false"
)

# Extract data paths from the base evaluation config (robustly)
TYPE_PATH=$(awk -F': ' '/^\s*type_path:/ {gsub("'", "", $2); print $2}' "$BASE_EVAL_CONFIG")
MGMT_PATH=$(awk -F': ' '/^\s*mgmt_path:/ {gsub("'", "", $2); print $2}' "$BASE_EVAL_CONFIG")
OS_PATH=$(awk -F': ' '/^\s*os_path:/ {gsub("'", "", $2); print $2}' "$BASE_EVAL_CONFIG")
TEXT_COL=$(awk -F': ' '/^\s*text_column:/ {gsub("'", "", $2); print $2}' "$BASE_EVAL_CONFIG")

echo "=========================================="
echo "Starting Sequential Evaluation Pipeline"
echo "=========================================="
echo "Evaluation data:"
echo "  - Tumor type: $TYPE_PATH"
echo "  - MGMT: $MGMT_PATH"
echo "  - OS: $OS_PATH"
echo ""

# Process each experiment
for MODEL_NAME in "${ALL_EXPERIMENTS[@]}"; do
    echo "=========================================="
    echo "Processing: $MODEL_NAME"
    echo "=========================================="
    
    FINETUNED_MODEL_PATH="$RESULTS_BASE_DIR/$MODEL_NAME/final"
    FINETUNED_MODEL_SAFETENSORS="$FINETUNED_MODEL_PATH/model.safetensors"
    
    # Check if model exists
    if [ ! -f "$FINETUNED_MODEL_SAFETENSORS" ]; then
        echo "⚠️  Model not found: $FINETUNED_MODEL_SAFETENSORS"
        echo "   Skipping $MODEL_NAME"
        continue
    fi
    
    # --- 1. Generate Embeddings ---
    echo ""
    echo "1. Generating embeddings for: $MODEL_NAME"
    echo "----------------------------------------"
    
    EMBEDDING_DIR="$EMBEDDINGS_BASE_DIR/$MODEL_NAME"
    
    # Create temporary config for embedding generation
    TEMP_GEN_CONFIG=$(mktemp)
    cat > "$TEMP_GEN_CONFIG" << EOL
global:
  seed: 42

data:
  tumor_type_path: $TYPE_PATH
  mgmt_path: $MGMT_PATH
  survival_path: $OS_PATH
  text_column: $TEXT_COL

model:
  jina:
    name: 'jinaai/jina-embeddings-v3'
    checkpoint_path: '$FINETUNED_MODEL_SAFETENSORS'
    trust_remote_code: true
    batch_size: 32
    max_length: 2048
    task: 'text-matching'
    device_ids: 0

output:
  output_dir: '$EMBEDDING_DIR'
EOL

    uv run python "$GLIOMA_SCRIPT" generate --config "$TEMP_GEN_CONFIG"
    rm "$TEMP_GEN_CONFIG"
    echo "✓ Embedding generation complete"
    
    # --- 2. Run Evaluation ---
    echo ""
    echo "2. Evaluating model: $MODEL_NAME"
    echo "----------------------------------------"
    
    EVAL_RESULTS_DIR="results/experiments/${MODEL_NAME}"
    
    # Create temporary eval config
    TEMP_EVAL_CONFIG=$(mktemp)
    cat > "$TEMP_EVAL_CONFIG" << EOL
global:
  seed: 7

model:
  base_model: 'jinaai/jina-embeddings-v3'
  finetuned_model: '$FINETUNED_MODEL_PATH'
  trust_remote_code: true

embedding:
  directory: '$EMBEDDING_DIR'
  batch_size: 32
  max_length: 2048
  task: 'text-matching'

data:
  type_path: $TYPE_PATH
  mgmt_path: $MGMT_PATH
  os_path: $OS_PATH
  text_column: $TEXT_COL

evaluation:
  n_splits: 5
  random_state: 7
  n_estimators: 1000
  baseline_model: 'jina_base'

output:
  results_dir: '$EVAL_RESULTS_DIR'
EOL

    uv run python "$GLIOMA_SCRIPT" evaluate --config "$TEMP_EVAL_CONFIG"
    rm "$TEMP_EVAL_CONFIG"
    echo "✓ Evaluation complete"
    echo ""
done

echo "=========================================="
echo "All evaluations completed successfully!"
echo "=========================================="
echo ""
echo "Results saved to: results/experiments/"
echo ""
echo "To compare results across experiments:"
echo "  python scripts/experiments/compare_results.py"
