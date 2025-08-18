#!/bin/bash
# This script runs fine-tuning jobs for GPU group 2 (GPUs 4,5,6,7)
# Run this in its own terminal or tmux pane

set -e

# --- Configuration ---
FINETUNE_SCRIPT="glioma.py"
CONFIG_DIR="scripts/experiments/configs"
GPUS="4,5,6,7"
PORT="29501"

# Define the list of jobs for this worker
JOB_QUEUE=(
    "$CONFIG_DIR/task_separation.yaml"
    "$CONFIG_DIR/batch_sampler_default.yaml"
    "$CONFIG_DIR/lora_false.yaml"
)

# --- Main Execution ---
echo "=========================================="
echo "Starting GPU Group 2 on GPUs: $GPUS"
echo "=========================================="
export CUDA_VISIBLE_DEVICES="$GPUS"
n_gpus=$(echo "$GPUS" | tr -cd ',' | wc -c)
n_gpus=$((n_gpus + 1))

for config_file in "${JOB_QUEUE[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running job: $(basename $config_file)"
    echo "Config: $config_file"
    echo "GPUs: $GPUS (${n_gpus} GPUs)"
    echo "Started at: $(date)"
    echo "=========================================="
    
    uv run torchrun \
        --nproc_per_node="$n_gpus" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="localhost:$PORT" \
        "$FINETUNE_SCRIPT" finetune --config "$config_file"
    
    echo ""
    echo "✓ Job completed: $(basename $config_file) at $(date)"
done

echo ""
echo "=========================================="
echo "All jobs for GPU Group 2 are complete!"
echo "=========================================="
