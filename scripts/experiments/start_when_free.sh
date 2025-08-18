#!/bin/bash
# Start this script now - it will wait and run experiments when GPUs are free

echo "=========================================="
echo "Auto-Start Experiment Runner"
echo "Started at: $(date)"
echo "=========================================="
echo ""
echo "Will:"
echo "  1. Create dataset splits NOW"
echo "  2. Start Group 1 in 45 minutes on GPUs 4,5,6,7"
echo "  3. Start Group 2 in 3 hours 10 minutes on GPUs 0,1,2,3"
echo "  4. Run evaluation after all training completes"
echo ""

# Create log directory
mkdir -p scripts/experiments/logs

# Step 1: Create dataset splits NOW
echo "Creating dataset splits..."
python scripts/experiments/create_dataset_splits.py \
    --train-path /home/neob/ai/paper/data/training/mnrl/train.csv \
    --eval-path /home/neob/ai/paper/data/training/mnrl/eval.csv \
    --output-dir /home/neob/ai/paper/data/training/mnrl/splits \
    --seed 42

echo "✓ Dataset splits created"
echo ""

# Kill any existing sessions
tmux kill-session -t auto_exp_1 2>/dev/null || true
tmux kill-session -t auto_exp_2 2>/dev/null || true

# Start Group 1 (waits 45 minutes then runs on GPUs 4,5,6,7)
tmux new-session -d -s auto_exp_1 \
    "echo 'Waiting 45 minutes (40 min + 5 min buffer)...' && \
     sleep 2700 && \
     echo 'Starting Group 1 at: $(date)' && \
     export CUDA_VISIBLE_DEVICES='4,5,6,7' && \
     bash scripts/experiments/run_group_1.sh 2>&1 | \
     tee scripts/experiments/logs/auto_group_1_$(date +%Y%m%d_%H%M%S).log"

# Start Group 2 (waits 3 hours 10 minutes then runs on GPUs 0,1,2,3)  
tmux new-session -d -s auto_exp_2 \
    "echo 'Waiting 3 hours 10 minutes (3 hr + 10 min buffer)...' && \
     sleep 11400 && \
     echo 'Starting Group 2 at: $(date)' && \
     export CUDA_VISIBLE_DEVICES='0,1,2,3' && \
     bash scripts/experiments/run_group_2.sh 2>&1 | \
     tee scripts/experiments/logs/auto_group_2_$(date +%Y%m%d_%H%M%S).log"

echo "✓ Both groups scheduled!"

# Start evaluation monitor (waits for all experiments to complete)
tmux new-session -d -s auto_eval \
    "echo 'Waiting for all experiments to complete...' && \
     echo 'Will check every 30 minutes' && \
     while true; do \
         completed=0; \
         for exp in baseline dataset_10k dataset_40k task_classification task_separation batch_sampler_default lora_false; do \
             if [ -f \"/data/users/neob/finetuning/results/\$exp/final/model.safetensors\" ]; then \
                 ((completed++)); \
             fi; \
         done; \
         echo \"\$(date): \$completed/7 experiments completed\"; \
         if [ \$completed -eq 7 ]; then \
             echo 'All experiments complete! Starting evaluation...'; \
             bash scripts/experiments/run_evaluation_sequential.sh 2>&1 | tee scripts/experiments/logs/auto_eval_\$(date +%Y%m%d_%H%M%S).log; \
             break; \
         fi; \
         sleep 1800; \
     done"

echo "✓ Evaluation monitor scheduled!"
echo ""
echo "Monitor with:"
echo "  tmux attach -t auto_exp_1   # Group 1 (starts in 45 min)"
echo "  tmux attach -t auto_exp_2   # Group 2 (starts in 3 hr 10 min)"
echo "  tmux attach -t auto_eval    # Evaluation (starts when training done)"
echo ""
echo "You can now close this terminal - everything will run automatically"
