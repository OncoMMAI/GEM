# Experiment Pipeline

This directory contains scripts for running parallel fine-tuning experiments with controlled single-variable design.

## Experiment Design

Each experiment changes exactly ONE parameter from the baseline:

| Experiment            | Variable Changed | Value                                                 |
| --------------------- | ---------------- | ----------------------------------------------------- |
| baseline              | -                | Full dataset, text-matching, NO_DUPLICATES, LoRA=true |
| dataset_10k           | Dataset Size     | 10k samples (~13%)                                    |
| dataset_40k           | Dataset Size     | 40k samples (~50%)                                    |
| task_classification   | Task Adapter     | classification                                        |
| task_separation       | Task Adapter     | separation                                            |
| batch_sampler_default | Batch Sampler    | DEFAULT                                               |
| lora_false            | LoRA             | false (full fine-tuning)                              |

## Essential Scripts

### 1. Data Preparation

```bash
python scripts/experiments/create_dataset_splits.py \
  --train-path data/training/mnrl/train.csv \
  --eval-path data/training/mnrl/eval.csv \
  --output-dir data/training/mnrl/splits
```

Creates dataset splits (10k, 40k, full) from the full MNRL dataset.

### 2. Parallel Training

```bash
# Start each group in its own terminal (or tmux pane)
bash scripts/experiments/run_group_1.sh
bash scripts/experiments/run_group_2.sh
```

- Group 1: baseline, dataset_10k, dataset_40k, task_classification
- Group 2: task_separation, batch_sampler_default, lora_false

### 3. Monitoring

You can optionally use `scripts/experiments/start_when_free.sh` to schedule runs when GPUs free up.

### 4. Sequential Evaluation

```bash
# Run after all training completes
RESULTS_BASE_DIR=models/results \
EMBEDDINGS_BASE_DIR=embeddings/experiments \
bash scripts/experiments/run_evaluation_sequential.sh
```

Generates embeddings and evaluates all models on three tasks: tumor_type, mgmt, survival.

### 5. Results Comparison

```bash
python scripts/experiments/compare_results.py
```

Creates comparison tables and plots.

## Directory Structure

```
scripts/experiments/
├── configs/                 # Experiment configurations
├── logs/                    # Training logs
├── run_group_1.sh          # GPU group 1 runner
├── run_group_2.sh          # GPU group 2 runner
├── launch_parallel_experiment.sh  # Main launcher
├── monitor_experiments.sh   # Progress monitor
├── run_evaluation_sequential.sh   # Evaluation pipeline
├── prepare_experiments.sh   # Dataset preparation
├── create_dataset_splits.py # Dataset splitting logic
└── compare_results.py      # Results analysis
```

## Quick Start

```bash
# 1. Prepare data
bash scripts/experiments/prepare_experiments.sh

# 2. Launch training
bash scripts/experiments/launch_parallel_experiment.sh

# 3. Monitor (in another terminal)
bash scripts/experiments/monitor_experiments.sh

# 4. After training completes, evaluate
bash scripts/experiments/run_evaluation_sequential.sh

# 5. Compare results
python scripts/experiments/compare_results.py
```
