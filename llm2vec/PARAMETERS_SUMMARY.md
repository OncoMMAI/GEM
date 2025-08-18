# LLM2Vec Parameters Summary

This document provides a quick reference for all configurable parameters and hardcoded values in the LLM2Vec project.

## Quick Reference: Key Parameters by Category

### 1. Model Selection
- **Base Models Supported**: 
  - Llama-3.1-8B-Instruct (and other Llama variants)
  - Mistral models
  - Gemma models
  - Qwen2 models
- **Model Path**: `model_name_or_path` in config files
- **PEFT Adapter Path**: `peft_model_name_or_path` for loading fine-tuned adapters

### 2. Training Hyperparameters

#### Batch Sizes
- `per_device_train_batch_size`: 4-8 (typical)
- `per_device_eval_batch_size`: 4-8 (typical)
- `gradient_accumulation_steps`: 8-16 (to simulate larger batches)

#### Learning Rates
- MNTP: `5e-5`
- SimCSE: `3e-5`
- Supervised: `3e-5`

#### Training Duration
- `num_train_epochs`: 5 (typical)
- `warmup_steps`: 100-300
- `stop_after_n_steps`: 1000 (for debugging/testing)

### 3. LoRA Configuration
- `lora_r`: 16 (rank, can be 8, 16, 32)
- `lora_alpha`: 32 (typically 2 * lora_r)
- `lora_dropout`: 0.05 (default)

### 4. Sequence Lengths
- `max_seq_length`: 2048 (can be 512, 1024, 2048, 4096)
- `doc_max_length`: 400 (for document encoding)

### 5. Model Architecture
- `bidirectional`: true/false (enable bidirectional attention)
- `pooling_mode`: "mean", "max", "cls", "last"
- `attn_implementation`: "flash_attention_2", "sdpa", "eager"

### 6. Training Strategies

#### MNTP-Specific
- `mlm_probability`: 0.2 (20% token masking)
- `mask_token_type`: "blank"
- `data_collator_type`: "default"

#### SimCSE-Specific
- `simcse_dropout`: 0.3
- `loss_scale`: 20

#### Evaluation
- `evaluation_strategy`: "steps" or "epoch"
- `eval_steps`: 50-100
- `logging_steps`: 50

### 7. Memory and Performance
- `gradient_checkpointing`: true (for memory efficiency)
- `torch_dtype`: "bfloat16" (or "float16", "float32")
- `disable_tqdm`: false

### 8. Saving Configuration
- `save_steps`: 50-100
- `save_total_limit`: 2 (max checkpoints to keep)
- `save_only_model`: true (don't save optimizer state)
- `overwrite_output_dir`: true

### 9. Data Paths
- Training data: `train_file` or `dataset_file_path`
- Validation data: `validation_file`
- Output directory: `output_dir`
- Cache directory: `cache_dir`

### 10. Distributed Training
- `nproc_per_node`: Number of GPUs (1-8)
- Default master port: 29500
- Backend: NCCL (hardcoded for GPU training)

## Hardcoded Values and Defaults

### In Training Scripts
1. **Random Seeds**: 42 (in most configs)
2. **Train/Eval Split**: 90/10 for supervised data preparation
3. **K-Fold CV**: 5 folds (in evaluation)
4. **Random Forest Trees**: 1000 (n_estimators)
5. **Class Weight**: "balanced" (in RandomForestClassifier)

### In Model Implementation
1. **Padding Token**: Set to EOS token
2. **Padding Side**: "left"
3. **Trust Remote Code**: true (for custom models)
4. **Remove Unused Columns**: false (for custom datasets)

### File Naming Conventions
1. **Checkpoint Naming**: "checkpoint-{step}"
2. **Final Model Dir**: Usually ends with "_2048" (sequence length)
3. **Data File Extensions**: .csv for all data files

## Environment-Specific Settings

### GPU Configuration
```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python script.py

# Multi-GPU
torchrun --nproc_per_node=8 script.py config.json
```

### Required Environment Variables
```bash
export TOKENIZERS_PARALLELISM=false  # Avoid warnings
export TRANSFORMERS_CACHE=/path/to/cache  # Model cache
export HF_HOME=/path/to/huggingface  # HuggingFace home
```

## Configuration File Templates

### Minimal MNTP Config
```json
{
    "model_name_or_path": "/path/to/model",
    "train_file": "data/train.csv",
    "learning_rate": 5e-5,
    "output_dir": "output/mntp"
}
```

### Minimal SimCSE Config
```json
{
    "model_name_or_path": "/path/to/model",
    "peft_model_name_or_path": "/path/to/mntp/checkpoint",
    "dataset_file_path": "data/train.csv",
    "learning_rate": 3e-5,
    "output_dir": "output/simcse"
}
```

### Minimal Supervised Config
```json
{
    "model_name_or_path": "/path/to/model",
    "peft_model_name_or_path": "/path/to/simcse/checkpoint",
    "train_file": "data/pairs_train.csv",
    "validation_file": "data/pairs_eval.csv",
    "learning_rate": 3e-5,
    "output_dir": "output/supervised"
}
```

## Performance Benchmarks (Typical Values)

### Training Speed (per 1000 steps)
- MNTP: ~2-3 hours (8x A100 GPUs)
- SimCSE: ~1-2 hours (8x A100 GPUs)
- Supervised: ~1-2 hours (8x A100 GPUs)

### Memory Usage (per GPU)
- With gradient checkpointing: ~20-30GB
- Without gradient checkpointing: ~40-50GB
- Varies with sequence length and batch size

### Model Sizes
- Base Llama-3.1-8B: ~15GB
- With LoRA adapters: +50-200MB
- Full fine-tuned: Same as base

## Common Modifications

### For Limited GPU Memory
```json
{
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "gradient_checkpointing": true,
    "max_seq_length": 512
}
```

### For Faster Training
```json
{
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2"
}
```

### For Better Quality
```json
{
    "lora_r": 32,
    "learning_rate": 2e-5,
    "num_train_epochs": 10,
    "max_seq_length": 4096
}
```

## Data Requirements

### Minimum Dataset Sizes
- MNTP: 10,000+ texts recommended
- SimCSE: 10,000+ texts recommended
- Supervised: 5,000+ pairs recommended

### Data Format Examples

#### MNTP/SimCSE Format (CSV)
```csv
text
"This is a clinical report about..."
"Patient presents with symptoms..."
```

#### Supervised Format (CSV)
```csv
sentence1,sentence2,label
"MRI report text...","Pathology report text...",1.0
"MRI report text...","Different pathology text...",0.0
```

## Debugging Parameters

### Enable Verbose Logging
```json
{
    "logging_steps": 1,
    "disable_tqdm": false,
    "logging_dir": "./logs"
}
```

### Quick Testing
```json
{
    "stop_after_n_steps": 10,
    "save_steps": 5,
    "eval_steps": 5
}
```

### Deterministic Training
```json
{
    "seed": 42,
    "data_seed": 42,
    "deterministic": true
}
```