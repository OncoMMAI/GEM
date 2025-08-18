# LLM2Vec Configuration Guide

This document provides a comprehensive overview of all configurable parameters in the LLM2Vec project.

## Overview

The LLM2Vec project implements various training approaches for converting LLMs into text embedding models:
- **MNTP (Masked Next Token Prediction)**: Unsupervised pre-training approach
- **SimCSE**: Contrastive learning approach  
- **Supervised**: Supervised fine-tuning with positive/negative pairs

## Training Configuration Files

### 1. MNTP Configuration (`train_configs/mntp/MetaLlama3.1.json`)

```json
{
    "model_name_or_path": "/path/to/Llama-3.1-8B-Instruct",
    "train_file": "data/pretraining/mntp_corpus_train.csv",
    "validation_file": "data/pretraining/mntp_corpus_eval.csv",
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 16,
    "learning_rate": 5e-5,
    "num_train_epochs": 5,
    "warmup_steps": 100,
    "do_train": true,
    "do_eval": true,
    "evaluation_strategy": "steps",
    "eval_steps": 50,
    "logging_steps": 50,
    "max_seq_length": 2048,
    "mask_token_type": "blank",
    "data_collator_type": "default",
    "mlm_probability": 0.2,
    "overwrite_output_dir": true,
    "output_dir": "output/clinical_llama3.1_mntp_2048",
    "save_steps": 50,
    "stop_after_n_steps": 1000,
    "lora_r": 16,
    "gradient_checkpointing": true,
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2",
    "cache_dir": "/path/to/cache"
}
```

#### MNTP-Specific Parameters:
- `mask_token_type`: Type of masking token ("blank", "cls", or custom)
- `data_collator_type`: Data collator type for MLM ("default" or custom)
- `mlm_probability`: Probability of masking tokens (0.0-1.0)

### 2. SimCSE Configuration (`train_configs/simcse/MetaLlama3.1.json`)

```json
{
    "model_name_or_path": "/path/to/Llama-3.1-8B-Instruct",
    "peft_model_name_or_path": "/path/to/mntp_checkpoint",
    "simcse_dropout": 0.3,
    "bidirectional": true,
    "pooling_mode": "mean",
    "dataset_name": "MedicalData",
    "dataset_file_path": "data/pretraining/mntp_corpus_train.csv",
    "remove_unused_columns": false,
    "learning_rate": 3e-5,
    "loss_scale": 20,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 16,
    "do_train": true,
    "disable_tqdm": false,
    "max_seq_length": 2048,
    "overwrite_output_dir": true,
    "output_dir": "output/clinical_llama3.1_simcse_2048",
    "logging_steps": 50,
    "save_steps": 100,
    "save_only_model": true,
    "stop_after_n_steps": 1000,
    "lora_r": 16,
    "gradient_checkpointing": true,
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2",
    "seed": 42
}
```

#### SimCSE-Specific Parameters:
- `simcse_dropout`: Dropout rate for SimCSE training (0.0-1.0)
- `loss_scale`: Scaling factor for the loss function
- `pooling_mode`: Pooling strategy ("mean", "max", "cls", "last")
- `bidirectional`: Enable bidirectional attention

### 3. Supervised Configuration (`train_configs/supervised/MetaLlama3.1.json`)

```json
{
    "model_name_or_path": "/path/to/Llama-3.1-8B-Instruct",
    "peft_model_name_or_path": "/path/to/simcse_checkpoint",
    "dataset_name": "MedicalPairData",
    "train_file": "data/finetuning/supervised_pairs_train.csv",
    "validation_file": "data/finetuning/supervised_pairs_eval.csv",
    "bidirectional": true,
    "pooling_mode": "mean",
    "remove_unused_columns": false,
    "learning_rate": 3e-5,
    "num_train_epochs": 5,
    "warmup_steps": 300,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 8,
    "do_train": true,
    "do_eval": true,
    "eval_strategy": "steps",
    "disable_tqdm": false,
    "max_seq_length": 2048,
    "overwrite_output_dir": true,
    "output_dir": "output/clinical_llama3.1_supervised_2048",
    "logging_steps": 50,
    "eval_steps": 100,
    "save_steps": 100,
    "save_total_limit": 2,
    "save_only_model": true,
    "stop_after_n_steps": 1000,
    "lora_r": 16,
    "gradient_checkpointing": true,
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2",
    "seed": 42
}
```

## Common Training Parameters

### Model Configuration
- `model_name_or_path`: Base model path (local or HuggingFace)
- `peft_model_name_or_path`: Path to PEFT adapter weights (for SimCSE/Supervised)
- `cache_dir`: Directory for caching downloaded models

### LoRA Configuration
- `lora_r`: LoRA rank (typically 8, 16, or 32)
- `lora_alpha`: LoRA scaling factor (default: 2 * lora_r)
- `lora_dropout`: LoRA dropout rate (default: 0.05)
- `lora_modules`: List of modules to apply LoRA (auto-detected if None)

### Training Hyperparameters
- `learning_rate`: Learning rate (typical range: 1e-5 to 5e-4)
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Training batch size per device
- `per_device_eval_batch_size`: Evaluation batch size per device
- `gradient_accumulation_steps`: Steps to accumulate gradients
- `warmup_steps`: Number of warmup steps
- `seed`: Random seed for reproducibility

### Sequence Configuration
- `max_seq_length`: Maximum sequence length (512, 1024, 2048, etc.)
- `doc_max_length`: Maximum document length for encoding (default: 400)

### Optimization Settings
- `gradient_checkpointing`: Enable gradient checkpointing for memory efficiency
- `torch_dtype`: Data type for model weights ("float32", "float16", "bfloat16")
- `attn_implementation`: Attention implementation ("flash_attention_2", "sdpa", "eager")

### Training Control
- `do_train`: Whether to run training
- `do_eval`: Whether to run evaluation
- `evaluation_strategy`: When to evaluate ("steps", "epoch", "no")
- `eval_steps`: Evaluation frequency (if strategy is "steps")
- `stop_after_n_steps`: Stop training after N steps (for debugging)

### Logging and Saving
- `output_dir`: Directory to save model checkpoints
- `overwrite_output_dir`: Whether to overwrite existing output directory
- `logging_steps`: Frequency of logging
- `save_steps`: Frequency of saving checkpoints
- `save_total_limit`: Maximum number of checkpoints to keep
- `save_only_model`: Save only model weights (not optimizer state)

### Data Configuration
- `train_file`: Path to training data
- `validation_file`: Path to validation data
- `dataset_name`: Name of dataset class to use
- `dataset_file_path`: Path to dataset file (for custom datasets)
- `remove_unused_columns`: Remove columns not required by model

## LLM2Vec Model Parameters

When loading a model with `LLM2Vec.from_pretrained()`:

```python
model = LLM2Vec.from_pretrained(
    base_model_name_or_path="path/to/base/model",
    peft_model_name_or_path="path/to/peft/adapter",  # Optional
    merge_peft=False,  # Whether to merge PEFT weights
    enable_bidirectional=True,  # Enable bidirectional attention
    pooling_mode="mean",  # Pooling strategy
    max_length=512,  # Maximum sequence length
    doc_max_length=400,  # Maximum document length
    skip_instruction=True,  # Skip instruction formatting
    device_map="auto",  # Device placement
    torch_dtype=torch.bfloat16,  # Model dtype
    trust_remote_code=True,  # Trust remote code
    cache_dir="/path/to/cache"  # Cache directory
)
```

## Data Preparation Parameters

### Supervised Data Creation (`prepare_supervised_data.py`)
- Input: `matches.csv` with matched radiology-pathology reports
- Output: `supervised_pairs_train.csv` and `supervised_pairs_eval.csv`
- Train/eval split: 90/10 (configurable via `test_size` parameter)
- Random seed: 42 (configurable)

### Data Format Requirements

#### MNTP Data Format
CSV with single column:
- `text`: Raw text for masked language modeling

#### SimCSE Data Format
CSV with single column:
- `text`: Raw text (positive pairs created via dropout)

#### Supervised Data Format
CSV with columns:
- `sentence1`: First text in pair
- `sentence2`: Second text in pair
- `label`: 1.0 for positive pairs, 0.0 for negative pairs

## Evaluation Configuration

### Full Evaluation Script (`run_full_evaluation.py`)
Configuration via YAML file:

```yaml
model:
  stages:
    - name: "base"
      path: "/path/to/base/model"
    - name: "mntp"
      path: "/path/to/mntp/checkpoint"
    - name: "simcse"
      path: "/path/to/simcse/checkpoint"
    - name: "supervised"
      path: "/path/to/supervised/checkpoint"
  
  pooling_mode: "mean"
  max_length: 2048
  batch_size: 32

evaluation:
  n_splits: 5  # K-fold cross-validation
  n_estimators: 1000  # Random Forest trees
  random_state: 42
  
data:
  tumor_type_path: "data/eval/tumor_type_eval.csv"
  mgmt_path: "data/eval/mgmt_eval.csv"
  survival_path: "data/eval/os_eval.csv"
  text_column: "report"
  label_columns:
    tumor_type: "tumor_type"
    mgmt: "mgmt_methylation"
    survival: "overall_survival"

output:
  results_dir: "results/llm2vec_evaluation"
  save_embeddings: true
  save_predictions: true
```

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: GPU devices to use
- `TOKENIZERS_PARALLELISM`: Set to "false" to avoid warnings
- `TRANSFORMERS_CACHE`: Cache directory for transformers
- `HF_HOME`: HuggingFace home directory

## Command-Line Usage

### Training Commands

```bash
# MNTP Training
torchrun --nproc_per_node=8 experiments/run_mntp.py train_configs/mntp/MetaLlama3.1.json

# SimCSE Training
torchrun --nproc_per_node=8 experiments/run_simcse.py train_configs/simcse/MetaLlama3.1.json

# Supervised Training
torchrun --nproc_per_node=8 experiments/run_supervised.py train_configs/supervised/MetaLlama3.1.json
```

### Multi-GPU Configuration
- `--nproc_per_node`: Number of GPUs to use
- `--master_port`: Port for distributed training (default: 29500)
- `--master_addr`: Master address for distributed training (default: localhost)

## Model Support

Currently supported models:
- Llama family (Llama-2, Llama-3, Llama-3.1)
- Mistral family
- Gemma family
- Qwen2 family

Each model family has specific bidirectional implementations in:
- `llm2vec/models/bidirectional_llama.py`
- `llm2vec/models/bidirectional_mistral.py`
- etc.

## Advanced Configuration

### Custom Dataset Implementation

To use custom datasets, create a class in `llm2vec/dataset/`:

```python
class CustomDataset:
    def __init__(self, file_path, tokenizer, max_length):
        # Implementation
        pass
    
    def __len__(self):
        # Return dataset size
        pass
    
    def __getitem__(self, idx):
        # Return tokenized example
        pass
```

### Loss Function Configuration

Available loss functions in `llm2vec/loss/`:
- `HardNegativeNLLLoss`: For supervised training with hard negatives
- Additional custom loss functions can be implemented

### Experiment Tracking

The `experiment_utils.py` provides utilities for:
- Generating experiment IDs from parameters
- Parsing experiment IDs back to parameters
- Consistent naming across experiments

## Performance Tuning

### Memory Optimization
- Enable `gradient_checkpointing` for large models
- Reduce `max_seq_length` if running out of memory
- Use `torch_dtype="bfloat16"` for faster training
- Adjust `gradient_accumulation_steps` to simulate larger batches

### Speed Optimization
- Use `attn_implementation="flash_attention_2"` (requires flash-attn package)
- Enable `torch.compile` for PyTorch 2.0+
- Use larger batch sizes when possible
- Enable mixed precision training

### Quality Optimization
- Increase `lora_r` for more model capacity
- Use longer `max_seq_length` for better context
- Adjust `learning_rate` and `warmup_steps`
- Experiment with different `pooling_mode` options

## Troubleshooting

Common issues and solutions:
1. **OOM errors**: Reduce batch size or sequence length
2. **Slow training**: Enable flash attention, check GPU utilization
3. **Poor performance**: Check data quality, adjust hyperparameters
4. **Loading errors**: Ensure correct model paths and versions