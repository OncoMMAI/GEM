# Clinical Text Embeddings for Glioma Prognostication using LLM2Vec

Code here is adapted from [McGill-NLP/llm2vec: Code for 'LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders'](https://github.com/McGill-NLP/llm2vec)

This repository implements a multi-stage fine-tuning pipeline that transforms Llama-3.1 8B Instruct into a specialized clinical text encoder for neuro-oncology. The project generates high-quality embeddings from MRI and pathology reports for glioma patients, enabling accurate prediction of tumor type, survival outcomes, and molecular markers.

## Project Overview

The central objective is to transform a general-purpose, instruction-tuned Large Language Model (LLM) into a highly specialized text encoder tailored for the clinical neuro-oncology domain. We employ the LLM2Vec methodology with three progressive fine-tuning stages to create embeddings from clinical reports for downstream prognostication tasks.

### Key Results

| Task                          | Best Model     | Accuracy    | F1-Macro    | AUC-ROC     |
| ----------------------------- | -------------- | ----------- | ----------- | ----------- |
| **Tumor Type Classification** | Full Fine-tune | 0.77 ± 0.08 | 0.71 ± 0.11 | 0.93 ± 0.03 |
| **One-Year Survival**         | SimCSE         | 0.70 ± 0.02 | 0.69 ± 0.02 | 0.78 ± 0.02 |
| **MGMT Status**               | SimCSE         | 0.60 ± 0.03 | 0.53 ± 0.03 | 0.63 ± 0.05 |

**Key Finding**: The optimal training strategy is task-dependent. Supervised fine-tuning excels at tumor classification, while unsupervised SimCSE better captures survival-related features.

## Methodology: Three-Stage LLM2Vec Pipeline

### Stage 1: In-Domain Pre-training via Masked Next Token Prediction (MNTP)

- **Objective**: Adapt Llama-3.1 to clinical radiology/pathology dialect
- **Method**: Bidirectional attention + masked token prediction
- **Corpus**: 81,000 training + 9,000 evaluation clinical reports
- **Max Sequence Length**: 768 tokens
- **Output**: `clinical_mntp_only` model

### Stage 2: Unsupervised Contrastive Learning (SimCSE)

- **Objective**: Learn semantic similarity without explicit labels
- **Method**: Dropout-based positive pairs with in-batch negatives
- **Instruction Prefix**: "Encode this medical report for semantic similarity comparison:"
- **Max Sequence Length**: 512 tokens
- **Output**: `clinical_mntp_simcse` model

### Stage 3: Supervised Contrastive Fine-tuning

- **Objective**: Distinguish matched vs unmatched patient reports
- **Dataset**: 175,000 training + 19,500 evaluation pairs
  - Positive pairs: Same patient MRI + pathology reports
  - Hard negatives: Different patients, same cancer type
- **Instruction Prefix**: "Compare these medical reports for semantic similarity:"
- **Output**: `clinical_full_finetune` model

## System Requirements

### Hardware Requirements

- **GPU**: NVIDIA GPU with minimum 24GB VRAM (e.g., A5000, RTX 4090, V100, A100)
- **Multi-GPU**: 8x GPUs recommended for distributed training
- **RAM**: Minimum 32GB system memory
- **Storage**: 100GB free disk space

### Software Requirements

- **Operating System**: Ubuntu 20.04+ or compatible Linux
- **CUDA**: Version 11.8 or higher
- **Python**: Version 3.8-3.10
- **PyTorch**: 2.0+ with Flash Attention support

## Installation

### Option 1: Using pip

```bash
# Clone repository
git clone https://github.com/benedictneo/glioma_embedding.git
cd glioma_embedding/llm2vec

# Install dependencies
pip install -e .
pip install flash-attn --no-build-isolation
```

### Option 2: Using uv (recommended)

```bash
# Clone repository
git clone https://github.com/benedictneo/glioma_embedding.git
cd glioma_embedding/llm2vec

# Install with uv
uv venv
uv pip install -e .
uv pip install flash-attn --no-build-isolation
```

After installation, download the Llama-3.1 base model and place it in: `models/meta-llama/Llama-3.1-8B-Instruct/`

## Configuration and Parameters

For detailed configuration options and parameters:

- **[CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)**: Comprehensive guide to all configuration files, training parameters, and advanced settings
- **[PARAMETERS_SUMMARY.md](PARAMETERS_SUMMARY.md)**: Quick reference for key parameters, typical values, and performance tuning tips

## Training Commands

### Stage 1: MNTP Training

```bash
# Single GPU
torchrun --nproc_per_node=1 experiments/run_mntp.py train_configs/mntp/MetaLlama3.1.json

# Multi-GPU (8 GPUs)
torchrun --nproc_per_node=8 experiments/run_mntp.py train_configs/mntp/MetaLlama3.1.json
```

### Stage 2: SimCSE Training

```bash

# Train with clinical corpus (modify dataset_file_path in config)
torchrun --nproc_per_node=8 experiments/run_simcse.py train_configs/simcse/MetaLlama3.1.json
```

### Stage 3: Supervised Training

```bash
# Prepare matched report pairs in cache/clinical-pairs/
# Format: anchor, positive, negative columns in CSV

torchrun --nproc_per_node=8 experiments/run_supervised.py train_configs/supervised/MetaLlama3.1.json
```

## Quick Start Examples

### Loading Pre-trained Models

```python
from llm2vec import LLM2Vec
import torch

# Load fine-tuned model
l2v = LLM2Vec.from_pretrained(
    "path/to/mntp/checkpoint",
    peft_model_name_or_path="path/to/supervised/adapter",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

# Generate embeddings
texts = ["Clinical report text here..."]
embeddings = l2v.encode(texts)
```

### Clinical Report Encoding

```python
# Generate clinical embeddings with instruction
instruction = "Encode this medical report for semantic similarity comparison:"
reports = [[instruction, report_text] for report_text in clinical_reports]
embeddings = l2v.encode(reports)

# Use for downstream tasks
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1000, random_state=42)
clf.fit(embeddings, labels)
```

## Model Outputs

```
models/
├── mntp/
│   └── checkpoint-final/
├── simcse/
│   └── checkpoint-1000/
└── supervised/
    └── checkpoint-final/
```

## Evaluation and Visualization

### Comprehensive Model Evaluation (`eval.py`)

Performs end-to-end evaluation of LLM2Vec models across all training stages:

- Evaluates MNTP, SimCSE, and supervised fine-tuned models
- Tests on three clinical tasks: tumor type classification, survival prediction, MGMT status
- Uses k-fold cross-validation with Random Forest classifiers
- Generates performance metrics (accuracy, precision, recall, F1, AUC-ROC)
- Saves embeddings for reuse and detailed results in JSON/Markdown formats

```bash
python eval.py --config config.yml
```

### Embedding Visualization (`visualize.py`)

Creates t-SNE visualizations to analyze embedding quality progression:

- Visualizes embeddings from all three training stages
- Shows tumor type clustering patterns
- Generates individual stage plots and side-by-side comparisons
- Filters oligoastrocytoma class for consistency with evaluation
- Outputs high-resolution figures for publication

```bash
python visualize.py --config config.yml
```

## Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce batch size or enable gradient checkpointing
2. **Slow Training**: Enable Flash Attention 2, use mixed precision
3. **Poor Convergence**: Adjust learning rate, check data quality
4. **DDP Errors**: Ensure NCCL backend, check GPU visibility

For detailed performance tuning tips, see [PARAMETERS_SUMMARY.md](PARAMETERS_SUMMARY.md#performance-tuning).

## License

Apache License 2.0 - see LICENSE file for details.
