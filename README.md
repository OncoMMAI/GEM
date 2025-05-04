# A Clinically Aligned Embedding Model for Glioma Prognostication via Radiology-Pathology Report Matching

This repository contains the implementation and evaluation framework for fine-tuning JINA embeddings for glioma prognostication tasks.

## Abstract

Large Language Models (LLMs) demonstrate strong general text comprehension but remain limited in oncology due to insufficient contextual alignment. We pilot embedding alignment through radiology-pathology report matching to enhance oncology-specific representations. Using Multiple Negative Ranking Loss (MNRL) and Low-Rank Adaptation (LoRA)-adapted fine-tuning of jina-embeddings-v3, we aim to create a context-aware embedding model and evaluate its efficacy across glioma prognostication tasks, including tumor type classification, MGMT methylation prediction, and overall survival prediction. We also compare our fine-tuned embeddings against zero-shot predictions from BioMistral, a specialized biomedical language model.

## Repository Structure

```
glioma_embedding/
├── LICENSE                      # Apache 2.0 license file
├── README.md                    # This documentation file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation script
├── setup_env.sh                 # Environment setup script
├── glioma.py                    # Command-line interface
│
├── configs/                     # Configuration files
│   ├── dataset_config.yaml      # Dataset configuration
│   ├── evaluation_config.yaml   # Evaluation configuration
│   ├── finetune_config.yaml     # Fine-tuning configuration
│   ├── generate_config.yaml     # Embedding generation configuration
│   ├── inference_config.yaml    # Inference configuration
│   └── visualize_config.yaml    # Visualization configuration
│
├── data/                        # Data directory
│   ├── eval/                    # Evaluation datasets
│   │   ├── mgmt_eval.csv        # MGMT methylation data
│   │   ├── os_eval.csv          # Overall survival data
│   │   └── tumor_type_eval.csv  # Tumor type classification data
│   ├── raw/                     # Raw synthetic data
│   │   ├── mri/                 # Synthetic MRI reports by tumor type
│   │   ├── pathology/           # Synthetic pathology reports by tumor type
│   │   └── report.csv           # Raw report file
│   └── training/                # Training datasets
│       ├── matches.csv          # Matched MRI-pathology reports
│       └── mnrl/                # MNRL training data
│           ├── train.csv        # Training data
│           └── eval.csv         # Validation data
│
├── embeddings/                  # Generated embeddings
│   ├── mgmt/                    # MGMT prediction embeddings
│   ├── os/                      # Overall survival embeddings
│   └── tumor_type/              # Tumor type classification embeddings
│
├── figures/                     # Generated figures and visualizations
├── models/                      # Trained model checkpoints
├── notebooks/
│   ├── 00_synthetic_data.ipynb          # Synthetic data generation
│   ├── 01_create_dataset.ipynb          # Dataset creation
│   ├── 02_biomistral_inference.ipynb    # BioMistral inference
│   ├── 02_finetune.ipynb                # Fine-tuning notebook
│   ├── 03_get_embeddings.ipynb          # Embedding generation
│   ├── 03_visualize_embeddings.ipynb    # Embedding visualization
│   ├── 04_eval.ipynb                    # Evaluation notebook
│   └── 05_kaplan_meier.ipynb            # Survival analysis
│
├── results/                     # Evaluation results
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── data/                    # Data processing modules
│   │   ├── __init__.py
│   │   └── data_pipeline.py     # Data processing pipeline
│   ├── embeddings/              # Embedding generation modules
│   │   ├── __init__.py
│   │   ├── generate.py          # Generate embeddings
│   │   └── visualize.py         # Visualize embeddings
│   ├── models/                  # Model training and evaluation
│   │   ├── __init__.py
│   │   ├── evaluation.py        # Evaluation metrics and procedures
│   │   ├── finetune.py          # Fine-tuning implementation
│   │   └── inference.py         # Inference implementation with BioMistral
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       ├── config.py            # Configuration utilities
│       ├── exceptions.py        # Custom exceptions
│       ├── logging.py           # Logging utilities
│       ├── metrics.py           # Evaluation metrics
│       └── seed.py              # Random seed management
```

## Installation

The codebase requires Python 3.8 or higher and uses the UV package manager. To set up the environment:

```bash
# Clone the repository
git clone https://github.com/benedictneo/glioma_embedding.git
cd glioma_embedding

# Run setup script (creates environment and installs dependencies)
bash setup_env.sh
```

The setup script will install UV, create all necessary directories, and install required dependencies.

## Workflow

The workflow consists of six main steps, each controlled by a corresponding configuration file. Each configuration file corresponds to specific modules in the codebase and enables reproducible experiments.

- **dataset_config.yaml**: Configuration for data processing pipeline
- **finetune_config.yaml**: Settings for MNRL and LoRA fine-tuning
- **generate_config.yaml**: Configuration for embedding generation
- **inference_config.yaml**: BioMistral model parameters for inference
- **evaluation_config.yaml**: Parameters for model evaluation
- **visualize_config.yaml**: Settings for embedding visualization

Below is an overview of each step and the commands to execute them.

### 1. Data Preparation (`dataset_config.yaml`)

The data preparation phase includes cleaning clinical reports, matching radiology and pathology reports, and creating training datasets.

```bash
# Process raw clinical data
uv run python glioma.py prepare-data --config configs/dataset_config.yaml

# With custom parameters
uv run python glioma.py prepare-data --input path/to/data --output data/processed --train_split 0.8
```

The data pipeline performs the following steps:

1. Loads and cleans raw MRI and pathology reports
2. Extracts cancer type information from report metadata
3. Matches reports based on patient identifiers and cancer type
4. Creates anchor-positive pairs for Multiple Negative Ranking Loss (MNRL)
5. Splits the data by patient to ensure no data leakage between train/test sets

### 2. Fine-tuning (`finetune_config.yaml`)

Fine-tune the JINA embeddings model using Multiple Negative Ranking Loss (MNRL) and Low-Rank Adaptation (LoRA):

```bash
# Standard fine-tuning on single GPU
uv run torchrun --nproc_per_node=1 glioma.py finetune --config configs/finetune_config.yaml

# Multi-GPU training with Distributed Data Parallel
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=4 glioma.py finetune --config configs/finetune_config.yaml
```

### 3. Embedding Generation (`generate_config.yaml`)

Generate embeddings from clinical texts using various models (JINA, NVIDIA NV-Embed, Gatortron):

```bash
# Generate embeddings with default configuration (all models)
uv run python glioma.py generate --config configs/generate_config.yaml

# Generate embeddings for specific model and task
uv run python glioma.py generate --model jina --task tumor_type --gpus 0

# Generate embeddings with multiple GPUs
uv run python glioma.py generate --model nvidia --gpus 0,1,2,3
```

### 4. BioMistral Inference (`inference_config.yaml`)

Run zero-shot inference using the BioMistral language model for glioma classification tasks:

```bash
# start the server
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve \
    BioMistral-7B \
    --gpu-memory-utilization=0.95 \
    --tensor-parallel-size=4 \
    --trust-remote-code

# On a new terminal, run inference on all tasks
uv run python glioma.py inference --config configs/inference_config.yaml

# sample call
curl -X POST "http://localhost:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data '{
        "model": "BioMistral-7B",
        "messages": [
            {
                "role": "user",
                "content": "Classify this brain tumor into exactly one of these categories: glioblastoma, astrocytoma, oligodendroglioma, or oligoastrocytoma.\n\nReport:\nThere is a large, irregular mass in the right temporal lobe measuring approximately 4.8 x 3.9 x 4.2 cm. The lesion demonstrates heterogeneous enhancement with central necrosis on post-contrast images. Significant surrounding vasogenic edema is present, causing mass effect and midline shift of approximately 7mm to the left. The mass shows restricted diffusion and demonstrates thick, irregular ring enhancement. Multiple areas of hemorrhage are noted within the tumor on susceptibility-weighted imaging.\n\nOutput format:\nCLASSIFICATION: [tumor type]"
            }
        ],
        "temperature": 0.0,
        "max_tokens": 50
    }'
```

### 5. Evaluation (`evaluation_config.yaml`)

Evaluate embedding model performance on glioma prognostication tasks:

```bash
# Run evaluation on all tasks
uv run python glioma.py evaluate --config configs/evaluation_config.yaml

# Run evaluation on specific task
uv run python glioma.py evaluate --task tumor_type
```

### 6. Visualization (`visualize_config.yaml`)

Visualize embeddings using dimension reduction techniques:

```bash
uv run python glioma.py visualize --config configs/visualize_config.yaml

uv run python glioma.py visualize --method tsne --task tumor_type

# Create visualizations for specific embedding type
uv run python glioma.py visualize --embeddings_dir embeddings/tumor_type --output_dir figures/tumor_type
```

## Models

### Embedding Models

The codebase supports multiple embedding models for generating text representations:

1. **JINA Embeddings (v3)**: Primary model used for fine-tuning with MNRL
2. **NVIDIA NV-Embed (v2)**: Medical domain embedding model
3. **Gatortron-large**: Specialized medical language model

### Inference Model

For zero-shot inference on glioma classification tasks:

- **BioMistral-7B**: Specialized biomedical large language model used for zero-shot classification of tumor type, MGMT methylation status, and survival prediction

## Data Availability

The datasets used in this research contain sensitive clinical information and are not publicly available in this repository. However, researchers interested in accessing the data for academic purposes can do so through a Material Transfer Agreement (MTA) with the University of California San Francisco (UCSF).

To request access to the data:

1. Contact the corresponding author with a brief description of your research purpose
2. Complete the UCSF MTA application process
3. Upon approval, you will receive instructions for secure data transfer

All data usage must comply with HIPAA regulations and the terms specified in the MTA.

The code in this repository can be adapted to work with similar clinical text datasets following the same structure as described in the documentation.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{neo2024clinically,
  title={A Clinically Aligned Embedding Model for Glioma Prognostication via Radiology-Pathology Report Matching},
  author={Neo, Benedict and Liu, Bo and Interian, Yannet and Braunstein, Steve and Lupo, Janine and Morin, Olivier and Lin, Hui},
  journal={},
  year={2024}
}
```

## Contributors

- Benedict Neo, University of California San Francisco, University of San Francisco
- Bo Liu, University of California San Francisco, University of California San Francisco-UC Berkeley
- Yannet Interian, University of San Francisco
- Steve Braunstein, University of California San Francisco
- Janine Lupo, University of California San Francisco
- Olivier Morin, University of California San Francisco
- Hui Lin, University of California San Francisco, UCSF-UC Berkeley Joint Program in Computational Precision Health

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
