## A Clinically Aligned Embedding Model for Glioma Prognostication via Radiology–Pathology Report Matching

This repository contains code for our clinically aligned embedding framework. We align radiology and pathology reports to produce robust clinical embeddings tailored for glioma prognostication, using Multiple Negatives Ranking Loss (MNRL) with LoRA-adapted fine‑tuning of `jinaai/jina-embeddings-v3`. We benchmark across tumor type classification, MGMT methylation prediction, and 1‑year survival prediction.

### Abstract

Large language models exhibit strong general text understanding yet lack oncology‑specific alignment needed for clinical prognostication. We propose a clinically aligned embedding model trained on matched radiology–pathology report pairs using MNRL with LoRA on `jina-embeddings-v3`. Across three neuro‑oncology tasks (tumor type, MGMT status, one‑year survival), the fine‑tuned encoder achieves strong, balanced performance with modest compute.

### Table of Contents

- Overview and Contributions
- Reproducibility Quickstart
- Environment and System Requirements
- Data and Preprocessing
- Training and Ablations
- Embedding Generation and Evaluation
- Report Alignment Evaluation
- Visualization
- Ethical, Privacy, and Compliance Notes
- Citation and License

### Overview and Contributions

- Clinically aligned embeddings via report-pair training (radiology ↔ pathology) with MNRL.
- Efficient LoRA + optional main-parameter updates on `jina-embeddings-v3` (1024‑d embeddings, long context).
- Strong performance across 3 clinical tasks; robust ablative analyses (losses, dataset size, sampling, LoRA).
- Cross-modal retrieval evaluation quantifying alignment quality (Recall@K, MRR, MAP).
- Full scripts for end‑to‑end reproducibility (data → train → embed → evaluate → visualize).

---

## Reproducibility Quickstart

1. Clone and set up the base environment (uses uv; pip also works):

```bash
git clone https://github.com/benedictneo/glioma_embedding.git
cd glioma_embedding
bash setup_env.sh
```

2. Prepare data (uses included synthetic samples by default):

```bash
uv run python glioma.py prepare-data --config configs/dataset_config.yaml
```

3. Fine‑tune (single GPU):

```bash
uv run python glioma.py finetune --config configs/finetune_config.yaml
```

Multi‑GPU (DDP):

```bash
uv run python glioma.py finetune --config configs/finetune_config.yaml --ddp --num_gpus 4 --gpus 0,1,2,3
```

4. Generate embeddings:

```bash
uv run python glioma.py generate --config configs/generate_config.yaml
```

5. Evaluate downstream classification:

```bash
uv run python glioma.py evaluate --config configs/evaluation_config.yaml
```

6. Evaluate report alignment quality:

```bash
uv run python glioma.py align --config configs/alignment_config.yaml
```

7. Visualize:

```bash
uv run python glioma.py visualize --config configs/visualize_config.yaml
```

To reproduce all ablations/sequential evaluation:

```bash
# Create dataset splits (10k, 40k) for ablations
python scripts/experiments/create_dataset_splits.py \
  --train-path data/training/mnrl/train.csv \
  --eval-path data/training/mnrl/eval.csv \
  --output-dir data/training/mnrl/splits

# Launch training groups (two terminals)
bash scripts/experiments/run_group_1.sh
bash scripts/experiments/run_group_2.sh

# After training completes, evaluate all experiments
RESULTS_BASE_DIR=models/results \
EMBEDDINGS_BASE_DIR=embeddings/experiments \
bash scripts/experiments/run_evaluation_sequential.sh

# Compare results
python scripts/experiments/compare_results.py
```

---

## Environment and System Requirements

Hardware

- NVIDIA GPU with ≥24GB VRAM recommended; multi‑GPU optional for faster training.
- RTX 6000 Ada tested (CUDA 12.5 drivers).

Software

- Python ≥3.10. uv is used for fast, reproducible envs.
- PyTorch 2.3.1. On Linux GPU servers, install CUDA 12.1 wheels (compatible with 12.5 drivers).
- Transformers 4.45+, SentenceTransformers 5.x.
- Optional: FlashAttention‑2 on Linux (`flash-attn>=2.5.8`).

Install (Linux GPU server)

```bash
# Already done by setup_env.sh, but manual steps:
uv venv
uv pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
uv pip install -r requirements.txt
uv pip install 'flash-attn>=2.5.8' --no-build-isolation  # Linux only, optional
uv pip install -e .
```

Notes

- macOS/CPU: setup skips FlashAttention and installs CPU Torch automatically.
- Hugging Face trust_remote_code is used for `jina-embeddings-v3` to enable encode/task adapters.
- bitsandbytes is optional; it's pinned Linux‑only in requirements and not required for this pipeline.

Reproducibility notes

- We fix random seeds (default 7) across Python/NumPy/PyTorch; some CUDA ops may remain nondeterministic per PyTorch guidance.
- Results can vary slightly by GPU/driver; folds and metrics are reported with mean ± SD.

### Recommended environments (uv)

To avoid dependency conflicts, we recommend three separate uv environments:

- Jina fine‑tune + embeddings (this repo): `uv venv .venv-jina; source .venv-jina/bin/activate`

  - Install Torch CUDA wheels on Linux: `uv pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1`
  - `uv pip install -r requirements.txt`
  - Optional speedup (Linux): `uv pip install 'flash-attn>=2.5.8' --no-build-isolation`
  - `uv pip install -e .`

---

## Data and Preprocessing

- Synthetic data is provided under `data/raw/*` and `data/eval/*` to exercise the full pipeline.
- Real clinical data is not included; see Data Availability under Ethics/Compliance for MTA process.
- Data prep uses `configs/dataset_config.yaml` and `src/data/data_pipeline.py`:
  - Lowercases, basic PHI scrubbing (synthetic), standardizes whitespace and formats
  - Matches radiology ↔ pathology reports by patient and cancer type
  - Creates MNRL anchor–positive pairs; splits by patient to avoid leakage

Run:

```bash
uv run python glioma.py prepare-data --config configs/dataset_config.yaml
```

---

## Training and Ablations

Fine‑tuning uses `src/models/finetune.py` with MNRL and SentenceTransformers trainer over `jinaai/jina-embeddings-v3`.

- LoRA adapters are enabled; `lora_main_params_trainable: true` also fine‑tunes base params (per ablations).
- Batch sampler defaults to `NO_DUPLICATES` for in‑batch negatives.
- Mixed precision (bf16) where supported; gradient checkpointing enabled.

Single‑GPU:

```bash
uv run python glioma.py finetune --config configs/finetune_config.yaml
```

Multi‑GPU (DDP):

```bash
uv run python glioma.py finetune --config configs/finetune_config.yaml --ddp --num_gpus 4 --gpus 0,1,2,3
```

Reproducing ablations (`scripts/experiments/configs/*.yaml`):

```bash
python scripts/experiments/create_dataset_splits.py \
  --train-path data/training/mnrl/train.csv \
  --eval-path data/training/mnrl/eval.csv \
  --output-dir data/training/mnrl/splits
bash scripts/experiments/run_group_1.sh
bash scripts/experiments/run_group_2.sh
```

---

## Embedding Generation and Evaluation

Generate embeddings from base or fine‑tuned models:

```bash
uv run python glioma.py generate --config configs/generate_config.yaml
```

Supported models (configured in `configs/generate_config.yaml`):

| Model | Key | Dimensionality |
|---|---|---|
| Jina Embeddings v3 (base) | `jina_base` | 1024 |
| NV-Embed-v2 | `nvidia` | 4096 |
| GatorTron | `gatortron` | 3584 |
| MedEmbed | `medembed` | 384 |
| BioClinicalBERT | `bioclinicalbert` | 768 |
| PubMedBERT | `pubmedbert` | 768 |
| ModernBERT | `modernbert` | 768 |
| Jina finetuned variants (MNRL, Contrastive, Triplet) | `jina_finetuned_variants` | 1024 |

Notes

- For fine‑tuned Jina, set `checkpoint_path` to your checkpoint. We accept either a direct file (`model.safetensors`) or a directory containing the SentenceTransformers layout (auto‑resolved).
- Set `output.skip_existing: true` to avoid regenerating existing embeddings.
- Embeddings are saved under `embeddings/{tumor_type,os,mgmt}/` as `.npy` files.

Evaluate embeddings (10‑fold stratified CV with Random Forest):

```bash
uv run python glioma.py evaluate --config configs/evaluation_config.yaml
```

Notes

- Evaluation reads `embedding.directory` from the config and looks under `tumor_type/`, `os/`, `mgmt/` subdirectories for `.npy` files.
- To evaluate only specific models, set `evaluation.models` in the config to a list of filenames (without `.npy`). Omit to evaluate all.
- Results include per-model accuracy, precision, recall, F1-macro, AUCROC, and statistical significance testing (paired t-test, Cohen's d) against a configurable baseline model.
- Results are saved as markdown tables and JSON to `results/`.

Sequential experiment evaluation:

```bash
RESULTS_BASE_DIR=models/results \
EMBEDDINGS_BASE_DIR=embeddings/experiments \
bash scripts/experiments/run_evaluation_sequential.sh
python scripts/experiments/compare_results.py
```

---

## Report Alignment Evaluation

To quantify the quality of the learned radiology–pathology alignment, we provide a cross-modal retrieval evaluation (`src/models/alignment.py`).

Given a held-out set of paired radiology and pathology reports, the evaluation:

1. Embeds all unique radiology reports (queries) and pathology reports (gallery) using each model variant
2. Computes cosine similarity between every query–gallery pair
3. For each radiology report, ranks all pathology reports and checks whether the correct match (same patient) appears in the top K

Reported metrics:

- **Recall@K** (R@1, R@5, R@10, R@20): proportion of queries where a correct pathology report appears in the top K results
- **MRR** (Mean Reciprocal Rank): average of 1/rank of the first correct match
- **MAP** (Mean Average Precision): average precision across all relevant rank positions

Run:

```bash
uv run python glioma.py align --config configs/alignment_config.yaml
```

The config (`configs/alignment_config.yaml`) specifies which model variants to compare (e.g., base vs. MNRL vs. Contrastive vs. Triplet). Results are saved as a markdown table and JSON to `results/`.

---

## Visualization

```bash
uv run python glioma.py visualize --config configs/visualize_config.yaml
```

Produces t‑SNE projections with class overlays and confidence ellipses for all three tasks (tumor type, MGMT, survival); outputs to `figures/`.

---

## Ethical, Privacy, and Compliance Notes

- No clinical data is included in this repository. Synthetic samples are provided to demonstrate structure and pipeline functionality.
- Access to real clinical data requires a HIPAA‑compliant Data Use Agreement or MTA with UCSF. Contact the corresponding author.
- All methods are for research use only and not intended for clinical decision‑making without appropriate validation and regulatory review.

---

## Repository Structure

```
glioma_embedding/
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── setup_env.sh
├── glioma.py                        # CLI entry point
├── configs/
│   ├── finetune_config.yaml          # Fine-tuning configuration
│   ├── generate_config.yaml          # Embedding generation (base + finetuned models)
│   ├── evaluation_config.yaml        # Downstream classification evaluation
│   ├── alignment_config.yaml         # Cross-modal retrieval alignment evaluation
│   ├── visualize_config.yaml         # t-SNE visualization
│   ├── dataset_config.yaml           # Data preparation
│   └── inference_config.yaml         # LLM inference
├── data/
│   ├── eval/                         # Evaluation CSVs (tumor_type, mgmt, os)
│   ├── training/                     # Training pairs (MNRL anchor–positive)
│   └── raw/                          # Raw synthetic reports
├── src/
│   ├── data/data_pipeline.py         # Data preparation and pair generation
│   ├── models/
│   │   ├── finetune.py               # MNRL fine-tuning with SentenceTransformers
│   │   ├── evaluation.py             # k-fold classification evaluation
│   │   ├── alignment.py              # Cross-modal retrieval evaluation
│   │   └── inference.py              # LLM inference
│   ├── embeddings/
│   │   ├── generate.py               # Embedding generation (Jina, NV-Embed, etc.)
│   │   └── visualize.py              # t-SNE visualization
│   └── utils/                        # Config, logging, metrics, seeding
├── embeddings/                       # Generated embeddings (.npy)
├── figures/                          # Visualization outputs
├── models/                           # Model checkpoints
├── results/                          # Evaluation results
└── scripts/experiments/              # Ablation experiment scripts
```

---

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
