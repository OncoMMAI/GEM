## A Clinically Aligned Embedding Model for Glioma Diagnostic and Prognostic Tasks via Radiology-Pathology Report Matching

This repository contains the code for GEM (glioma embedding model). We fine-tune `jinaai/jina-embeddings-v3` on naturally paired radiology-pathology reports with Multiple-Negatives Ranking Loss (MNRL), evaluate cross-modal report retrieval, and probe the frozen embeddings on tumor type, MGMT promoter methylation, and one-year survival.

### Abstract

Background: Integrating radiology with histopathology is central to neuro-oncology, yet general-purpose language models do not align these report types. We tested whether label-free alignment of naturally paired radiology-pathology reports yields transferable neuro-oncology representations.

Methods: We assembled 97,398 naturally paired radiology-pathology report pairs from 4,250 glioma patients and fine-tuned jina-embeddings-v3 on 80% of patients (850 held out) using Multiple-Negatives Ranking Loss, producing the glioma embedding model (GEM). With the encoder frozen, lightweight classifiers on radiology-report embeddings alone were evaluated for three-class tumor type, O6-methylguanine-DNA methyltransferase (MGMT) promoter methylation, and one-year survival against strong encoder baselines and zero-shot decoder large language models (LLMs).

Results: This label-free, institution-specific alignment raised held-out cross-modal retrieval Recall@1 5.7-fold (6.3% to 35.6%). GEM reached 86.9 ± 1.6% accuracy (area under the receiver operating characteristic curve [AUROC] 96.3 ± 0.8%) for three-class tumor type, versus 85.2 ± 1.6% (AUROC 94.6 ± 0.9%) for the untuned base encoder, and exceeded the other encoder baselines and zero-shot decoder LLMs, with similar results under patient-grouped cross-validation. On the smaller MGMT methylation and one-year survival cohorts, it was comparable to the best baselines (AUROC 78.6 ± 12.8% and 75.4 ± 6.7%); these smaller-cohort estimates remain exploratory. A decoder-to-encoder (LLM2Vec) pipeline reached comparable accuracy only at 19-fold higher compute.

Conclusions: Cross-modal alignment of institutional reports offers an efficient route to neuro-oncology representations; external and prospective validation is required before clinical use.

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
- Full fine-tuning of `jina-embeddings-v3` (1024-d embeddings, 8,192-token context); LoRA-only adaptation is included as an ablation.
- Strong performance across 3 clinical tasks; robust ablative analyses (losses, dataset size, sampling, LoRA).
- Cross-modal retrieval evaluation quantifying alignment quality (Recall@K, MRR, MAP).
- Full scripts for end‑to‑end reproducibility (data → train → embed → evaluate → visualize).

---

## Reproducibility Quickstart

1. Clone and set up the base environment (uses uv; pip also works):

```bash
git clone https://github.com/OncoMMAI/GEM.git
cd GEM
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

## Reproducing the paper

All encoder results use the configs in `configs/` with the evaluation defaults (10-fold report-level CV, seed 7, random forest with 1,000 trees, MGMT reports without a recorded status dropped, unknown one-year outcomes dropped).

| Paper item | How to reproduce |
|---|---|
| Table 1 (retrieval) | `python -m src.models.alignment --config configs/alignment_config.yaml` |
| Table 2, Supp. S5-S7, S14 (encoders) | `python -m src.models.evaluation --config configs/evaluation_config.yaml` (3-class tumor type: `data.type_exclude_labels: [3]`) |
| Paired 95% CIs vs untuned JINAv3 | `python scripts/paper/table2_ci.py --results results` (uses the `*_oof.npz` files from the step above) |
| Supp. S13, patient-grouped | same as Table 2 with `evaluation.patient_grouped_cv: true` |
| Supp. S13, unseen patients | also set `evaluation.exclude_patients_from: data/training/mnrl/train.csv` |
| Retrieval robustness (same-histology, BM25, CIs) | `python scripts/paper/retrieval_robustness.py` |
| Table 2 decoders (GPT-4o, BioMistral-7B) | `python scripts/paper/decoder_inference.py --model gpt-4o` (needs `AZURE_OPENAI_API_KEY`) or `--model biomistral` (local vLLM) |
| Survival labels | `notebooks/01_create_dataset.ipynb` (time zero = diagnosis date; unknown outcomes left empty) |

The clinical data cannot be shared (see Ethical, Privacy, and Compliance Notes); the files in `data/` are small synthetic stand-ins that show the expected columns (the real tumor-type file also has an integer `class_label` column: 0 glioblastoma, 1 astrocytoma, 2 oligodendroglioma, 3 oligoastrocytoma).

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

- `lora_main_params_trainable: true` (default) updates all backbone parameters, which is the configuration used for GEM; set it to `false` for the LoRA-only ablation.
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

Evaluate embeddings (stratified k-fold CV with a random forest probe; `n_splits` is set in the config):

```bash
uv run python glioma.py evaluate --config configs/evaluation_config.yaml
```

Notes

- Evaluation reads `embedding.directory` from the config and looks under `tumor_type/`, `os/`, `mgmt/` subdirectories for `.npy` files.
- To evaluate only specific models, set `evaluation.models` in the config to a list of filenames (without `.npy`). Omit to evaluate all.
- Results include per-model accuracy, balanced accuracy, no-information rate, precision, recall, F1-macro, AUROC, and PR-AUC. Fold-level markers against the baseline model are descriptive only, because CV folds share training data.
- Folds are report-level by default, matching the published results. Set `evaluation.patient_grouped_cv: true` to use patient-disjoint folds (`StratifiedGroupKFold` on the patient ID column). With grouped folds, `n_splits` is reduced automatically when a class has fewer patients than folds (MGMT).
- Set `evaluation.exclude_patients_from: data/training/mnrl/train.csv` to evaluate only patients the encoder never saw during fine-tuning. Fold AUROC is averaged over folds whose test set contains both classes.
- MGMT reports with no recorded status are dropped by default. The published MGMT results counted them as unmethylated; set `evaluation.mgmt_missing_as_negative: true` to reproduce that.
- `notebooks/01_create_dataset.ipynb` leaves the one-year survival label empty for patients with no survival time or with less than a year of follow-up while alive; evaluation drops these. The published survival results coded them as non-survivors.
- For inference on paired comparisons, `src/utils/stats.py` provides patient-level aggregation, bootstrap 95% CIs, exact McNemar, and DeLong tests on pooled out-of-fold predictions.
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
GEM/
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
│   └── utils/                        # Config, logging, metrics, statistics (stats.py), seeding
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

## Citation

If you use this code, please cite:

> Neo B, Liu B, Interian Y, Braunstein SE, Morin O, Lupo J, Lin H. A clinically aligned embedding model for glioma diagnostic and prognostic tasks via radiology-pathology report matching. *Manuscript under review.*

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
