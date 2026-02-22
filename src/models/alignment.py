# -*- coding: utf-8 -*-

"""
Cross-modal retrieval evaluation for radiology–pathology report alignment.

Evaluates how well a finetuned embedding model can match radiology reports
to their corresponding pathology reports from the same patient, using
standard information-retrieval metrics (Recall@K, MRR, MAP).

This provides an intrinsic measure of alignment quality, independent of
downstream classification tasks.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModel

from src.utils.config import get_config
from src.utils.logging import setup_logger, disable_external_loggers

logger = logging.getLogger("glioma.alignment")


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _load_jina_model(
    model_name: str = "jinaai/jina-embeddings-v3",
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    trust_remote_code: bool = True,
    cache_dir: Optional[str] = None,
) -> torch.nn.Module:
    """Load a Jina model, optionally applying a finetuned checkpoint."""
    model_kwargs = {"trust_remote_code": trust_remote_code}
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir

    model = AutoModel.from_pretrained(model_name, **model_kwargs).to(device)

    if checkpoint_path:
        candidate = Path(checkpoint_path)
        resolved: Optional[Path] = None
        if candidate.is_file():
            resolved = candidate
        elif candidate.is_dir():
            st_layout = candidate / "0_Transformer" / "model.safetensors"
            if st_layout.exists():
                resolved = st_layout
            else:
                for p in candidate.rglob("model.safetensors"):
                    resolved = p
                    break
        if resolved and resolved.exists():
            logger.info(f"Loading checkpoint from {resolved}")
            state_dict = load_file(str(resolved))
            model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning(f"Checkpoint not found at {checkpoint_path}, using base weights")

    model.eval()
    return model


def _embed_texts(
    model: torch.nn.Module,
    texts: List[str],
    task: str = "text-matching",
    batch_size: int = 32,
    max_length: int = 2048,
    device: str = "cuda",
) -> np.ndarray:
    """Embed a list of texts and return the embedding matrix."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            emb = model.encode(
                batch,
                task=task,
                max_length=max_length,
                device=device,
                show_progress_bar=False,
            )
            all_embeddings.append(emb)
    return np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def _compute_retrieval_metrics(
    sim_matrix: np.ndarray,
    query_mrns: np.ndarray,
    gallery_mrns: np.ndarray,
    k_values: Tuple[int, ...] = (1, 5, 10, 20),
) -> Dict[str, float]:
    """Compute retrieval metrics from a similarity matrix.

    Args:
        sim_matrix: (n_queries, n_gallery) cosine similarity matrix
        query_mrns: MRN for each query (radiology report)
        gallery_mrns: MRN for each gallery item (pathology report)
        k_values: K values for Recall@K

    Returns:
        Dictionary with Recall@K, MRR, and MAP
    """
    n_queries = sim_matrix.shape[0]
    reciprocal_ranks = []
    average_precisions = []
    recall_at_k = {k: 0 for k in k_values}

    for i in range(n_queries):
        scores = sim_matrix[i]
        ranked_indices = np.argsort(-scores)
        ranked_mrns = gallery_mrns[ranked_indices]

        # Relevance: gallery item belongs to the same patient
        relevant = ranked_mrns == query_mrns[i]
        n_relevant = relevant.sum()
        if n_relevant == 0:
            continue

        # Recall@K
        for k in k_values:
            if relevant[:k].any():
                recall_at_k[k] += 1

        # MRR: reciprocal rank of the first relevant item
        first_relevant = np.where(relevant)[0][0]
        reciprocal_ranks.append(1.0 / (first_relevant + 1))

        # MAP: average precision across all relevant items
        cumulative_relevant = np.cumsum(relevant).astype(float)
        precision_at_positions = cumulative_relevant / np.arange(1, len(relevant) + 1)
        average_precisions.append(
            (precision_at_positions * relevant).sum() / n_relevant
        )

    n_evaluated = len(reciprocal_ranks)
    metrics = {}
    for k in k_values:
        metrics[f"R@{k}"] = recall_at_k[k] / n_evaluated if n_evaluated else 0.0
    metrics["MRR"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    metrics["MAP"] = float(np.mean(average_precisions)) if average_precisions else 0.0
    metrics["n_queries"] = n_evaluated
    metrics["n_gallery"] = sim_matrix.shape[1]

    return metrics


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_alignment(config: Dict) -> None:
    """Run cross-modal retrieval evaluation for report alignment.

    The eval CSV must contain columns: anchor (radiology), positive (pathology),
    mrn (patient identifier).
    """
    data_cfg = config["data"]
    models_cfg = config["models"]
    output_dir = Path(config["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = config.get("global", {}).get("cache_dir")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Load eval pairs and deduplicate
    # ------------------------------------------------------------------
    eval_path = data_cfg["eval_pairs_path"]
    logger.info(f"Loading eval pairs from {eval_path}")
    df = pd.read_csv(eval_path)
    logger.info(f"Loaded {len(df)} pairs from {df['mrn'].nunique()} patients")

    # Unique radiology reports (queries) and pathology reports (gallery)
    rad_df = df[["anchor", "mrn"]].drop_duplicates(subset="anchor").reset_index(drop=True)
    path_df = df[["positive", "mrn"]].drop_duplicates(subset="positive").reset_index(drop=True)

    rad_texts = rad_df["anchor"].tolist()
    rad_mrns = rad_df["mrn"].values
    path_texts = path_df["positive"].tolist()
    path_mrns = path_df["mrn"].values

    logger.info(
        f"Unique reports — radiology (queries): {len(rad_texts)}, "
        f"pathology (gallery): {len(path_texts)}"
    )

    # ------------------------------------------------------------------
    # Evaluate each model variant
    # ------------------------------------------------------------------
    all_results = {}
    k_values = tuple(config.get("evaluation", {}).get("k_values", [1, 5, 10, 20]))

    for model_cfg in models_cfg:
        name = model_cfg["name"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating alignment: {name}")
        logger.info(f"{'='*60}")

        model = _load_jina_model(
            model_name=model_cfg.get("model_name", "jinaai/jina-embeddings-v3"),
            checkpoint_path=model_cfg.get("checkpoint_path"),
            device=device,
            trust_remote_code=model_cfg.get("trust_remote_code", True),
            cache_dir=cache_dir,
        )

        task = model_cfg.get("task", "text-matching")
        batch_size = model_cfg.get("batch_size", 32)
        max_length = model_cfg.get("max_length", 2048)

        logger.info("Embedding radiology reports...")
        rad_emb = _embed_texts(model, rad_texts, task=task, batch_size=batch_size,
                               max_length=max_length, device=device)
        logger.info("Embedding pathology reports...")
        path_emb = _embed_texts(model, path_texts, task=task, batch_size=batch_size,
                                max_length=max_length, device=device)

        # L2 normalise for cosine similarity via dot product
        rad_emb = rad_emb / np.linalg.norm(rad_emb, axis=1, keepdims=True)
        path_emb = path_emb / np.linalg.norm(path_emb, axis=1, keepdims=True)

        sim_matrix = rad_emb @ path_emb.T
        logger.info(f"Similarity matrix shape: {sim_matrix.shape}")

        metrics = _compute_retrieval_metrics(sim_matrix, rad_mrns, path_mrns, k_values)
        all_results[name] = metrics

        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"  {k}: {v}")

        # Free GPU memory between models
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    _save_results(all_results, output_dir, k_values)
    logger.info(f"\nAlignment evaluation complete. Results in {output_dir}")


def _save_results(
    results: Dict[str, Dict],
    output_dir: Path,
    k_values: Tuple[int, ...],
) -> None:
    """Save alignment results as JSON and a markdown table."""
    # JSON
    json_path = output_dir / "alignment_retrieval_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics to {json_path}")

    # Markdown table
    metric_cols = [f"R@{k}" for k in k_values] + ["MRR", "MAP"]
    headers = ["Model"] + metric_cols
    rows = ["|" + "|".join(headers) + "|"]
    rows.append("|" + "|".join(["---"] * len(headers)) + "|")

    for model_name, metrics in results.items():
        cells = [model_name]
        for col in metric_cols:
            val = metrics.get(col, 0.0)
            cells.append(f"{val * 100:.1f}")
        rows.append("|" + "|".join(cells) + "|")

    table = "\n".join(rows)
    md_path = output_dir / "alignment_retrieval_results.md"
    md_path.write_text(table)
    logger.info(f"Saved table to {md_path}")
    logger.info("\n" + table)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    log = setup_logger(name="glioma.alignment", level=logging.INFO,
                       log_file="alignment_eval.log")
    disable_external_loggers()
    config = get_config()
    evaluate_alignment(config)


if __name__ == "__main__":
    main()
