"""
End-to-end evaluation script for glioma classification models.

This script performs a full evaluation pipeline:
1.  Loads configuration from a YAML file.
2.  Iterates through different fine-tuning stages of an LLM2Vec model.
3.  For each model stage, it loads clinical data for multiple classification tasks.
4.  Generates text embeddings for the data using the specific model.
5.  Immediately performs k-fold cross-validation using a RandomForestClassifier,
    using the exact evaluation logic required.
6.  Aggregates and saves the final performance metrics and result tables.
"""

import argparse
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from llm2vec import LLM2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def setup_logging():
    """Sets up a basic logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: str) -> dict:
    """Loads the YAML configuration file."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {path}")
        raise
    except Exception as e:
        logging.error(f"Error reading YAML configuration: {e}")
        raise


def evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int,
    random_state: int,
) -> Dict[str, float]:
    """
    Evaluate a single fold with a RandomForestClassifier.
    This function is an exact copy of the logic in your evaluation script.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,  # Changed from -1 to avoid multiprocessing issues
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro,
    }

    if len(np.unique(y_test)) == 2:
        metrics["aucroc"] = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        metrics["aucroc"] = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")

    return metrics


def run_kfold_evaluation(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_splits: int,
    n_estimators: int,
    random_state: int,
) -> Dict[str, Any]:
    """
    Run k-fold cross-validation and return aggregated metrics.
    This function is an exact copy of the logic in your evaluation script.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []

    for train_idx, test_idx in tqdm(
        skf.split(embeddings, labels),
        total=n_splits,
        desc=f"Running {n_splits}-fold CV",
        leave=False,
    ):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        metrics = evaluate_fold(
            X_train, y_train, X_test, y_test, n_estimators, random_state
        )
        fold_metrics.append(metrics)

    avg_metrics = {}
    for key in fold_metrics[0].keys():
        values = [f[key] for f in fold_metrics]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f"{key}_std"] = np.std(values)
        avg_metrics[f"{key}_values"] = values

    return avg_metrics


def load_model_for_stage(
    stage: str, checkpoints: dict, base_model_path: str, device: str, dtype: torch.dtype
) -> LLM2Vec:
    """Loads the base model and appropriate adapters for each stage."""
    logging.info(f"--- Loading model for stage: {stage.upper()} ---")

    # Determine max_length based on stage
    max_length = 2048

    logging.info(f"Using max_length={max_length} for {stage} stage")

    if stage == "mntp":
        # Load base model with MNTP adapter
        l2v = LLM2Vec.from_pretrained(
            base_model_path,
            peft_model_name_or_path=checkpoints["mntp"],
            device_map=device,
            torch_dtype=dtype,
            max_length=max_length,
            attn_implementation="flash_attention_2",  # Match training config
            # merge_peft defaults to False, which is correct for evaluation
        )
    elif stage == "simcse":
        # Load base model with SimCSE adapter
        # Note: This assumes SimCSE was trained on top of base model (not MNTP-merged)
        # If SimCSE was trained on MNTP-merged model, you'd need to adjust accordingly
        l2v = LLM2Vec.from_pretrained(
            base_model_path,
            peft_model_name_or_path=checkpoints["simcse"],
            device_map=device,
            torch_dtype=dtype,
            max_length=max_length,
            attn_implementation="flash_attention_2",  # Match training config
        )
    elif stage == "supervised":
        # Load base model with supervised adapter
        # Based on your training log, this was trained starting from MNTP checkpoint
        l2v = LLM2Vec.from_pretrained(
            base_model_path,
            peft_model_name_or_path=checkpoints["supervised"],
            device_map=device,
            torch_dtype=dtype,
            max_length=max_length,
            attn_implementation="flash_attention_2",  # Match training config
        )
    else:
        # Fallback to base model only
        l2v = LLM2Vec.from_pretrained(
            base_model_path,
            device_map=device,
            torch_dtype=dtype,
            max_length=max_length,
            attn_implementation="flash_attention_2",  # Match training config
        )

    return l2v


def format_results_table(results: Dict[str, Any], task_name: str) -> str:
    """Formats evaluation results into a markdown table."""
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Macro", "AUC-ROC"]
    table = [
        f"### Results for Task: {task_name.replace('_', ' ').title()}",
        "| " + " | ".join(headers) + " |",
        "|---" * len(headers) + "|",
    ]

    for model_name, metrics in sorted(results.items()):
        acc = f"{metrics.get('accuracy', 0) * 100:.2f} ± {metrics.get('accuracy_std', 0) * 100:.2f}"
        precision = f"{metrics.get('precision', 0) * 100:.2f} ± {metrics.get('precision_std', 0) * 100:.2f}"
        recall = f"{metrics.get('recall', 0) * 100:.2f} ± {metrics.get('recall_std', 0) * 100:.2f}"
        f1 = f"{metrics.get('f1_macro', 0) * 100:.2f} ± {metrics.get('f1_macro_std', 0) * 100:.2f}"
        auc = f"{metrics.get('aucroc', 0) * 100:.2f} ± {metrics.get('aucroc_std', 0) * 100:.2f}"
        row = f"| {model_name} | {acc} | {precision} | {recall} | {f1} | {auc} |"
        table.append(row)

    return "\n".join(table)


def main(config_path: str):
    """Main entry point for the end-to-end evaluation pipeline."""
    setup_logging()
    config = load_config(config_path)

    base_model_path = "/data/users/neob/Llama-3.1-8B-Instruct"
    checkpoints = {
        "mntp": "/data/users/neob/llm2vec/output/clinical_llama3.1_mntp_2048",
        "simcse": "/data/users/neob/llm2vec/output/clinical_llama3.1_simcse_2048/checkpoint-400",
        "supervised": "/data/users/neob/llm2vec/output/clinical_llama3.1_supervised_2048/MedicalPairData_train_m-Llama-3.1-8B-Instruct_p-mean_b-512_l-2048_bidirectional-True_e-5_s-42_w-300_lr-3e-05_lora_r-16/checkpoint-800",
    }
    stages_to_run = {
        # "Stage 1: MNTP Only": "mntp",
        # "Stage 2: MNTP + SimCSE": "simcse",
        "Stage 3: Full Finetune": "supervised",
    }
    tasks = {
        "tumor_type": {"path": config["data"]["type_path"], "label": "tumor_type"},
        "survival": {"path": config["data"]["os_path"], "label": "one_year_survival"},
        "mgmt": {"path": config["data"]["mgmt_path"], "label": "mgmt_label"},
    }

    data_cfg, eval_cfg, embed_cfg = (
        config["data"],
        config["evaluation"],
        config["embedding"],
    )
    text_column = data_cfg["text_column"]

    all_results = {task_name: {} for task_name in tasks}

    for model_name, stage in stages_to_run.items():
        model = load_model_for_stage(
            stage,
            checkpoints,
            base_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16,
        )

        for task_name, task_info in tasks.items():
            logging.info(
                f"--- Evaluating Model '{model_name}' on Task '{task_name}' ---"
            )

            df = pd.read_csv(task_info["path"])
            # Special handling for mgmt_label as per your original script
            if task_name == "mgmt":
                df["mgmt_label"] = (
                    df["mgmt"].notna() & (df["mgmt"] == "positive")
                ).astype(int)

            texts = df[text_column].fillna("").tolist()
            labels = df[task_info["label"]].values

            # -------------------------------------------------------------
            # Tumor-type task: drop the "oligoastrocytoma" class so that we
            # evaluate on exactly three classes (glioblastoma, astrocytoma,
            # oligodendroglioma) to keep the evaluation consistent with the
            # updated visualisations.
            # -------------------------------------------------------------
            if task_name == "tumor_type":
                filter_mask = df["tumor_type"] != "oligoastrocytoma"
                valid_mask = pd.notna(labels) & filter_mask
            else:
                valid_mask = pd.notna(labels)

            # Convert to a NumPy boolean array for safe indexing
            valid_mask_np = np.asarray(valid_mask, dtype=bool)

            valid_texts = [text for i, text in enumerate(texts) if valid_mask_np[i]]
            valid_labels = labels[valid_mask_np]

            # Check if embeddings already exist
            embeddings_dir = Path(config["output"]["results_dir"]) / "embeddings"
            embeddings_dir.mkdir(exist_ok=True, parents=True)

            embedding_filename = f"{model_name}_{task_name}_embeddings.npy"
            embedding_path = embeddings_dir / embedding_filename

            if embedding_path.exists() and not config.get("force_regenerate", False):
                logging.info(
                    f"Found existing embeddings for {model_name}/{task_name}, loading from disk..."
                )
                embeddings = torch.from_numpy(np.load(embedding_path))

                # If the stored embeddings were generated *before* the new 3-class
                # filtering, their first dimension will match the full dataset
                # length.  In that case we need to apply the same mask here so
                # that the embeddings align with `valid_labels`.
                if embeddings.shape[0] == len(valid_mask_np):
                    embeddings = embeddings[valid_mask_np]

                logging.info(f"Successfully loaded embeddings from {embedding_path}")
            else:
                logging.info(
                    f"Generating embeddings for {len(valid_texts)} valid samples..."
                )
                embeddings = model.encode(
                    valid_texts, batch_size=embed_cfg["batch_size"]
                )

                nan_count = np.isnan(embeddings.cpu().numpy()).any(axis=1).sum()
                if nan_count > 0:
                    logging.warning(
                        f"Found {nan_count} samples with NaN embeddings ({nan_count / len(embeddings) * 100:.1f}%)"
                    )
                    logging.warning(
                        "This may indicate issues with certain text inputs that the model cannot process"
                    )

                # Save embeddings
                np.save(embedding_path, embeddings.cpu().numpy())
                logging.info(f"Saved embeddings to {embedding_path}")

            metrics = run_kfold_evaluation(
                embeddings.cpu().numpy(),
                valid_labels,
                n_splits=eval_cfg["n_splits"],
                n_estimators=eval_cfg["n_estimators"],
                random_state=eval_cfg["random_state"],
            )
            all_results[task_name][model_name] = metrics

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(exist_ok=True, parents=True)

    logging.info("\n--- FINAL RESULTS ---")
    for task_name, task_results in all_results.items():
        results_table = format_results_table(task_results, task_name)
        logging.info("\n" + results_table)

        # Save results
        (results_dir / f"{task_name}_results.md").write_text(results_table + "\n")
        with open(results_dir / f"{task_name}_metrics.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {
                m: {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in met.items()
                }
                for m, met in task_results.items()
            }
            json.dump(serializable_results, f, indent=2)

    logging.info(f"\nAll evaluation results saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full evaluation pipeline for glioma classification."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)
