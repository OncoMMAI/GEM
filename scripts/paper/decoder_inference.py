"""Zero-shot decoder inference used for the NOA paper (GPT-4o via Azure OpenAI, BioMistral-7B via local vLLM).

Run from the repo root: python scripts/paper/decoder_inference.py --model gpt-4o
Requires AZURE_OPENAI_API_KEY for GPT-4o. Letter probabilities are read from the generated answer token.
"""
import os
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import numpy as np
import pandas as pd
from openai import OpenAI, AzureOpenAI
from sklearn.model_selection import KFold
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Sentinel for missing letter logprobs. Avoid -9999 from the API (means "not in top-k").
_MISSING_LOGPROB = -100.0


def get_task_labels(task: str) -> List[str]:
    """Get the ordered list of labels for a given task."""
    if task == "tumor_type":
        return ["glioblastoma", "astrocytoma", "oligodendroglioma", "oligoastrocytoma"]
    if task == "tumor_type_3_class":
        return ["glioblastoma", "astrocytoma", "oligodendroglioma"]
    if task == "survival":
        return ["NOT_SURVIVE", "SURVIVE"]
    if task == "mgmt":
        return ["NEGATIVE", "POSITIVE"]
    raise ValueError(f"Unknown task for labels: {task}")


def get_task_prompt_with_letters(clinical_note: str, task: str) -> str:
    """Get the prompt for a given task using the expert context but with direct letter output."""
    
    if task == "tumor_type" or task == "tumor_type_3_class":
        labels = get_task_labels(task)
        # Create letter mapping
        letter_options = "\n".join([f"{chr(ord('A') + i)}: {label}" for i, label in enumerate(labels)])
        
        return f"""You are an expert neuro-radiologist. Based on the following clinical notes, classify the brain tumor type.

Clinical Notes:
{clinical_note}

Choose from:
{letter_options}

Answer with a single letter:"""
    
    elif task == "survival":
        return f"""You are an expert neuro-oncologist. Based on the following clinical notes, predict whether the patient will survive beyond 1 year from diagnosis.

Clinical Notes:
{clinical_note}

Choose from:
A: NOT_SURVIVE
B: SURVIVE

Answer with a single letter:"""
    
    elif task == "mgmt":
        return f"""You are an expert molecular pathologist. Based on the following clinical notes, predict whether the patient's tumor has MGMT promoter methylation.

Clinical Notes:
{clinical_note}

Choose from:
A: NEGATIVE
B: POSITIVE

Answer with a single letter:"""
    
    raise ValueError(f"Unknown task: {task}")


def _clean_letter_token(token: Optional[str], letter_to_idx: Dict[str, int]) -> Optional[str]:
    """Strip whitespace/punct from a logprob token; return A/B/C if valid."""
    if not token:
        return None
    cleaned = re.sub(r"[^A-Z]", "", token.upper())
    if len(cleaned) == 1 and cleaned in letter_to_idx:
        return cleaned
    return None


def _logprob_is_usable(logprob: float) -> bool:
    """Reject API placeholders like -9999.0 that are not real probabilities."""
    return logprob is not None and np.isfinite(logprob) and logprob > -500.0


def get_model_prediction_hybrid(
    clinical_note: str, task: str, client, model_name: str
) -> Tuple[int, np.ndarray]:
    """
    Get model prediction using expert context with direct letter output.
    Extract logprobs from the letter token position (not blindly content[0]).
    """
    labels = get_task_labels(task)
    prompt = get_task_prompt_with_letters(clinical_note, task)
    letter_to_idx = {chr(ord("A") + i): i for i in range(len(labels))}
    n_labels = len(labels)

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
            logprobs=True,
            top_logprobs=20,
        )

        response_text = (completion.choices[0].message.content or "").strip().upper()
        pred_idx = -1
        for char in response_text:
            if char in letter_to_idx:
                pred_idx = letter_to_idx[char]
                break

        probs = np.ones(n_labels, dtype=float) / n_labels
        logprobs_dict = {letter: _MISSING_LOGPROB for letter in letter_to_idx}

        lp = getattr(completion.choices[0], "logprobs", None)
        content = getattr(lp, "content", None) if lp else None
        if content:
            # Prefer the content step whose generated token is a letter (skip leading space/etc.)
            letter_step = None
            for step in content:
                cleaned = _clean_letter_token(getattr(step, "token", None), letter_to_idx)
                if cleaned is not None:
                    letter_step = step
                    # If text parse failed, use this token as the prediction
                    if pred_idx == -1:
                        pred_idx = letter_to_idx[cleaned]
                    break
            if letter_step is None:
                letter_step = content[0]

            # Record the chosen token's own logprob when usable
            chosen = _clean_letter_token(getattr(letter_step, "token", None), letter_to_idx)
            chosen_lp = getattr(letter_step, "logprob", None)
            if chosen is not None and _logprob_is_usable(chosen_lp):
                logprobs_dict[chosen] = float(chosen_lp)

            for tlp in getattr(letter_step, "top_logprobs", None) or []:
                cleaned = _clean_letter_token(getattr(tlp, "token", None), letter_to_idx)
                if cleaned is None:
                    continue
                raw_lp = getattr(tlp, "logprob", None)
                if not _logprob_is_usable(raw_lp):
                    continue
                # Keep the best (least negative) logprob if duplicates appear
                if (
                    logprobs_dict[cleaned] == _MISSING_LOGPROB
                    or float(raw_lp) > logprobs_dict[cleaned]
                ):
                    logprobs_dict[cleaned] = float(raw_lp)

            logprob_values = np.array(
                [logprobs_dict[letter] for letter in sorted(letter_to_idx.keys())],
                dtype=float,
            )
            # If every letter is missing, keep uniform; otherwise softmax over observed letters
            if np.any(logprob_values > _MISSING_LOGPROB):
                exp_logp = np.exp(logprob_values - np.max(logprob_values))
                probs = exp_logp / exp_logp.sum()

            logger.debug("Extracted letter logprobs: %s -> probs %s", logprobs_dict, probs)

        return pred_idx, probs

    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        return -1, np.ones(n_labels, dtype=float) / n_labels


def process_batch_hybrid(
    clinical_notes: List[str], task: str, client, model_name: str
) -> List[tuple]:
    """Process a batch of clinical notes and return predictions and probabilities."""
    results = []
    
    for note in clinical_notes:
        pred_idx, probs = get_model_prediction_hybrid(note, task, client, model_name)
        results.append((pred_idx, probs))
    
    return results


def calculate_metrics_simple(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Dict:
    """Calculate evaluation metrics following EleutherAI approach."""
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    
    # Filter valid predictions
    valid_mask = y_pred != -1
    if not np.any(valid_mask):
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_macro": 0,
            "aucroc": np.nan,
        }
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    y_pred_proba_valid = y_pred_proba[valid_mask]
    
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true_valid, y_pred_valid)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        y_true_valid, y_pred_valid, average="macro", zero_division=0
    )
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1_macro"] = f1_macro
    
    # Calculate AUROC from probabilities
    try:
        if len(np.unique(y_true_valid)) == 2:
            metrics["aucroc"] = roc_auc_score(y_true_valid, y_pred_proba_valid[:, 1])
        else:
            metrics["aucroc"] = roc_auc_score(
                y_true_valid, y_pred_proba_valid, multi_class="ovr",
                labels=list(range(y_pred_proba_valid.shape[1])),
            )
    except Exception as e:
        logger.warning("AUROC failed: %s", e)
        metrics["aucroc"] = np.nan
    
    return metrics


def format_results_table(fold_metrics: List[Dict], task_name: str, model_name: str) -> str:
    """Format results as a markdown table."""
    metric_order = ["accuracy", "precision", "recall", "f1_macro", "aucroc"]
    metric_display = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_macro": "F1-macro",
        "aucroc": "AUCROC",
    }

    headers = ["Model"] + [f"{metric_display[m]} (%)" for m in metric_order]

    table = ["|" + "|".join(headers) + "|"]
    table.append("|" + "|".join(["---"] * len(headers)) + "|")

    # Calculate mean and std for each metric
    row = [model_name]
    for metric in metric_order:
        values = [fold[metric] for fold in fold_metrics]
        if not np.isnan(values).all():
            mean_value = np.nanmean(values) * 100
            std_value = np.nanstd(values) * 100
            row.append(f"{mean_value:.1f} ± {std_value:.1f}")
        else:
            row.append("N/A")

    table.append("|" + "|".join(row) + "|")

    return "\n".join(table)


def setup_client_and_model(model_arg: str):
    """Sets up the client and model name based on the command-line argument."""
    if model_arg.lower() == "biomistral":
        model_name = "BioMistral-7B"
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc123",
        )
        logger.info(f"Using local model: {model_name}")
        return client, model_name
    elif model_arg.lower() == "gpt-4o":
        model_name = "gpt-4o-2024-11-20"
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-10-21",
            azure_endpoint="https://unified-api.ucsf.edu/general",
        )
        logger.info(f"Using Azure OpenAI model: {model_name}")
        return client, model_name
    else:
        raise ValueError("Invalid model specified. Choose 'biomistral' or 'gpt-4o'.")


def main():
    parser = argparse.ArgumentParser(description="Run inference with LLMs - Expert context with letter-based output.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["biomistral", "gpt-4o"],
        help="Model to use for inference.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=30,
        help="Number of samples to process in each batch (default: 10).",
    )
    args = parser.parse_args()

    client, model_name = setup_client_and_model(args.model)

    # Load datasets
    tumor_df = pd.read_csv("data/eval/tumor_type_eval.csv")
    mgmt_df = pd.read_csv("data/eval/mgmt_eval.csv")
    os_df = pd.read_csv("data/eval/os_eval.csv")

    # MGMT: drop missing labels (do NOT map NaN -> 0)
    n_mgmt_raw = len(mgmt_df)
    mgmt_df = mgmt_df[mgmt_df["mgmt"].notna()].copy()
    mgmt_df["mgmt_label"] = (mgmt_df["mgmt"].str.lower() == "positive").astype(int)
    logger.info(
        "MGMT cohort: kept %d / %d rows with non-null mgmt "
        "(%d positive, %d negative)",
        len(mgmt_df),
        n_mgmt_raw,
        int((mgmt_df["mgmt_label"] == 1).sum()),
        int((mgmt_df["mgmt_label"] == 0).sum()),
    )

    # Filter 3-class tumor task
    tumor_df_3class = tumor_df[tumor_df["tumor_type"] != "oligoastrocytoma"].copy()

    # Create results directory
    results_dir = Path(f"results/{model_name}_hybrid")
    results_dir.mkdir(exist_ok=True, parents=True)
    preds_dir = results_dir / "per_sample"
    preds_dir.mkdir(exist_ok=True)

    # Define tasks - using 3-class tumor type to match letter-based results
    tasks = {
        "tumor_type_3_class": {"df": tumor_df_3class, "label_col": "class_label"},
        "survival": {"df": os_df.copy(), "label_col": "one_year_survival"},
        "mgmt": {"df": mgmt_df.copy(), "label_col": "mgmt_label"},
    }

    # Run evaluation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_results = []

    for task_name, task_config in tasks.items():
        logger.info(f"\nProcessing {task_name} task...")

        df = task_config["df"]
        valid_mask = df[task_config["label_col"]].notna()
        df = df[valid_mask].reset_index(drop=True)
        logger.info(
            "Task %s: %d samples; label counts %s",
            task_name,
            len(df),
            df[task_config["label_col"]].value_counts().to_dict(),
        )

        # Get text column
        if "report" in df.columns:
            text_col = "report"
        elif "text" in df.columns:
            text_col = "text"
        elif "Anon Report Text" in df.columns:
            text_col = "Anon Report Text"
        else:
            raise ValueError("No valid text column found")

        fold_metrics = []
        task_y_true = []
        task_y_pred = []
        task_y_prob = []
        task_fold_id = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df), 1):
            logger.info(f"Processing fold {fold_idx}...")
            test_df = df.iloc[test_idx]

            clinical_notes = test_df[text_col].tolist()

            predictions = []
            probabilities = []

            batches = [
                clinical_notes[i : i + args.batch_size]
                for i in range(0, len(clinical_notes), args.batch_size)
            ]

            for batch in tqdm(batches, desc=f"Fold {fold_idx} batches"):
                batch_results = process_batch_hybrid(batch, task_name, client, model_name)
                for pred_idx, probs in batch_results:
                    predictions.append(pred_idx)
                    probabilities.append(probs)

            predictions = np.array(predictions)
            probabilities = np.array(probabilities)
            true_labels = test_df[task_config["label_col"]].values.astype(int)

            metrics = calculate_metrics_simple(true_labels, predictions, probabilities)
            fold_metrics.append(metrics)

            task_y_true.append(true_labels)
            task_y_pred.append(predictions)
            task_y_prob.append(probabilities)
            task_fold_id.append(np.full(len(true_labels), fold_idx, dtype=int))

            logger.info(
                f"Fold {fold_idx} - Accuracy: {metrics['accuracy']:.3f}, "
                f"Precision: {metrics['precision']:.3f}, "
                f"Recall: {metrics['recall']:.3f}, "
                f"F1: {metrics['f1_macro']:.3f}, "
                f"AUROC: {metrics['aucroc']:.3f}"
            )

        # Persist per-sample outputs so AUROC can be re-audited without a re-run
        y_true_all = np.concatenate(task_y_true)
        y_pred_all = np.concatenate(task_y_pred)
        y_prob_all = np.vstack(task_y_prob)
        fold_all = np.concatenate(task_fold_id)
        np.savez_compressed(
            preds_dir / f"{task_name}_predictions.npz",
            y_true=y_true_all,
            y_pred=y_pred_all,
            y_prob=y_prob_all,
            fold=fold_all,
        )
        logger.info(
            "Saved per-sample predictions to %s",
            preds_dir / f"{task_name}_predictions.npz",
        )

        results_table = format_results_table(fold_metrics, task_name, model_name)
        task_results = (
            f"\n## {task_name.replace('_', ' ').title()} Task Results\n\n"
            f"{results_table}\n"
        )
        all_results.append(task_results)

        logger.info(f"\nResults for {task_name} task:\n{results_table}")

    # Save results
    output_file = results_dir / "results_hybrid_style.md"
    with open(output_file, "w") as f:
        f.write(f"# Results for {model_name} - Expert Context with Letter-Based Output\n")
        f.write("\n".join(all_results))

    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
