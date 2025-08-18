# -*- coding: utf-8 -*-

"""
Inference module for glioma classification using large language models.

This module performs zero-shot inference using LLMs for glioma classification tasks.
It evaluates the model's performance on type classification, MGMT methylation
prediction, and survival prediction using clinical text reports using BioMistral.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.utils.config import get_config
from src.utils.logging import setup_logger, disable_external_loggers
# calculate_metrics no longer used in letter-based path


# Default prompt templates if not specified in config (letter-based, concise)
DEFAULT_PROMPT_TEMPLATES = {
    "tumor_type": """You are an expert neuro-radiologist. Classify the brain tumor type.

Clinical Notes:
{text}

Choose from:
A: glioblastoma
B: astrocytoma
C: oligodendroglioma

Answer with a single letter.""",
    "survival": """You are an expert neuro-oncologist. Predict 1-year survival.

Clinical Notes:
{text}

Choose from:
A: NOT_SURVIVE
B: SURVIVE

Answer with a single letter.""",
    "mgmt": """You are an expert molecular pathologist. Predict MGMT promoter methylation.

Clinical Notes:
{text}

Choose from:
A: NEGATIVE
B: POSITIVE

Answer with a single letter.""",
}


class LLMInferenceClient:
    """Client for performing inference with OpenAI-compatible API endpoints."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        max_tokens: int = 50,
        temperature: float = 0.0,
        max_retries: int = 3,
        prompt_templates: Optional[Dict[str, str]] = None,
    ):
        """Initialize the LLM inference client.

        Args:
            model_name: Name of the model to use
            base_url: Base URL for the API endpoint
            api_key: API key for authentication
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            max_retries: Maximum number of retries on failure
            prompt_templates: Custom prompt templates for different tasks
        """
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.prompt_templates = prompt_templates or DEFAULT_PROMPT_TEMPLATES
        self.logger = logging.getLogger("glioma.inference")

    def get_prediction(self, clinical_note: str, task: str) -> str:
        """Get model prediction for a single example.

        Args:
            clinical_note: Clinical note text
            task: Task type (tumor_type, survival, mgmt)

        Returns:
            Model's prediction as a string
        """
        if task not in self.prompt_templates:
            # Try alternate keys
            alternate_mapping = {"type": "tumor_type"}
            mapped_task = alternate_mapping.get(task, task)
            if mapped_task not in self.prompt_templates:
                raise ValueError(f"Unknown task: {task}")
            task = mapped_task

        prompt = self.prompt_templates[task].format(text=clinical_note)

        # Add retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return completion.choices[0].message.content
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Failed after {self.max_retries} attempts: {e}")
                    return ""
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2**attempt)  # Exponential backoff

    def parse_prediction(self, prediction: str, task: str) -> int:
        """Parse model output into numerical labels.

        Args:
            prediction: Raw prediction from the model
            task: Task type (tumor_type, survival, mgmt)

        Returns:
            Integer label (-1 if parsing failed)
        """
        if not prediction:
            return -1

        prediction = prediction.lower().strip()

        # Normalize task name
        if task == "type":
            task = "tumor_type"

        if task == "tumor_type":
            if "glioblastoma" in prediction:
                return 0
            elif "astrocytoma" in prediction:
                return 1
            elif "oligodendroglioma" in prediction:
                return 2
            # Removed oligoastrocytoma from the label set

        elif task == "survival":
            if "survive" in prediction:
                return 1
            elif "not_survive" in prediction:
                return 0

        elif task == "mgmt":
            if "positive" in prediction:
                return 1
            elif "negative" in prediction:
                return 0

        # Parsing failed
        self.logger.warning(f"Failed to parse prediction: {prediction}")
        return -1

    def process_batch(self, clinical_notes: List[str], task: str) -> List[int]:
        """Process a batch of examples.

        Args:
            clinical_notes: List of clinical note texts
            task: Task type (type, survival, mgmt)

        Returns:
            List of integer labels
        """
        predictions = []
        for note in clinical_notes:
            raw_prediction = self.get_prediction(note, task)
            label = self.parse_prediction(raw_prediction, task)
            predictions.append(label)
        return predictions

    # --- Letter-based prompting with logprobs for calibrated AUROC ---
    def _get_task_labels(self, task: str) -> list:
        if task == "tumor_type":
            return ["glioblastoma", "astrocytoma", "oligodendroglioma"]
        if task == "survival":
            return ["NOT_SURVIVE", "SURVIVE"]
        if task == "mgmt":
            return ["NEGATIVE", "POSITIVE"]
        raise ValueError(f"Unknown task: {task}")

    def _build_letter_prompt(self, clinical_note: str, task: str) -> str:
        labels = self._get_task_labels(task)
        options = "\n".join(
            [f"{chr(ord('A') + i)}: {lab}" for i, lab in enumerate(labels)]
        )
        if task == "tumor_type":
            header = (
                "You are an expert neuro-radiologist. Classify the brain tumor type."
            )
        elif task == "survival":
            header = "You are an expert neuro-oncologist. Predict 1-year survival."
        else:
            header = "You are an expert molecular pathologist. Predict MGMT promoter methylation."
        return f"""{header}

Clinical Notes:
{clinical_note}

Choose from:
{options}

Answer with a single letter:"""

    def predict_letter_with_probs(
        self, clinical_note: str, task: str
    ) -> tuple[int, "np.ndarray"]:
        import numpy as np  # local import to avoid global dependency if not used

        labels = self._get_task_labels(task)
        prompt = self._build_letter_prompt(clinical_note, task)
        letter_to_idx = {chr(ord("A") + i): i for i in range(len(labels))}

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5,
                logprobs=True,
                top_logprobs=10,
            )

            # Extract predicted letter
            response_text = completion.choices[0].message.content or ""
            response_text = response_text.strip().upper()
            pred_idx = -1
            for ch in response_text:
                if ch in letter_to_idx:
                    pred_idx = letter_to_idx[ch]
                    break

            # Default probabilities
            probs = np.ones(len(labels), dtype=float) / len(labels)

            # Extract logprobs if provided for the first output token
            lp = getattr(completion.choices[0], "logprobs", None)
            if lp and getattr(lp, "content", None):
                top = lp.content[0].top_logprobs
                logprob_map = {ltr: -100.0 for ltr in letter_to_idx}
                for entry in top:
                    tok = (entry.token or "").upper()
                    # Keep only A/B/C letters from token
                    if len(tok) == 1 and tok in letter_to_idx:
                        logprob_map[tok] = entry.logprob
                logvals = np.array(
                    [logprob_map[ltr] for ltr in sorted(letter_to_idx.keys())]
                )
                expv = np.exp(logvals - np.max(logvals))
                probs = expv / expv.sum()

            return pred_idx, probs
        except Exception as e:
            self.logger.error(f"Letter prediction failed: {e}")
            return -1, np.ones(len(labels), dtype=float) / len(labels)

    def get_predictions(
        self,
        df: pd.DataFrame,
        task: str,
        text_col: Optional[str] = None,
        batch_size: int = 10,
        num_workers: int = 4,
    ) -> np.ndarray:
        """Get predictions for all examples in parallel batches.

        Args:
            df: DataFrame containing clinical notes
            task: Task type (type, survival, mgmt)
            text_col: Column name containing the text (if None, automatically detected)
            batch_size: Number of examples to process in each batch
            num_workers: Number of parallel workers

        Returns:
            Array of predictions
        """
        # Use text column from dataframe
        if text_col is None:
            text_columns = ["report", "text", "Anon Report Text"]
            text_col = next((col for col in text_columns if col in df.columns), None)

        if text_col is None:
            raise ValueError("No valid text column found in dataframe")

        clinical_notes = df[text_col].tolist()
        self.logger.info(
            f"Processing {len(clinical_notes)} notes with {num_workers} workers"
        )

        # Create batches
        batches = [
            clinical_notes[i : i + batch_size]
            for i in range(0, len(clinical_notes), batch_size)
        ]

        # Disable verbose logging from external libraries
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Process batches in parallel
        predictions = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for batch in batches:
                futures.append(executor.submit(self.process_batch, batch, task))

            for future in tqdm(
                futures,
                total=len(batches),
                desc=f"Getting {task} predictions",
            ):
                batch_predictions = future.result()
                predictions.extend(batch_predictions)

        return np.array(predictions)


def format_results_table(results: Dict[str, Dict], model_name: str) -> str:
    """Format results as a markdown table.

    Args:
        results: Dictionary of fold results
        model_name: Name of the model

    Returns:
        Markdown formatted table
    """
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
        values = [
            fold_results[metric]
            for fold_results in results.values()
            if metric in fold_results
        ]
        if values:
            mean_value = np.mean(values) * 100
            std_value = np.std(values) * 100
            row.append(f"{mean_value:.1f} ± {std_value:.1f}")
        else:
            row.append("N/A")

    table.append("|" + "|".join(row) + "|")

    return "\n".join(table)


def run_inference_task(
    task_name: str,
    df: pd.DataFrame,
    label_col: str,
    client: LLMInferenceClient,
    text_col: Optional[str] = None,
    n_splits: int = 10,
    random_state: int = 42,
    batch_size: int = 10,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """Run inference for a specific task using letter prompts and logprob-based AUROC.

    Returns a dict of fold -> metrics.
    """
    logger = logging.getLogger("glioma.inference")
    logger.info(f"\nProcessing {task_name} task...")

    # Filter out rows with null labels
    valid_mask = df[label_col].notna()
    valid_df = df[valid_mask]

    logger.info(f"Dataset has {len(valid_df)} valid samples (out of {len(df)} total)")
    logger.info(f"Label distribution: {valid_df[label_col].value_counts().to_dict()}")

    # Determine text column
    if text_col is None:
        for c in ["report", "text", "Anon Report Text"]:
            if c in valid_df.columns:
                text_col = c
                break
    if text_col is None:
        raise ValueError("No valid text column found in dataframe")

    # K-Fold CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results: Dict[int, Dict[str, float]] = {}
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    for fold_idx, (_, test_idx) in enumerate(kf.split(valid_df), 1):
        logger.info(f"Processing fold {fold_idx}/{n_splits}...")
        test_df = valid_df.iloc[test_idx]

        # Collect predictions and probabilities
        preds: list[int] = []
        probs: list[np.ndarray] = []
        notes = test_df[text_col].tolist()

        for i in range(0, len(notes), batch_size):
            batch = notes[i : i + batch_size]
            for note in batch:
                pred_idx, pvec = client.predict_letter_with_probs(note, task_name)
                preds.append(pred_idx)
                probs.append(pvec)

        y_true = test_df[label_col].values.astype(int)
        y_pred = np.array(preds)
        y_prob = np.vstack(probs) if probs else np.zeros((0, 0))

        # Compute metrics (filter invalid)
        valid_pred = y_pred != -1
        if not np.any(valid_pred):
            fold_results[fold_idx] = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_macro": 0.0,
                "aucroc": float("nan"),
            }
            continue

        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            roc_auc_score,
        )

        yt = y_true[valid_pred]
        yp = y_pred[valid_pred]
        yp_prob = y_prob[valid_pred] if y_prob.size else None

        acc = accuracy_score(yt, yp)
        pr, rc, f1, _ = precision_recall_fscore_support(
            yt, yp, average="macro", zero_division=0
        )
        # AUROC from probabilities when available
        try:
            uniq = np.unique(yt)
            if yp_prob is not None and yp_prob.size:
                if len(uniq) == 2:
                    auc = roc_auc_score(yt, yp_prob[:, 1])
                else:
                    auc = roc_auc_score(yt, yp_prob, multi_class="ovr")
            else:
                auc = float("nan")
        except Exception:
            auc = float("nan")

        fold_results[fold_idx] = {
            "accuracy": acc,
            "precision": pr,
            "recall": rc,
            "f1_macro": f1,
            "aucroc": auc,
        }

        y_true_all.append(yt)
        y_pred_all.append(yp)

    overall_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    overall_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([])

    return {
        "fold_results": fold_results,
        "overall_y_true": overall_true,
        "overall_y_pred": overall_pred,
    }


def run_inference(config: Dict[str, Any]) -> None:
    """Run inference for all tasks.

    Args:
        config: Configuration dictionary
    """
    logger = logging.getLogger("glioma.inference")

    # Model configuration
    model_config = config["model"]
    model_name = model_config["name"]
    base_url = model_config["base_url"]
    api_key = model_config["api_key"]

    # Inference configuration
    inference_config = config["inference"]
    batch_size = inference_config.get("batch_size", 10)
    max_tokens = inference_config.get("max_tokens", 50)
    temperature = inference_config.get("temperature", 0.0)
    num_workers = inference_config.get("num_threads", 4)
    max_retries = inference_config.get("max_retries", 3)

    # Evaluation configuration (align with provided decoder inference defaults)
    eval_config = config["evaluation"]
    n_splits = eval_config.get("n_splits", 10)
    random_state = eval_config.get("random_state", 42)

    # Output configuration
    output_dir = Path(config["output"]["results_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get prompt templates from config if available
    prompt_templates = None
    if "prompts" in config:
        prompt_templates = config["prompts"]
        logger.info("Using custom prompt templates from config")

    # Initialize client
    client = LLMInferenceClient(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries,
        prompt_templates=prompt_templates,
    )

    # Load datasets
    data_config = config["data"]
    logger.info("Loading datasets...")

    try:
        type_df = pd.read_csv(data_config["type_path"])
        logger.info(f"Loaded type data: {len(type_df)} samples")
    except Exception as e:
        logger.error(f"Failed to load type data: {e}")
        type_df = None

    try:
        mgmt_df = pd.read_csv(data_config["mgmt_path"])
        mgmt_df["survival_label"] = (mgmt_df["os_days"] > 364).astype(int)
        mgmt_df["mgmt_label"] = (
            mgmt_df["mgmt"].notna() & (mgmt_df["mgmt"] == "positive")
        ).astype(int)
        logger.info(f"Loaded MGMT/OS data: {len(mgmt_df)} samples")
    except Exception as e:
        logger.error(f"Failed to load MGMT/OS data: {e}")
        mgmt_df = None

    try:
        os_df = pd.read_csv(data_config["os_path"])
        logger.info(f"Loaded OS full data: {len(os_df)} samples")
    except Exception as e:
        logger.error(f"Failed to load OS full data: {e}")
        os_df = None

    # Get text column from config
    text_column = data_config.get("text_column", "report")
    logger.info(f"Using '{text_column}' as the text column for all datasets")

    # Define tasks
    tasks = {}

    if type_df is not None:
        tasks["tumor_type"] = {
            "df": type_df,
            "label_col": "tumor_type",
            "text_col": text_column,
        }
        if isinstance(type_df["tumor_type"].iloc[0], str):
            label_mapping = {
                "glioblastoma": 0,
                "astrocytoma": 1,
                "oligodendroglioma": 2,
            }
            type_df["tumor_type"] = type_df["tumor_type"].str.lower().map(label_mapping)

    if os_df is not None:
        tasks["survival"] = {
            "df": os_df,
            "label_col": "one_year_survival",
            "text_col": text_column,
        }

    if mgmt_df is not None:
        tasks["mgmt"] = {
            "df": mgmt_df,
            "label_col": "mgmt_label",
            "text_col": text_column,
        }

    # Run inference for each task
    all_results = []

    for task_name, task_config in tasks.items():
        task_out = run_inference_task(
            task_name=task_name,
            df=task_config["df"],
            label_col=task_config["label_col"],
            client=client,
            text_col=task_config.get("text_col"),
            n_splits=n_splits,
            random_state=random_state,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        fold_results = task_out["fold_results"]
        # Format results table
        results_table = format_results_table(fold_results, model_name)
        all_results.append(f"# {task_name.title()}\n\n{results_table}\n")

        # Save fold metrics
        task_file = output_dir / f"{task_name}_results.json"
        with open(task_file, "w") as f:
            json.dump(fold_results, f, indent=2)

        # Save classification report and confusion matrix
        try:
            from sklearn.metrics import classification_report, confusion_matrix

            y_true = task_out["overall_y_true"]
            y_pred = task_out["overall_y_pred"]
            if y_true.size and y_pred.size:
                unique_labels = np.unique(y_true)
                target_names = [str(lbl) for lbl in sorted(unique_labels)]
                report_txt = classification_report(
                    y_true, y_pred, target_names=target_names, digits=3
                )
                cm = confusion_matrix(y_true, y_pred, labels=sorted(unique_labels))
                import pandas as _pd

                cm_df = _pd.DataFrame(cm, index=target_names, columns=target_names)
                (output_dir / f"{task_name}_classification_report.txt").write_text(
                    report_txt
                )
                cm_df.to_csv(output_dir / f"{task_name}_confusion_matrix.csv")
        except Exception as e:
            logger.warning(f"Failed detailed report for {task_name}: {e}")

        logger.info(f"\nResults for {task_name} task:\n{results_table}")

    # Save all results to a single markdown file
    output_file = output_dir / "all_results.md"
    with open(output_file, "w") as f:
        f.write("\n".join(all_results))

    logger.info(f"\nAll results saved to {output_file}")


def main() -> None:
    """Main entry point for inference module."""
    # Setup logger
    logger = setup_logger(
        name="glioma.inference", level=logging.INFO, log_file="inference.log"
    )

    # Disable verbose logging from external libraries
    disable_external_loggers()

    # Load configuration
    config = get_config()

    # Run inference
    run_inference(config)


if __name__ == "__main__":
    main()
