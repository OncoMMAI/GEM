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
from src.utils.metrics import calculate_metrics


# Default prompt templates if not specified in config
DEFAULT_PROMPT_TEMPLATES = {
    "tumor_type": """You are an expert neuro-radiologist. Based on the following clinical notes, classify the brain tumor type into one of these categories: glioblastoma, astrocytoma, oligodendroglioma, or oligoastrocytoma.

Clinical Notes:
{text}

Provide your classification in the following format:
CLASSIFICATION: [tumor type]

Only output one of the four specified categories.""",
    "survival": """You are an expert neuro-oncologist. Based on the following clinical notes, predict whether the patient will survive beyond 1 year from diagnosis.

Clinical Notes:
{text}

Provide your prediction in the following format:
PREDICTION: [SURVIVE/NOT_SURVIVE]""",
    "mgmt": """You are an expert molecular pathologist. Based on the following clinical notes, predict whether the patient's tumor has MGMT promoter methylation (positive) or not (negative).

Clinical Notes:
{text}

Provide your prediction in the following format:
MGMT_STATUS: [POSITIVE/NEGATIVE]""",
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
            elif "oligoastrocytoma" in prediction:
                return 3

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
) -> Dict[int, Dict[str, float]]:
    """Run inference for a specific task.

    Args:
        task_name: Name of the task (type, survival, mgmt)
        df: DataFrame containing the data
        label_col: Column name containing the labels
        client: LLM inference client
        text_col: Column name containing the text (if None, automatically detected)
        n_splits: Number of folds for cross-validation
        random_state: Random seed for reproducibility
        batch_size: Number of examples to process in each batch
        num_workers: Number of parallel workers

    Returns:
        Dictionary of fold results
    """
    logger = logging.getLogger("glioma.inference")
    logger.info(f"\nProcessing {task_name} task...")

    # Filter out rows with null labels
    valid_mask = df[label_col].notna()
    valid_df = df[valid_mask]

    logger.info(f"Dataset has {len(valid_df)} valid samples (out of {len(df)} total)")

    # Get label distribution
    label_counts = valid_df[label_col].value_counts()
    logger.info(f"Label distribution: {label_counts.to_dict()}")

    # Setup k-fold cross validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = {}

    # Run each fold
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(valid_df), 1):
        logger.info(f"\nProcessing fold {fold_idx}/{n_splits}...")

        # Get test data for this fold
        test_df = valid_df.iloc[test_idx].copy()

        # Get predictions for test set
        predictions = client.get_predictions(
            test_df,
            task_name,
            text_col=text_col,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Get true labels
        true_labels = test_df[label_col].values

        # Filter out invalid predictions
        valid_pred_mask = predictions != -1
        if not np.all(valid_pred_mask):
            logger.warning(
                f"Filtered out {np.sum(~valid_pred_mask)} invalid predictions "
                f"(out of {len(predictions)})"
            )
            predictions = predictions[valid_pred_mask]
            true_labels = true_labels[valid_pred_mask]

        # Calculate metrics for this fold
        fold_results[fold_idx] = calculate_metrics(true_labels, predictions)

        # Log fold results
        log_metrics = {k: f"{v:.4f}" for k, v in fold_results[fold_idx].items()}
        logger.info(f"Fold {fold_idx} metrics: {log_metrics}")

    # Calculate and log average metrics
    avg_metrics = {}
    for metric in next(iter(fold_results.values())).keys():
        values = [fold[metric] for fold in fold_results.values() if metric in fold]
        avg = np.mean(values)
        std = np.std(values)
        avg_metrics[metric] = (avg, std)
        logger.info(f"Average {metric}: {avg:.4f} ± {std:.4f}")

    return fold_results


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

    # Evaluation configuration
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
            "label_col": "class_label",
            "text_col": text_column,
        }

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
        fold_results = run_inference_task(
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

        # Format results table
        results_table = format_results_table(fold_results, model_name)
        all_results.append(f"# {task_name.title()} Task Results\n\n{results_table}\n")

        # Save individual task results
        task_file = output_dir / f"{task_name}_results.json"
        with open(task_file, "w") as f:
            json.dump(fold_results, f, indent=2)

        logger.info(f"\nResults for {task_name} task:")
        logger.info("\n" + results_table)

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
