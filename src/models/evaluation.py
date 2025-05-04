# -*- coding: utf-8 -*-

"""
Evaluation module for glioma classification models.

This module evaluates the performance of embedding models on multiple glioma
classification tasks including type classification, MGMT methylation prediction,
and survival prediction. It implements k-fold cross-validation with random forest
classifiers and calculates statistical significance and effect sizes between models.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

from src.utils.config import get_config
from src.utils.logging import setup_logger, disable_external_loggers
from src.utils.metrics import format_results_table


def load_embeddings(embedding_path: Path) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Load embedding file and extract model name.

    Args:
        embedding_path: Path to the embedding file

    Returns:
        Tuple containing:
            - numpy array of embeddings (None if loading failed)
            - model name derived from filename (None if loading failed)
    """
    if embedding_path.suffix == ".npy":
        try:
            embeddings = np.load(embedding_path)
            model_name = embedding_path.stem
            return embeddings, model_name
        except Exception as e:
            logging.error(f"Failed to load embeddings from {embedding_path}: {e}")

    return None, None


def evaluate_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 1000,
    random_state: int = 7,
) -> Dict:
    """Evaluate a single fold for both tumor_type and survival metrics.

    Args:
        X_train: Training data embeddings
        X_test: Test data embeddings
        y_train: Training data labels
        y_test: Test data labels
        n_estimators: Number of trees in the random forest
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing classification metrics
    """
    # Initialize classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    # Fit and predict
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro,
    }

    # Add AUC for binary and multiclass classification
    unique_classes = np.unique(y_test)
    if len(unique_classes) == 2:
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    metrics["aucroc"] = auc

    return metrics


def run_kfold_evaluation(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    n_estimators: int = 1000,
    random_state: int = 7,
) -> Dict:
    """Run k-fold cross validation.

    Args:
        embeddings: Input embeddings
        labels: Target labels
        n_splits: Number of folds
        n_estimators: Number of trees in random forest
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing averaged metrics and values for each fold
    """
    logger = logging.getLogger("glioma.evaluation")
    logger.info(f"Running {n_splits}-fold evaluation on {len(embeddings)} samples")

    # Setup stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []

    # Evaluate each fold
    for fold_idx, (train_idx, test_idx) in tqdm(
        enumerate(skf.split(embeddings, labels)),
        total=n_splits,
        desc="Running k-fold evaluation",
    ):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        metrics = evaluate_fold(
            X_train,
            X_test,
            y_train,
            y_test,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        fold_metrics.append(metrics)

        # Log metrics for this fold
        if fold_idx == 0:  # Only log detailed metrics for first fold
            logger.debug(f"Fold {fold_idx + 1} metrics: {metrics}")

    # Average metrics across folds
    avg_metrics = {}
    for metric in fold_metrics[0].keys():
        values = [fold[metric] for fold in fold_metrics]
        avg_metrics[metric] = np.mean(values)
        avg_metrics[f"{metric}_std"] = np.std(values)

        # Store all values for statistical testing
        avg_metrics[f"{metric}_values"] = values

        # Log average metrics
        logger.info(
            f"Average {metric}: {avg_metrics[metric]:.4f} ± {avg_metrics[f'{metric}_std']:.4f}"
        )

    return avg_metrics


def evaluate_task(
    task_name: str,
    task_config: Dict,
    output_dir: Path,
    n_splits: int = 5,
    n_estimators: int = 1000,
    random_state: int = 7,
    baseline_model: str = "jina_base",
) -> None:
    """Evaluate embeddings for a specific task.

    Args:
        task_name: Name of the task (tumor_type, survival, mgmt)
        task_config: Task configuration dictionary
        output_dir: Directory to save results
        n_splits: Number of folds for cross-validation
        n_estimators: Number of trees in random forest
        random_state: Random seed for reproducibility
        baseline_model: Name of the baseline model for comparison
    """
    logger = logging.getLogger("glioma.evaluation")
    logger.info(f"\nEvaluating {task_name} classification...")

    results = {}
    embedding_files = list(task_config["embeddings_dir"].glob("*.npy"))

    if not embedding_files:
        logger.warning(f"No embedding files found in {task_config['embeddings_dir']}")
        return

    logger.info(f"Found {len(embedding_files)} embedding files for {task_name} task")

    # Load dataframe
    df = task_config["df"]
    label_col = task_config["label_col"]

    # Filter out rows with null labels
    valid_mask = df[label_col].notna()
    valid_df = df[valid_mask]

    logger.info(f"Dataset has {len(valid_df)} valid samples (out of {len(df)} total)")

    # Get label distribution
    label_counts = valid_df[label_col].value_counts()
    logger.info(f"Label distribution: {label_counts.to_dict()}")

    # Evaluate each embedding file
    for emb_file in embedding_files:  # Remove sorted() to match notebook
        embeddings, model_name = load_embeddings(emb_file)
        if embeddings is None:
            continue

        if len(embeddings) != len(df):
            logger.warning(
                f"Embeddings length ({len(embeddings)}) doesn't match dataframe length ({len(df)}). "
                f"Skipping {emb_file}"
            )
            continue

        # Filter embeddings to match valid labels
        labels = valid_df[label_col].values
        embeddings = embeddings[
            valid_mask
        ]  # Renamed from valid_embeddings to match notebook

        logger.info(f"Evaluating {model_name}...")
        metrics = run_kfold_evaluation(
            embeddings,
            labels,
            n_splits=n_splits,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        results[model_name] = metrics

    if not results:
        logger.warning(f"No valid results for {task_name} task")
        return

    # Save results
    results_table = format_results_table(
        results, task=task_name, baseline_model=baseline_model
    )
    output_file = output_dir / f"{task_name}_results.md"
    output_file.write_text(results_table)

    # Save raw metrics
    metrics_file = output_dir / f"{task_name}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")
    logger.info("\nResults Table:")
    logger.info("\n" + results_table)


def evaluate_models(config: Dict) -> None:
    """Evaluate all models on all tasks.

    Args:
        config: Configuration dictionary
    """
    logger = logging.getLogger("glioma.evaluation")

    # Get paths from config
    data_config = config["data"]
    eval_config = config["evaluation"]
    output_config = config["output"]

    # Create results directory
    results_dir = Path(output_config["results_dir"])
    results_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Results will be saved to: {results_dir}")

    # Load label data
    logger.info("Loading data...")

    try:
        type_df = pd.read_csv(data_config["type_path"])
        logger.info(f"Loaded type data: {len(type_df)} samples")
    except Exception as e:
        logger.error(f"Failed to load type data: {e}")
        type_df = None

    try:
        mgmt_df = pd.read_csv(data_config["mgmt_path"])
        mgmt_df["mgmt_label"] = (
            mgmt_df["mgmt"].notna() & (mgmt_df["mgmt"] == "positive")
        ).astype(int)
        logger.info(f"Loaded MGMT data: {len(mgmt_df)} samples")
    except Exception as e:
        logger.error(f"Failed to load MGMT data: {e}")
        mgmt_df = None

    try:
        os_df = pd.read_csv(data_config["os_path"])
        logger.info(f"Loaded OS data: {len(os_df)} samples")
    except Exception as e:
        logger.error(f"Failed to load OS data: {e}")
        os_df = None

    # Evaluation configuration
    n_splits = eval_config.get("n_splits", 5)
    random_state = eval_config.get("random_state", 7)
    n_estimators = eval_config.get("n_estimators", 1000)
    baseline_model = eval_config.get("baseline_model", "jina_base")

    # Define tasks
    tasks = {}

    if type_df is not None:
        tasks["tumor_type"] = {
            "embeddings_dir": Path("embeddings/tumor_type"),  # Updated path
            "df": type_df,
            "label_col": "tumor_type",
            "id_col": "MRN",
        }

    if os_df is not None:
        tasks["survival"] = {
            "embeddings_dir": Path("embeddings/survival"),
            "df": os_df,
            "label_col": "one_year_survival",  # Using column name from notebook
            "id_col": "Patient_mrn",  # Updated to match notebook
        }

    if mgmt_df is not None:
        tasks["mgmt"] = {
            "embeddings_dir": Path("embeddings/mgmt"),
            "df": mgmt_df,
            "label_col": "mgmt_label",
            "id_col": "mrn",
        }

    # Evaluate each task
    for task_name, task_config in tasks.items():
        evaluate_task(
            task_name=task_name,
            task_config=task_config,
            output_dir=results_dir,
            n_splits=n_splits,
            n_estimators=n_estimators,
            random_state=random_state,
            baseline_model=baseline_model,
        )

    logger.info("Evaluation completed!")


def main() -> None:
    """Main entry point for evaluation module."""
    # Setup logger
    logger = setup_logger(
        name="glioma.evaluation", level=logging.INFO, log_file="evaluation.log"
    )

    # Disable verbose logging from external libraries
    disable_external_loggers()

    # Load configuration
    config = get_config()

    # Run evaluation
    evaluate_models(config)


if __name__ == "__main__":
    main()
