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
    classification_report,
    confusion_matrix,
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
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

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

    unique_classes = np.unique(y_test)
    if len(unique_classes) == 2:
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    metrics["aucroc"] = auc

    return metrics, y_pred


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

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []
    all_y_true: list = []
    all_y_pred: list = []

    for fold_idx, (train_idx, test_idx) in tqdm(
        enumerate(skf.split(embeddings, labels)),
        total=n_splits,
        desc="Running k-fold evaluation",
    ):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        metrics, y_pred = evaluate_fold(
            X_train,
            X_test,
            y_train,
            y_test,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        fold_metrics.append(metrics)
        all_y_true.append(y_test)
        all_y_pred.append(y_pred)

        # Log metrics for this fold
        if fold_idx == 0:  # Only log detailed metrics for first fold
            logger.debug(f"Fold {fold_idx + 1} metrics: {metrics}")

    avg_metrics = {}
    for metric in fold_metrics[0].keys():
        values = [fold[metric] for fold in fold_metrics]
        avg_metrics[metric] = np.mean(values)
        avg_metrics[f"{metric}_std"] = np.std(values)

        avg_metrics[f"{metric}_values"] = values
        logger.info(
            f"Average {metric}: {avg_metrics[metric]:.4f} ± {avg_metrics[f'{metric}_std']:.4f}"
        )

    avg_metrics["overall_y_true"] = np.concatenate(all_y_true)
    avg_metrics["overall_y_pred"] = np.concatenate(all_y_pred)

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
        task_config: Task configuration dictionary containing:
            - embeddings_dir: Path to directory with .npy files
            - df: DataFrame with labels
            - label_col: Column name for labels
        output_dir: Directory to save results
        n_splits: Number of folds for cross-validation
        n_estimators: Number of trees in random forest
        random_state: Random seed for reproducibility
        baseline_model: Name of the baseline model for comparison
    """
    logger = logging.getLogger("glioma.evaluation")
    logger.info(f"\nEvaluating {task_name} classification...")

    results = {}
    embedding_files = sorted(task_config["embeddings_dir"].glob("*.npy"))

    if not embedding_files:
        logger.warning(f"No embedding files found in {task_config['embeddings_dir']}")
        return

    # Optional allowlist: evaluate only the specified models
    include_models = task_config.get("include_models")
    if include_models:
        include_set = {m if m.endswith(".npy") else f"{m}.npy" for m in include_models}
        embedding_files = [f for f in embedding_files if f.name in include_set]

    logger.info(f"Found {len(embedding_files)} embedding files for {task_name} task")

    df = task_config["df"]
    label_col = task_config["label_col"]

    valid_mask = df[label_col].notna()
    valid_df = df[valid_mask]
    logger.info(f"Dataset has {len(valid_df)} valid samples (out of {len(df)} total)")

    label_counts = valid_df[label_col].value_counts()
    logger.info(f"Label distribution: {label_counts.to_dict()}")

    exclude_labels = task_config.get("exclude_labels")

    for emb_file in embedding_files:
        embeddings, model_name = load_embeddings(emb_file)
        if embeddings is None:
            continue

        if len(embeddings) != len(df):
            logger.warning(
                f"Embeddings length ({len(embeddings)}) doesn't match dataframe "
                f"length ({len(df)}), skipping {model_name}"
            )
            continue

        labels = valid_df[label_col].values
        filtered_embeddings = embeddings[valid_mask]

        if exclude_labels:
            keep = ~np.isin(labels, exclude_labels)
            labels = labels[keep]
            filtered_embeddings = filtered_embeddings[keep]

        logger.info(
            f"Evaluating {model_name} ({filtered_embeddings.shape[0]} samples, "
            f"{filtered_embeddings.shape[1]}d)..."
        )
        metrics = run_kfold_evaluation(
            filtered_embeddings,
            labels,
            n_splits=n_splits,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        results[model_name] = metrics

        try:
            y_true = metrics.pop("overall_y_true", None)
            y_pred = metrics.pop("overall_y_pred", None)

            if y_true is not None and y_pred is not None:
                unique_labels = np.unique(y_true)
                target_names = [str(lab) for lab in sorted(unique_labels)]

                report_text = classification_report(
                    y_true, y_pred, target_names=target_names, digits=3
                )
                logger.info(f"\nClassification report for {model_name}:\n{report_text}")

                cm = confusion_matrix(y_true, y_pred, labels=sorted(unique_labels))
                cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
                logger.info(f"\nConfusion matrix for {model_name}:\n{cm_df}\n")

                report_file = (
                    output_dir / f"{task_name}_{model_name}_classification_report.txt"
                )
                report_file.write_text(report_text)
                cm_file = output_dir / f"{task_name}_{model_name}_confusion_matrix.csv"
                cm_df.to_csv(cm_file)
        except Exception as e:
            logger.warning(f"Failed detailed reporting for {model_name}: {e}")

    if not results:
        logger.warning(f"No valid results for {task_name} task")
        return

    results_table = format_results_table(
        results, task=task_name, baseline_model=baseline_model
    )
    output_file = output_dir / f"{task_name}_results.md"
    output_file.write_text(results_table)

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
    embedding_base_dir = None
    if "embedding" in config and isinstance(config["embedding"], dict):
        # Allow experiments to override where embeddings are read from
        embedding_base_dir = config["embedding"].get("directory")

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
    include_models = eval_config.get("models")

    # Define tasks
    tasks = {}

    if type_df is not None:
        tumor_emb_dir = (
            Path(embedding_base_dir) / "tumor_type"
            if embedding_base_dir
            else Path("embeddings/tumor_type")
        )
        task_cfg = {
            "embeddings_dir": tumor_emb_dir,
            "df": type_df,
            "label_col": "class_label",
        }
        if include_models:
            task_cfg["include_models"] = include_models
        type_exclude = data_config.get("type_exclude_labels")
        if type_exclude:
            task_cfg["exclude_labels"] = type_exclude
        tasks["tumor_type"] = task_cfg

    if os_df is not None:
        survival_emb_dir = (
            Path(embedding_base_dir) / "os"
            if embedding_base_dir
            else Path("embeddings/os")
        )
        task_cfg = {
            "embeddings_dir": survival_emb_dir,
            "df": os_df,
            "label_col": "one_year_survival",
        }
        if include_models:
            task_cfg["include_models"] = include_models
        tasks["survival"] = task_cfg

    if mgmt_df is not None:
        mgmt_emb_dir = (
            Path(embedding_base_dir) / "mgmt"
            if embedding_base_dir
            else Path("embeddings/mgmt")
        )
        task_cfg = {
            "embeddings_dir": mgmt_emb_dir,
            "df": mgmt_df,
            "label_col": "mgmt_label",
        }
        if include_models:
            task_cfg["include_models"] = include_models
        tasks["mgmt"] = task_cfg

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
