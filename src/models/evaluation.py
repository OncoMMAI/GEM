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
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
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
        # Reported alongside accuracy because the MGMT cohort is imbalanced
        # enough that accuracy can sit below the majority-class rate.
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "no_information_rate": float(
            np.bincount(y_test.astype(int)).max() / len(y_test)
        ),
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro,
    }

    # clf.classes_ gives the column order of predict_proba; relying on it (and
    # passing labels= explicitly) keeps the score columns aligned with y_test
    # even when a fold is missing a class.
    classes = clf.classes_
    if len(classes) == 2:
        pos_col = int(np.where(classes == classes.max())[0][0])
        metrics["aucroc"] = roc_auc_score(y_test, y_pred_proba[:, pos_col])
        metrics["pr_auc"] = average_precision_score(
            (y_test == classes.max()).astype(int), y_pred_proba[:, pos_col]
        )
    else:
        metrics["aucroc"] = roc_auc_score(
            y_test, y_pred_proba, multi_class="ovr", labels=classes
        )
        onehot = np.stack([(y_test == c).astype(int) for c in classes], axis=1)
        metrics["pr_auc"] = average_precision_score(
            onehot, y_pred_proba, average="macro"
        )

    return metrics, y_pred, y_pred_proba, classes


def run_kfold_evaluation(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    n_estimators: int = 1000,
    random_state: int = 7,
    groups: Optional[np.ndarray] = None,
) -> Dict:
    """Run k-fold cross validation.

    Args:
        embeddings: Input embeddings
        labels: Target labels
        n_splits: Number of folds
        n_estimators: Number of trees in random forest
        random_state: Random seed for reproducibility
        groups: Patient identifier per sample. When supplied, splits are made
            with StratifiedGroupKFold so that all reports from one patient stay
            on the same side of the split. Without it, patients with multiple
            reports appear in both train and test and the scores are optimistic.

    Returns:
        Dictionary containing averaged metrics, per-fold values, and pooled
        out-of-fold predictions/probabilities for patient-level inference.
    """
    logger = logging.getLogger("glioma.evaluation")
    logger.info(f"Running {n_splits}-fold evaluation on {len(embeddings)} samples")

    if groups is None:
        logger.warning(
            "No patient groups supplied: folds are report-level, so reports from "
            "the same patient can appear in both train and test. Scores from this "
            "run are not leakage-controlled."
        )
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        split_iter = splitter.split(embeddings, labels)
    else:
        groups = np.asarray(groups)
        logger.info(
            f"Patient-grouped CV over {len(np.unique(groups))} patients "
            f"({len(embeddings)} reports)"
        )
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        split_iter = splitter.split(embeddings, labels, groups=groups)

    fold_metrics = []
    all_y_true: list = []
    all_y_pred: list = []
    oof_proba = None
    oof_groups: list = []

    for fold_idx, (train_idx, test_idx) in tqdm(
        enumerate(split_iter),
        total=n_splits,
        desc="Running k-fold evaluation",
    ):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        if groups is not None:
            overlap = np.intersect1d(groups[train_idx], groups[test_idx])
            if overlap.size:
                raise AssertionError(
                    f"fold {fold_idx + 1}: {overlap.size} patients in both splits"
                )

        metrics, y_pred, y_proba, classes = evaluate_fold(
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

        # Pool out-of-fold probabilities in original sample order so that
        # McNemar/DeLong can be run on identical cases across models.
        all_classes = np.unique(labels)
        if oof_proba is None:
            oof_proba = np.full((len(embeddings), len(all_classes)), np.nan)
        col_of = {c: i for i, c in enumerate(all_classes)}
        for j, c in enumerate(classes):
            oof_proba[test_idx, col_of[c]] = y_proba[:, j]

        if groups is not None:
            oof_groups.append(groups[test_idx])

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
    avg_metrics["oof_proba"] = oof_proba
    avg_metrics["oof_labels"] = labels
    avg_metrics["oof_groups"] = groups
    avg_metrics["patient_grouped"] = groups is not None

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

        # Patient identifier, so folds can be made patient-disjoint. The column
        # is named inconsistently across the three eval files.
        patient_col = next(
            (c for c in ("mrn", "MRN", "patient_id") if c in valid_df.columns), None
        )
        # Patient-grouped CV is opt-in. The published results used report-level
        # stratified folds; set `patient_grouped_cv: true` under `evaluation` in
        # the config for the leakage-controlled variant.
        use_groups = task_config.get("patient_grouped_cv", False)
        groups = valid_df[patient_col].values if (patient_col and use_groups) else None
        if use_groups and patient_col is None:
            logger.error(
                f"{task_name}: no patient identifier column found in "
                f"{list(valid_df.columns)}; cannot run patient-grouped CV"
            )

        if exclude_labels:
            keep = ~np.isin(labels, exclude_labels)
            labels = labels[keep]
            filtered_embeddings = filtered_embeddings[keep]
            if groups is not None:
                groups = groups[keep]

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
            groups=groups,
        )
        results[model_name] = metrics

        # Persist pooled out-of-fold predictions so McNemar/DeLong can be run
        # across models on identical cases (see src/utils/stats.py). These are
        # arrays, so they must not reach the JSON summary.
        oof = {
            "proba": metrics.pop("oof_proba", None),
            "labels": metrics.pop("oof_labels", None),
            "groups": metrics.pop("oof_groups", None),
        }
        metrics["patient_grouped"] = bool(metrics.pop("patient_grouped", False))
        if oof["proba"] is not None:
            np.savez(
                output_dir / f"{task_name}_{model_name}_oof.npz",
                **{k: v for k, v in oof.items() if v is not None},
            )

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

    # Report-level folds by default (matches the published results);
    # set evaluation.patient_grouped_cv: true for patient-disjoint folds.
    patient_grouped_cv = eval_config.get("patient_grouped_cv", False)
    for _cfg in tasks.values():
        _cfg["patient_grouped_cv"] = patient_grouped_cv

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
