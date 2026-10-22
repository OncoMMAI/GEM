"""Evaluation metrics and statistical testing utilities."""

import numpy as np
import logging
from typing import Dict, List, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from scipy import stats

# Setup logger
logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
) -> Dict[str, float]:
    """Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional, for AUC)
        
    Returns:
        Dictionary of metrics including accuracy, precision, recall, F1, and AUC
    """
    metrics = {}
    
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    # Precision, recall, F1
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1_macro"] = f1_macro
    
    # AUC calculation if probability estimates are provided
    unique_classes = np.unique(y_true)
    if y_prob is not None:
        if len(unique_classes) == 2:
            # Binary classification
            metrics["aucroc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            # Multi-class classification
            metrics["aucroc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
    
    return metrics


def calculate_cohens_d(baseline: List[float], comparison: List[float]) -> float:
    """Calculate Cohen's d effect size.
    
    Args:
        baseline: Baseline model performance values
        comparison: Comparison model performance values
        
    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(baseline), len(comparison)
    var1, var2 = np.var(baseline, ddof=1), np.var(comparison, ddof=1)
    
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = (np.mean(comparison) - np.mean(baseline)) / pooled_se
    return d


def calculate_significance(
    baseline_values: List[float], comparison_values: List[float]
) -> Tuple[float, str]:
    """Calculate statistical significance using paired t-test.
    
    Args:
        baseline_values: Baseline model performance values
        comparison_values: Comparison model performance values
        
    Returns:
        Tuple containing p-value and combined significance+effect marker
    """
    t_stat, p_value = stats.ttest_rel(baseline_values, comparison_values)
    effect_size = calculate_cohens_d(baseline_values, comparison_values)

    if abs(effect_size) > 0.8:
        effect_marker = "‡"  # Large effect
    elif abs(effect_size) > 0.5:
        effect_marker = "†"  # Medium effect
    elif abs(effect_size) > 0.2:
        effect_marker = "§"  # Small effect
    else:
        effect_marker = ""
    
    # Statistical significance markers (matching notebook)
    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = ""
    
    return p_value, significance + effect_marker


def format_results_table(
    results: Dict[str, Dict], 
    task: str,
    baseline_model: str = "jina_base_text_match",
) -> str:
    """Format results as a markdown table with significance testing.
    
    Args:
        results: Dictionary of model results
        task: Task name
        baseline_model: Name of the baseline model for comparison (default: jina_base_text_match)
        
    Returns:
        Markdown formatted table as a string
    """
    metric_order = [
        "accuracy",
        "balanced_accuracy",
        "no_information_rate",
        "precision",
        "recall",
        "f1_macro",
        "aucroc",
        "pr_auc",
    ]
    metric_display = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced acc.",
        "no_information_rate": "NIR",
        "precision": "Precision",
        "recall": "Recall",
        "f1_macro": "F1-macro",
        "aucroc": "AUCROC",
        "pr_auc": "PR-AUC",
    }
    # The no-information rate is a property of the cohort, not the model, so it
    # is displayed for reference but never significance-tested.
    no_test = {"no_information_rate"}
    
    metrics = [m for m in metric_order if m in next(iter(results.values()))]
    headers = ["Model"] + [f"{metric_display[m]} (%)" for m in metrics]
    
    table = ["|" + "|".join(headers) + "|"]
    table.append("|" + "|".join(["---"] * len(headers)) + "|")
    
    if baseline_model not in results:
        baseline_model = list(results.keys())[0]
        logger.info(
            f"Baseline model '{baseline_model}' not found. Using '{baseline_model}' as baseline."
        )
    
    baseline_results = results[baseline_model]
    
    sorted_models = sorted(results.keys(), key=lambda x: (x != baseline_model, x))
    
    for model in sorted_models:
        model_results = results[model]
        row = [model]
        for metric in metrics:
            value = model_results[metric] * 100  # Convert to percentage
            std = model_results.get(f"{metric}_std", 0) * 100
            
            if model != baseline_model and metric not in no_test:
                _, sig = calculate_significance(
                    baseline_results[f"{metric}_values"],
                    model_results[f"{metric}_values"],
                )
            else:
                sig = ""
            
            row.append(f"{value:.1f} ± {std:.1f}{sig}")
        
        table.append("|" + "|".join(row) + "|")
    
    table.append("|" + "|".join([""] * len(headers)) + "|")
    table.append("|" + "**Legend:**" + "|".join([""] * (len(headers) - 1)) + "|")
    table.append(
        "|" + "* p < 0.05, ** p < 0.01" + "|".join([""] * (len(headers) - 1)) + "|"
    )
    table.append(
        "|"
        + "§ small effect, † medium effect, ‡ large effect"
        + "|".join([""] * (len(headers) - 1))
        + "|"
    )
    table.append(
        "|"
        + "Fold-level markers are DESCRIPTIVE ONLY: CV folds share ~60% of their "
        + "training data, so fold means are not independent. For inference, use "
        + "McNemar / DeLong on patient-level out-of-fold predictions "
        + "(src/utils/stats.py)."
        + "|".join([""] * (len(headers) - 1))
        + "|"
    )

    return "\n".join(table)