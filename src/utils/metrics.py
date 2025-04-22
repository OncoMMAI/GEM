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
    
    # Basic metrics
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
    
    # Pooled standard deviation (using 'se' as in the notebook)
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
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
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(baseline_values, comparison_values)
    
    # Effect size
    effect_size = calculate_cohens_d(baseline_values, comparison_values)
    
    # Effect size interpretation
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
    # Define metric order and display names
    metric_order = ["accuracy", "precision", "recall", "f1_macro", "aucroc"]
    metric_display = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_macro": "F1-macro",
        "aucroc": "AUCROC",
    }
    
    # Filter metrics that exist in results
    metrics = [m for m in metric_order if m in next(iter(results.values()))]
    headers = ["Model"] + [f"{metric_display[m]} (%)" for m in metrics]
    
    table = ["|" + "|".join(headers) + "|"]
    table.append("|" + "|".join(["---"] * len(headers)) + "|")
    
    # Use first model as baseline if baseline_model not found
    if baseline_model not in results:
        baseline_model = list(results.keys())[0]
        logger.info(
            f"Baseline model '{baseline_model}' not found. Using '{baseline_model}' as baseline."
        )
    
    baseline_results = results[baseline_model]
    
    # Sort models to ensure baseline is first
    sorted_models = sorted(results.keys(), key=lambda x: (x != baseline_model, x))
    
    for model in sorted_models:
        model_results = results[model]
        row = [model]
        for metric in metrics:
            value = model_results[metric] * 100  # Convert to percentage
            std = model_results.get(f"{metric}_std", 0) * 100
            
            # Calculate significance if not baseline
            if model != baseline_model:
                _, sig = calculate_significance(
                    baseline_results[f"{metric}_values"],
                    model_results[f"{metric}_values"],
                )
            else:
                sig = ""
            
            row.append(f"{value:.1f} ± {std:.1f}{sig}")
        
        table.append("|" + "|".join(row) + "|")
    
    # Add legend
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
    
    return "\n".join(table)