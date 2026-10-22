"""Patient-level inference for model comparison.

Fold-level Cohen's d is not a valid significance surrogate: 5-fold training
sets overlap by ~60%, so fold means are not independent and the pooled fold
s.d. understates variance. The functions here operate on pooled out-of-fold
predictions at the patient level, where each patient contributes exactly one
observation and paired tests are well defined.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def no_information_rate(y_true: Sequence) -> float:
    """Accuracy achieved by always predicting the majority class."""
    y = np.asarray(y_true)
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


def aggregate_to_patient(
    patient_ids: Sequence,
    y_true: Sequence,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse report-level scores to one prediction per patient.

    Labels are patient-level while scoring is report-level, so patients with
    more reports are otherwise weighted more heavily and the effective sample
    size is overstated. Probabilities are averaged across a patient's reports.

    Returns:
        (unique patient ids, patient label, patient probability matrix)
    """
    pid = np.asarray(patient_ids)
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    if p.ndim == 1:
        p = p[:, None]

    uniq = np.unique(pid)
    pat_y = np.empty(len(uniq), dtype=y.dtype)
    pat_p = np.empty((len(uniq), p.shape[1]), dtype=float)
    for i, u in enumerate(uniq):
        m = pid == u
        labels = np.unique(y[m])
        if len(labels) > 1:
            raise ValueError(f"patient {u} has conflicting labels {labels}")
        pat_y[i] = labels[0]
        pat_p[i] = np.nanmean(p[m], axis=0)
    return uniq, pat_y, pat_p


def _score(metric: str, y: np.ndarray, prob: np.ndarray) -> float:
    binary = prob.shape[1] == 2
    pred = np.argmax(prob, axis=1)
    if metric == "accuracy":
        return accuracy_score(y, pred)
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y, pred)
    if metric == "f1_macro":
        return f1_score(y, pred, average="macro", zero_division=0)
    if metric == "auroc":
        return roc_auc_score(y, prob[:, 1]) if binary else roc_auc_score(
            y, prob, multi_class="ovr"
        )
    if metric == "pr_auc":
        if binary:
            return average_precision_score(y, prob[:, 1])
        onehot = np.eye(prob.shape[1])[y]
        return average_precision_score(onehot, prob, average="macro")
    raise ValueError(f"unknown metric {metric}")


def bootstrap_ci(
    y_true: Sequence,
    y_prob: np.ndarray,
    metric: str = "auroc",
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 7,
) -> Dict[str, float]:
    """Patient-level bootstrap percentile CI for a headline metric."""
    rng = np.random.default_rng(random_state)
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    n = len(y)
    point = _score(metric, y, p)

    boots: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            boots.append(_score(metric, y[idx], p[idx]))
        except ValueError:
            continue

    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "metric": metric,
        "estimate": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_patients": int(n),
        "n_boot_effective": len(boots),
    }


def mcnemar_test(
    y_true: Sequence, pred_a: Sequence, pred_b: Sequence
) -> Dict[str, float]:
    """Exact McNemar test on paired correctness for the same patients."""
    y = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y
    b_ok = np.asarray(pred_b) == y

    n01 = int(np.sum(~a_ok & b_ok))  # only B correct
    n10 = int(np.sum(a_ok & ~b_ok))  # only A correct
    n_disc = n01 + n10

    if n_disc == 0:
        p = 1.0
    else:
        p = float(min(1.0, 2 * stats.binom.cdf(min(n01, n10), n_disc, 0.5)))

    return {
        "acc_a": float(a_ok.mean()),
        "acc_b": float(b_ok.mean()),
        "acc_diff_b_minus_a": float(b_ok.mean() - a_ok.mean()),
        "n_only_a_correct": n10,
        "n_only_b_correct": n01,
        "p_value": p,
    }


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_test(
    y_true: Sequence, score_a: Sequence, score_b: Sequence
) -> Dict[str, float]:
    """DeLong test for two correlated binary-task AUROCs on identical cases.

    Returns each AUROC, their difference, a 95% CI on the difference and a
    two-sided p-value.
    """
    y = np.asarray(y_true)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("delong_test requires a binary label")
    pos = y == classes[1]

    scores = np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)])
    x, yv = scores[:, pos], scores[:, ~pos]
    m, n = x.shape[1], yv.shape[1]

    tx = np.vstack([_midrank(r) for r in x])
    ty = np.vstack([_midrank(r) for r in yv])
    tz = np.vstack([_midrank(np.concatenate([xi, yi])) for xi, yi in zip(x, yv)])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    cov = np.atleast_2d(cov)

    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[1] - aucs[0])
    se = float(np.sqrt(max(var_diff, 0.0)))

    if se == 0:
        p, lo, hi = 1.0, diff, diff
    else:
        z = diff / se
        p = float(2 * stats.norm.sf(abs(z)))
        lo, hi = diff - 1.96 * se, diff + 1.96 * se

    return {
        "auroc_a": float(aucs[0]),
        "auroc_b": float(aucs[1]),
        "auroc_diff_b_minus_a": diff,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": p,
    }


def compare_models(
    y_true: Sequence,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    patient_ids: Optional[Sequence] = None,
) -> Dict[str, Dict]:
    """Full paired comparison of two models on pooled out-of-fold predictions.

    Both models must have been scored on the same cases in the same order.
    """
    y = np.asarray(y_true)
    pa = np.asarray(prob_a, dtype=float)
    pb = np.asarray(prob_b, dtype=float)

    if patient_ids is not None:
        _, y, pa = aggregate_to_patient(patient_ids, y, pa)
        _, _, pb = aggregate_to_patient(patient_ids, np.asarray(y_true), pb)

    out: Dict[str, Dict] = {
        "no_information_rate": {"value": no_information_rate(y)},
        "mcnemar": mcnemar_test(y, np.argmax(pa, 1), np.argmax(pb, 1)),
    }
    for metric in ("accuracy", "balanced_accuracy", "f1_macro", "auroc", "pr_auc"):
        out[f"{metric}_a"] = bootstrap_ci(y, pa, metric)
        out[f"{metric}_b"] = bootstrap_ci(y, pb, metric)

    if pa.shape[1] == 2:
        out["delong"] = delong_test(y, pa[:, 1], pb[:, 1])

    return out


def cluster_bootstrap_ci(
    y_true: Sequence,
    prob_a: np.ndarray,
    groups: Sequence,
    metric: str = "auroc",
    prob_b: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 7,
) -> Dict[str, float]:
    """Report-level metric with a patient-cluster bootstrap CI.

    Resamples patients (all of a patient's reports together), so CIs account for
    several reports per patient. If `prob_b` is given, returns the paired
    difference metric(a) - metric(b) on identical resamples.
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y_true)
    pa = np.asarray(prob_a, dtype=float)
    pb = None if prob_b is None else np.asarray(prob_b, dtype=float)
    g = np.asarray(groups)
    patients = np.unique(g)
    rows = {p: np.flatnonzero(g == p) for p in patients}

    def stat(idx):
        a = _score(metric, y[idx], pa[idx])
        return a if pb is None else a - _score(metric, y[idx], pb[idx])

    point = stat(np.arange(len(y)))
    boots: List[float] = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[p] for p in rng.choice(patients, len(patients))])
        if len(np.unique(y[idx])) < len(np.unique(y)):
            continue
        try:
            boots.append(stat(idx))
        except ValueError:
            continue
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"metric": metric, "estimate": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "n_patients": int(len(patients)), "n_boot_effective": len(boots)}
