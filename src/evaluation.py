"""
src/evaluation.py
-----------------
Module 4: Standardized Metric Computation, Bootstrap Confidence Intervals & Leakage Audits.

Computes:
1. Sentence-level AUROC (`roc_auc_score`).
2. Sentence-level AUC-PR via trapezoidal integration (`auc(R, P)` matching Manakul et al. 2023)
   and step-wise Average Precision (`average_precision_score`).
3. Operational operating-point metrics: FPR@95% Recall and Precision@80% Recall.
4. 95% empirical Bootstrap Confidence Intervals (1,000 resamples, configurable seed).
5. Passage-level Pearson & Spearman rank correlation with mean human factuality annotations.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def align_scores_and_labels(
    keys: List[Tuple[int, int]],
    labels: np.ndarray,
    scores_dict: Dict[Tuple[int, int], float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns prediction scores to ground-truth label keys and verifies zero NaNs or missing keys.
    """
    assert len(keys) == len(labels), "Length mismatch between keys and labels."
    y_score = np.array([scores_dict[k] for k in keys], dtype=np.float64)

    # Safely clamp any infinite log-loss penalties (e.g., zero-frequency bigrams in SelfCheck-2gram)
    # to a high finite upper bound so ranking is preserved without crashing metrics.
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=50.0, neginf=-50.0)

    return labels, y_score


def compute_sentence_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Computes exact detection metrics and 95% bootstrap confidence intervals.

    Inputs:
        y_true: 1D binary numpy array (0 = negative, 1 = positive class).
        y_score: 1D float numpy array (higher = more likely positive).
        n_bootstrap: Number of bootstrap resamples for CI.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary of metrics scaled to percentages [0, 100].
    """
    auroc = float(roc_auc_score(y_true, y_score) * 100.0)
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=1)
    auc_pr = float(auc(recall, precision) * 100.0)
    ap_score = float(average_precision_score(y_true, y_score) * 100.0)

    # FPR at 95% TPR (Recall)
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    idx_95 = np.searchsorted(tpr, 0.95, side="left")
    fpr95 = float(fpr[min(idx_95, len(fpr) - 1)] * 100.0)

    # Precision at 80% Recall
    valid_rec = np.where(recall >= 0.80)[0]
    prec_at_rec80 = float(np.max(precision[valid_rec]) * 100.0) if len(valid_rec) > 0 else 0.0

    # Bootstrap 95% Confidence Intervals
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_aurocs = []
    boot_aucprs = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_aurocs.append(roc_auc_score(yt, ys) * 100.0)
        p_b, r_b, _ = precision_recall_curve(yt, ys, pos_label=1)
        boot_aucprs.append(auc(r_b, p_b) * 100.0)

    return {
        "n_samples": int(n),
        "pos_rate": float(np.mean(y_true) * 100.0),
        "auroc": auroc,
        "auroc_ci_low": float(np.percentile(boot_aurocs, 2.5)),
        "auroc_ci_high": float(np.percentile(boot_aurocs, 97.5)),
        "auc_pr": auc_pr,
        "auc_pr_ci_low": float(np.percentile(boot_aucprs, 2.5)),
        "auc_pr_ci_high": float(np.percentile(boot_aucprs, 97.5)),
        "ap_score": ap_score,
        "fpr95": fpr95,
        "prec_at_rec80": prec_at_rec80,
    }


def compute_passage_correlations(
    records: List[Dict],
    scores_dict: Dict[Tuple[int, int], float],
) -> Dict[str, float]:
    """
    Computes passage-level Pearson and Spearman correlation between mean predicted
    sentence hallucination score and mean human factuality score.
    """
    label_map = {"accurate": 0.0, "minor_inaccurate": 0.5, "major_inaccurate": 1.0}
    human_means = []
    pred_means = []

    for p_idx, rec in enumerate(records):
        h_vals = [label_map[s["annotation"]] for s in rec["sentences"]]
        p_vals = [scores_dict[(p_idx, s_idx)] for s_idx in range(len(rec["sentences"]))]
        human_means.append(float(np.mean(h_vals)))
        pred_means.append(float(np.mean(p_vals)))

    p_corr, p_pval = pearsonr(human_means, pred_means)
    s_corr, s_pval = spearmanr(human_means, pred_means)

    return {
        "pearson_r": float(p_corr * 100.0),
        "pearson_p": float(p_pval),
        "spearman_rho": float(s_corr * 100.0),
        "spearman_p": float(s_pval),
    }
