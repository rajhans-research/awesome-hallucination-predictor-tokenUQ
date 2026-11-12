"""
src/visualization.py
--------------------
Module 6: Publication-Grade Visualizations.

Generates:
1. `fig1_roc_curves.pdf` & `.png`: ROC curves with AUROC values in legend.
2. `fig2_pr_curves.pdf` & `.png`: Precision-Recall curves with AUC-PR values.
3. `fig3_pareto_frontier.pdf` & `.png`: (Wall-Clock Latency vs. AUROC) Pareto curves.
4. `fig4_high_certainty_breakdown.pdf` & `.png`: High-certainty failure analysis.
"""

import os
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def set_academic_plot_style():
    """Configures clean, publication-ready Matplotlib aesthetics."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 13,
        "lines.linewidth": 1.75,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })


def plot_roc_curves(
    y_true: np.ndarray,
    method_scores: Dict[str, np.ndarray],
    method_aurocs: Dict[str, float],
    output_dir: str,
    title: str = "Sentence-Level Hallucination Detection ROC Curves",
) -> None:
    set_academic_plot_style()
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=300)

    color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for idx, (m_name, scores) in enumerate(method_scores.items()):
        fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
        auroc = method_aurocs.get(m_name, 0.0)
        c = color_cycle[idx % len(color_cycle)]
        ls = "-" if "token" in m_name else "--"
        ax.plot(fpr, tpr, color=c, linestyle=ls, label=f"{m_name} (AUROC={auroc:.1f}%)")

    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="Random Chance (50.0%)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.92)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig1_roc_curves.pdf"), format="pdf")
    fig.savefig(os.path.join(output_dir, "fig1_roc_curves.png"), format="png", dpi=300)
    plt.close(fig)


def plot_pr_curves(
    y_true: np.ndarray,
    method_scores: Dict[str, np.ndarray],
    method_aucprs: Dict[str, float],
    output_dir: str,
    title: str = "Sentence-Level Precision-Recall Curves (Positive = Hallucination)",
) -> None:
    set_academic_plot_style()
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=300)

    base_rate = float(np.mean(y_true) * 100.0)
    color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for idx, (m_name, scores) in enumerate(method_scores.items()):
        p, r, _ = precision_recall_curve(y_true, scores, pos_label=1)
        auc_pr = method_aucprs.get(m_name, 0.0)
        c = color_cycle[idx % len(color_cycle)]
        ls = "-" if "token" in m_name else "--"
        ax.plot(r, p, color=c, linestyle=ls, label=f"{m_name} (AUC-PR={auc_pr:.1f}%)")

    ax.axhline(base_rate / 100.0, color="grey", linestyle=":", label=f"Base Rate ({base_rate:.1f}%)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left", framealpha=0.92)
    plt.tight_layout()

    fig.savefig(os.path.join(output_dir, "fig2_pr_curves.pdf"), format="pdf")
    fig.savefig(os.path.join(output_dir, "fig2_pr_curves.png"), format="png", dpi=300)
    plt.close(fig)


def plot_pareto_frontier(
    points: List[Dict[str, Any]],
    output_dir: str,
    title: str = "The Cost-Quality Pareto Frontier: Latency vs. AUROC",
) -> None:
    set_academic_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=300)

    sp_pts = [p for p in points if p["type"] == "Single-Pass (1x)"]
    mp_pts = [p for p in points if p["type"] == "Multi-Pass (Nx)"]

    # Single-pass points
    ax.scatter(
        [p["latency_ms"] for p in sp_pts],
        [p["auroc"] for p in sp_pts],
        color="#1f77b4",
        marker="o",
        s=90,
        zorder=5,
        label="Single-Pass Token UQ (1x gen)",
    )
    for p in sp_pts:
        ax.annotate(
            p["name"],
            (p["latency_ms"], p["auroc"]),
            textcoords="offset points",
            xytext=(6, -2),
            fontsize=8,
            fontweight="bold" if "keyword" in p["name"] or "hybrid" in p["name"] else "normal",
        )

    # Multi-pass points
    ax.scatter(
        [p["latency_ms"] for p in mp_pts],
        [p["auroc"] for p in mp_pts],
        color="#d62728",
        marker="^",
        s=90,
        zorder=5,
        label="Multi-Pass Sampling (N=2..20 gen)",
    )
    for p in mp_pts:
        ax.annotate(
            p["name"],
            (p["latency_ms"], p["auroc"]),
            textcoords="offset points",
            xytext=(-6, 6),
            fontsize=7.5,
        )

    # Draw Pareto envelope
    sorted_pts = sorted(points, key=lambda x: x["latency_ms"])
    pareto_pts = []
    max_auc = -1.0
    for p in sorted_pts:
        if p["auroc"] > max_auc:
            pareto_pts.append(p)
            max_auc = p["auroc"]

    ax.plot(
        [p["latency_ms"] for p in pareto_pts],
        [p["auroc"] for p in pareto_pts],
        color="#2ca02c",
        linestyle="-.",
        linewidth=2.0,
        label="Pareto Optimal Frontier",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Wall-Clock Scoring Latency per Passage (ms, log scale)")
    ax.set_ylabel("Sentence AUROC (%)")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.92)
    plt.tight_layout()

    fig.savefig(os.path.join(output_dir, "fig3_pareto_frontier.pdf"), format="pdf")
    fig.savefig(os.path.join(output_dir, "fig3_pareto_frontier.png"), format="png", dpi=300)
    plt.close(fig)


def plot_high_certainty_breakdown(
    bin_labels: List[str],
    token_failure_recalls: Dict[str, float],
    output_dir: str,
    title: str = "High-Certainty Hallucination Error Recovery Analysis",
) -> None:
    set_academic_plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

    methods = list(token_failure_recalls.keys())
    recalls = list(token_failure_recalls.values())
    colors = ["#1f77b4" if "token" in m else "#d62728" for m in methods]

    bars = ax.barh(methods, recalls, color=colors, height=0.55, edgecolor="black", linewidth=0.8)
    ax.set_xlim([0, 105])
    ax.set_xlabel("Detection Recall on High-Certainty Hallucinations (%)")
    ax.set_title(title)

    for bar, val in zip(bars, recalls):
        ax.text(
            val + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_high_certainty_breakdown.pdf"), format="pdf")
    fig.savefig(os.path.join(output_dir, "fig4_high_certainty_breakdown.png"), format="png", dpi=300)
    plt.close(fig)
