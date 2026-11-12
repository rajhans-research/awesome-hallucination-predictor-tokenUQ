"""
scripts/run_experiment.py
-------------------------
Gate G3 Execution Script: Full Experimental Pipeline & Deeper Analyses.

Executes:
1. All single-pass baselines & proposed token methods:
   `token_mean`, `token_min`, `token_entropy5`, `token_ppl5`, `token_max_ppl5`,
   `token_keyword` (Ours), `token_hybrid_a50` (Ours), `token_hybrid_a70` (Ours).
2. Multi-pass consistency budget sweep across N in {2, 3, 5, 10, 20}:
   `selfcheck_1gram`, `selfcheck_2gram`, `selfcheck_tfidf`.
3. Standardized metrics calculation with 1,000 bootstrap resamples across
   protocols (`nonfact`, `nonfact_star`, `major_only`).
4. Deeper Analysis 1: Latency-AUROC Pareto Frontier mapping.
5. Deeper Analysis 2: High-Certainty Hallucination Error Analysis (Simhi et al.).
6. Generates publication-ready figures (PNG & PDF) and export tables.
"""

import os
import sys
import time
import logging
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    load_processed_dataset,
    extract_labels_for_protocol,
    audit_dataset_integrity,
)
from src.baseline_methods import run_single_pass_baselines
from src.proposed_method import (
    run_proposed_single_pass_methods,
    run_budget_sweep_multi_pass,
)
from src.evaluation import (
    align_scores_and_labels,
    compute_sentence_metrics,
    compute_passage_correlations,
)
from src.visualization import (
    plot_roc_curves,
    plot_pr_curves,
    plot_pareto_frontier,
    plot_high_certainty_breakdown,
)


def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("gate_g3_experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    seed = 42
    t_global_start = time.perf_counter()

    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "wikibio_processed.json")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    log_path = os.path.join(results_dir, "logs", "gate_g3_experiment_run.log")
    pred_path = os.path.join(results_dir, "predictions", "experiment_predictions.csv")
    metrics_path = os.path.join(results_dir, "metrics", "experiment_metrics.csv")
    results_csv_path = os.path.join(results_dir, "results.csv")
    fig_dir = os.path.join(results_dir, "figures")
    paper_fig_dir = os.path.join(PROJECT_ROOT, "paper", "figures")

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(paper_fig_dir, exist_ok=True)

    logger = setup_logger(log_path)
    logger.info("=== GATE G3: FULL EXPERIMENTAL RUN STARTED ===")
    logger.info(f"Random Seed: {seed}")

    records = load_processed_dataset(data_path)
    audit = audit_dataset_integrity(records)
    logger.info(
        f"Loaded dataset: {audit['num_passages']} passages, {audit['total_sentences']} sentences."
    )

    all_scores = {}
    all_timings = {}

    if os.path.exists(pred_path) and os.path.exists(metrics_path) and os.path.exists(results_csv_path):
        logger.info(f"Existing predictions and metrics found. Loading cached results...")
        df_preds = pd.read_csv(pred_path)
        pred_rows = df_preds.to_dict(orient="records")
        for col in df_preds.columns:
            if col not in ["passage_id", "wiki_bio_test_idx", "sentence_idx", "annotation", "num_tokens"]:
                all_scores[col] = {(int(r["passage_id"]), int(r["sentence_idx"])): float(r[col]) for _, r in df_preds.iterrows()}
        df_metrics = pd.read_csv(metrics_path)
    else:
        # 1. Single-Pass Baselines
        logger.info("[1/5] Evaluating Single-Pass Baselines...")
        sp_base_scores, sp_base_times = run_single_pass_baselines(records)
        all_scores.update(sp_base_scores)
        all_timings.update(sp_base_times)

        # 2. Proposed Single-Pass Methods
        logger.info("[2/5] Evaluating Proposed Single-Pass Methods (Keyword & Hybrid)...")
        sp_prop_scores, sp_prop_times = run_proposed_single_pass_methods(records)
        all_scores.update(sp_prop_scores)
        all_timings.update(sp_prop_times)

        # 3. Multi-Pass Sampling Sweep across N in {2, 3, 5, 10, 20}
        logger.info("[3/5] Evaluating Multi-Pass Sampling Sweep (N in {2, 3, 5, 10, 20})...")
        sample_budgets = [2, 3, 5, 10, 20]
        mp_scores, mp_times = run_budget_sweep_multi_pass(
            records, sample_budgets=sample_budgets, seed=seed
        )
        all_scores.update(mp_scores)
        all_timings.update(mp_times)

        # Log timings
        for m_name, elapsed in all_timings.items():
            lat = (elapsed / len(records)) * 1000.0
            logger.info(f"  {m_name:<28} | Total: {elapsed:6.3f}s | Latency/passage: {lat:6.2f}ms")

        # 4. Save per-sentence predictions CSV
        pred_rows = []
        for p_idx, rec in enumerate(records):
            for s_idx, sent in enumerate(rec["sentences"]):
                row = {
                    "passage_id": p_idx,
                    "wiki_bio_test_idx": rec["wiki_bio_test_idx"],
                    "sentence_idx": s_idx,
                    "annotation": sent["annotation"],
                    "num_tokens": len(sent["token_logprobs"]),
                }
                for m_name, s_dict in all_scores.items():
                    row[m_name] = s_dict[(p_idx, s_idx)]
                pred_rows.append(row)
        pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
        logger.info(f"Saved complete prediction store to: {pred_path}")

        # 5. Standardized Metric Computation across protocols
        logger.info("[4/5] Computing Metrics & Bootstrap 95% Confidence Intervals...")
        protocols = ["nonfact", "nonfact_star", "major_only"]
        metric_rows = []

        for proto in protocols:
            keys, y_true = extract_labels_for_protocol(records, protocol=proto)
            logger.info(f"\n--- Protocol: {proto} (N={len(y_true)}, PosRate={y_true.mean()*100:.2f}%) ---")
            for m_name, s_dict in all_scores.items():
                yt, ys = align_scores_and_labels(keys, y_true, s_dict)
                m = compute_sentence_metrics(yt, ys, n_bootstrap=1000, seed=seed)
                corr = compute_passage_correlations(records, s_dict)

                # Determine generator forward pass count
                if "N20" in m_name:
                    gen_passes = 20
                elif "N10" in m_name:
                    gen_passes = 10
                elif "N5" in m_name:
                    gen_passes = 5
                elif "N3" in m_name:
                    gen_passes = 3
                elif "N2" in m_name:
                    gen_passes = 2
                else:
                    gen_passes = 1

                m.update({
                    "method": m_name,
                    "protocol": proto,
                    "generation_passes": gen_passes,
                    "latency_ms_per_passage": (all_timings[m_name] / len(records)) * 1000.0,
                    "pearson_r": corr["pearson_r"],
                    "spearman_rho": corr["spearman_rho"],
                })
                metric_rows.append(m)
                logger.info(
                    f"  {m_name:<28} | AUROC={m['auroc']:5.2f}% [{m['auroc_ci_low']:5.2f}, {m['auroc_ci_high']:5.2f}] | "
                    f"AUC-PR={m['auc_pr']:5.2f}% | Latency={m['latency_ms_per_passage']:5.2f}ms"
                )

        df_metrics = pd.DataFrame(metric_rows)
        df_metrics.to_csv(metrics_path, index=False)
        df_metrics.to_csv(results_csv_path, index=False)
        logger.info(f"Saved full evaluation metrics to: {metrics_path} and {results_csv_path}")

    # 6. Deeper Analysis 1: The Cost-Quality Pareto Frontier
    logger.info("\n[5/5] Generating Deeper Analyses & Publication Figures...")
    df_nonfact = df_metrics[df_metrics["protocol"] == "nonfact"].copy()

    pareto_points = []
    for _, row in df_nonfact.iterrows():
        m_type = "Single-Pass (1x)" if row["generation_passes"] == 1 else "Multi-Pass (Nx)"
        pareto_points.append({
            "name": row["method"],
            "type": m_type,
            "latency_ms": row["latency_ms_per_passage"],
            "auroc": row["auroc"],
            "auc_pr": row["auc_pr"],
            "passes": row["generation_passes"],
        })

    plot_pareto_frontier(pareto_points, fig_dir, title="Cost-Quality Pareto Frontier (NonFact Protocol)")

    # Plot ROC & PR curves for primary representative methods
    keys_nf, y_true_nf = extract_labels_for_protocol(records, protocol="nonfact")
    primary_methods = [
        "token_mean",
        "token_min",
        "token_keyword",
        "token_hybrid_a50",
        "selfcheck_1gram_max_N20",
        "selfcheck_2gram_max_N20",
        "selfcheck_tfidf_N20",
    ]
    rep_scores = {}
    rep_aurocs = {}
    rep_aucprs = {}
    for m in primary_methods:
        _, ys = align_scores_and_labels(keys_nf, y_true_nf, all_scores[m])
        rep_scores[m] = ys
        m_row = df_nonfact[df_nonfact["method"] == m].iloc[0]
        rep_aurocs[m] = m_row["auroc"]
        rep_aucprs[m] = m_row["auc_pr"]

    plot_roc_curves(y_true_nf, rep_scores, rep_aurocs, fig_dir)
    plot_pr_curves(y_true_nf, rep_scores, rep_aucprs, fig_dir)

    # 7. Deeper Analysis 2: High-Certainty Hallucination Error Breakdown (Simhi et al.)
    # Define high-certainty hallucinations: Sentences labeled major_inaccurate where token_mean is LOW (<= 25th percentile of token_mean on hallucinations)
    df_preds = pd.DataFrame(pred_rows)
    df_major = df_preds[df_preds["annotation"] == "major_inaccurate"]
    q25_mean = df_major["token_mean"].quantile(0.25)
    high_cert_hallucinations = df_major[df_major["token_mean"] <= q25_mean]
    n_hc = len(high_cert_hallucinations)
    logger.info(f"\n--- Deeper Analysis: High-Certainty Hallucinations (N={n_hc} sentences) ---")
    logger.info(f"Threshold: token_mean <= {q25_mean:.4f} (Bottom 25% uncertainty of false text)")

    # Measure what percentage of these false negatives are detected by other methods at their respective 80% recall thresholds
    hc_recalls = {}
    for m in primary_methods:
        all_s = np.nan_to_num(df_preds[m].values.astype(float), posinf=50.0, neginf=-50.0)
        all_y = (df_preds["annotation"] != "accurate").astype(int).values
        # Operating threshold for 80% overall recall
        prec, rec, thresholds = precision_recall_curve(all_y, all_s, pos_label=1)
        # Find threshold where recall is approx 0.80
        idx_t = np.searchsorted(rec[::-1], 0.80)
        idx_t = len(thresholds) - 1 - idx_t
        idx_t = max(0, min(idx_t, len(thresholds) - 1))
        t_val = thresholds[idx_t]

        hc_scores = np.nan_to_num(high_cert_hallucinations[m].values.astype(float), posinf=50.0, neginf=-50.0)
        detected = (hc_scores >= t_val).sum()
        rec_hc = (detected / n_hc) * 100.0
        hc_recalls[m] = rec_hc
        logger.info(f"  {m:<26} detected {detected:3d}/{n_hc} high-certainty errors ({rec_hc:5.1f}% recall)")

    plot_high_certainty_breakdown(
        bin_labels=list(hc_recalls.keys()),
        token_failure_recalls=hc_recalls,
        output_dir=fig_dir,
    )

    # Copy figures to paper/figures/
    for fname in os.listdir(fig_dir):
        shutil.copy2(os.path.join(fig_dir, fname), os.path.join(paper_fig_dir, fname))
    logger.info(f"Synchronized all figures to paper directory: {paper_fig_dir}")

    total_time = time.perf_counter() - t_global_start
    logger.info(f"\n=== GATE G3 EXPERIMENTAL RUN COMPLETE in {total_time:.2f}s ===")


if __name__ == "__main__":
    main()
