"""
scripts/run_baseline.py
-----------------------
Gate G2 Execution Script: Validates Single-Pass Token Baselines and Multi-Pass
SelfCheck-1gram Baseline against Manakul et al. (EMNLP 2023).

Outputs:
- results/predictions/baseline_predictions.csv
- results/metrics/baseline_validation_metrics.csv
- results/logs/gate_g2_baseline_run.log
"""

import os
import sys
import time
import logging
from datetime import datetime
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    load_processed_dataset,
    extract_labels_for_protocol,
    audit_dataset_integrity,
)
from src.baseline_methods import (
    run_single_pass_baselines,
    run_selfcheck_ngram_baseline,
)
from src.evaluation import (
    align_scores_and_labels,
    compute_sentence_metrics,
    compute_passage_correlations,
)


def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("gate_g2_baseline")
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
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "wikibio_processed.json")
    log_path = os.path.join(PROJECT_ROOT, "results", "logs", "gate_g2_baseline_run.log")
    pred_path = os.path.join(PROJECT_ROOT, "results", "predictions", "baseline_predictions.csv")
    metrics_path = os.path.join(PROJECT_ROOT, "results", "metrics", "baseline_validation_metrics.csv")

    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    logger = setup_logger(log_path)
    logger.info("=== GATE G2: BASELINE VALIDATION RUN STARTED ===")
    logger.info(f"Random Seed: {seed}")

    t_start = time.perf_counter()
    records = load_processed_dataset(data_path)
    audit = audit_dataset_integrity(records)
    logger.info(
        f"Dataset Loaded: {audit['num_passages']} passages, "
        f"{audit['total_sentences']} sentences | Classes: {audit['class_counts']}"
    )

    # 1. Single-pass baselines
    logger.info("Running Single-Pass Token Baselines...")
    all_scores, timings = run_single_pass_baselines(records)
    for m_name, elapsed in timings.items():
        logger.info(f"  Method={m_name:<16} | Total Time={elapsed:.4f}s | Latency/passage={(elapsed/len(records))*1000:.2f}ms")

    # 2. Multi-pass SelfCheck-1gram (N=20) baseline
    logger.info("Running Multi-Pass SelfCheck-1gram (N=20) Baseline...")
    ngram_avg, ngram_max, ngram_time = run_selfcheck_ngram_baseline(
        records, num_samples=20, n=1, seed=seed
    )
    all_scores["selfcheck_1gram_avg_N20"] = ngram_avg
    all_scores["selfcheck_1gram_max_N20"] = ngram_max
    timings["selfcheck_1gram_avg_N20"] = ngram_time
    timings["selfcheck_1gram_max_N20"] = ngram_time
    logger.info(
        f"  Method=selfcheck_1gram (N=20) | Total Time={ngram_time:.4f}s | Latency/passage={(ngram_time/len(records))*1000:.2f}ms"
    )

    # 3. Export per-sentence predictions CSV
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
    logger.info(f"Saved sentence-level predictions to: {pred_path}")

    # 4. Evaluate across protocols
    protocols = ["nonfact", "nonfact_star", "major_only"]
    metric_rows = []

    for proto in protocols:
        keys, y_true = extract_labels_for_protocol(records, protocol=proto)
        logger.info(f"\n--- Protocol: {proto} (N_sents={len(y_true)}, PosRate={y_true.mean()*100:.2f}%) ---")
        for m_name, s_dict in all_scores.items():
            yt, ys = align_scores_and_labels(keys, y_true, s_dict)
            m = compute_sentence_metrics(yt, ys, n_bootstrap=1000, seed=seed)
            corr = compute_passage_correlations(records, s_dict)
            m.update({
                "method": m_name,
                "protocol": proto,
                "latency_ms_per_passage": (timings[m_name] / len(records)) * 1000.0,
                "pearson_r": corr["pearson_r"],
                "spearman_rho": corr["spearman_rho"],
            })
            metric_rows.append(m)
            logger.info(
                f"  {m_name:<24} | AUROC={m['auroc']:5.2f}% [{m['auroc_ci_low']:5.2f}, {m['auroc_ci_high']:5.2f}] | "
                f"AUC-PR={m['auc_pr']:5.2f}% | Pearson={corr['pearson_r']:5.2f}%"
            )

    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    logger.info(f"\nSaved baseline validation metrics to: {metrics_path}")
    logger.info(f"=== GATE G2 VALIDATION COMPLETE in {time.perf_counter() - t_start:.2f}s ===")


if __name__ == "__main__":
    main()
