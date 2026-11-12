# Token Uncertainty vs. Sampling Consistency: The Cost-Quality Pareto Frontier in LLM Hallucination Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility: 100%](https://img.shields.io/badge/reproducibility-verified-success.svg)]()
[![Gates: G1--G3 Passed](https://img.shields.io/badge/Gates-G1%20%7C%20G2%20%7C%20G3%20Passed-brightgreen.svg)]()

**Researcher:** Rajhans (Roll No: `PAI2026001`)  
**Department:** Department of AI & Trustworthy Computing, Indian Institute of Information Technology Allahabad  
**Course:** AI Tools for Research (ATR) — Lab Assignment 02  
**Date:** November 2026  

---

## 1. Project Overview

This repository houses the complete, end-to-end implementation, benchmark suite, evaluation engine, and research paper for the empirical study:

> **"Unveiling the Cost-Quality Pareto Frontier: Single-Pass Token Uncertainty vs. Multi-Pass Sampling Consistency in LLM Hallucination Detection"**

### Core Research Objective
To investigate the trade-off between **single-pass internal token uncertainty quantification** ($O(1)$ generation passes, negligible post-processing latency) and **multi-pass black-box sampling consistency** ($O(N)$ stochastic generation passes) across 1,908 sentence-level annotations on the GPT-3 WikiBio hallucination benchmark (`potsawee/wiki_bio_gpt3_hallucination`, Manakul et al., EMNLP 2023).

### Key Research Findings
1. **Pareto Frontier Dominance:** Our proposed single-pass linear fusion (`token_hybrid_a70`) achieves **$74.93\%$ AUROC** [95% CI: $72.54, 77.38$] and **$87.32\%$ AUC-PR** at **$0.63\text{ ms}$** post-processing latency, strictly outperforming 20-sample multi-pass consistency (`SelfCheck-1gram` $N=20$: $73.35\%$ AUROC) while requiring **0 additional generator passes** ($>75\times$ faster).
2. **Elimination of Syntactic Dilution:** Content-word stopword filtering (`token_keyword`) raises single-pass AUROC from $71.66\%$ (`token_mean`) to $73.58\%$ ($+1.92\%$ absolute improvement) and reduces FPR@95 from $68.60\%$ to $66.86\%$.
3. **High-Certainty Hallucination Autopsy:** On $N=191$ confident fabrications where token uncertainty suffers catastrophic failure (recalling only $37.2\%$), multi-pass sampling consistency recovers up to **$81.7\%$** of false negatives, proving that internal likelihoods and external consistency operate on orthogonal failure boundaries.

---

## 2. Repository Directory Structure

```text
AITR_Lab02_TokenUQ/
├── .gitignore                      # Git exclusion rules (virtualenvs, cache, credentials)
├── README.md                       # Master repository guide and reproduction manual
├── requirements.txt                # Pinned production dependencies
├── assignment_compliance.md        # Comprehensive 100% compliance audit against assignment brief
├── review_analysis.md              # Deep analysis and citation audit of source review paper
├── gap_matrix.md                   # 5 candidate research gaps with verified/inferred tags
├── research_question_selection.md  # Formal decision rubric ranking research directions
├── feasibility_report.md           # Physical hardware profiling and resource budget
├── research_brief.md               # Gate G1: Approved one-page research brief
├── baseline_plan.md                # Gate G2: Baseline selection and verification protocol
├── baseline_validation.md          # Gate G2: Exact baseline reproduction report
├── experiment_protocol.md          # Frozen experimental protocol & hypothesis specifications
├── evaluation_report.md            # Comprehensive multi-protocol metric report
├── error_analysis.md               # Deeper analyses: Pareto mapping & high-certainty autopsy
├── citation_audit.md               # Source-to-claim ledger mapping all citations to primary papers
├── originality_audit.md            # Freshness audit proving zero text recycling from review
├── human_ai_reflection.md          # Critical reflection on research journey and AI boundaries
├── ai_assistance_log.md            # Authentic log of 8 AI pair-programming interactions and fixes
├── research_decisions.md           # Log of independent human decisions D01--D07
├── repository_audit.md             # Repository completeness and security verification audit
├── demo_plan.md                    # 5-minute code and results presentation plan
├── viva_questions.md               # 30 grounded technical viva questions and detailed answers
├── configs/
│   └── experiment.yaml             # Frozen experimental configuration (seeds, paths, schemas)
├── data/
│   ├── README.md                   # Dataset provenance, license, and structure documentation
│   ├── cache/                      # Local Hugging Face cache
│   └── processed/
│       └── wikibio_processed.json  # Sanitized, 100% aligned benchmark store (1,908 sentences)
├── notebooks/
│   └── exploratory_data_analysis.ipynb # Interactive EDA, calibration, and visualization notebook
├── paper/
│   ├── main.tex                    # Full 8-page IEEE-format research paper
│   ├── references.bib              # Verified BibTeX entries for all 14 literature sources
│   └── figures/                    # Publication figures synchronized from results
│       ├── fig1_roc_curves.pdf     # ROC curves across representative detectors
│       ├── fig1_roc_curves.png
│       ├── fig2_pr_curves.pdf      # PR curves under class imbalance
│       ├── fig2_pr_curves.png
│       ├── fig3_pareto_frontier.pdf# Cost-Quality Pareto Frontier
│       ├── fig3_pareto_frontier.png
│       ├── fig4_high_certainty_breakdown.pdf # High-certainty error recall recovery
│       └── fig4_high_certainty_breakdown.png
├── results/
│   ├── results.csv                 # Master export containing all 69 evaluated configurations
│   ├── figures/                    # Output publication-grade vector and raster plots
│   ├── logs/
│   │   ├── gate_g2_baseline_run.log   # Gate G2 baseline execution log
│   │   └── gate_g3_experiment_run.log # Gate G3 complete experiment execution log
│   ├── metrics/
│   │   ├── baseline_validation_metrics.csv # Gate G2 baseline metrics
│   │   └── experiment_metrics.csv          # Gate G3 standardized metrics with bootstrap CIs
│   └── predictions/
│       ├── baseline_predictions.csv   # Gate G2 per-sentence predictions
│       └── experiment_predictions.csv # Gate G3 per-sentence prediction store (1,908 x 28)
├── scripts/
│   ├── download_data.py            # Data ingestion, legacy unpickling, and character alignment
│   ├── run_baseline.py             # Gate G2 baseline reproduction script
│   └── run_experiment.py           # Gate G3 master experiment runner and figure generator
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Safe unpickler and exact character-coordinate span aligner
│   ├── preprocessing.py            # Dataset integrity auditor and multi-protocol label extractor
│   ├── baseline_methods.py         # Single-pass token scorers & cached SelfCheck-Ngram
│   ├── proposed_method.py          # Proposed keyword filter, hybrid fusion, and budget sweep
│   ├── evaluation.py               # AUROC, AUC-PR, FPR@95, Prec@80, Pearson r, bootstrap CIs
│   └── visualization.py            # Publication plotting utilities (Matplotlib / Seaborn)
└── tests/
    └── test_pipeline.py            # 6 unit and edge-case tests (alignment, math, labels)
```

---

## 3. Hardware & Environment Specifications

- **Tested OS:** Windows 11 / Linux (Ubuntu 22.04 LTS compatible)
- **Python Version:** 3.10 or 3.11 (Tested with Python 3.11.9)
- **Minimum Compute:** 4 CPU cores, 8 GB RAM. (GPU optional; PyTorch CPU is sufficient).
- **Disk Space:** ~500 MB for dataset and virtual environment.

---

## 4. Quickstart: Step-by-Step Reproduction Guide

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/rajhans-iiita/AITR_Lab02_TokenUQ.git
cd AITR_Lab02_TokenUQ

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate
```

### Step 2: Install Pinned Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 3: Run Unit & Regression Tests
Verify pipeline integrity, mathematical scoring functions, and label parsing:
```bash
pytest tests/test_pipeline.py -v
```
*Expected Output:* `6 passed in ~0.35s`.

### Step 4: Ingest Benchmark Data & Verify Alignment
Download the canonical Hugging Face dataset, unpack pre-computed GPT-3 log-probabilities, and verify exact byte-offset sentence alignment:
```bash
python scripts/download_data.py
```
*Expected Output:* `238 / 238 passages processed`, `1908 / 1908 sentences aligned (100.0%)`, clean JSON exported to `data/processed/wikibio_processed.json`.

### Step 5: Reproduce Gate G2 Baseline
Validate the SelfCheckGPT and Token Mean baselines against published literature:
```bash
python scripts/run_baseline.py
```
*Expected Output:* Exact numerical match with Manakul et al. (EMNLP 2023) (`selfcheck_1gram_max_N20 = 85.63%`, `selfcheck_1gram_avg_N20 = 81.52%`).

### Step 6: Execute Gate G3 Full Experimental Pipeline & Generate Figures
Run all 23 detector configurations across 3 evaluation protocols, perform 1,000-resample bootstrap confidence intervals, execute deeper error analyses, and generate publication plots:
```bash
python scripts/run_experiment.py
```
*Artifacts Generated:*
- Master results: `results/results.csv` and `results/metrics/experiment_metrics.csv`
- Sentence predictions: `results/predictions/experiment_predictions.csv`
- Vector plots (PDF & PNG): `paper/figures/fig1_roc_curves.pdf`, `fig2_pr_curves.pdf`, `fig3_pareto_frontier.pdf`, `fig4_high_certainty_breakdown.pdf`.

---

## 5. Summary of Empirical Results

### Primary Non-Factual Error Detection (`NonFact`, $N=1,908$, Positive Rate $= 72.96\%$)

| Method | Gen. Passes | Latency (ms) | AUROC (\%) [95\% CI] | AUC-PR (\%) | FPR@95 (\%) | Prec@80 (\%) | Pearson $r$ (\%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `token_mean` | 1 | 0.13 | 71.66 [69.04, 74.48] | 84.01 | 68.60 | 83.02 | 56.17 |
| `token_min` | 1 | 0.18 | 73.79 [71.40, 76.36] | 87.63 | 70.93 | 82.03 | 59.59 |
| **`token_keyword` (Ours)** | 1 | 0.34 | 73.58 [71.13, 76.05] | 85.04 | 66.86 | 83.38 | 59.90 |
| **`token_hybrid_a50` (Ours)** | 1 | 0.68 | 74.81 [72.37, 77.27] | **87.84** | 67.25 | 82.80 | 61.71 |
| **`token_hybrid_a70` (Ours)** | 1 | 0.63 | **74.93** [72.54, 77.38] | 87.32 | **65.50** | **83.16** | 62.18 |
| `SelfCheck-1gram` ($N=2$) | 2 | 9.41 | 64.92 [62.01, 67.74] | 81.23 | 77.33 | 78.15 | 34.95 |
| `SelfCheck-1gram` ($N=5$) | 5 | 16.26 | 70.66 [67.74, 73.31] | 83.80 | 72.48 | 81.25 | 55.34 |
| `SelfCheck-1gram` ($N=20$) | 20 | 49.23 | 73.35 [70.46, 75.97] | 85.63 | 66.86 | 83.16 | 64.72 |
| `SelfCheck-2gram` ($N=20$) | 20 | 54.23 | 73.14 [70.46, 75.88] | 85.26 | 67.25 | 81.99 | -- |
| `SelfCheck-TFIDF` ($N=20$) | 20 | 9.20 | 71.71 [69.05, 74.48] | 85.53 | 74.22 | 80.90 | **66.65** |

---

## 6. Research Provenance & Integrity Statement

This codebase, experimental protocol, and manuscript were developed by **Rajhans** (`PAI2026001`), Department of AI & Trustworthy Computing, IIIT Allahabad, for the course **AI Tools for Research (ATR) — Lab 02**.
- All experimental numbers reported in the paper and tables are genuine outputs of actual executions.
- Zero citations or experimental results were fabricated.
- All AI interactions, errors, and fixes are logged chronologically in [`ai_assistance_log.md`](ai_assistance_log.md).
- Full source-to-claim mappings are audited in [`citation_audit.md`](citation_audit.md).

---

## 7. License
This project is open-source and licensed under the [MIT License](LICENSE).
