# Repository Audit & Reproducibility Checklist

**Student Researcher:** Rajhans (Roll No: `PAI2026001`)  
**Affiliation:** Department of AI & Trustworthy Computing, Indian Institute of Information Technology Allahabad  
**Course:** AI Tools for Research (ATR) — Lab 02  
**Date:** November 13, 2026  
**Artifact Version:** 1.0 (Frozen Gate G3 Audit)  

---

## 1. Audit Overview
This audit verifies the structural integrity, dependency pinning, execution repeatability, credential safety, and standards compliance of the research repository `AITR_Lab02_TokenUQ`.

---

## 2. Structural & File Completeness Checklist

| Component / Subdirectory | Target Path | Actual Status | Size / Count | Purpose & Verification |
| :--- | :--- | :---: | :---: | :--- |
| **Root Documentation** | `README.md` | **PRESENT** | ~8 KB | Comprehensive quickstart, directory layout, and results. |
| **Dependencies** | `requirements.txt` | **PRESENT** | 12 packages | Pinned versions for NumPy, PyTorch, Scikit-learn, SpaCy. |
| **Git Exclusion** | `.gitignore` | **PRESENT** | 205 B | Excludes `.venv/`, `__pycache__/`, raw `.bin`, `.DS_Store`. |
| **Configuration** | `configs/experiment.yaml` | **PRESENT** | 1.8 KB | Frozen random seed (`42`), filepaths, and schema. |
| **Dataset Store** | `data/processed/wikibio_processed.json` | **PRESENT** | 10.64 MB | 238 passages, 1,908 sentence annotations, 100% token aligned. |
| **Data Documentation**| `data/README.md` | **PRESENT** | 2.5 KB | Benchmark provenance, license, schema, and rater instructions. |
| **Source Modules** | `src/*.py` | **PRESENT** | 7 files | `data_loader`, `preprocessing`, `baseline_methods`, `proposed_method`, `evaluation`, `visualization`, `__init__`. |
| **Execution Scripts** | `scripts/*.py` | **PRESENT** | 3 files | `download_data.py`, `run_baseline.py`, `run_experiment.py`. |
| **Unit & Edge Tests** | `tests/test_pipeline.py` | **PRESENT** | 6 tests | Tested via `pytest`: 6/6 passed in 0.334s. |
| **Interactive Notebook**| `notebooks/exploratory_data_analysis.ipynb` | **PRESENT** | Valid JSON | Demonstrates data loading, logprob calibration, and plots. |
| **Manuscript** | `paper/main.tex` | **PRESENT** | ~22 KB | 8-page IEEE conference format with all authentic numbers. |
| **Bibliography** | `paper/references.bib` | **PRESENT** | 14 entries | Verified BibTeX citations for all review & baseline papers. |
| **Paper Figures** | `paper/figures/*.pdf` & `.png` | **PRESENT** | 8 files | Fig 1 (ROC), Fig 2 (PR), Fig 3 (Pareto), Fig 4 (High-Cert). |
| **Results Master** | `results/results.csv` | **PRESENT** | 18.7 KB | 69 evaluated configurations with bootstrap CIs. |
| **Predictions Store** | `results/predictions/experiment_predictions.csv` | **PRESENT** | 842.9 KB | 1,908 rows $\times$ 28 columns (all detector predictions). |
| **Execution Logs** | `results/logs/*.log` | **PRESENT** | 2 files | Exact logs of Gate G2 and Gate G3 executions. |
| **Provenance Audits**| `citation_audit.md`, `originality_audit.md`, `human_ai_reflection.md` | **PRESENT** | 3 files | Source-to-claim mapping, freshness check, AI reflection. |
| **Evaluation Reports**| `evaluation_report.md`, `error_analysis.md` | **PRESENT** | 2 files | Full metrics across protocols, Pareto & high-certainty autopsy. |
| **Presentation & Viva**| `demo_plan.md`, `viva_questions.md`, `final_review.md` | **PRESENT** | 3 files | 5-min demo walkthrough, 30 viva Q&As, peer review. |

---

## 3. Dependency & Environment Isolation
- **Virtual Environment:** Configured in `.venv` (Python 3.11.9).
- **Core Dependencies:**
  - `torch`: 2.15.0+cpu (CPU execution verified, no CUDA OOM risks)
  - `transformers`: 4.57.3
  - `selfcheckgpt`: 0.1.7 (with optimized lightweight tokenizer cache)
  - `spacy`: 3.8.0 with `en_core_web_sm`
  - `scikit-learn`: 1.8.0
  - `scipy`: 1.16.3
  - `numpy`: 2.2.6
  - `pandas`: 2.3.3
  - `matplotlib`: 3.10.8
  - `pyyaml`: 6.0.3
- **Reproducibility Verification:** All random seeds frozen to `42`. Deterministic bootstrapping tested.

---

## 4. Credential & Security Hygiene Audit
- **Legacy OpenAI SDK Objects:** Legacy `.bin` files downloaded from the original author archive contained historical API key metadata in `_init_args`.
- **Mitigation Implemented:** `_SafeOpenAIObject` explicitly stripped `_init_args` during unpickling. Data was transformed into a pure, clean JSON dictionary (`data/processed/wikibio_processed.json`).
- **Disk Cleanliness:** All raw `.bin` directories were permanently purged from disk.
- **Git Security:** `.gitignore` explicitly prevents `.bin`, `.env`, `.pem`, and credential files from being tracked.
- **Audit Outcome:** **ZERO active or inactive API keys or secrets exist in the repository.**

---

## 5. Audit Sign-off
The repository satisfies all structural, statistical, reproducibility, and safety requirements specified in `atr_final_assignment.pdf`.
