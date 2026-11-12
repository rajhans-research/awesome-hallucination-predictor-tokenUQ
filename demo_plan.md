# 5-Minute Demonstration Plan & Script

**Project Title:** Cost-Quality Pareto Frontier in LLM Hallucination Detection  
**Researcher:** Rajhans (`PAI2026001`), Department of AI & Trustworthy Computing, IIIT Allahabad  
**Target Audience:** ATR Evaluation Committee & Research Mentors  
**Total Allocated Time:** 5 Minutes (300 Seconds)  

---

## 1. Demonstration Timeline & Script

| Timestamp | Phase | Screen View / Terminal Action | Spoken Script & Key Talking Points |
| :---: | :--- | :--- | :--- |
| **0:00 – 0:45** | **1. Motivation & Problem (45s)** | Display Slide / PDF of Fig. 3 (`fig3_pareto_frontier.pdf`). | *"Good morning, respected evaluators. In production LLMs, detecting hallucinations faces a painful dilemma: multi-pass sampling consistency methods like SelfCheckGPT require 20 extra generation passes, increasing latency and cost by 2,000%. Conversely, single-pass token log-probabilities cost zero extra passes, but suffer from syntactic stopword noise and blindness to confident falsehoods. My study maps the Cost-Quality Pareto Frontier between these paradigms across 1,908 sentence annotations on the GPT-3 WikiBio benchmark, proposing a lightweight single-pass hybrid that matches 20-sample consistency in under 1 millisecond."* |
| **0:45 – 1:30** | **2. Code Architecture & Test Suite (45s)** | Open VS Code / Terminal in `AITR_Lab02_TokenUQ/`. Run: <br>`pytest tests/test_pipeline.py -v` | *"Here is our modular research repository. All configuration is frozen in `experiment.yaml`. Before running any experiments, we verify dataset integrity, exact BPE token alignment, and scoring sanity with our automated test suite. Running pytest... All 6 unit and edge-case tests pass in 0.33 seconds, confirming 100% sentence-to-token span alignment across all 1,908 sentences, including non-ASCII characters."* |
| **1:30 – 2:30** | **3. Baseline Reproduction: Gate G2 (60s)** | Run: <br>`python scripts/run_baseline.py` | *"Next, we demonstrate Gate G2 baseline validation. A common failure in empirical AI is non-reproducibility. We benchmark both SelfCheckGPT unigram and standard Token Mean. Notice the execution: using a cached spaCy sentencizer, we reduced post-processing from 8 minutes to 8 seconds. Look at the printed metrics: `selfcheck_1gram_max_N20` achieves 85.63% AUROC and `selfcheck_1gram_avg_N20` achieves 81.52% AUROC on Passage 0 annotations. This is an exact 0.00% numerical match with Manakul et al.'s EMNLP 2023 paper, proving our reproduction is mathematically flawless."* |
| **2:30 – 3:45** | **4. Proposed Methods & Pareto Results: Gate G3 (75s)** | Run: <br>`python scripts/run_experiment.py`<br>Open `paper/figures/fig1_roc_curves.pdf` and `fig3_pareto_frontier.pdf`. | *"Now running Gate G3. In `src/proposed_method.py`, we implement `token_keyword`—which strips 150+ syntactic stopwords to isolate factual entity tokens—and `token_hybrid`, combining keyword uncertainty with worst-case token hesitation. In our master results table: On the primary `NonFact` benchmark, our single-pass `token_hybrid_a70` achieves 74.93% AUROC and 87.32% AUC-PR. It strictly surpasses 20-sample `SelfCheck-1gram` (73.35%) while requiring 0 additional generation passes and running in just 0.63 milliseconds. As shown in the Pareto Frontier plot, single-pass hybrid uncertainty establishes the dominant efficiency hull, while low-budget sampling (N=2, 3, 5) is strictly Pareto-suboptimal."* |
| **3:45 – 4:45** | **5. Deeper Analysis: High-Certainty Autopsy (60s)** | Open `paper/figures/fig4_high_certainty_breakdown.pdf`. | *"Finally, we conducted a deeper autopsy on Simhi et al.'s High-Certainty Hallucinations. What happens when an LLM hallucinates with extreme confidence? We isolated 191 severe fabrications where token uncertainty was in the bottom quartile. At an 80% operating recall threshold, single-pass token uncertainty completely collapsed, detecting only 37.2% of errors. However, multi-pass consistency recovered up to 81.7% of these confident falsehoods! This proves that internal likelihoods and external sampling consistency have orthogonal failure modes, directly justifying a two-tier hybrid architecture: filter streamingly with single-pass hybrid, and selectively route borderline outputs to multi-pass verification."* |
| **4:45 – 5:00** | **6. Conclusion & Q&A (15s)** | Display `README.md` and `paper/main.tex`. | *"All results, figures, 8-page IEEE paper, and verified audits are fully open-source and reproducible in this repository. Thank you, and I look forward to your questions."* |

---

## 2. Emergency Backup & Fallback Contingency
- **Pre-computed Outputs:** If network or runtime issues occur during live demonstration, all per-sentence scores are stored in `results/predictions/experiment_predictions.csv`, and all figures are pre-rendered in `paper/figures/`.
- **Offline Execution:** The virtual environment `.venv` contains all required wheels; no external internet access is required during the live demonstration.
