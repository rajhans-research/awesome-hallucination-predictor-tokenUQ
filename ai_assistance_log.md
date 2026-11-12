# Authentic AI Assistance Log (`ai_assistance_log.md`)

**Course:** AI Tools for Research (AITR) — Lab 02  
**Researcher:** Rajhans (PAI2026001, IIIT Allahabad)  
**AI Assistant:** Antigravity (Powered by DeepMind Gemini Architecture)  
**Logging Protocol:** Maintain traceable records of significant AI interactions, proposals, human validations, corrections, and execution outcomes.

---

## Log Entries

### Entry 01: Literature Synthesis & Document Extraction
- **Date:** 2026-11-04
- **Tool:** Antigravity (Multi-Modal Document Inspection & OCR)
- **Prompt:** "Phase 0 — Read and Understand Both Documents: Extract assignment requirements and review paper findings."
- **Task:** Extract all specifications from `atr_final_assignment.pdf` and `PAI_01_Rajhans.pdf`.
- **AI Suggestion:** Structured extraction into `assignment_compliance.md` and `review_analysis.md`, mapping the two-dimensional taxonomy (model access vs. granularity) and verifying citations [1]–[14].
- **Human Decision:** Approved. Directed AI to verify all 14 references against live repositories/DOIs rather than assuming correctness.
- **Verification Method:** Web searches verified GitHub repositories and publication status for SelfCheckGPT, LM-Polygraph, CCP, Focus, and CHOKE.
- **Code / File Change:** Created `assignment_compliance.md` and `review_analysis.md`.
- **Actual Outcome:** Successfully established 100% compliant specifications and verified baseline literature.

---

### Entry 02: Hardware Diagnostic & Feasibility Reality Check
- **Date:** 2026-11-04
- **Tool:** Antigravity (Command Execution / PowerShell)
- **Prompt:** "Check hardware diagnostics, GPU, and Python environment before selecting research direction."
- **Task:** Determine physical computing limits on the local host.
- **AI Suggestion:** Run PowerShell commands to inspect Python, Torch, and `nvidia-smi`.
- **Human Decision:** Execute commands and observe actual terminal output.
- **Verification Method:** PowerShell executed `nvidia-smi` and Python version check.
- **Observed Fact / AI Correction:**
  - *Observation:* Host has an NVIDIA GeForce GTX 1050 with 4,096 MiB (4 GB) VRAM. Python 3.11.9 is installed without PyTorch in base.
  - *Correction / Impact:* AI had initially considered local fine-tuning or running Llama-3-8B locally; this was immediately rejected because 4GB VRAM would trigger an Out-of-Memory (OOM) error. The research plan was strictly pivoted to utilize either lightweight models ($\le 0.5\text{B}$) or the canonical `wiki_bio_gpt3_hallucination` benchmark with pre-computed GPT-3 outputs.
- **Code / File Change:** Documented in `feasibility_report.md`.
- **Actual Outcome:** Prevented a major project failure where code would crash due to insufficient GPU memory.

---

### Entry 03: Gap Matrix & Research Question Selection
- **Date:** 2026-11-04
- **Tool:** Antigravity (Academic Research Planning)
- **Prompt:** "Phase 1 & Phase 2: Develop 5 evidence-based research gaps, rank top 3 directions, and recommend 1 primary and 1 backup."
- **Task:** Formulate testable research questions grounded in the review paper.
- **AI Suggestion:** Formulate 5 gaps covering keyword filtering, high-certainty errors, attention propagation, grey-box top-$k$ truncation, and cost-quality Pareto analysis. Recommend the Pareto analysis (Direction H + B) as primary.
- **Human Decision:** Approved. The Pareto analysis directly addresses the call in Section 7.4 of the review paper, while the keyword-filtered token uncertainty serves as both an internal component and an ideal zero-overhead backup.
- **Verification Method:** Cross-checked against Section 6.4, Section 7.4, and Table 2 of `PAI_01_Rajhans.pdf`.
- **Code / File Change:** Created `gap_matrix.md`, `research_question_selection.md`, and `research_brief.md`.
- **Actual Outcome:** Prepared Gate G1 Research Brief ready for formal human review.

---

### Entry 04: Legacy Pickle Deserialization & Security Hygiene Fix
- **Date:** 2026-11-12
- **Tool:** Antigravity (Python Debugging & Security Inspection)
- **Prompt:** "Verify alignment and load the GPT-3 token log-probability `.bin` files from the SelfCheckGPT archive."
- **Task:** Unpickle 238 binary completion files (`10488.bin`, etc.) in Python 3.11 with OpenAI SDK v3.45.0.
- **AI Suggestion / Initial Error:** Standard `pickle.load()` crashed immediately with `ModuleNotFoundError: No module named 'openai.openai_object'` because the archive was saved using OpenAI v0.27. Additionally, first unpickling attempt via a naive `dict` subclass failed with `TypeError: dict expected at most 1 argument, got 5`.
- **Human / Pair Debugging Decision:** Implemented `_SafeOpenAIObject` accepting `*args, **kwargs` and `_LegacyOpenAIUnpickler`. Upon inspecting unpickled fields, discovered `_init_args` in the public author archive contained legacy OpenAI key metadata (`sk-...`). Immediately designed `_SafeOpenAIObject` and `load_clean_logprob_bin` to discard `_init_args`, exported clean `data/processed/wikibio_processed.json`, and added raw `.bin` files to `.gitignore`.
- **Verification Method:** Verified `238 / 238` files loaded cleanly and scanned the serialized `wikibio_processed.json` (confirmed zero API keys).
- **Code / File Change:** `src/data_loader.py`, `src/preprocessing.py`, `.gitignore`.
- **Actual Outcome:** 100% dataset ingestion with zero legacy SDK errors or credential leaks.

---

### Entry 05: Multi-Byte UTF-8 Token Alignment Bug Fix
- **Date:** 2026-11-12
- **Tool:** Antigravity (Alignment Audit)
- **Prompt:** "Verify 100% sentence-to-token alignment across all 1,908 sentences."
- **Task:** Map human-annotated sentence strings to GPT-3 BPE token spans.
- **AI Suggestion / Initial Error:** Naive string search using `"".join(tokens)` (mirroring the original `probability-based-baselines.ipynb`) failed on 3 sentences in passage `30773` (`Týr` / `Sirið Stenberg`) due to multi-byte UTF-8 replacement characters in raw BPE token lists, and hit `UnicodeEncodeError: 'charmap' codec can't encode` on Windows console.
- **Human / Pair Debugging Decision:** Replaced BPE string concatenation with exact API `text_offset` character-coordinate alignment against `choices[0]['text']`, and enforced `sys.stdout.reconfigure(encoding="utf-8")`.
- **Verification Method:** Ran alignment audit across all 238 passages; confirmed `1908 / 1908` exact sentence alignments.
- **Code / File Change:** Updated `align_sentences_to_tokens()` in `src/data_loader.py` and added Unicode unit test in `tests/test_pipeline.py`.
- **Actual Outcome:** 100.0% sentence alignment and $+0.80\%$ AUC-PR gain on `token_mean` (`84.01%` vs published `83.21%`) over the original paper's notebook.

---

### Entry 06: Eliminating `SelfCheckNgram` SpaCy Reloading Bottleneck (50x Speedup)
- **Date:** 2026-11-12
- **Tool:** Antigravity (Performance Profiling & Code Inspection)
- **Prompt:** "Run `scripts/run_baseline.py` and inspect latency."
- **Task:** Benchmark `SelfCheckNgram(n=1)` across 238 passages.
- **Observed Bottleneck:** Baseline execution stalled for $> 8$ minutes because `selfcheckgpt.modeling_ngram.UnigramModel.__init__` invokes `spacy.load("en_core_web_sm")` on every passage (238 reloads) and runs full neural dependency parsing/NER just for word tokenization. Additionally, `scipy.stats.entropy` inside a Python loop took 11.7s for top-5 entropy.
- **Human / Pair Debugging Decision:** Canceled stalled run, replaced `scipy.stats.entropy` with vectorized NumPy (`0.88s`), and cached a lightweight `spacy.blank("en") + sentencizer` tokenizer for `SelfCheckNgram` while explicitly verifying numerical equivalence against the uncached `en_core_web_sm` parser on Passage 0 (`Exact match: [4.6485, 4.3993, 4.6913, 4.5689]`).
- **Verification Method:** Executed full `scripts/run_baseline.py` in `95.10s`; `selfcheck_1gram_avg_N20` reproduced `81.52%` and `selfcheck_1gram_max_N20` reproduced `85.63%` — exact 0.00% deviation from Manakul et al. (EMNLP 2023).
- **Code / File Change:** `src/baseline_methods.py`, `results/logs/gate_g2_baseline_run.log`.
- **Actual Outcome:** Gate G2 baseline validated cleanly with 6/6 passing unit tests.

---

### Entry 07: Bigram Log-Loss Infinite Penalty & Scikit-Learn Metric Clamping
- **Date:** 2026-11-13
- **Tool:** Antigravity (Statistical Pipeline Execution)
- **Prompt:** "Execute `scripts/run_experiment.py` across all 23 detector configurations."
- **Task:** Run 1,000-resample bootstrap AUROC/AUC-PR across all protocols.
- **Observed Errors:** 
  1. For small sampling budgets ($N=2$), rare test bigrams had zero occurrences in samples, producing $\ln(0.0) = -\infty$, which inverted to $+\infty$ in `max_neg_logprob`. Scikit-learn's `assert_all_finite()` in `precision_recall_curve` raised `ValueError: Input contains infinity`.
  2. Missing import of `precision_recall_curve` in `run_experiment.py` interrupted the initial execution at the high-certainty evaluation stage.
- **Human / Pair Debugging Decision:**
  - Added `from sklearn.metrics import precision_recall_curve`.
  - Implemented finite clamping via `np.nan_to_num(y_score, posinf=50.0, neginf=-50.0)` in `src/evaluation.py` and `scripts/run_experiment.py`. This preserves the maximal anomaly ranking of absent n-grams without crashing numerical algorithms.
  - Added prediction caching logic in `scripts/run_experiment.py` to prevent redundant 7-minute feature recomputations upon downstream script adjustments.
- **Verification Method:** Execution completed cleanly with exit code 0 (`GATE G3 EXPERIMENTAL RUN COMPLETE in 7.68s`); generated all 4 publication figures and populated `results/results.csv`.
- **Code / File Change:** `src/evaluation.py`, `scripts/run_experiment.py`.
- **Actual Outcome:** Flawless generation of all empirical tables, ROC/PR curves, Pareto frontiers, and high-certainty autopsy charts.

---

### Entry 08: Research Paper Drafting, Provenance Audits & Reproducibility Packaging
- **Date:** 2026-11-13
- **Tool:** Antigravity (LaTeX Writing, Citation Verification, Packaging)
- **Prompt:** "Draft complete research paper `paper/main.tex`, references, audits, and package repository into submission zip."
- **Task:** Produce complete IEEE-compliant manuscript, citation audit ledger, originality audit, reflection, demo plan, and zip archive.
- **AI Suggestion / Collaboration:** Drafted all 11 mandatory sections in `paper/main.tex` strictly embedding real experimental numbers from `results/results.csv`. Verified all 14 literature references in `paper/references.bib`. Structured `citation_audit.md` and `originality_audit.md`.
- **Human Review / Decision:** Verified that all numbers in LaTeX tables match `results.csv` down to the exact decimal point (e.g., $74.93\%$ AUROC for `token_hybrid_a70`, $81.7\%$ recovery of high-certainty falsehoods). Directed generation of interactive Jupyter notebook (`exploratory_data_analysis.ipynb`) and packaged the entire workspace into `AITR_Lab02_PAI2026001_TokenUQ.zip`.
- **Code / File Change:** `paper/main.tex`, `paper/references.bib`, `citation_audit.md`, `originality_audit.md`, `human_ai_reflection.md`, `README.md`, `demo_plan.md`, `viva_questions.md`.
- **Actual Outcome:** 100% compliant, fully reproducible research submission package ready for evaluation and viva demonstration.
