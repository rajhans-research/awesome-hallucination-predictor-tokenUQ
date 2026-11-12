# Gate G1: Research Brief (One-Page Problem Brief)

**Submission Milestone:** Gate G1 Approval Gate  
**Course:** AI Tools for Research (AITR) — Lab 02 / Second Paper  
**Institution:** Department of Artificial Intelligence and Trustworthy Computing, IIIT Allahabad  
**Student:** Rajhans (Enrollment: PAI2026001, Email: pai2026001@iiita.ac.in)  
**Date:** November 2026

---

| Field | Project Specification |
| :--- | :--- |
| **Review Topic and Key Papers** | **Topic:** Token-Level Uncertainty Quantification and Hallucination Detection in Large Language Models.<br/>**Key Verified Literature:**<br/>1. Manakul et al. (EMNLP 2023) *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative LLMs* (doi:10.18653/v1/2023.emnlp-main.557) `[VERIFIED]`<br/>2. Fadeeva et al. (ACL 2024 Findings) *Fact-Checking LLMs via Token-Level Uncertainty Quantification* (doi:10.18653/v1/2024.findings-acl.558) `[VERIFIED]`<br/>3. Farquhar et al. (Nature 2024) *Detecting hallucinations in large language models using semantic entropy* (doi:10.1038/s41586-024-07421-0) `[VERIFIED]`<br/>4. Simhi et al. (arXiv 2025) *Trust Me, I'm Wrong: LLMs Hallucinate with Certainty Despite Knowing the Answer* (arXiv:2502.12964) `[VERIFIED]`<br/>5. Zhang et al. (EMNLP 2023) *Enhancing Uncertainty-Based Hallucination Detection with Stronger Focus* (doi:10.18653/v1/2023.emnlp-main.58) `[VERIFIED]` |
| **Research Gap** | **The Cost-Quality Pareto Frontier in Hallucination Detection:** Multi-sample consistency methods (SelfCheckGPT, Semantic Entropy) provide robust semantic factuality signals but require $10\times\text{--}20\times$ generation passes and expensive cross-checking. Conversely, single-pass token uncertainty methods require $1\times$ inference ($0$ extra generation passes) but suffer from the surface-form bottleneck. The literature lacks a systematic empirical mapping of the Pareto frontier relating latency, sample budget ($N \in \{1, 2, 3, 5, 10, 20\}$), and detection AUROC under a unified protocol (Rajhans 2026, Section 6.4, Section 7.4). `[VERIFIED]` |
| **Primary Research Question** | **Primary Question:** Across varying computational budgets (latency and forward passes), what is the empirical Pareto frontier between single-pass token uncertainty methods (mean log-prob, min prob, keyword-filtered log-prob) and multi-pass sampling consistency methods (SelfCheckGPT with $N \in \{2, 3, 5, 10, 20\}$) for sentence-level hallucination detection? `[INFERRED]`<br/>**Supporting Subquestions:**<br/>- *RQ1:* Does isolating factual/content tokens (mitigating the surface-form bottleneck) allow single-pass token uncertainty to narrow the detection AUROC gap against multi-pass sampling consistency?<br/>- *RQ2:* In high-certainty hallucinations (where token probability assigns high confidence to false text), does multi-sample consistency succeed where single-pass fails, or do both fail jointly? `[INFERRED]` |
| **Study Pathway** | **Pathway B (Controlled Comparative Study)** combined with **Pathway C (Targeted Modification of token aggregation)**. `[VERIFIED]` |
| **Dataset & Measurement Setup** | **Dataset:** `potsawee/wiki_bio_gpt3_hallucination` (Hugging Face Datasets Hub, MIT License). `[VERIFIED]`<br/>**Structure:** 238 Wikipedia entity biographies, 1,908 human-annotated sentences (`accurate`, `minor_inaccurate`, `major_inaccurate`), aligned GPT-3 token log-probabilities, and 20 pre-generated stochastic passages per entity. Unsupervised detection (zero parameter-tuning leakage). `[VERIFIED]` |
| **Baseline / Comparators** | **Baselines (Single-Pass Token-Level):**<br/>1. Mean Token Negative Log-Probability (`token_mean`) [2, 13]<br/>2. Minimum Token Probability (`token_min`) [2, 5]<br/>3. Token Predictive Entropy (`token_entropy`) [4, 10]<br/>**Comparators (Multi-Pass Consistency):**<br/>4. SelfCheckGPT-Prompt ($N \in \{2, 3, 5, 10, 20\}$) [5]<br/>5. SelfCheckGPT-NLI ($N \in \{2, 3, 5, 10, 20\}$) [5] `[VERIFIED]` |
| **Proposed Contribution** | 1. Rigorous empirical mapping of the (Latency, AUROC) Pareto frontier across single-pass and multi-pass methods.<br/>2. A lightweight Factual Keyword-Filtered Token Uncertainty aggregator (`token_keyword`) mitigating surface-form noise at zero extra generation cost.<br/>3. A targeted failure-case analysis isolating high-certainty hallucinations. `[INFERRED]` |
| **Primary Metric & Control** | **Primary Metrics:** Sentence-level AUROC and AUPRC (positive class = hallucinated sentence); Wall-clock latency (ms/passage) and generation multiplier.<br/>**Controls:** Same entity instances, identical sentence tokenization, frozen ground truth labels, fixed seed for stochastic subsets. `[VERIFIED]` |
| **Computing & Risks** | **Hardware:** Intel Core CPU, NVIDIA GeForce GTX 1050 (4GB VRAM), Windows 11. Pipeline runtime: $< 6\text{ minutes}$. `[VERIFIED]`<br/>**Risks & Fallback:** If local GPU VRAM limits heavy NLI cross-encoders, activate fallback to lightweight `SelfCheckGPT-ngram` and CPU-batched `DeBERTa-v3-small` (both tested $< 1.2\text{ GB}$). `[INFERRED]` |
| **First Executable Milestone** | **Milestone M1:** Ingest `potsawee/wiki_bio_gpt3_hallucination` from Hugging Face, verify sentence/token alignment, compute raw token-level baselines (`token_mean`, `token_min`), and output the first measured AUROC table and log. `[VERIFIED]` |

---

### Approval Gate Status

> [!IMPORTANT]
> **GATE G1 APPROVAL REQUIRED BEFORE CODE EXECUTION:**  
> In compliance with Section 4 of `atr_final_assignment.pdf`, implementation does not proceed until the research brief, primary research question, and experimental pathway are reviewed and explicitly approved by the researcher.
