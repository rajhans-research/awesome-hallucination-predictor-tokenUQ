# Research Question Selection & Experimental Direction Ranking

**Review Base:** `PAI_01_Rajhans.pdf`  
**Assignment Reference:** `atr_final_assignment.pdf` (Section 3: Pathways; Section 4: Gate G1)  
**Author / Researcher:** Rajhans (PAI2026001, IIIT Allahabad)

---

## 1. Evaluation of Candidate Directions

We evaluate the eight suggested assignment directions against the empirical findings, theoretical framework, and open gaps established in the author's literature review:

| Direction | Literature Grounding in Review | Feasibility on Student Setup | Dataset & Code Availability | Recommendation Status |
| :--- | :--- | :--- | :--- | :--- |
| **A. Reproduce & extend token-level UQ** | High (Sec. 4.1, 5.2, CCP [2]) | High (CPU / 4GB GPU) | High (WikiBio GPT-3 / Hugging Face) | **Strong Candidate** |
| **B. Compare token prob, entropy, & semantic UQ** | High (Sec. 4.2, 5.1, [3, 4, 5]) | High (Pre-computed samples available) | High (WikiBio, LM-Polygraph) | **Strong Candidate** |
| **C. Attention / historical propagation** | Moderate (Sec. 4.4, 5.3, Focus [14]) | Moderate (Needs white-box attention tensors) | Moderate (Requires running local open LLM) | Feasible |
| **D. Model or dataset shifts** | Moderate (Sec. 5.5, 6.5) | Moderate (Requires running multiple models) | Moderate | Less focused |
| **E. High-confidence hallucinations** | High (Sec. 4.5, 7.1, CHOKE [9]) | Moderate (Needs activation probing) | High (technion-cs-nlp repo) | Compelling Deeper Analysis |
| **F. Lightweight uncertainty detector** | High (Sec. 4.3, 7.4) | High (Fast single-pass feature extraction) | High | Integrated into Primary |
| **G. Restricted model access (Grey-box)** | High (Sec. 3.1, 6.3, EPR [8]) | High (Logit truncation simulation) | High | Viable Alternative |
| **H. Computational efficiency vs. performance** | Very High (Sec. 6.4, 7.4, [5]) | High (Exact latency & memory benchmarking) | High (Benchmarked datasets) | **Primary Driver** |

---

## 2. Ranking the Top Three Most Feasible Research Directions

### Rank 1 (Top Recommendation — Primary Direction):
**Direction H + B + F: The Cost-Quality Pareto Frontier — Single-Pass Token Uncertainty vs. Multi-Pass Sampling Consistency for Hallucination Detection**
- **Approved Pathway:** **Pathway B (Controlled Comparative Study)** with elements of **Pathway C (Targeted Modification)**.
- **Why this ranks #1:**
  1. *Direct Literature Justification:* Section 7.4 of the author's review paper specifically frames this exact research priority:
     > *"No single signal appears sufficient across all hallucination types... The research challenge is to combine them without simply increasing computational cost. A useful design objective is therefore a Pareto analysis between detection quality, latency, model access, and interpretability."* (Rajhans, 2026, p. 8).
  2. *Defensible Research Question:* Addresses the fundamental tension between cheap single-pass token probabilities ($1\times$ inference) and expensive multi-pass stochastic consistency ($10\times\text{--}20\times$ inference + NLI scoring).
  3. *Empirical Grounding:* Evaluated on the canonical `potsawee/wiki_bio_gpt3_hallucination` benchmark, containing 238 biographies with human sentence annotations, aligned token probabilities, and 20 stochastic samples per passage.
  4. *Rigor and Reproducibility:* Both token-level metrics (mean log-prob, min prob, token entropy, factual keyword-filtered log-prob) and sampling-based metrics (SelfCheckGPT with varying sample counts $N \in \{2, 3, 5, 10, 20\}$) can be benchmarked under an identical, frozen protocol with real wall-clock latency, memory tracking, and AUROC/AUPRC evaluation.
  5. *Deeper Analysis:* Includes a targeted error analysis on high-certainty hallucinations (Section 4.5 / Simhi et al. [9]), examining when token uncertainty fails and whether sampling consistency recovers those errors.

---

### Rank 2 (Secondary Recommendation — Backup Direction):
**Direction F + A + C: Keyword-Filtered Token Uncertainty — Mitigating the Surface-Form Bottleneck in Single-Pass Hallucination Detection**
- **Approved Pathway:** **Pathway C (Targeted Modification)**.
- **Why this ranks #2:**
  1. *Literature Justification:* Section 4.2 and Section 5.2 of the review detail how non-informative syntactic tokens ("the", "is", "born") dilute token negative log-probabilities.
  2. *Defensible Research Question:* Does isolating factual and named-entity tokens via POS/NER filtering improve sentence-level hallucination detection AUROC over unweighted token probability without increasing computational overhead?
  3. *Why it serves as the ideal backup:* It relies strictly on single-pass token outputs and standard NLP tagging (e.g., spaCy or regex POS masks), requiring zero stochastic generation and zero NLI inference. If environment constraints or package incompatibilities prevent running multi-pass NLI models, this direction provides an immediate, robust fallback.

---

### Rank 3 (Third Ranked Direction):
**Direction G: Simulating Grey-Box Constraints — Hallucination Detection via Truncated Top-$k$ Entropy and Entropy Production Rate**
- **Approved Pathway:** **Pathway A (Reproduction + Extension)**.
- **Why this ranks #3:**
  1. *Literature Justification:* Directly addresses Section 3.1 and Section 6.3 of the review regarding commercial API limits (only top-$k$ logprobs exposed).
  2. *Defensible Research Question:* How does predictive entropy truncation at $k \in \{1, 3, 5, 10, 20\}$ degrade AUROC compared to full-vocabulary white-box entropy?
  3. *Feasibility:* Can be executed on any open-source causal LLM (e.g., Qwen2.5-0.5B or GPT-2) by mathematically truncating logits prior to entropy calculation.

---

## 3. Final Recommendation & Rationale

**We recommend PRIMARY DIRECTION 1 (The Cost-Quality Pareto Frontier).**

### Academic & Technical Justification:
1. **Perfect Continuity:** Synthesizes the two major paradigms reviewed in Rajhans (2026): White/Grey-box token-level uncertainty (Fadeeva et al. [2], Xiao & Wang [13]) versus Black-box multi-sample consistency (Manakul et al. [5], Farquhar et al. [3]).
2. **Defensible, Non-Trivial Contribution:** Instead of naively claiming that one method is "better" in isolation, it maps the operational trade-off curve across latency, memory, and detection accuracy under fixed compute budgets.
3. **High Reproducibility on Student Hardware:** Can be executed cleanly on the local system (Intel CPU + GTX 1050 4GB) in minutes without risk of GPU OOM or unfulfillable compute dependencies.
4. **Immediate Fallback:** If any multi-sample scoring component fails, the protocol cleanly folds back into Backup Direction 2 (pure single-pass token filtering) without altering the experimental foundation.
