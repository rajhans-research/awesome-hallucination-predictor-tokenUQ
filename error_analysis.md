# Error Analysis & Deeper Evaluations: Failure Modes in Token vs. Multi-Pass Hallucination Detection

**Student Researcher:** Rajhans (Roll No: `PAI2026001`)  
**Affiliation:** Department of AI & Trustworthy Computing, Indian Institute of Information Technology Allahabad  
**Course:** AI Tools for Research (ATR) — Lab 02  
**Date:** November 13, 2026  
**Artifact Version:** 1.0 (Frozen Gate G3 Evaluation)  

---

## 1. Introduction

While macroscopic metrics such as AUROC and AUC-PR summarize ranking quality, they obscure the underlying failure modes that dictate operational safety. This error analysis investigates two specific evaluation dimensions directly motivated by the research gaps identified in our structured review (`PAI_01_Rajhans.pdf`):

1. **Deeper Analysis 1 — The Cost-Quality Pareto Frontier:** Quantifying the diminishing marginal return of multi-pass stochastic generation versus single-pass internal features across computational budgets.
2. **Deeper Analysis 2 — High-Certainty Hallucination Error Breakdown:** Rigorously testing the hypothesis formulated by Simhi et al. (2024) and Rajhans (2025) regarding the systemic blindness of token log-probabilities to confidently fabricated parametric hallucinations.
3. **Qualitative Case Studies:** In-depth linguistic and probabilistic autopsy of true positives, false positives, and false negatives.

---

## 2. Deeper Analysis 1: The Cost-Quality Pareto Frontier

### 2.1 Theoretical Trade-off
Production hallucination detection faces a strict trade-off between inference budget (FLOPs, wall-clock latency, financial API token cost) and detection fidelity. Multi-pass consistency estimators (e.g., SelfCheckGPT, Semantic Entropy) incur an $O(N)$ multiplication of generator inference costs, requiring $N$ stochastic passes per response. Single-pass uncertainty extraction incurs $O(1)$ generation passes and negligible post-processing overhead.

### 2.2 Empirical Pareto Points (`NonFact` Protocol)

```
        AUROC (%)
  76.0 |                                              [SelfCheck-TFIDF N=20: 71.71%]
       |
  75.0 |                   * token_hybrid_a70 (74.93%, 0.63ms, 1x)
       |                   * token_hybrid_a50 (74.81%, 0.68ms, 1x)
  74.0 |         * token_min (73.79%, 0.18ms, 1x)     [SelfCheck-1gram N=20: 73.35%, 49.2ms, 20x]
       |         * token_keyword (73.58%, 0.34ms, 1x) [SelfCheck-2gram N=20: 73.14%, 54.2ms, 20x]
  72.0 |   * token_mean (71.66%, 0.13ms, 1x)          [SelfCheck-1gram N=10: 72.38%, 27.8ms, 10x]
       |
  70.0 |                                              [SelfCheck-1gram N=5: 70.66%, 16.3ms, 5x]
  68.0 |                                              [SelfCheck-1gram N=3: 68.40%, 12.2ms, 3x]
  65.0 |                                              [SelfCheck-1gram N=2: 64.92%, 9.4ms, 2x]
       +-------------------------------------------------------------------------------------------->
         0.1ms      0.5ms      1.0ms                   10ms             30ms           55ms  (Log Latency)
```

### 2.3 Pareto Frontier Dominance Analysis
- **Dominant Pareto Set:** The empirical Pareto-optimal frontier is defined by:
  1. `token_mean`: 0.13 ms, AUROC 71.66% (Ultra-fast lower bound)
  2. `token_min`: 0.18 ms, AUROC 73.79% (Fastest high-precision point)
  3. `token_hybrid_a70`: 0.63 ms, AUROC 74.93% (Optimal quality point)
- **Sub-optimality of Low-Budget Multi-Pass:** All multi-pass sampling configurations with $N \le 5$ are **strictly dominated** by single-pass `token_min` and `token_hybrid`. Generating 2, 3, or 5 extra stochastic samples yields lower AUROC ($64.92\%$ to $70.66\%$) while consuming $10\times$ to $100\times$ more post-processing latency and $2\times$ to $5\times$ more generation inference.
- **Pareto Inversion:** Multi-pass `SelfCheck-1gram` with $N=20$ samples achieves $73.35\%$ AUROC at $49.23\text{ ms}$ post-processing latency + 20 generator forward passes. Our proposed single-pass `token_hybrid_a70` achieves $74.93\%$ AUROC (+1.58% higher) at $0.63\text{ ms}$ post-processing latency with **0 additional generator passes**.

---

## 3. Deeper Analysis 2: High-Certainty Hallucinations Breakdown

### 3.1 Motivation & Operational Definition
As established in `PAI_01_Rajhans.pdf` (Gap 1) and Simhi et al. (2024), Large Language Models frequently output non-factual statements with deceptive overconfidence due to parametric memorization biases and training distribution artifacts. When an LLM generates a hallucination with high confidence, its token-level negative log-probabilities remain low, causing single-pass uncertainty detectors to suffer catastrophic false negatives.

To rigorously evaluate this failure boundary, we define:
- **High-Certainty Hallucinations ($N_{\text{HC}} = 191$ sentences):** Sentences annotated by human raters as `major_inaccurate` where the baseline token uncertainty metric (`token_mean`) is within the **bottom quartile** of uncertainty for fabricated text ($\text{token\_mean} \le 0.3919$).
- **Experimental Test:** We set an operating threshold for each detector corresponding to an **$80\%$ overall recall** on the full dataset, and measure what percentage of these 191 high-certainty false negatives are successfully detected.

### 3.2 Measured Recall on High-Certainty Fabrications

| Method Family | Method Name | Passes ($N$) | Overall Recall Target | High-Certainty Errors Detected | High-Certainty Recall (%) | Failure Mode Summary |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Token-Level** | `token_mean` | 1 | 80.0% | 71 / 191 | **37.2%** | Severe false negative rate ($62.8\%$ missed) |
| **Token-Level** | `token_keyword` | 1 | 80.0% | 69 / 191 | **36.1%** | Diluted by confident keyword tokens |
| **Token-Level** | `token_min` | 1 | 80.0% | 81 / 191 | **42.4%** | Catches single local hesitation token |
| **Token-Level** | `token_hybrid_a50`| 1 | 80.0% | 75 / 191 | **39.3%** | Moderate recovery |
| **Multi-Pass** | `selfcheck_1gram_max_N20` | 20 | 80.0% | 141 / 191 | **73.8%** | **+36.6% absolute recovery** over `token_mean` |
| **Multi-Pass** | `selfcheck_2gram_max_N20` | 20 | 80.0% | 148 / 191 | **77.5%** | **+40.3% absolute recovery** over `token_mean` |
| **Multi-Pass** | `selfcheck_tfidf_N20` | 20 | 80.0% | 156 / 191 | **81.7%** | **+44.5% absolute recovery** over `token_mean` |

### 3.3 Core Scientific Finding
This empirical result provides direct quantitative validation of the central theoretical premise:
1. **Single-pass token uncertainty methods collapse on high-certainty hallucinations.** When the generator confidently hallucinates, token-level recall plummets from the designed $80.0\%$ operating baseline down to $37.2\%$. Over $62\%$ of dangerous major fabrications slip past undetected.
2. **Multi-pass consistency serves as an orthogonal, non-redundant error detector.** Because stochastic sampling at temperature $T > 0$ forces the model to explore alternate decoding paths, inconsistent facts diverge across sampled completions even when the greedy completion was produced with high probability. Consequently, `selfcheck_tfidf_N20` successfully recovers **$81.7\%$** of the errors missed by `token_mean`.
3. **The Tiered Architecture Recommendation:** Rather than relying exclusively on either single-pass or multi-pass methods, trustworthy LLM pipelines should implement a **two-tier cascading detector**:
   - **Tier 1 (Streaming Gate):** Evaluate all tokens via single-pass `token_hybrid_a70` at zero additional generation cost ($0.63\text{ ms}$). High-uncertainty sentences are immediately flagged.
   - **Tier 2 (Audit Gate):** Ambiguous or high-stakes outputs passing Tier 1 are conditionally routed to multi-pass consistency sampling (`SelfCheck-TFIDF`), catching high-certainty parametric hallucinations.

---

## 4. Qualitative Error Case Studies

### 4.1 Case Study 1: The Confident Biography Hallucination (False Negative for Token UQ, True Positive for Multi-Pass)
- **Subject:** Passage 0 (`wiki_bio_test_idx: 62464`, John Peter Russell)
- **Sentence 0:** *"John Peter Russell was an Australian impressionist painter who was born in Sydney in 1858."*
- **Human Annotation:** `major_inaccurate` (Actual birth year: 1858; place: Sydney; but generated details in subsequent sentences fabricate his schooling, career, and family lineage).
- **Sentence 2:** *"He was educated at the Sydney Grammar School and then studied art at the Slade School of Fine Art in London."*
- **Token Uncertainty:** `token_mean = 0.3918` (Low uncertainty, 24th percentile). The model emits *"Slade School of Fine Art"* with high confidence because it is a prominent art institution frequently associated with British and Australian painters.
- **Single-Pass Score:** Missed by `token_mean` (predicted score: 0.392 vs threshold 0.441).
- **Multi-Pass Consistency Score:** `selfcheck_tfidf_N20 = 0.960`, `selfcheck_1gram_max_N20 = 7.61`. Across 20 stochastic samples, the generator alternative samples mention *"Royal Academy"*, *"Julian Academy in Paris"*, and *"East London School"*. The inconsistency across samples immediately exposes the fabrication.

### 4.2 Case Study 2: Syntactic Function Word Hesitation (False Positive for `token_mean`, Corrected by `token_keyword`)
- **Subject:** Passage 14 (`wiki_bio_test_idx: 12948`, Professional Athlete)
- **Sentence 1:** *"In addition to his playing career, he also served as a player-coach for the team."*
- **Human Annotation:** `accurate`
- **Token Breakdown:**
  - Token `" In"`: logprob = -0.05
  - Token `" addition"`: logprob = -0.12
  - Token `" to"`: logprob = -0.02
  - Token `" player"`: logprob = -0.85
  - Token `"-"`: logprob = -1.95 (Rare hyphenation token causes high negative logprob)
  - Token `"coach"`: logprob = -0.45
- **Single-Pass Baseline:** `token_mean` is inflated by the subword hyphenation token, incorrectly pushing the sentence across the hallucination threshold (False Positive).
- **Proposed `token_keyword`:** Isolates content words `{"playing", "career", "served", "player", "coach", "team"}`. The robust factual content lowers the keyword uncertainty score, correctly classifying the sentence as accurate (True Negative).

---

## 5. Summary of Failure Taxonomies

| Failure Type | Observed Mechanism | Mitigation Strategy |
| :--- | :--- | :--- |
| **High-Certainty Hallucination** | Model memorizes an incorrect co-occurrence pattern; outputs false token with high softmax confidence. | Multi-pass stochastic sampling consistency (SelfCheckGPT, Semantic Entropy). |
| **Tokenization Artifact / Rare Subword** | Hyphens, foreign accents, or rare morphological subwords receive low probability despite factual accuracy. | Content-word filtering (`token_keyword`) and length-normalized scoring. |
| **Syntactic Filler Noise** | Formulaic transitions (*"Furthermore, it is worth noting that..."*) inflate token counts and mask factual errors. | Stopword masking using validated closed-class syntactic lexical sets. |
| **Stochastic Sampling Collusion** | In low-temperature sampling, multi-pass generators repeat the same prompt-induced hallucination. | Semantic clustering with non-zero temperature ($T \ge 0.5$) and token-level verification fusion. |
