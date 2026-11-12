# Evidence-Based Research Gap Matrix

**Review Base:** `PAI_01_Rajhans.pdf` (*Token-Level Uncertainty as a Predictor of Hallucination in Large Language Models: A Structured Review*)  
**Status Tags:** `[VERIFIED]` = Backed by peer-reviewed literature or direct empirical measurement; `[INFERRED]` = Plausible deduction based on literature evidence requiring validation; `[UNKNOWN]` = Insufficient empirical evidence.

---

## The 5 Candidate Research Gaps & Experimental Formulations

### Candidate Gap 1: Fine-Grained Token Filtering vs. Naive Sequence Aggregation
- **Review Evidence:** Section 4.1 (Eq. 2), Section 4.3 (CCP), Section 6.1 (Surface-form ambiguity), Section 7.1 / Table 2 ("Fine-grained localization: Sequence scores may hide a small false claim"), Pages 4, 5, 8. `[VERIFIED]`
- **Original Literature:** Fadeeva et al. (ACL 2024 Findings) [2]; Zhang et al. (EMNLP 2023) [14]. `[VERIFIED]`
- **Research Problem:** Autoregressive generation produces sequences where the vast majority of tokens are functional words (articles, prepositions, copulas) that have high predictive certainty. When computing sequence-level uncertainty via unweighted arithmetic mean, these boilerplate tokens overwhelm and dilute the uncertainty spikes that occur at informative factual slots (e.g., proper nouns, dates, numerical values). `[VERIFIED]`
- **Testable Research Question:** Does isolating factual and keyword tokens via part-of-speech (POS) and named-entity (NER) filtering yield statistically significant improvements in sentence-level hallucination detection AUROC compared to unweighted mean token negative log-probability and minimum token probability? `[INFERRED]`
- **Proposed Method:** Implement a lexical-selective token aggregator that filters tokens by POS tags (nouns, proper nouns, numerals) and content masks, comparing: (a) Arithmetic mean, (b) Minimum probability, (c) Length-penalized mean, and (d) Factual-token filtered mean. `[INFERRED]`
- **Dataset:** `potsawee/wiki_bio_gpt3_hallucination` (Hugging Face) with sentence-level human annotations (Accurate, Minor Inaccurate, Major Inaccurate) and aligned token log-probabilities. `[VERIFIED]`
- **Baseline Methods:**
  1. Sequence Mean Token Negative Log-Probability (`avg_logprob`)
  2. Sequence Minimum Token Log-Probability (`min_prob`)
  3. Predictive Token Entropy (`token_entropy`) `[VERIFIED]`
- **Evaluation Metrics:** AUROC, AUPRC (Positive class = Hallucination), Precision at 80% Recall, False Positive Rate at 95% Specificity. `[VERIFIED]`
- **Hardware & Memory:** Local CPU or NVIDIA GeForce GTX 1050 (4GB VRAM). Dataset processing takes < 15 seconds; token evaluation is immediate. `[VERIFIED]`
- **Implementation Complexity:** Low to Moderate. Clean Python implementation, modular architecture, fast turnaround.
- **Expected Research Value:** Directly demonstrates how the surface-form bottleneck can be mitigated in single-pass inference without requiring expensive multi-sample generation or external knowledge retrieval. `[INFERRED]`
- **Risks & Limitations:** Dependency on accuracy of token-to-word alignment; POS tagger errors on noisy text. `[INFERRED]`
- **Feasibility Under Student Compute:** **10/10 (Extremely Feasible).** Fully runnable locally.

---

### Candidate Gap 2: High-Certainty Hallucinations and Logit Margin Diagnostics
- **Review Evidence:** Section 4.5 ("Calibration and high-certainty hallucinations"), Section 6.2, Section 7.1 / Table 2 ("High-certainty errors: Wrong outputs may have low uncertainty"), Pages 5, 7, 8. Reference: Simhi et al. (2025) [9]. `[VERIFIED]`
- **Original Literature:** Simhi et al. (arXiv 2025, *Trust Me, I'm Wrong*) [9]; Azaria & Mitchell (arXiv 2023) [1]. `[VERIFIED]`
- **Research Problem:** Current uncertainty methods operate on the premise that uncertainty correlates inversely with factuality. However, post-training alignment (RLHF/DPO) and parametric memorization induce "hallucinations with certainty" where the model outputs factually false tokens with near-zero predictive entropy. In these failure modes, standard thresholding produces false negatives. `[VERIFIED]`
- **Testable Research Question:** When an LLM produces a high-certainty hallucination ($u_i < \tau$), does the logit margin ($\text{logit}_{(1)} - \text{logit}_{(2)}$) across transformer layers or residual-stream linear probing reveal factuality degradation that output-layer token probability misses? `[INFERRED]`
- **Proposed Method:** Replicate a controlled factual probing setup (TruthfulQA or CHOKE subsets) using a small open model (Pythia-160M / Qwen2.5-0.5B), extract intermediate residual-stream activations, and train a lightweight logistic probe to classify high-certainty hallucinations. `[INFERRED]`
- **Dataset:** Simhi et al. CHOKE dataset / TruthfulQA factual subset. `[VERIFIED]`
- **Baseline Methods:**
  1. Output token predictive entropy ($H_i$)
  2. Output token negative log-probability ($u_i$)
  3. Sequence perplexity `[VERIFIED]`
- **Evaluation Metrics:** High-certainty AUROC (conditioned on low entropy), Brier Score, Probe Accuracy. `[VERIFIED]`
- **Hardware & Memory:** 4GB GPU can fit 0.5B models; 7B/8B models require Google Colab T4. Activation saving requires 2–5 GB disk space. `[VERIFIED]`
- **Implementation Complexity:** High. Requires layer-wise activation hooks, hidden-state extraction, and cross-validated linear probing.
- **Expected Research Value:** Mechanistic insight into internal model representations during confident hallucination. `[INFERRED]`
- **Risks & Limitations:** Probing requires training linear classifiers, introducing risk of probe overfitting or dataset shortcut learning. `[INFERRED]`
- **Feasibility Under Student Compute:** **6/10 (Moderate).** Possible locally with 0.5B models, but constrained by 4GB VRAM.

---

### Candidate Gap 3: Attention-Aware Historical Uncertainty Propagation vs. Static Token Scoring
- **Review Evidence:** Section 4.4 ("Autoregressive error propagation", Figure 2), Section 5.3, Section 7 (Table 2), Pages 5, 6, 8. References: Zhang et al. (EMNLP 2023) [14]; Vazhentsev et al. (ICML 2026) [11]. `[VERIFIED]`
- **Original Literature:** Zhang et al. (EMNLP 2023, *Focus*) [14]; Xiao & Wang (EACL 2021) [13]. `[VERIFIED]`
- **Research Problem:** Due to autoregressive conditioning, an early hallucinated token becomes part of the context for subsequent tokens, allowing later incorrect tokens to be generated with high confidence. Static token detectors that score tokens independently cannot capture this temporal cascade. `[VERIFIED]`
- **Testable Research Question:** Does propagating uncertainty from preceding tokens via transformer cross-token self-attention weights improve span-level and sentence-level hallucination detection AUROC compared to unweighted historical moving-window averages? `[INFERRED]`
- **Proposed Method:** Implement a simplified attention-propagation module that extracts self-attention weights from the last layer of an open LLM, constructs an attention-directed uncertainty graph, and computes attention-propagated risk scores for downstream tokens. `[INFERRED]`
- **Dataset:** WikiBio GPT-3 / HaluEval summarization subset. `[VERIFIED]`
- **Baseline Methods:**
  1. Raw Token Negative Log-Prob
  2. Exponential Moving Average of Historical Uncertainty
  3. Static Window Uncertainty `[VERIFIED]`
- **Evaluation Metrics:** AUROC, AUPRC, Latency overhead (ms/passage), GPU memory delta. `[VERIFIED]`
- **Hardware & Memory:** GTX 1050 (4GB) can run GPT-2 / Qwen2.5-0.5B with attention extraction. Attention matrix storage requires careful batching. `[VERIFIED]`
- **Implementation Complexity:** Moderate to High. Extracting and storing multi-head attention tensors ($B \times H \times T \times T$) requires tensor manipulation.
- **Expected Research Value:** Validates whether attention mechanics can overcome autoregressive error cascades in small-scale models. `[INFERRED]`
- **Risks & Limitations:** Attention weights are often diffuse or dominated by attention sinks (initial tokens), potentially diluting propagation signal. `[INFERRED]`
- **Feasibility Under Student Compute:** **7/10 (Feasible).**

---

### Candidate Gap 4: Grey-Box Top-$k$ Entropy Approximation vs. Full-Vocabulary Uncertainty
- **Review Evidence:** Section 3.1, Section 4.1 (Eq. 3), Section 6.3 ("Access and deployment constraints"), Section 7 (Table 2), Pages 3, 4, 7, 8. Reference: Moslonka et al. (arXiv 2025) [8]. `[VERIFIED]`
- **Original Literature:** Moslonka et al. (arXiv 2025, *Learned Hallucination Detection in Black-Box LLMs Using Token-Level Entropy Production Rate*) [8]; Vashurin et al. (TACL 2025) [10]. `[VERIFIED]`
- **Research Problem:** Production commercial LLM APIs (e.g., OpenAI, Anthropic, Mistral) restrict probability outputs to top-$k$ logprobs (typically $k=5$), hiding the remaining 32,000–128,000 vocabulary logits. Full predictive entropy cannot be computed directly. It remains unknown how severely top-$k$ truncation degrades hallucination detection, and whether renormalized top-$k$ entropy or top-$k$ entropy production rate (EPR) can serve as a lossless surrogate. `[VERIFIED]`
- **Testable Research Question:** How does the hallucination detection AUROC of renormalized top-$k$ predictive entropy scale as $k$ varies from $1$ to $20$ compared to exact full-vocabulary entropy, and at what value of $k$ does detection performance saturate? `[INFERRED]`
- **Proposed Method:** Using an open-weight model, compute both exact full-vocabulary predictive entropy $H_{\text{full}}$ and top-$k$ renormalized entropy $H_{\text{top-}k}$ across $k \in \{1, 2, 3, 5, 10, 20\}$, measuring Spearman rank correlation and AUROC degradation on hallucinated vs factual passages. `[INFERRED]`
- **Dataset:** WikiBio GPT-3 hallucination dataset / HaluEval QA. `[VERIFIED]`
- **Baseline Methods:**
  1. Full-Vocabulary Predictive Entropy ($H_{\text{full}}$)
  2. Raw Top-1 Log-Probability
  3. Uniform Tail-Assumed Entropy `[VERIFIED]`
- **Evaluation Metrics:** AUROC, AUPRC, Spearman rank correlation $\rho(H_{\text{full}}, H_{\text{top-}k})$, Relative AUROC Delta. `[VERIFIED]`
- **Hardware & Memory:** Local CPU or GTX 1050 (4GB VRAM). Very lightweight. `[VERIFIED]`
- **Implementation Complexity:** Low. Pure mathematical estimators applied to output logits.
- **Expected Research Value:** Directly bridges the gap between academic white-box research and commercial API deployment constraints. `[INFERRED]`
- **Risks & Limitations:** Findings on small models may not perfectly extrapolate to 70B+ proprietary APIs. `[INFERRED]`
- **Feasibility Under Student Compute:** **9/10 (Highly Feasible).**

---

### Candidate Gap 5: The Cost-Quality Pareto Frontier: Single-Pass Token Uncertainty vs. Multi-Pass Sampling Consistency
- **Review Evidence:** Section 1, Section 2.2, Section 5.1, Section 6.4 ("Computational overhead"), Section 7.4 ("Pareto analysis between detection quality, latency, model access, and interpretability"), Pages 1, 2, 5, 7, 8. References: Manakul et al. (EMNLP 2023) [5]; Farquhar et al. (Nature 2024) [3]; Fadeeva et al. (ACL 2024) [2]. `[VERIFIED]`
- **Original Literature:** Manakul et al. (EMNLP 2023, *SelfCheckGPT*) [5]; Farquhar et al. (Nature 2024, *Semantic Entropy*) [3]; Fadeeva et al. (ACL 2024 Findings, *CCP*) [2]. `[VERIFIED]`
- **Research Problem:** Multi-sample consistency methods like SelfCheckGPT (sampling 10–20 stochastic continuations and computing sentence-level NLI or BERTScore consistency) achieve high hallucination detection performance, but at massive computational cost ($10\times\text{--}20\times$ inference passes + $N \times S$ NLI inference calls). Conversely, single-pass token uncertainty methods (mean log-prob, min prob, token entropy) require $0$ extra generative passes ($1\times$ cost), but are considered less semantically robust. Despite this acknowledged trade-off, no systematic study maps the **empirical Pareto frontier** between computational latency, memory footprint, and hallucination detection AUROC across single-pass token metrics and varying sampling budgets ($N \in \{1, 2, 3, 5, 10, 20\}$). `[VERIFIED]`
- **Testable Research Question:** Across varying inference budgets (FLOPs and wall-clock latency), where does the empirical Pareto frontier lie between single-pass token uncertainty aggregations (mean, min, keyword-filtered, CCP) and multi-pass sampling consistency (SelfCheckGPT with $N \in \{2, 3, 5, 10, 20\}$ samples) for sentence-level hallucination detection? `[INFERRED]`
- **Proposed Method:** Evaluate both method families under a standardized, frozen protocol on the verified WikiBio GPT-3 hallucination dataset (which contains generated passages, human labels, token log-probabilities, and 20 stochastic samples per passage). Compute AUROC, AUPRC, inference latency (seconds per passage), and forward pass multipliers to plot the exact Pareto curves. `[INFERRED]`
- **Dataset:** `potsawee/wiki_bio_gpt3_hallucination` (Hugging Face) with 238 passages, human annotations (Major, Minor, Accurate), token logprobs, and 20 pre-generated stochastic samples per concept. `[VERIFIED]`
- **Baseline Methods:**
  1. Single-Pass Unweighted Mean Token Negative Log-Prob (`token_mean`)
  2. Single-Pass Minimum Token Probability (`token_min`)
  3. Single-Pass Token Predictive Entropy (`token_entropy`)
  4. Single-Pass Keyword/Factual-Filtered Token Probability (`token_keyword`)
  5. Multi-Pass SelfCheckGPT-Prompt ($N \in \{2, 3, 5, 10, 20\}$)
  6. Multi-Pass SelfCheckGPT-NLI ($N \in \{2, 3, 5, 10, 20\}$) `[VERIFIED]`
- **Evaluation Metrics:**
  - Detection Quality: Sentence-level AUROC, AUPRC (Hallucination class), Brier Score.
  - Resource Efficiency: Wall-clock latency (ms/passage), Number of model forward passes, Peak memory (MB).
  - Pareto Optimality: Identification of non-dominated methods in (Latency, AUROC) space. `[VERIFIED]`
- **Hardware & Memory:** Local CPU / GTX 1050 (4GB VRAM). The entire experiment can be fully executed and reproduced locally in under 10 minutes because the benchmark dataset provides authentic pre-generated outputs, token log-probabilities, and samples, supplemented by local model verification scripts. `[VERIFIED]`
- **Implementation Complexity:** Moderate. Clean modular implementation covering data loading, feature extraction, baseline implementations, evaluation metrics, and publication-ready Pareto visualization.
- **Expected Research Value:** Provides the research and practitioner community with concrete, quantitative guidance on the true operational efficiency trade-off, answering Section 7.4 of the review paper directly. `[INFERRED]`
- **Risks & Limitations:** Evaluated on synthetic biographical text (WikiBio GPT-3); generalization to open-domain QA should be noted as a limitation. `[INFERRED]`
- **Feasibility Under Student Compute:** **10/10 (Ideal Feasibility).** 100% reproducible on local hardware with zero risk of out-of-memory or timeout.
