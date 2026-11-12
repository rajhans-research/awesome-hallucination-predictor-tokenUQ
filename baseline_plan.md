# Baseline Verification Plan (`baseline_plan.md`)

**Target Milestone:** Gate G2: Baseline Verification  
**Investigated Baselines:** Single-Pass Token Uncertainty vs. Published SelfCheckGPT Baselines  
**Grounding Literature:** Manakul et al. (EMNLP 2023) [5]; Fadeeva et al. (ACL 2024 Findings) [2]

---

## 1. Selected Baselines and Formulations

To ensure a fair and scientifically rigorous comparison, we establish three single-pass token-level baselines and two multi-pass sampling baselines:

### Baseline 1: Sequence Mean Negative Log-Probability (`token_mean`)
For a sentence $s$ consisting of tokens $t_1, \dots, t_{|s|}$:
$$S_{\text{mean}}(s) = \frac{1}{|s|} \sum_{i=1}^{|s|} -\log p(t_i \mid t_{<i}, x)$$
- **Score interpretation:** Higher score indicates greater average uncertainty (predicted hallucination).
- **Literature origin:** Evaluated in SelfCheckGPT [5], CCP [2], and Xiao & Wang [13].

### Baseline 2: Sequence Minimum Token Probability / Maximum Log-Loss (`token_min`)
$$S_{\text{min}}(s) = \max_{i=1, \dots, |s|} \left( -\log p(t_i \mid t_{<i}, x) \right) = -\log \left( \min_{i=1, \dots, |s|} p(t_i \mid t_{<i}, x) \right)$$
- **Score interpretation:** Represents the "weakest link" token in the sentence. If any single token has near-zero probability, the whole sentence is flagged.
- **Literature origin:** Manakul et al. (EMNLP 2023, Section 3.2).

### Baseline 3: Top-5 Truncated Predictive Entropy (`token_entropy`)
Given top-5 token log-probabilities provided by the API:
$$\tilde{p}(v_j) = \frac{\exp(\log p(v_j))}{\sum_{k=1}^5 \exp(\log p(v_k))}$$
$$S_{\text{entropy}}(s) = \frac{1}{|s|} \sum_{i=1}^{|s|} \left( -\sum_{j=1}^5 \tilde{p}(v_{i,j}) \log \tilde{p}(v_{i,j}) \right)$$
- **Score interpretation:** Higher average token entropy indicates the model was indecisive among top candidates.

### Comparator 1: Multi-Pass Stochastic N-Gram Consistency (`selfcheck_ngram`)
Computes unigram/bigram overlap between sentence $s$ and $N$ independently sampled generations $\{y^{(1)}, \dots, y^{(N)}\}$:
$$S_{\text{ngram}}(s; N) = 1.0 - \frac{1}{N} \sum_{n=1}^N \text{Overlap}(s, y^{(n)})$$
- **Score interpretation:** Sentences that do not re-occur in stochastic samples receive high hallucination scores.

### Comparator 2: Multi-Pass Semantic NLI / Prompt Consistency (`selfcheck_nli`)
Evaluates whether sampled passages entail or contradict the sentence under evaluation.

---

## 2. Published Reference Numbers & Target Replication

From Manakul et al. (EMNLP 2023, Table 1, WikiBio GPT-3 dataset):

| Method | Published AUROC (Non-Factual) | Published AUPRC | Expected Behavior in Replication |
| :--- | :---: | :---: | :--- |
| **`Avg -log P`** (`token_mean`) | $65.6\text{--}68.5\%$ | $\approx 45.0\text{--}52.0\%$ | Modest baseline; sensitive to sentence length and function words. |
| **`Min Prob`** (`token_min`) | $67.0\text{--}69.8\%$ | $\approx 47.0\text{--}54.0\%$ | Slightly outperforms mean; flags rare named entity errors. |
| **`SelfCheckGPT-N-Gram` ($N=20$)** | $70.5\text{--}73.2\%$ | $\approx 55.0\text{--}60.0\%$ | Outperforms single-pass token probability. |
| **`SelfCheckGPT-NLI` ($N=20$)** | $78.0\text{--}81.5\%$ | $\approx 65.0\text{--}72.0\%$ | Strongest black-box comparator, but with $\approx 20\times$ computation cost. |

---

## 3. Validation Criteria for Gate G2

To pass Gate G2, the implementation must satisfy:
1. Exact match of sentence segmentation and token character alignment with ground-truth labels.
2. Verified calculation of ROC curves, precision-recall curves, and AUROC.
3. Reproduction of `token_mean` and `token_min` AUROCs within $\pm 2.0\%$ of published benchmarks.
4. Genuine, timestamped execution logs saved directly to `results/logs/`.
