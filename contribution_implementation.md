# Contribution Implementation Report (`contribution_implementation.md`)

**Pathway Chosen:** Pathway B (Controlled Comparative Study) combined with Pathway C (Targeted Modification of Token Aggregator).  
**Source Code:** [`src/proposed_method.py`](file:///C:/Users/RAJHANS/.gemini/antigravity/scratch/AITR_Lab02_TokenUQ/src/proposed_method.py)  
**Supporting Review Section:** `PAI_01_Rajhans.pdf`, Section 4.2 ("The surface-form bottleneck"), Section 4.3 ("Claim-conditioned uncertainty"), Section 7.4 ("Hybrid uncertainty signals and Pareto analysis").

---

## 1. Scientific Motivation & Theoretical Grounding

In standard LLM autoregressive generation:
$$p(t_i \mid t_{<i}, x)$$
The majority of generated tokens are high-frequency closed-class syntactic stopwords (e.g., articles, prepositions, conjunctions, copulas like `"the"`, `"was"`, `"in"`). These words have high predictive confidence and near-zero negative log-probabilities.

When a sentence-level uncertainty score is calculated via naive arithmetic mean:
$$S_{\text{mean}}(s) = \frac{1}{|s|} \sum_{i=1}^{|s|} -\log p(t_i \mid t_{<i}, x)$$
The high-confidence syntactic filler tokens dilute and overwhelm the genuine uncertainty spikes that manifest on factual entity slots (e.g., birth dates, locations, organizations).

### Our Proposed Innovation: Factual Keyword-Filtered Token Uncertainty (`token_keyword`)
We isolate the set of factual semantic content tokens:
$$\mathcal{K}(s) = \{ t_i \in s \mid t_i \notin \mathcal{V}_{\text{stop}} \land \text{isalnum}(t_i) \}$$
The uncertainty score is computed strictly over this filtered semantic core:
$$S_{\text{keyword}}(s) = \frac{1}{|\mathcal{K}(s)|} \sum_{t_i \in \mathcal{K}(s)} -\log p(t_i \mid t_{<i}, x)$$
If $|\mathcal{K}(s)| = 0$, the function gracefully reverts to sentence mean.

### Peak-Content Hybrid Fusion (`token_hybrid`)
Recognizing that hallucinated entities may consist of a single catastrophic token (captured by $S_{\text{min}}(s) = \max_{i} (-\log p_i)$), we also introduce a parameter-free convex combination:
$$S_{\text{hybrid}}(s) = \alpha \cdot S_{\text{keyword}}(s) + (1 - \alpha) \cdot S_{\text{min}}(s)$$
with $\alpha \in \{0.5, 0.7\}$.

### Multi-Pass Budget Sweep ($N \in \{2, 3, 5, 10, 20\}$)
To empirically construct the **Cost-Quality Pareto Frontier**, we sweep sample budgets $N$ across:
1. `SelfCheck-1gram`
2. `SelfCheck-2gram`
3. `SelfCheck-TFIDF` (Lexical-semantic consistency via TF-IDF cosine distance)
measuring both AUROC and wall-clock inference latency per passage.
