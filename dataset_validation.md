# Dataset Validation & Protocol Specification (`dataset_validation.md`)

**Target Benchmark:** `potsawee/wiki_bio_gpt3_hallucination`  
**Originating Publication:** Potsawee Manakul, Adian Liusie, and Mark J. F. Gales. *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* EMNLP 2023. (doi:10.18653/v1/2023.emnlp-main.557) `[VERIFIED]`  
**Host / Repository:** [Hugging Face Datasets: potsawee/wiki_bio_gpt3_hallucination](https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination) `[VERIFIED]`  
**License:** MIT License `[VERIFIED]`

---

## 1. Dataset Provenance & Structure

The dataset was constructed to evaluate hallucination detection methods on long-form natural language generation without relying on external databases. 

### Generation Protocol:
- **Seed Prompts:** 238 individual names sampled randomly from the WikiBio test set.
- **Generation Model:** OpenAI GPT-3 (`text-davinci-003`), using standard prompt:  
  `"This is a Wikipedia passage about {concept}:"`
- **Primary Response:** Generated with greedy decoding / low temperature ($T \approx 0.0\text{--}0.2$), yielding a fluent, factual-sounding biography.
- **Stochastic Samples:** For each concept, 20 additional responses were generated independently at temperature $T=1.0$ to provide alternative generations for consistency checking.
- **Token Information:** Log-probabilities of every generated token in the primary response were recorded directly from the API output.

---

## 2. Annotation & Ground Truth Label Schema

Human annotators labeled each individual sentence within the generated passages against the authoritative Wikipedia reference article for the corresponding subject:

| Label Category | Code / Value | Definition in Benchmark | Binary Label Mapping (Primary) | Sensitivity Alternative |
| :--- | :---: | :--- | :---: | :---: |
| **Accurate** | `accurate` | All factual assertions in the sentence are fully verified by reference evidence. | `0` (Factual) | `0` (Factual) |
| **Minor Inaccurate** | `minor_inaccurate` | Sentence is mostly factual but contains minor errors (e.g., slight date mismatch, imprecise role). | Excluded from primary binary / or `0` | `1` (Hallucinated) |
| **Major Inaccurate** | `major_inaccurate` | Sentence contains significant factual fabrications (e.g., invented career, false alma mater, wrong nationality). | `1` (Hallucinated) | `1` (Hallucinated) |

### Dataset Partitioning & Leakage Prevention:
- **Total Passages:** 238 entities.
- **Total Annotated Sentences:** 1,908 sentences.
- **Zero-Leakage Protocol:** All uncertainty and consistency methods are zero-shot / unsupervised detectors that compute confidence directly from token distributions or sample consistency. No detector parameters are trained or fitted on the ground truth labels. For threshold-dependent metrics (e.g. Precision@Recall), a strict 5-fold stratified cross-validation protocol is applied where thresholds are determined only on the train fold and evaluated on the test fold.

---

## 3. Data Integrity & Verification Checklist

- [x] Official repository located and confirmed live on Hugging Face.
- [x] MIT License permits open academic inspection, modification, and reproduction.
- [x] Schema includes passage text, sentence segmentations, human ratings, token logprobs, and 20 alternate samples.
- [x] Exact token boundaries and log-probabilities match sentence character spans.
- [x] No data contamination or training leakage risks present.
