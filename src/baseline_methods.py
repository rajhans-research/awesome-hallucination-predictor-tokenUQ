"""
src/baseline_methods.py
-----------------------
Module 3: Verified Baseline Uncertainty & Consistency Detectors.

Implements:
1. Single-Pass Token Uncertainty Baselines (from GPT-3 token logprobs):
   - `token_mean`: Mean negative log-probability across sentence tokens.
   - `token_min`: Maximum negative log-probability (weakest-link token probability).
   - `token_entropy5`: Mean top-5 normalized predictive entropy (or perplexity 2^H).
   - `token_max_entropy5`: Maximum top-5 normalized predictive entropy (or perplexity 2^H).
2. Multi-Pass Black-Box Consistency Comparator:
   - `selfcheck_ngram`: Official SelfCheckGPT n-gram consistency over N stochastic samples,
     using cached lightweight `spacy.blank("en") + sentencizer` to eliminate redundant disk I/O
     while preserving exact tokenization parity.

All methods return scores oriented such that HIGHER score = HIGHER hallucination likelihood.
"""

import time
from typing import Any, Dict, List, Tuple
import numpy as np
import spacy
import selfcheckgpt.modeling_ngram as _sc_ngram_mod
from selfcheckgpt.modeling_selfcheck import SelfCheckNgram


_CACHED_BLANK_NLP = None


def _get_lightweight_nlp():
    """Returns a cached lightweight spaCy English tokenizer + sentencizer."""
    global _CACHED_BLANK_NLP
    if _CACHED_BLANK_NLP is None:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        _CACHED_BLANK_NLP = nlp
    return _CACHED_BLANK_NLP


def score_token_mean(sentence_entry: Dict[str, Any]) -> float:
    """
    Mean negative log-probability of tokens in a sentence.
    Higher value indicates higher uncertainty (lower average probability).
    """
    logprobs = sentence_entry["token_logprobs"]
    if not logprobs:
        return 0.0
    return float(-np.mean(logprobs))


def score_token_min(sentence_entry: Dict[str, Any]) -> float:
    """
    Maximum negative log-probability (equivalent to minimum token probability)
    across tokens in a sentence.
    Higher value indicates a lower-probability weakest-link token.
    """
    logprobs = sentence_entry["token_logprobs"]
    if not logprobs:
        return 0.0
    return float(-np.min(logprobs))


def _compute_token_entropies5(
    top_logprobs: List[Dict[str, float]],
    use_perplexity: bool = False
) -> List[float]:
    """
    Computes normalized top-5 Shannon entropy (base 2) or perplexity (2^H)
    for each token position using pure NumPy for minimal overhead.
    """
    vals: List[float] = []
    for top5 in top_logprobs:
        if not top5:
            vals.append(0.0)
            continue
        lp_arr = np.fromiter(top5.values(), dtype=np.float64)
        lp_shifted = lp_arr - np.max(lp_arr)
        probs = np.exp(lp_shifted)
        probs = probs / np.sum(probs)
        nonzero = probs[probs > 0.0]
        h = float(-np.sum(nonzero * np.log2(nonzero)))
        vals.append(float(2.0 ** h) if use_perplexity else h)
    return vals


def score_token_entropy5(sentence_entry: Dict[str, Any], use_perplexity: bool = False) -> float:
    """
    Mean top-5 predictive entropy (or perplexity 2^H) across sentence tokens.
    """
    vals = _compute_token_entropies5(sentence_entry["top_logprobs"], use_perplexity=use_perplexity)
    return float(np.mean(vals)) if vals else 0.0


def score_token_max_entropy5(sentence_entry: Dict[str, Any], use_perplexity: bool = False) -> float:
    """
    Maximum top-5 predictive entropy (or perplexity 2^H) across sentence tokens.
    """
    vals = _compute_token_entropies5(sentence_entry["top_logprobs"], use_perplexity=use_perplexity)
    return float(np.max(vals)) if vals else 0.0


def run_single_pass_baselines(
    records: List[Dict[str, Any]]
) -> Tuple[Dict[str, Dict[Tuple[int, int], float]], Dict[str, float]]:
    """
    Evaluates all single-pass token-level baselines across all passages and records
    exact scoring wall-clock latencies.

    Returns:
        scores_by_method: Dict[method_name -> Dict[(passage_idx, sentence_idx) -> float]]
        timings_sec: Dict[method_name -> float total wall-clock seconds]
    """
    methods = {
        "token_mean": score_token_mean,
        "token_min": score_token_min,
        "token_entropy5": lambda s: score_token_entropy5(s, use_perplexity=False),
        "token_ppl5": lambda s: score_token_entropy5(s, use_perplexity=True),
        "token_max_ppl5": lambda s: score_token_max_entropy5(s, use_perplexity=True),
    }

    scores_by_method: Dict[str, Dict[Tuple[int, int], float]] = {m: {} for m in methods}
    timings_sec: Dict[str, float] = {}

    for m_name, fn in methods.items():
        t0 = time.perf_counter()
        for p_idx, rec in enumerate(records):
            for s_idx, sent in enumerate(rec["sentences"]):
                scores_by_method[m_name][(p_idx, s_idx)] = fn(sent)
        timings_sec[m_name] = time.perf_counter() - t0

    return scores_by_method, timings_sec


def run_selfcheck_ngram_baseline(
    records: List[Dict[str, Any]],
    num_samples: int = 20,
    n: int = 1,
    seed: int = 42,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float], float]:
    """
    Runs the official `SelfCheckNgram` detector across all passages using `num_samples`
    stochastic samples per passage. Uses a cached lightweight spaCy tokenizer to avoid
    reloading `en_core_web_sm` from disk 238 times.

    Returns:
        avg_scores: Dict[(passage_idx, sentence_idx) -> avg_neg_logprob]
        max_scores: Dict[(passage_idx, sentence_idx) -> max_neg_logprob]
        elapsed_sec: Total wall-clock seconds for scoring.
    """
    nlp = _get_lightweight_nlp()
    _sc_ngram_mod.spacy.load = lambda *args, **kwargs: nlp

    rng = np.random.default_rng(seed)
    scorer = SelfCheckNgram(n=n)
    avg_scores: Dict[Tuple[int, int], float] = {}
    max_scores: Dict[Tuple[int, int], float] = {}

    t0 = time.perf_counter()
    for p_idx, rec in enumerate(records):
        sentences = [s["sentence"] for s in rec["sentences"]]
        all_samples = rec["samples"]
        if num_samples < len(all_samples):
            chosen_idx = rng.choice(len(all_samples), size=num_samples, replace=False)
            sampled_passages = [all_samples[i] for i in chosen_idx]
        else:
            sampled_passages = all_samples

        res = scorer.predict(
            sentences=sentences,
            passage=rec["passage_text"],
            sampled_passages=sampled_passages,
        )
        sent_avg = res["sent_level"]["avg_neg_logprob"]
        sent_max = res["sent_level"]["max_neg_logprob"]
        for s_idx, (a_val, m_val) in enumerate(zip(sent_avg, sent_max)):
            avg_scores[(p_idx, s_idx)] = float(a_val)
            max_scores[(p_idx, s_idx)] = float(m_val)

    elapsed_sec = time.perf_counter() - t0
    return avg_scores, max_scores, elapsed_sec
