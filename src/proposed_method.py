"""
src/proposed_method.py
----------------------
Module 5: Proposed Research Contribution.

Implements:
1. Factual Keyword-Filtered Token Uncertainty (`token_keyword`):
   Mitigates the surface-form bottleneck (Rajhans 2026, Sec 4.2) by stripping
   high-frequency syntactic functional stopwords and punctuation, computing
   negative log-probabilities strictly over informative semantic content tokens
   (proper nouns, entities, dates, numbers, descriptors).
2. Selective Peak-Content Fusion (`token_hybrid`):
   Fuses content-word uncertainty with weakest-link peak uncertainty (`token_min`).
3. Multi-Pass Consistency Scaled by Sample Budget ($N \in \{2, 3, 5, 10, 20\}$):
   - `SelfCheck-1gram`
   - `SelfCheck-2gram`
   - `SelfCheck-TFIDF` (Lexical-semantic overlap inconsistency)
"""

import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import selfcheckgpt.modeling_ngram as _sc_ngram_mod
from selfcheckgpt.modeling_selfcheck import SelfCheckNgram
from src.baseline_methods import _get_lightweight_nlp


# Standard closed-class English stopwords for zero-dependency, ultra-fast filtering
ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
    "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves"
}


def is_factual_content_token(token_str: str) -> bool:
    """
    Checks if a token represents an informative factual content token
    (e.g., proper noun, year, entity, descriptor) rather than functional syntax.
    """
    cleaned = token_str.strip().lower()
    if not cleaned or not re.search(r"\w", cleaned):
        return False
    if cleaned in ENGLISH_STOPWORDS:
        return False
    return True


def score_token_keyword(sentence_entry: Dict[str, Any]) -> float:
    """
    Computes negative log-probability strictly over factual/content tokens.
    If no content tokens exist, falls back gracefully to sentence mean.
    """
    tokens = sentence_entry["tokens"]
    logprobs = sentence_entry["token_logprobs"]
    if not logprobs:
        return 0.0

    content_logprobs = [
        lp for t, lp in zip(tokens, logprobs) if is_factual_content_token(t)
    ]
    if content_logprobs:
        return float(-np.mean(content_logprobs))
    return float(-np.mean(logprobs))


def score_token_hybrid(
    sentence_entry: Dict[str, Any], alpha: float = 0.5
) -> float:
    """
    Fuses content-word uncertainty with weakest-link peak uncertainty:
    Score = alpha * Keyword_Uncertainty + (1 - alpha) * Min_Token_Uncertainty.
    """
    s_key = score_token_keyword(sentence_entry)
    logprobs = sentence_entry["token_logprobs"]
    s_min = float(-np.min(logprobs)) if logprobs else 0.0
    return float(alpha * s_key + (1.0 - alpha) * s_min)


def compute_selfcheck_tfidf_inconsistency(
    sentences: List[str], sampled_passages: List[str]
) -> List[float]:
    """
    Evaluates sentence consistency against sampled passages via TF-IDF cosine distance.
    Returns 1.0 - mean_cosine_similarity across sampled passages.
    """
    if not sampled_passages or not sentences:
        return [0.5 for _ in sentences]

    corpus = sampled_passages + sentences
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return [0.5 for _ in sentences]

    n_samples = len(sampled_passages)
    sample_vecs = tfidf_matrix[:n_samples]
    sent_vecs = tfidf_matrix[n_samples:]

    sim_matrix = cosine_similarity(sent_vecs, sample_vecs) # (S, N)
    mean_sim = np.mean(sim_matrix, axis=1) # (S,)
    inconsistency = [float(1.0 - s) for s in mean_sim]
    return inconsistency


def run_proposed_single_pass_methods(
    records: List[Dict[str, Any]]
) -> Tuple[Dict[str, Dict[Tuple[int, int], float]], Dict[str, float]]:
    """
    Computes scores for proposed single-pass methods across all passages.
    """
    methods = {
        "token_keyword": score_token_keyword,
        "token_hybrid_a50": lambda s: score_token_hybrid(s, alpha=0.5),
        "token_hybrid_a70": lambda s: score_token_hybrid(s, alpha=0.7),
    }

    scores: Dict[str, Dict[Tuple[int, int], float]] = {m: {} for m in methods}
    timings: Dict[str, float] = {}

    for m_name, fn in methods.items():
        t0 = time.perf_counter()
        for p_idx, rec in enumerate(records):
            for s_idx, sent in enumerate(rec["sentences"]):
                scores[m_name][(p_idx, s_idx)] = fn(sent)
        timings[m_name] = time.perf_counter() - t0

    return scores, timings


def run_budget_sweep_multi_pass(
    records: List[Dict[str, Any]],
    sample_budgets: List[int] = [2, 3, 5, 10, 20],
    seed: int = 42,
) -> Tuple[Dict[str, Dict[Tuple[int, int], float]], Dict[str, float]]:
    """
    Runs multi-pass consistency detectors across varying sample budgets N.
    """
    nlp = _get_lightweight_nlp()
    _sc_ngram_mod.spacy.load = lambda *args, **kwargs: nlp

    rng = np.random.default_rng(seed)
    scorer_1g = SelfCheckNgram(n=1)
    scorer_2g = SelfCheckNgram(n=2)

    scores: Dict[str, Dict[Tuple[int, int], float]] = {}
    timings: Dict[str, float] = {}

    for N in sample_budgets:
        m_1g = f"selfcheck_1gram_max_N{N}"
        m_2g = f"selfcheck_2gram_max_N{N}"
        m_tfidf = f"selfcheck_tfidf_N{N}"

        scores[m_1g] = {}
        scores[m_2g] = {}
        scores[m_tfidf] = {}

        t0_1g, t0_2g, t0_tf = 0.0, 0.0, 0.0

        for p_idx, rec in enumerate(records):
            sents = [s["sentence"] for s in rec["sentences"]]
            all_samples = rec["samples"]
            if N < len(all_samples):
                chosen = rng.choice(len(all_samples), size=N, replace=False)
                sub_samples = [all_samples[i] for i in chosen]
            else:
                sub_samples = all_samples

            # 1-gram
            t_a = time.perf_counter()
            res_1g = scorer_1g.predict(sents, rec["passage_text"], sub_samples)
            t0_1g += time.perf_counter() - t_a
            for s_idx, val in enumerate(res_1g["sent_level"]["max_neg_logprob"]):
                scores[m_1g][(p_idx, s_idx)] = float(val)

            # 2-gram
            t_b = time.perf_counter()
            res_2g = scorer_2g.predict(sents, rec["passage_text"], sub_samples)
            t0_2g += time.perf_counter() - t_b
            for s_idx, val in enumerate(res_2g["sent_level"]["max_neg_logprob"]):
                scores[m_2g][(p_idx, s_idx)] = float(val)

            # TF-IDF
            t_c = time.perf_counter()
            tf_scores = compute_selfcheck_tfidf_inconsistency(sents, sub_samples)
            t0_tf += time.perf_counter() - t_c
            for s_idx, val in enumerate(tf_scores):
                scores[m_tfidf][(p_idx, s_idx)] = float(val)

        timings[m_1g] = t0_1g
        timings[m_2g] = t0_2g
        timings[m_tfidf] = t0_tf

    return scores, timings
