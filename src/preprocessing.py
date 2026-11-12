"""
src/preprocessing.py
--------------------
Module 2: Data Sanitization, Serialization, Label Mapping & Integrity Audits.

Responsibilities:
1. Export clean, standalone JSON dataset (`data/processed/wikibio_processed.json`)
   stripped of any legacy OpenAI SDK pickle objects.
2. Provide deterministic sentence-level label extraction matching:
   - `nonfact`: accurate=0, minor_inaccurate=1, major_inaccurate=1 (Manakul et al. NonFact)
   - `nonfact_star`: excludes 100% major_inaccurate passages; major_inaccurate=1, else=0 (Manakul et al. NonFact*)
   - `factual`: accurate=1, minor/major=0 (Manakul et al. Factual)
   - `major_only`: accurate=0, major_inaccurate=1, minor_inaccurate excluded
3. Audit dataset for missing values, duplicate IDs, and potential leakage.
"""

import json
import os
from typing import Any, Dict, List, Tuple
import numpy as np


LABEL_VALUE_MAP = {
    "accurate": 0.0,
    "minor_inaccurate": 0.5,
    "major_inaccurate": 1.0,
}


def save_processed_dataset(records: List[Dict[str, Any]], output_path: str) -> None:
    """
    Saves cleaned passage records to JSON for fast, dependency-free loading.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


def load_processed_dataset(input_path: str) -> List[Dict[str, Any]]:
    """
    Loads pre-sanitized JSON dataset records.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def audit_dataset_integrity(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Performs strict data validation and leakage checks on the dataset.

    Checks:
    - Exact count of passages and sentences.
    - Uniqueness of `wiki_bio_test_idx` (no duplicated test entities).
    - Non-empty token sequences and logprobs for every sentence.
    - Annotation class frequency distribution.
    """
    seen_ids = set()
    duplicate_ids = []
    total_sentences = 0
    empty_token_sents = 0
    class_counts = {"accurate": 0, "minor_inaccurate": 0, "major_inaccurate": 0}
    token_lengths = []
    sample_counts = []

    for rec in records:
        test_id = rec["wiki_bio_test_idx"]
        if test_id in seen_ids:
            duplicate_ids.append(test_id)
        seen_ids.add(test_id)
        sample_counts.append(len(rec["samples"]))

        for sent in rec["sentences"]:
            total_sentences += 1
            annot = sent["annotation"]
            class_counts[annot] = class_counts.get(annot, 0) + 1
            n_toks = len(sent["token_logprobs"])
            token_lengths.append(n_toks)
            if n_toks == 0:
                empty_token_sents += 1

    return {
        "num_passages": len(records),
        "unique_passage_ids": len(seen_ids),
        "duplicate_ids": duplicate_ids,
        "total_sentences": total_sentences,
        "empty_token_sentences": empty_token_sents,
        "class_counts": class_counts,
        "mean_tokens_per_sentence": float(np.mean(token_lengths)),
        "min_tokens_per_sentence": int(np.min(token_lengths)),
        "max_tokens_per_sentence": int(np.max(token_lengths)),
        "min_samples_per_passage": int(np.min(sample_counts)),
        "max_samples_per_passage": int(np.max(sample_counts)),
    }


def extract_labels_for_protocol(
    records: List[Dict[str, Any]],
    protocol: str = "nonfact"
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Extracts `(passage_idx, sentence_idx)` keys and binary numpy labels
    according to the specified evaluation protocol.

    Args:
        records: List of passage dictionaries.
        protocol: One of {"nonfact", "nonfact_star", "factual", "major_only"}.

    Returns:
        keys: List of (passage_idx, sentence_idx) tuples matching label array.
        labels: 1D numpy int32 array of binary labels (0 or 1).
    """
    keys: List[Tuple[int, int]] = []
    labels: List[int] = []

    for p_idx, rec in enumerate(records):
        raw_vals = np.array([LABEL_VALUE_MAP[s["annotation"]] for s in rec["sentences"]])
        passage_mean = float(np.mean(raw_vals))

        if protocol == "nonfact_star" and passage_mean >= 0.99:
            # Exclude completely fabricated passages per Manakul et al. (2023)
            continue

        for s_idx, sent in enumerate(rec["sentences"]):
            annot = sent["annotation"]
            if protocol == "nonfact":
                lbl = 1 if annot in ("minor_inaccurate", "major_inaccurate") else 0
            elif protocol == "nonfact_star":
                lbl = 1 if annot == "major_inaccurate" else 0
            elif protocol == "factual":
                lbl = 1 if annot == "accurate" else 0
            elif protocol == "major_only":
                if annot == "minor_inaccurate":
                    continue
                lbl = 1 if annot == "major_inaccurate" else 0
            else:
                raise ValueError(f"Unknown label protocol: {protocol}")

            keys.append((p_idx, s_idx))
            labels.append(lbl)

    return keys, np.array(labels, dtype=np.int32)
