"""
src/data_loader.py
------------------
Module 1: Dataset Ingestion & Safe Log-Probability Unpickling.

Responsibilities:
1. Load the Hugging Face benchmark `potsawee/wiki_bio_gpt3_hallucination`.
2. Safely unpickle the original OpenAI GPT-3 (`text-davinci-003`) completion objects
   using a custom `Unpickler` compatible with OpenAI SDK v3+ while stripping all
   non-research header metadata (e.g., legacy API keys in `_init_args`).
3. Align every annotated sentence with its exact token indices using `text_offset`.
"""

import os
import pickle
from typing import Any, Dict, List, Optional
from datasets import load_dataset


class _SafeOpenAIObject(dict):
    """
    Lightweight dictionary wrapper replacing legacy `openai.openai_object.OpenAIObject`
    during unpickling. Intentionally discards positional `_init_args` to guarantee
    no legacy API keys or organization IDs persist in memory or processed files.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(kwargs)

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.update(state)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


class _LegacyOpenAIUnpickler(pickle.Unpickler):
    """Custom Unpickler remapping legacy OpenAI SDK v0.27 objects."""
    def find_class(self, module: str, name: str) -> Any:
        if module == "openai.openai_object" and name == "OpenAIObject":
            return _SafeOpenAIObject
        return super().find_class(module, name)


def load_clean_logprob_bin(bin_path: str) -> Dict[str, Any]:
    """
    Loads a single GPT-3 completion `.bin` file and extracts strictly the
    linguistic and probability fields required for research.

    Input:
        bin_path (str): Path to `<wiki_bio_test_idx>.bin`.

    Output (dict):
        {
            "text": str,
            "tokens": List[str],               # length T
            "token_logprobs": List[float],     # length T
            "top_logprobs": List[Dict[str, float]], # length T, each dict <= 5 items
            "text_offset": List[int]           # length T
        }
    """
    with open(bin_path, "rb") as f:
        raw_obj = _LegacyOpenAIUnpickler(f).load()

    choice = raw_obj["choices"][0]
    lp = choice["logprobs"]

    # Clean top_logprobs dicts strictly to token -> float logprob
    clean_top_logprobs = []
    for item in lp["top_logprobs"]:
        clean_dict = {str(k): float(v) for k, v in item.items() if k != "_init_args"}
        clean_top_logprobs.append(clean_dict)

    return {
        "text": str(choice["text"]),
        "tokens": [str(t) for t in lp["tokens"]],
        "token_logprobs": [float(v) if v is not None else 0.0 for v in lp["token_logprobs"]],
        "top_logprobs": clean_top_logprobs,
        "text_offset": [int(o) for o in lp["text_offset"]],
    }


def align_sentences_to_tokens(
    sentences: List[str],
    full_text: str,
    text_offsets: List[int]
) -> List[List[int]]:
    """
    Maps each sentence in `sentences` to its constituent token indices [i_1, ..., i_k]
    using character offsets (`text_offset`) relative to the first generated token.

    Inputs:
        sentences: List[str] of length S.
        full_text: str representing `choices[0]['text']`.
        text_offsets: List[int] of length T (character positions in prompt+text).

    Output:
        List[List[int]] of length S, where element j is the list of token indices
        belonging to sentence j.
    """
    if not text_offsets:
        return [[] for _ in sentences]

    base_offset = text_offsets[0]
    rel_offsets = [off - base_offset for off in text_offsets]

    aligned_indices: List[List[int]] = []
    cursor = 0

    for sent in sentences:
        pos = full_text.find(sent, cursor)
        if pos == -1:
            pos = full_text.find(sent.strip(), cursor)
        if pos == -1:
            pos = full_text.find(sent.strip())

        if pos == -1:
            aligned_indices.append([])
            continue

        end_pos = pos + len(sent)
        tok_idx = [i for i, off in enumerate(rel_offsets) if pos <= off < end_pos]
        aligned_indices.append(tok_idx)
        cursor = end_pos

    return aligned_indices


def load_benchmark_with_logprobs(
    cache_dir: str = "./data/cache",
    logprob_dir: str = "./data/gpt3_logprobs/gpt3_davinci_003_FEB15_v1_logprob_export",
    max_passages: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Ingests the Hugging Face `potsawee/wiki_bio_gpt3_hallucination` evaluation split
    and merges it with aligned GPT-3 token log-probabilities.

    Output:
        List of dicts (length 238 by default), each containing passage metadata,
        20 stochastic samples, aligned sentence texts, annotations, and token logprobs.
    """
    os.makedirs(cache_dir, exist_ok=True)
    hf_ds = load_dataset(
        "potsawee/wiki_bio_gpt3_hallucination",
        split="evaluation",
        cache_dir=cache_dir
    )

    num_rows = len(hf_ds) if max_passages is None else min(len(hf_ds), max_passages)
    records: List[Dict[str, Any]] = []

    for row_idx in range(num_rows):
        row = hf_ds[row_idx]
        test_idx = int(row["wiki_bio_test_idx"])
        sentences = [s.strip() for s in row["gpt3_sentences"]]
        annotations = list(row["annotation"])
        samples = list(row["gpt3_text_samples"])

        assert len(sentences) == len(annotations), (
            f"Mismatch at passage {test_idx}: {len(sentences)} sentences vs {len(annotations)} annotations"
        )

        bin_path = os.path.join(logprob_dir, f"{test_idx}.bin")
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Missing logprob binary file: {bin_path}")

        lp_data = load_clean_logprob_bin(bin_path)
        token_spans = align_sentences_to_tokens(
            row["gpt3_sentences"], lp_data["text"], lp_data["text_offset"]
        )

        sentence_entries = []
        for s_idx, (sent_text, annot, tok_ids) in enumerate(zip(sentences, annotations, token_spans)):
            assert len(tok_ids) > 0, (
                f"Zero tokens aligned for passage {test_idx}, sentence {s_idx}: '{sent_text}'"
            )
            sent_tokens = [lp_data["tokens"][i] for i in tok_ids]
            sent_logprobs = [lp_data["token_logprobs"][i] for i in tok_ids]
            sent_top_logprobs = [lp_data["top_logprobs"][i] for i in tok_ids]

            sentence_entries.append({
                "sentence_idx": s_idx,
                "sentence": sent_text,
                "annotation": annot,
                "tokens": sent_tokens,
                "token_logprobs": sent_logprobs,
                "top_logprobs": sent_top_logprobs,
            })

        records.append({
            "passage_id": row_idx,
            "wiki_bio_test_idx": test_idx,
            "passage_text": row["gpt3_text"],
            "wiki_bio_text": row["wiki_bio_text"],
            "sentences": sentence_entries,
            "samples": samples,
        })

    return records
