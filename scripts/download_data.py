"""
scripts/download_data.py
------------------------
Downloads/ingests the WikiBio GPT-3 Hallucination dataset, safely aligns GPT-3
token log-probabilities, audits integrity/leakage, and saves clean processed JSON.
"""

import os
import sys
import json
import time

# Allow importing from project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_benchmark_with_logprobs
from src.preprocessing import save_processed_dataset, audit_dataset_integrity


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()

    cache_dir = os.path.join(PROJECT_ROOT, "data", "cache")
    logprob_dir = os.path.join(
        PROJECT_ROOT, "data", "gpt3_logprobs", "gpt3_davinci_003_FEB15_v1_logprob_export"
    )
    output_path = os.path.join(PROJECT_ROOT, "data", "processed", "wikibio_processed.json")

    print("[1/3] Loading Hugging Face dataset and aligning GPT-3 logprobs...")
    records = load_benchmark_with_logprobs(cache_dir=cache_dir, logprob_dir=logprob_dir)

    print("[2/3] Auditing dataset integrity & leakage checks...")
    audit = audit_dataset_integrity(records)
    print(json.dumps(audit, indent=2))

    assert len(audit["duplicate_ids"]) == 0, f"Found duplicate IDs: {audit['duplicate_ids']}"
    assert audit["empty_token_sentences"] == 0, "Found sentences with 0 aligned tokens"

    print(f"[3/3] Saving sanitized dataset to {output_path}...")
    save_processed_dataset(records, output_path)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s. Clean dataset ready for Gate G2 evaluation.")


if __name__ == "__main__":
    main()
