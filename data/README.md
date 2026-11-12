# Data Directory (`data/`)

## Dataset Information
- **Name:** `potsawee/wiki_bio_gpt3_hallucination`
- **Origin:** EMNLP 2023 paper *SelfCheckGPT* (Manakul et al.)
- **Format:** Parquet / Arrow via Hugging Face `datasets`
- **License:** MIT License

## Directory Contents
- `cache/`: Local cache of downloaded partitions to ensure offline repeatability.
- `processed/`: Serialized structured JSONL containing:
  - `concept`: Subject name
  - `gpt3_text`: Generated biography
  - `sentences`: Individual sentences
  - `sentence_labels`: Integer labels (0 = Accurate, 1 = Major Inaccurate)
  - `token_logprobs`: Token strings and associated log-probabilities
  - `samples`: 20 sampled alternate completions
