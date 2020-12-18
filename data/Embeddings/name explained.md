> This document explains the naming and configuration of each embeddings

- **finBERT_with_special_tokens**
    - Sentences less or equal than 5 tokens (include special tokens) are removed.
    - The original sentence embeddings, *not* stacked by `transcriptid`. e.g., `{sentence_id: {seq_len:int, embedding:tensor}}`
    - *Include* special tokens: `[CLS]` and `[EOS]`

- **finBERT_with_special_tokens_all**
    - Sentences less or equal than 5 tokens (include special tokens) are removed.
    - The processed embeddings, *stacked* by `transcriptid`. e.g., `{transcriptid: tensor}`
    - *Include* special tokens: `[CLS]` and `[EOS]`
    - *All* component types are preserved.

- **finBERT_without_special_tokens**
    - Sentences less or equal than 5 tokens (include special tokens) are removed.
    - The original sentence embeddings, *not* stacked by `transcriptid`. e.g., `{sentence_id: {seq_len:int, embedding:tensor}}`
    - Does *not* include special tokens: `[CLS]` and `[EOS]`

- **finBERT_without_special_tokens_all**
    - Sentences less or equal than 5 tokens (include special tokens) are removed.
    - The processed embeddings, *stacked* by `transcriptid`. e.g., `{transcriptid: tensor}`
    - Does *not* include special tokens: `[CLS]` and `[EOS]`
    - *All* component types are preserved.