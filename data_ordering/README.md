# Data Ordering

This module contains score-based data ordering methods for the Data Efficacy pipeline.

Implemented methods:

- `shuffle`: random ordering baseline.
- `sorting`: score sorting / curriculum learning baseline.
- `folding`: Folding Ordering (FO).
- `zigzag`: Zig-zag Ordering (ZIG).
- `segment`: Segment Ordering (SEG).
- `stair`: Stair Ordering (STR).
- `saw`: Saw Ordering (SAW).

Run:

```bash
bash data_ordering/entry.sh \
  data/scored_data.jsonl \
  data/ordered_saw.jsonl \
  saw \
  data_ordering/config/saw.yaml
```

For paper-specific details, see:

- [Data Efficacy for Language Model Training](../docs/data_efficacy_for_language_model_training.md)
- [Demystifying Data Organization for Enhanced LLM Training](../docs/demystifying_data_organization_for_enhanced_llm_training.md)
