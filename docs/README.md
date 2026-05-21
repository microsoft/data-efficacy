# Documentation

This folder keeps paper-specific documentation for the Data Efficacy codebase. The root README introduces the shared motivation and pipeline, while each page here records the details needed for a specific paper.

## Papers

| Work | Status | Page |
| --- | --- | --- |
| Data Efficacy for Language Model Training (DELT) | arXiv 2025 | [delt.md](./delt.md) |
| Demystifying Data Organization for Enhanced LLM Training | ACL 2026 | [data_organization_acl2026.md](./data_organization_acl2026.md) |

## Adding New Work

When adding a new data efficacy paper:

- Add a new page under `docs/`.
- Put paper-specific figures under `docs/assets/<paper_slug>/`.
- Link the page from both this index and the root [README](../README.md).
- Keep reusable code in the shared modules, such as `data_scoring`, `data_selection`, `data_ordering`, `model_train`, or `model_eval`.
