# Data Efficacy

<p align="center">
 <img src="https://img.shields.io/badge/Task-Data_Efficacy-orange" alt="Task" />
 <img src="https://img.shields.io/badge/License-MIT-blue" alt="License" />
</p>

Large-scale model training benefits from data at scale, but the value of a dataset also depends on how effectively it is used. **Data Efficacy** studies how to turn available data into stronger training signal by scoring samples, selecting useful subsets, and organizing them into effective training sequences.

## Introduction

Large-scale model training depends heavily on data curation. Many data efficiency methods compute expensive sample-level scores for quality, difficulty, learnability, or relevance, but these scores are often used only once for filtering.

Data Efficacy aims to reuse such scores more fully across the training pipeline. In this repository, that shared pipeline is organized around four reusable stages:

- **Data Scoring** estimates sample-level utility.
- **Data Selection** chooses useful subsets under a data or compute budget.
- **Data Ordering** organizes selected samples into an effective training sequence.
- **Model Training and Evaluation** measure whether the curated data improves downstream performance.

![Data efficacy pipeline](./figures/data_efficacy_paradigm.png)

## News

- **2026/05**: Added the ACL 2026 follow-up work **Demystifying Data Organization for Enhanced LLM Training**, with new data organization methods under `data_ordering`.
- **2025/08**: Released the codebase for general-domain pre-training.
- **2025/06**: Released **Data Efficacy for Language Model Training (DELT)** on arXiv.

## Works

### Demystifying Data Organization for Enhanced LLM Training ([Paper](https://openreview.net/forum?id=i409rQuIfB) | [README](./docs/data_organization_acl2026.md))

This work studies how to organize scored training data (data ordering) and introduces practical guidances for boundary sharpening, cyclic scheduling, curriculum continuity, and local diversity.

### Data Efficacy for Language Model Training ([Paper](https://arxiv.org/abs/2506.21545) | [README](./docs/delt.md))

This work introduces a data efficacy pipeline for language model training that reuses sample-level scores across data scoring, data selection, and data ordering.

## Repo Structure

```text
.
├── data_scoring/      # Compute sample-level scores, including LQS and KenLM-based scoring.
├── data_selection/    # Select subsets with top-r, top-k, or threshold methods.
├── data_ordering/     # Organize scored data with sorting, folding, zig-zag, segment, STR, and SAW.
├── model_train/       # Train models on curated data.
├── model_eval/        # Evaluate trained models.
├── docs/              # Paper-specific documentation and assets.
├── figures/           # Figures used by repository documentation.
└── tests/             # Lightweight tests for reusable components.
```

## Installation

```bash
conda create -n data_efficacy python=3.10 -y
conda activate data_efficacy
pip install -r requirements.txt
```

For lightweight data ordering only, `numpy` and `pyyaml` are sufficient.

## Preparation

<details open>
<summary>Environment Variables</summary>

```bash
export HF_TOKEN="<your_huggingface_token>"
export WANDB_API_KEY="<your_wandb_apikey>"
```
</details>

<details open>
<summary>Dataset</summary>

```bash
python utils.py --content dataset --id $HF_DATASET_ID --save-dir $OUTPUT_DATA_PATH

# Example:
python utils.py \
  --content dataset \
  --id togethercomputer/RedPajama-Data-1T \
  --save-dir data/source-cc-1b.jsonl \
  --data-name common_crawl \
  --split-name train \
  --sample-size 500000
```

You can also use your own JSONL dataset.
</details>

<details open>
<summary>Model</summary>

```bash
python utils.py --content model --id $HF_MODEL_ID --save-dir $OUTPUT_MODEL_PATH

# Example:
python utils.py \
  --content model \
  --id Data-Selection/BSL-160M \
  --save-dir models/mistral-160m
```
</details>

## Pipeline Usage

The repository exposes each stage through a separate entry script. You can run the full scoring-selection-ordering-training pipeline or reuse only the stages needed by a specific paper.

<details open>
<summary>Data Scoring</summary>

Existing scoring methods include **Learnability-Quality Score** (`lqs`) and Perplexity (`kenlm`). For LQS details, see [data_scoring/lqs/README.md](./data_scoring/lqs/README.md).

```bash
bash data_scoring/entry.sh $INPUT_DATA_PATH $OUTPUT_DATA_PATH $METHOD $CONFIG_PATH

# Example:
bash data_scoring/entry.sh \
  data/source-cc-1b.jsonl \
  data/source-cc-1b_scored-lqs.jsonl \
  lqs \
  data_scoring/config/lqs.yaml
```
</details>

<details open>
<summary>Data Selection</summary>

Existing selection methods include **Top-R** (`top-r`), Threshold (`threshold`), and Top-K (`top-k`).

```bash
bash data_selection/entry.sh $INPUT_DATA_PATH $OUTPUT_DATA_PATH $METHOD $CONFIG_PATH

# Example:
bash data_selection/entry.sh \
  data/source-cc-1b_scored-lqs.jsonl \
  data/source-cc-1b_scored-lqs_selected-r1.0.jsonl \
  top-r \
  data_selection/config/top-r.yaml
```
</details>

<details open>
<summary>Data Ordering</summary>

Existing ordering methods include Sorting (`sorting`), Folding Ordering (`folding`), Zig-zag Ordering (`zigzag`), Segment Ordering (`segment`), Stair Ordering / STR (`stair`), Saw Ordering / SAW (`saw`), and Shuffle (`shuffle`). For the ACL 2026 data organization work, see [docs/data_organization_acl2026.md](./docs/data_organization_acl2026.md).

```bash
bash data_ordering/entry.sh $INPUT_DATA_PATH $OUTPUT_DATA_PATH $METHOD $CONFIG_PATH

# Example:
bash data_ordering/entry.sh \
  data/source-cc-1b_scored-lqs_selected-r1.0.jsonl \
  data/source-cc-1b_scored-lqs_selected-r1.0_ordered-saw.jsonl \
  saw \
  data_ordering/config/saw.yaml
```
</details>

<details open>
<summary>Model Training</summary>

```bash
bash model_train/entry.sh $INPUT_DATA_PATH $INPUT_MODEL_PATH $OUTPUT_MODEL_PATH $METHOD $CONFIG_PATH

# Example:
bash model_train/entry.sh \
  data/source-cc-1b_scored-lqs_selected-r1.0_ordered-saw.jsonl \
  models/mistral-160m \
  models/pretrain_mistral-160m_source-cc-1b_ordered-saw \
  pretrain \
  model_train/config/train.yaml
```
</details>

<details open>
<summary>Model Evaluation</summary>

```bash
bash model_eval/entry.sh $INPUT_MODEL_PATH $OUTPUT_RESULT_PATH $METHOD $CONFIG_PATH

# Example:
bash model_eval/entry.sh \
  models/pretrain_mistral-160m_source-cc-1b_ordered-saw \
  models/pretrain_mistral-160m_source-cc-1b_ordered-saw/result.yaml \
  lm_evaluation_harness \
  model_eval/config/general.yaml
```
</details>

## Citation

```bibtex
@article{dai2025data,
  title={Data Efficacy for Language Model Training},
  author={Yalun Dai and Yangyu Huang and Xin Zhang and Wenshan Wu and Chong Li and Wenhui Lu and Shijie Cao and Li Dong and Scarlett Li},
  journal={arXiv preprint arXiv:2506.21545},
  year={2025}
}

@inproceedings{dai2026demystifying,
  title={Demystifying Data Organization for Enhanced LLM Training},
  author={Yalun Dai and Yangyu Huang and Tongshen Yang and Yonghan Wang and Xin Zhang and Wenshan Wu and Qihao Zhao and Hao Li and Yuanyuan Gao and Kim-Hui Yap and Scarlett Li},
  booktitle={Proceedings of the Annual Meeting of the Association for Computational Linguistics},
  year={2026}
}
```

## License

This repository is licensed under the [MIT](./LICENSE) License.
