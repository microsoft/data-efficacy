# Data Efficacy for Language Model Training

<p align="center">
 <img src="https://img.shields.io/badge/Task-Data_Efficacy-orange" alt="Task" />
 <img src="https://img.shields.io/badge/Paper-arXiv-green" alt="Paper" />
 <img src="https://img.shields.io/badge/License-MIT-blue" alt="License" />
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2506.21545"><b>[Paper]</b></a> •
  <a href="https://huggingface.co/microsoft/DELT"><b>[HF Model]</b></a>
</p>

This page documents **Data Efficacy for Language Model Training (DELT)**, the original work supported by this repository. For the repository-wide overview and common running commands, see the root [README](../README.md).

DELT studies how pre-computed sample-level scores can be reused across **Data Scoring**, **Data Selection**, and **Data Ordering** to improve language model training.

![DELT results](../figures/fig1_result.jpg)

## Contributions

DELT frames data curation as a data efficacy problem: once sample-level scores are available, they should guide more than a one-time filtering decision.

The paper introduces a connected pipeline with three core components:

- **Learnability-Quality Scoring (LQS)** scores each sample by considering both learnability and quality from the gradient consistency perspective.
- **Score-based Data Selection** constructs training subsets from the scored corpus under a data budget.
- **Folding Ordering (FO)** reuses the same scores to organize selected data before training, mitigating forgetting and distribution bias during one-pass training.

![DELT paradigm](../figures/data_efficacy_paradigm.png)

## Learnability-Quality Scoring

LQS is the data scoring component of DELT. It estimates sample-level utility by combining learnability and quality signals, so the score can later be reused by both selection and ordering.

![Learnability-Quality Scoring](../figures/fig2_score.jpg)

Implementation entry point:

```bash
bash data_scoring/entry.sh \
  data/source-cc-1b.jsonl \
  data/source-cc-1b_scored-lqs.jsonl \
  lqs \
  data_scoring/config/lqs.yaml
```

## Folding Ordering

FO is the data ordering component introduced by DELT. It orders selected samples according to their scores with a folding pattern, so training can revisit different score regions and reduce distribution bias.

![Folding Ordering](../figures/fig3_order.jpg)

Implementation entry point:

```bash
bash data_ordering/entry.sh \
  data/source-cc-1b_scored-lqs_selected-r1.0.jsonl \
  data/source-cc-1b_scored-lqs_selected-r1.0_ordered-folding-l3.jsonl \
  folding \
  data_ordering/config/folding.yaml
```

## Modules

- [data_scoring](../data_scoring): LQS and KenLM-based scoring.
- [data_selection](../data_selection): top-r, top-k, and threshold selection.
- [data_ordering](../data_ordering): score-based ordering, including DELT's Folding Ordering.
- [model_train](../model_train): pre-training.
- [model_eval](../model_eval): evaluation.

## Citation

```bibtex
@article{dai2025data,
  title={Data Efficacy for Language Model Training},
  author={Yalun Dai and Yangyu Huang and Xin Zhang and Wenshan Wu and Chong Li and Wenhui Lu and Shijie Cao and Li Dong and Scarlett Li},
  journal={arXiv preprint arXiv:2506.21545},
  year={2025}
}
```
