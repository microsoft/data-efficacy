# Learnability-Quality Scoring (LQS) Method
This instruction shows the steps of Learnability-Quality Scoring (LQS).

## 0. Required action of Huggingface 
Please note that LQS involves downloading Hugging Face gated models/datasets, and you need to configure it as follows:

1. Request Access. Visit the model/dataset page on Hugging Face (such as [lima](https://huggingface.co/datasets/GAIR/lima)), log in, and click Request Access. Approval may be automatic or manual.
2. Get Access Token. Generate an access token at [Access Tokens](https://huggingface.co/settings/tokens), then log in via `huggingface-cli login` or use the token directly in commands.

For details, see [Gated Models](https://huggingface.co/docs/hub/models-gated) and [Gated Datasets](https://huggingface.co/docs/hub/datasets-gated).

## 1. Prepare dataset and model
Download the [Mistral 160M](https://huggingface.co/Data-Selection/BSL-160M) model. Then run tokenization for CC.
```bash
bash data_scoring/lqs/scripts/prepare_full_dataset.sh $DATA_PATH $CONFIG_PATH

# e.g. bash data_scoring/lqs/scripts/prepare_full_dataset.sh data/source-cc-1b.jsonl data_scoring/config/lqs.yaml
```

## 2. Download weight of data scorer (choose one between this and below step)
*Please note that the LQS-based data scorer weights are currently being prepared and are NOT yet available.*

If you want to run inference with our checkpoints directly, download the [checkpoints]() or use the following script, then follow the [reference](#4-score-data-examples).
``` bash
bash data_scoring/lqs/scripts/prepare_checkpoints.sh $CONFIG_PATH
# e.g. bash data_scoring/lqs/scripts/prepare_checkpoints.sh data_scoring/config/lqs.yaml
```

## 3. Train data scorer (choose one between this and above step)
The following steps train the data scorer from scratch. To use our checkpoints, please refer to [download the checkpoint](#2-download-weight-of-data-scorer-choose-one-between-this-and-below-steps) instead. Please note that this [step](#33-annotate-proxy-data) will require approximately 70GB of GPU memory. Please use a GPU with larger memory (e.g., A100 80G) to avoid OOM errors.

### 3.1 Prepare data for downstream loss calculation
Download the [lima](https://huggingface.co/datasets/GAIR/lima) dataset for downstream loss calculation. Then run tokenization for lima.
```bash
bash data_scoring/lqs/scripts/prepare_target_dataset.sh $CONFIG_PATH
# e.g. bash data_scoring/lqs/scripts/prepare_target_dataset.sh data_scoring/config/lqs.yaml
```

### 3.2 Sample proxy data from CC
```bash
bash data_scoring/lqs/scripts/proxy_data_sampling.sh $CONFIG_PATH
# e.g. bash data_scoring/lqs/scripts/proxy_data_sampling.sh data_scoring/config/lqs.yaml
```

### 3.3 Annotate proxy data
Please note that this implementation must run on GPUs with A100 80GB or larger memory.
```bash
bash data_scoring/lqs/scripts/proxy_data_annotation.sh $CONFIG_PATH
# e.g. bash data_scoring/lqs/scripts/proxy_data_annotation.sh data_scoring/config/lqs.yaml
```

### 3.4 Train data scorer
Download the [fairseq](https://huggingface.co/datasets/GAIR/lima) model for data scorer training.
```bash
bash data_scoring/lqs/scripts/train_data_scorer.sh $CONFIG_PATH
# e.g. bash data_scoring/lqs/scripts/train_data_scorer.sh data_scoring/config/lqs.yaml 
```

## 4. Score data examples
```bash
bash data_scoring/lqs/scripts/infer_data_scorer.sh $CONFIG_PATH $OUTPUT_DATA_PATH
# e.g. bash data_scoring/lqs/scripts/infer_data_scorer.sh data_scoring/config/lqs.yaml data/cc/lqs_scored_data.jsonl
```
