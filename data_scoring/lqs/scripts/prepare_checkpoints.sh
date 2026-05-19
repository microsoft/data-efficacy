#!/bin/bash

CONFIG_PATH=${1-"./data_scoring/config/lqs.yaml"}

export TF_CPP_MIN_LOG_LEVEL=3

echo "Currently unavailable, data checkpoints will be released soon."

# # download model for data scoring
# python data_scoring/lqs/tools/hf_download.py \
#     --lqs-process annotation_data \
#     --content model \
#     --config-path $CONFIG_PATH

# # download data scorer checkpoints
# python data_scoring/lqs/tools/hf_download.py \
#     --lqs-process checkpoint_download \
#     --content repo \
#     --config-path $CONFIG_PATH