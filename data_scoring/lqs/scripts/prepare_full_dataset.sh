#!/bin/bash

DATA_PATH=${1-"./redpajama_sample_1B/cc_en_head/"}
CONFIG_PATH=${2-"./data_scoring/config/lqs.yaml"}

export TF_CPP_MIN_LOG_LEVEL=3

# download model
python data_scoring/lqs/tools/hf_download.py \
    --lqs-process full_data \
    --content model \
    --config-path $CONFIG_PATH

# process data
python data_scoring/lqs/tools/process_data/pretrain_data_process.py \
    --lqs-process full_data \
    --data-path $DATA_PATH \
    --config-path $CONFIG_PATH