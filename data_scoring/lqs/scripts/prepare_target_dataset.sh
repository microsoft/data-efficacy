#!/bin/bash

CONFIG_PATH=${1-"./data_scoring/config/lqs.yaml"}

export TF_CPP_MIN_LOG_LEVEL=3

# download dataset
python data_scoring/lqs/tools/hf_download.py \
    --lqs-process target_data \
    --content dataset \
    --config-path $CONFIG_PATH

# process data (lima)
python data_scoring/lqs/tools/process_data/hf_data_process.py \
    --lqs-process target_data \
    --config-path $CONFIG_PATH