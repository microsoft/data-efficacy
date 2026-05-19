#!/bin/bash

CONFIG_PATH=${1-"./data_scoring/config/lqs.yaml"}

export TF_CPP_MIN_LOG_LEVEL=3

python data_scoring/lqs/tools/sample_proxy_data.py \
    --lqs-process proxy_data \
    --config-path $CONFIG_PATH \