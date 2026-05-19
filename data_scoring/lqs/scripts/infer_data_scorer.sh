#!/bin/bash

CONFIG_PATH=${1-"data_scoring/config/lqs.yaml"}
OUTPUT_DATA_PATH=${2-"data/cc/lqs_scored_data.jsonl"}

# convert to jsonl
python data_scoring/lqs/tools/token_data_bin2json.py \
    --lqs-process scorer_data_infer \
    --config-path $CONFIG_PATH

# infer
python data_scoring/lqs/infer_data_scorer.py --lqs-process scorer_data_infer --config ${CONFIG_PATH} --output-data-path ${OUTPUT_DATA_PATH}
