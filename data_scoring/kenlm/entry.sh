#!/bin/bash

INPUT_DATA_PATH=${1-"./data/original_data.jsonl"}
OUTPUT_DATA_PATH=${2-"./data/scored_kenlm_data.jsonl"}
CONFIG_PATH=${3-"./data_scoring/config/kenlm.yaml"}

python data_scoring/kenlm/entry.py \
    --input-data-path $INPUT_DATA_PATH \
    --output-data-path $OUTPUT_DATA_PATH \
    --config-path $CONFIG_PATH
