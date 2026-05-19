#!/bin/bash

INPUT_DATA_PATH=${1-"./data/scored_data.jsonl"}
OUTPUT_DATA_PATH=${2-"./data/selected_data.jsonl"}
METHOD=${3-"top-r"}
CONFIG_PATH=${4-"./data_selection/config/top-r.yaml"}

python data_selection/entry.py \
    --input_data_path $INPUT_DATA_PATH \
    --output_data_path $OUTPUT_DATA_PATH \
    --method $METHOD \
    --config_path $CONFIG_PATH
