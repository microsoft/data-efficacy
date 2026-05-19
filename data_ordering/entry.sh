#!/bin/bash
set -euo pipefail

INPUT_DATA_PATH=${1-"./data/selected_data.jsonl"}
OUTPUT_DATA_PATH=${2-"./data/ordered_data.jsonl"}
METHOD=${3-"folding"}
CONFIG_PATH=${4-"./data_ordering/config/folding.yaml"}
PYTHON_BIN=${PYTHON:-python3}

"$PYTHON_BIN" data_ordering/entry.py \
    --input_data_path "$INPUT_DATA_PATH" \
    --output_data_path "$OUTPUT_DATA_PATH" \
    --method "$METHOD" \
    --config_path "$CONFIG_PATH" \
