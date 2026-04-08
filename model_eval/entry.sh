#!/bin/bash

INPUT_MODEL_PATH=${1-"./model/output_model"}
OUTPUT_RESULT_PATH=${2-"./result/general.jsonl"}
METHOD=${3-"lm_evaluation_harness"} 
CONFIG_PATH=${4-"./model_eval/config/general.yaml"}

python model_eval/entry.py \
    --input_model_path $INPUT_MODEL_PATH \
    --output_result_path $OUTPUT_RESULT_PATH \
    --method $METHOD \
    --config $CONFIG_PATH \
