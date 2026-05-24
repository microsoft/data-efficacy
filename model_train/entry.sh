#!/bin/bash

INPUT_DATA_PATH=${1-'./data/ordered_data.jsonl'}
INPUT_MODEL_PATH=${2-'./model/input_model'}
OUTPUT_MODEL_PATH=${3-'./model/output_model'}
METHOD=${4-'pretrain'}
CONFIG_PATH=${5-'train.yaml'}

GPUS_PER_NODE=${6-1}
NNODES=${7-1}
#MASTER_PORT=${8-2030}
min_port=2048
max_port=65535
MASTER_PORT=$((RANDOM % (max_port - min_port + 1) + min_port))

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=16

CMD="deepspeed ${DISTRIBUTED_ARGS} model_train/entry.py \
    --data_path ${INPUT_DATA_PATH} \
    --model_path ${INPUT_MODEL_PATH} \
    --save ${OUTPUT_MODEL_PATH} \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH}"

echo ${CMD}
mkdir -p ${OUTPUT_MODEL_PATH}
${CMD}
