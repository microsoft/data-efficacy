#!/bin/bash

CONFIG_FILE=${1-"./data_scoring/config/lqs.yaml"}

GPUS_PER_NODE=${2-1}
MASTER_PORT=${3-2030}
NNODES=1
DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3

CMD="deepspeed ${DISTRIBUTED_ARGS} data_scoring/lqs/train_data_scorer.py --lqs-process scorer_data_training --config ${CONFIG_FILE}"

echo ${CMD}
${CMD}
