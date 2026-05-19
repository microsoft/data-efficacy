#!/bin/bash

# proxy data annotation
CONFIG_PATH=${1-"./data_scoring/config/lqs.yaml"}

MASTER_PORT=${2-2030}
GPUS_PER_NODE=${3-1}
NNODES=1
DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT"

export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=16

CMD="deepspeed ${DISTRIBUTED_ARGS} data_scoring/lqs/train_data_scorer.py --lqs-process annotation_data --config-path ${CONFIG_PATH}"

echo ${CMD}
${CMD}

# download model for data scoring
python data_scoring/lqs/tools/hf_download.py \
    --lqs-process annotation_data \
    --content model \
    --config-path $CONFIG_PATH

# prepare data for data scoring
python data_scoring/lqs/tools/prepare_data_scorer_train_data.py \
    --lqs-process annotation_data \
    --config-path $CONFIG_PATH \
