#!/bin/bash

INPUT_DATA_PATH=${1-"./data/original_data.jsonl"}
OUTPUT_DATA_PATH=${2-"./data/scored_data.jsonl"}
METHOD=${3-"lqs"} 
CONFIG_PATH=${4-"./data_scoring/config/lqs.yaml"}
TRAIN_SCORER=${5-"True"} # Whether to train the data scorer from scratch (True) or use downloaded checkpoints (False). Applies to the LQS method.


if [ -z "$1" ]; then
  SUPPORTED_METHODS=("lqs" "kenlm")
  echo "Error: No method specified. Please use one of the following: ${SUPPORTED_METHODS[*]}"
  exit 1
fi

if [ "$1" == "--help" ]; then
  echo "Usage: $0 <method>"
  echo "Available methods:"
  echo "  lqs  - Run LQS scoring script."
  exit 0
fi


if [ "$METHOD" == "kenlm" ]; then
  chmod +x ./data_scoring/kenlm/entry.sh
  if [ ! -x "./data_scoring/kenlm/entry.sh" ]; then
    echo "Error: Script .sh does not exist or is not executable."
    exit 1
  fi
  bash ./data_scoring/kenlm/entry.sh $INPUT_DATA_PATH $OUTPUT_DATA_PATH $CONFIG_PATH
fi

if [ "$METHOD" == "lqs" ]; then
  chmod +x ./data_scoring/lqs/entry.sh
  if [ ! -x "./data_scoring/lqs/entry.sh" ]; then
    echo "Error: Script .sh does not exist or is not executable."
    exit 1
  fi
  bash ./data_scoring/lqs/entry.sh $INPUT_DATA_PATH $OUTPUT_DATA_PATH $CONFIG_PATH $TRAIN_SCORER
fi