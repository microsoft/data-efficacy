#!/bin/bash

INPUT_DATA_PATH=${1-"./data/original_data.jsonl"}
OUTPUT_DATA_PATH=${2-"./data/scored_data.jsonl"}
CONFIG_PATH=${3-"./data_scoring/config/lqs.yaml"}
TRAIN_SCORER=${4-"True"} # Whether to train the data scorer from scratch (True) or use downloaded checkpoints (False). Applies to the LQS method.

# ------------------ Part 1: Data processing ------------------
# Step 1: prepare full dataset.
echo "Step 1: Preparing full dataset..."
bash data_scoring/lqs/scripts/prepare_full_dataset.sh $INPUT_DATA_PATH $CONFIG_PATH
echo "Step 1 completed: Full dataset prepared."

# ------------------ Part 2: LQS checkpoints downloading OR training ------------------
# TBD. LQS-based data scorer weights are in preparation and are not currently available.
if [[ "$TRAIN_SCORER" == "False" ]]; then
    echo "Step 2: Downloading LQS data scorer checkpoints..."
    bash data_scoring/lqs/scripts/prepare_checkpoints.sh $CONFIG_PATH
    echo "Step 2 completed: checkpoints downloading done."
elif [[ "$TRAIN_SCORER" == "True" ]]; then
    # Step 2: prepare target dataset.
    echo "Step 2: Preparing target dataset..."
    bash data_scoring/lqs/scripts/prepare_target_dataset.sh $CONFIG_PATH
    echo "Step 2 completed: Target dataset prepared."

    # Prepare (sampled and annotated) training dataset for scoring model.
    # Step 3: proxy data sampling.
    echo "Step 3: Proxy data sampling..."
    bash data_scoring/lqs/scripts/proxy_data_sampling.sh "$CONFIG_PATH"
    echo "Step 3 completed: Proxy data sampling done."

    # Step 4: proxy data annotation.
    echo "Step 4: Proxy data annotation..."
    bash data_scoring/lqs/scripts/proxy_data_annotation.sh "$CONFIG_PATH"
    echo "Step 4 completed: Proxy data annotation done."

    # Step 5: data scorer training.
    echo "Step 5: Training data scorer..."
    bash data_scoring/lqs/scripts/train_data_scorer.sh "$CONFIG_PATH"
    echo "Step 5 completed: Data scorer trained."
else
    echo "Invalid TRAIN_SCORER value: $TRAIN_SCORER (expected True or False)"
    exit 1
fi

# ------------------ Part 3: LQS scoring ------------------
# Step 6: full dataset scoring.
echo "Step 6: Scoring full dataset..."
bash data_scoring/lqs/scripts/infer_data_scorer.sh $CONFIG_PATH $OUTPUT_DATA_PATH
echo "Step 6 completed: Full dataset scored. Output saved to $OUTPUT_DATA_PATH."

# Final message
echo "LQS data scoring completed successfully!"
