#!/bin/bash

# Usage: bash exchange_rate.sh

# Configuration

DATASET="exchange_rate"
SEQ_LEN=512
BATCH_SIZE=64
LRATE=0.0001
PATIENCE=10
MAX_EPOCHS=100
DEVICE="cuda:0"

# TSMixer specific

USE_REVIN=True
NORM_TYPE="batch"
NUM_BLOCKS=2
DROPOUT=0.1
LOSS_NAME="mse"

# Multi-split specific

NUM_SPLITS=3
PRED_LEN=96

# Seeds to run

SEEDS=(1)

# Script path

SCRIPT="experiments/multi_split/tsmixer/tsmixer.py"

# Run experiments

for seed in "${SEEDS[@]}"; do
    echo "=============================================="
    echo "Running: Dataset=${DATASET}, seq_len=${SEQ_LEN}, pred_len=${PRED_LEN}, num_splits=${NUM_SPLITS}, seed=${seed}"
    echo "=============================================="

    python -u ${SCRIPT} \
        --dataset ${DATASET} \
        --seed ${seed} \
        --device ${DEVICE} \
        --lrate ${LRATE} \
        --patience ${PATIENCE} \
        --max_epoch ${MAX_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --seq_len ${SEQ_LEN} \
        --pred_len ${PRED_LEN} \
        --use_revin ${USE_REVIN} \
        --norm_type ${NORM_TYPE} \
        --num_blocks ${NUM_BLOCKS} \
        --dropout ${DROPOUT} \
        --loss_name ${LOSS_NAME} \
        --shuffle_train_val False \
        --mode train \
        --num_splits ${NUM_SPLITS}

    echo "Completed: pred_len=${PRED_LEN}, seed=${seed}"
    echo ""
done

echo "All experiments completed!"
