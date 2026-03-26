#!/bin/bash

# Usage: bash national_illness.sh

# Configuration

DATASET="national_illness"
SEQ_LEN=144
BATCH_SIZE=32
LRATE=0.01
PATIENCE=10
MAX_EPOCHS=100
DEVICE="cuda:0"

# SAMformer specific

USE_REVIN=True
LOSS_NAME="mse"

# Multi-split specific

NUM_SPLITS=1
PRED_LEN=24

# Seeds to run

SEEDS=(1)

# Script path

SCRIPT="experiments/multi_split/samformer/samformer.py"

# Run experiments

for seed in "${SEEDS[@]}"; do
    echo "=============================================="
    echo "Running: Dataset=${DATASET}, seq_len=${SEQ_LEN}, pred_len=${PRED_LEN}, rho=${RHO}, num_splits=${NUM_SPLITS}, seed=${seed}"
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
        --loss_name ${LOSS_NAME} \
        --mode train \
        --hessian_analysis True \
        --sam True \
        --num_splits ${NUM_SPLITS}

    echo "Completed: pred_len=${PRED_LEN}, seed=${seed}"
    echo ""
done

echo "All experiments completed!"
