#!/bin/bash

# Usage: bash exchange_rate.sh

# Configuration

DATASET="exchange_rate"
SEQ_LEN=512
BATCH_SIZE=32
LRATE=0.001
PATIENCE=10
MAX_EPOCHS=100
DEVICE="cuda:0"

# SAMformer specific

USE_REVIN=True
LOSS_NAME="mse"

# Prediction lengths to test

# PRED_LENS=(96 192 336 720)

PRED_LENS=(96 720)

# SAM rho values per prediction length (from Table 4)

declare -A RHO_VALUES
RHO_VALUES[96]=0.7
RHO_VALUES[192]=0.8
RHO_VALUES[336]=0.5
RHO_VALUES[720]=0.9

# Seeds to run

SEEDS=(1)

# Script path

SCRIPT="experiments/multi_split/samformer/samformer.py"

# Run experiments

for pred_len in "${PRED_LENS[@]}"; do
    rho=${RHO_VALUES[$pred_len]}
    for seed in "${SEEDS[@]}"; do
        echo "=============================================="
        echo "Running: Dataset=${DATASET}, seq_len=${SEQ_LEN}, pred_len=${pred_len}, rho=${rho}, seed=${seed}"
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
            --pred_len ${pred_len} \
            --use_revin ${USE_REVIN} \
            --loss_name ${LOSS_NAME} \
            --mode train \
            --hessian_analysis True \
            --sam True \
            --rho ${rho}

        echo "Completed: pred_len=${pred_len}, seed=${seed}"
        echo ""
    done
done

echo "All experiments completed!"
