#!/bin/bash

# Usage: bash exchange_rate.sh

# Configuration

DATASET="exchange_rate"
SEQ_LEN=512
LRATE=0.0005
PATIENCE=10
MAX_EPOCHS=100
DEVICE="cuda:0"

# Linear specific

USE_REVIN=False

# Prediction lengths to test

PRED_LENS=(96 720)

# Batch sizes per prediction length

declare -A BATCH_SIZES
BATCH_SIZES[96]=8
BATCH_SIZES[720]=32

# SAM rho values

declare -A RHO_VALUES
RHO_VALUES[96]=0.5
RHO_VALUES[720]=0.5

# Seeds to run

SEEDS=(1)

# Script path

SCRIPT="experiments/single_split/linear/linear.py"

# Run experiments

for pred_len in "${PRED_LENS[@]}"; do
    batch_size=${BATCH_SIZES[$pred_len]}
    rho=${RHO_VALUES[$pred_len]}
    for seed in "${SEEDS[@]}"; do
        echo "=============================================="
        echo "Running: Dataset=${DATASET}, seq_len=${SEQ_LEN}, pred_len=${pred_len}, rho=${rho}, batch_size=${batch_size}, seed=${seed}"
        echo "=============================================="

        python -u ${SCRIPT} \
            --dataset ${DATASET} \
            --seed ${seed} \
            --device ${DEVICE} \
            --lrate ${LRATE} \
            --patience ${PATIENCE} \
            --max_epochs ${MAX_EPOCHS} \
            --batch_size ${batch_size} \
            --seq_len ${SEQ_LEN} \
            --pred_len ${pred_len} \
            --use_revin ${USE_REVIN} \
            --hessian_analysis True \
            --sam True \
            --rho ${rho}

        echo "Completed: pred_len=${pred_len}, seed=${seed}"
        echo ""
    done
done

echo "All experiments completed!"
