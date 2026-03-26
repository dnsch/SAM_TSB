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

# NLinear specific

USE_REVIN=False

# Prediction lengths to test (different from other datasets)

PRED_LENS=(24 60)

# SAM rho values

declare -A RHO_VALUES
RHO_VALUES[24]=0.5
RHO_VALUES[60]=0.5

# Seeds to run

SEEDS=(1)

# Script path

SCRIPT="experiments/single_split/nlinear/nlinear.py"

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
            --max_epochs ${MAX_EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --seq_len ${SEQ_LEN} \
            --pred_len ${pred_len} \
            --use_revin ${USE_REVIN} \
            --hessian_analysis True \
            --scheduler none \
            --sam True \
            --rho ${rho}

        echo "Completed: pred_len=${pred_len}, seed=${seed}"
        echo ""
    done
done

echo "All experiments completed!"
