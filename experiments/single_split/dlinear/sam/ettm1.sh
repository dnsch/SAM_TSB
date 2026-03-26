#!/bin/bash

# Usage: bash run_dlinear_ettm1.sh

# Configuration

DATASET="ETTm1"
SEQ_LEN=336
BATCH_SIZE=8
LRATE=0.0001
PATIENCE=10
MAX_EPOCHS=100
DEVICE="cuda:0"

# DLinear specific

INDIVIDUAL=False
KERNEL_SIZE=25

# Prediction lengths to test

PRED_LENS=(96 192 336 720)

# Seeds to run

SEEDS=(1 2 3)

# Script path

SCRIPT="experiments/single_split/dlinear/dlinear.py"

# Run experiments

for pred_len in "${PRED_LENS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "=============================================="
        echo "Running: Dataset=${DATASET}, seq_len=${SEQ_LEN}, pred_len=${pred_len}, seed=${seed}"
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
            --individual ${INDIVIDUAL} \
            --kernel_size ${KERNEL_SIZE} \
            --hessian_analysis True \
            --sam True \
            --rho 0.5

        echo "Completed: pred_len=${pred_len}, seed=${seed}"
        echo ""
    done
done

echo "All experiments completed!"
