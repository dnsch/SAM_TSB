#!/bin/bash

# Usage: bash run_dlinear_ettm2.sh

# Configuration

DATASET="ETTm2"
SEQ_LEN=512
BATCH_SIZE=32
PATIENCE=10
MAX_EPOCHS=100
DEVICE="cuda:0"

# DLinear specific

INDIVIDUAL=False
KERNEL_SIZE=25

# Seeds to run

SEEDS=(1 2 3)

# Script path

SCRIPT="experiments/single_split/dlinear/dlinear.py"

# Prediction lengths with corresponding learning rates (from original paper)

# pred_len 96:  lr=0.001

# pred_len 192: lr=0.001

# pred_len 336: lr=0.01

# pred_len 720: lr=0.1

declare -A LRATES
LRATES[96]=0.001
LRATES[192]=0.001
LRATES[336]=0.01
LRATES[720]=0.1

PRED_LENS=(96 192 336 720)

# Run experiments

for pred_len in "${PRED_LENS[@]}"; do
    lrate=${LRATES[$pred_len]}
    for seed in "${SEEDS[@]}"; do
        echo "=============================================="
        echo "Running: Dataset=${DATASET}, seq_len=${SEQ_LEN}, pred_len=${pred_len}, lrate=${lrate}, seed=${seed}"
        echo "=============================================="

        python -u ${SCRIPT} \
            --dataset ${DATASET} \
            --seed ${seed} \
            --device ${DEVICE} \
            --lrate ${lrate} \
            --patience ${PATIENCE} \
            --max_epochs ${MAX_EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --seq_len ${SEQ_LEN} \
            --pred_len ${pred_len} \
            --individual ${INDIVIDUAL} \
            --kernel_size ${KERNEL_SIZE} \
            --hessian_analysis True

        echo "Completed: pred_len=${pred_len}, seed=${seed}"
        echo ""
    done
done

echo "All experiments completed!"
