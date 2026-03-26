#!/bin/bash

# Usage: bash run_dlinear_exchange.sh

# Configuration

DATASET="exchange_rate"
SEQ_LEN=512
LRATE=0.0005
PATIENCE=10
MAX_EPOCHS=100
DEVICE="cuda:0"

# DLinear specific

INDIVIDUAL=False
KERNEL_SIZE=25

# Seeds to run

SEEDS=(1)

# Script path

SCRIPT="experiments/single_split/dlinear/dlinear.py"

# Prediction lengths with corresponding batch sizes (from original paper)

# pred_len 96:  batch_size=8

# pred_len 192: batch_size=8

# pred_len 336: batch_size=32

# pred_len 720: batch_size=32

declare -A BATCH_SIZES
BATCH_SIZES[96]=8
BATCH_SIZES[192]=8
BATCH_SIZES[336]=32
BATCH_SIZES[720]=32

# PRED_LENS=(96 192 336 720)
PRED_LENS=(96 720)

# Run experiments

for pred_len in "${PRED_LENS[@]}"; do
    batch_size=${BATCH_SIZES[$pred_len]}
    for seed in "${SEEDS[@]}"; do
        echo "=============================================="
        echo "Running: Dataset=${DATASET}, seq_len=${SEQ_LEN}, pred_len=${pred_len}, batch_size=${batch_size}, seed=${seed}"
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
            --individual ${INDIVIDUAL} \
            --kernel_size ${KERNEL_SIZE} \
            --hessian_analysis True \
            --fsam True \
            --fsam_rho 0.5 \
            --fsam_sigma 1.0 \
            --fsam_lmbda 0.95

        echo "Completed: pred_len=${pred_len}, seed=${seed}"
        echo ""
    done
done

echo "All experiments completed!"
