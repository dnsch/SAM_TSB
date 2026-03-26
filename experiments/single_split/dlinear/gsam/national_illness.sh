#!/bin/bash

# Usage: bash run_dlinear_ili.sh

# Configuration

DATASET="national_illness"
SEQ_LEN=144
BATCH_SIZE=32
LRATE=0.01
PATIENCE=10
MAX_EPOCHS=100
DEVICE="cuda:0"

# DLinear specific

INDIVIDUAL=False
KERNEL_SIZE=25

# Prediction lengths to test (different from other datasets)

# PRED_LENS=(24 36 48 60)
PRED_LENS=(24 60)

# Seeds to run

SEEDS=(1)

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
            --gsam True \
            --gsam_alpha 0.4 \
            --gsam_rho_min 0.5 \
            --gsam_rho_max 0.5

        echo "Completed: pred_len=${pred_len}, seed=${seed}"
        echo ""
    done
done

echo "All experiments completed!"
