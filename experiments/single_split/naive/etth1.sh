#!/bin/bash

# Usage: bash etth1.sh

# Configuration

DATASET="ETTh1"
SEEDS=(1)

# Prediction lengths to test

PRED_LENS=(96 720)

# Script path

SCRIPT="experiments/single_split/naive/naive.py"

# Run experiments

for pred_len in "${PRED_LENS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "=============================================="
        echo "Running: Dataset=${DATASET}, pred_len=${pred_len}, seed=${seed}"
        echo "=============================================="

        python -u ${SCRIPT} \
            --dataset ${DATASET} \
            --seed ${seed} \
            --metrics mse \
            --pred_len ${pred_len}

        echo "Completed: pred_len=${pred_len}, seed=${seed}"
        echo ""
    done
done

echo "All experiments completed!"
