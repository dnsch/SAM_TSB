#!/bin/bash

# Usage: bash national_illness.sh

# Configuration

DATASET="national_illness"
SEEDS=(1)
SEASON_LENGTH=52

# Prediction lengths to test (different from other datasets)

PRED_LENS=(24 60)

# Script path

SCRIPT="experiments/single_split/seasonal_exponential_smoothing/seasonal_exponential_smoothing.py"

# Run experiments

for pred_len in "${PRED_LENS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "=============================================="
        echo "Running: Dataset=${DATASET}, pred_len=${pred_len}, season_length=${SEASON_LENGTH}, seed=${seed}"
        echo "=============================================="

        python -u ${SCRIPT} \
            --dataset ${DATASET} \
            --seed ${seed} \
            --metrics mse \
            --pred_len ${pred_len} \
            --season_length ${SEASON_LENGTH}

        echo "Completed: pred_len=${pred_len}, seed=${seed}"
        echo ""
    done
done

echo "All experiments completed!"
