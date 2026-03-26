#!/bin/bash

# Usage: bash etth2.sh

# Configuration

DATASET="ETTh2"
SEQ_LEN=512
BATCH_SIZE=32
LRATE=0.05
PATIENCE=10
MAX_EPOCHS=100
DEVICE="cuda:0"

# NLinear specific

USE_REVIN=False

# Prediction lengths to test

PRED_LENS=(96 720)

# GSAM rho values

declare -A RHO_VALUES
RHO_VALUES[96]=0.5
RHO_VALUES[720]=0.5

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
            --gsam True \
            --gsam_alpha 0.4 \
            --gsam_rho_min ${rho} \
            --gsam_rho_max ${rho}

        echo "Completed: pred_len=${pred_len}, seed=${seed}"
        echo ""
    done
done

echo "All experiments completed!"
