#!/bin/bash

# Usage: bash national_illness.sh

# Configuration (ILI not in paper, using reasonable estimates)

DATASET="national_illness"
SEQ_LEN=144
BATCH_SIZE=32
LRATE=0.01
PATIENCE=10
MAX_EPOCHS=100
DEVICE="cuda:0"

# SAMformer specific

USE_REVIN=True
LOSS_NAME="mse"

# Prediction lengths to test (different from other datasets)

# PRED_LENS=(24 36 48 60)
PRED_LENS=(24 60)

# SAM rho (using default from paper's sensitivity analysis)

RHO=0.7

# Seeds to run

SEEDS=(1)

# Script path

SCRIPT="experiments/single_split/samformer/samformer.py"

# Run experiments

for pred_len in "${PRED_LENS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "=============================================="
        echo "Running: Dataset=${DATASET}, seq_len=${SEQ_LEN}, pred_len=${pred_len}, rho=${RHO}, seed=${seed}"
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
            --rho ${RHO}

        echo "Completed: pred_len=${pred_len}, seed=${seed}"
        echo ""
    done
done

echo "All experiments completed!"
