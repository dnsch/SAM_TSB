#!/bin/bash

# Usage: bash exchange_rate_timekan_asam.sh

# Configuration

DATASET="exchange_rate"
SEQ_LEN=512
BATCH_SIZE=32
LRATE=0.001
PATIENCE=10
MAX_EPOCHS=100
DEVICE="cuda:0"

# TimeKAN specific

USE_REVIN=True
LOSS_NAME="mse"
E_LAYERS=2
D_MODEL=16
D_FF=32
DOWN_SAMPLING_LAYERS=3
DOWN_SAMPLING_WINDOW=2
BEGIN_ORDER=0

# Prediction lengths to test

PRED_LENS=(96 720)

# ASAM rho values

declare -A RHO_VALUES
RHO_VALUES[96]=0.5
RHO_VALUES[720]=0.5

# Seeds to run

SEEDS=(1)

# Script path

SCRIPT="experiments/single_split/timekan/timekan.py"

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
            --max_epoch ${MAX_EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --seq_len ${SEQ_LEN} \
            --pred_len ${pred_len} \
            --use_revin ${USE_REVIN} \
            --loss_name ${LOSS_NAME} \
            --e_layers ${E_LAYERS} \
            --d_model ${D_MODEL} \
            --down_sampling_layers ${DOWN_SAMPLING_LAYERS} \
            --down_sampling_window ${DOWN_SAMPLING_WINDOW} \
            --begin_order ${BEGIN_ORDER} \
            --mode train \
            --hessian_analysis True \
            --sam True \
            --rho ${rho} \
            --sam_adaptive True

        echo "Completed: pred_len=${pred_len}, seed=${seed}"
        echo ""
    done
done

echo "All experiments completed!"
