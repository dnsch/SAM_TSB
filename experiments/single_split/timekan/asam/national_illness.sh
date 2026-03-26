#!/bin/bash

# Usage: bash national_illness_timekan_asam.sh

# Configuration

DATASET="national_illness"
SEQ_LEN=144
BATCH_SIZE=32
LRATE=0.01
PATIENCE=10
MAX_EPOCHS=100
DEVICE="cuda:0"

# TimeKAN specific

USE_REVIN=True
LOSS_NAME="mse"
E_LAYERS=2
D_MODEL=16
D_FF=32
DOWN_SAMPLING_LAYERS=2
DOWN_SAMPLING_WINDOW=2
BEGIN_ORDER=0

# Prediction lengths to test (different from other datasets)

PRED_LENS=(24 60)

# ASAM rho values

declare -A RHO_VALUES
RHO_VALUES[24]=0.5
RHO_VALUES[60]=0.5

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
