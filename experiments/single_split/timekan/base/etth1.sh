#!/bin/bash

# Usage: bash etth1_timekan.sh

# Configuration

DATASET="ETTh1"
SEQ_LEN=512
DEVICE="cuda:0"

# TimeKAN specific

USE_REVIN=True
LOSS_NAME="mse"
E_LAYERS=2
D_MODEL=16
D_FF=32
DOWN_SAMPLING_WINDOW=2
MAX_EPOCHS=10
PATIENCE=10

# Prediction lengths to test

PRED_LENS=(96 720)

# Per pred_len hyperparameters

declare -A BATCH_SIZE_VALUES
BATCH_SIZE_VALUES[96]=128
BATCH_SIZE_VALUES[720]=128

declare -A DOWN_SAMPLING_LAYERS_VALUES
DOWN_SAMPLING_LAYERS_VALUES[96]=2
DOWN_SAMPLING_LAYERS_VALUES[720]=3

declare -A BEGIN_ORDER_VALUES
BEGIN_ORDER_VALUES[96]=0
BEGIN_ORDER_VALUES[720]=1

declare -A LRATE_VALUES
LRATE_VALUES[96]=0.01
LRATE_VALUES[720]=0.01

# RHO values (for consistency with SAM variants)

declare -A RHO_VALUES
RHO_VALUES[96]=0.5
RHO_VALUES[720]=0.5

# Seeds to run

SEEDS=(1)

# Script path

SCRIPT="experiments/single_split/timekan/timekan.py"

# Run experiments

for pred_len in "${PRED_LENS[@]}"; do
    batch_size=${BATCH_SIZE_VALUES[$pred_len]}
    down_sampling_layers=${DOWN_SAMPLING_LAYERS_VALUES[$pred_len]}
    begin_order=${BEGIN_ORDER_VALUES[$pred_len]}
    lrate=${LRATE_VALUES[$pred_len]}
    rho=${RHO_VALUES[$pred_len]}
    for seed in "${SEEDS[@]}"; do
        echo "=============================================="
        echo "Running: Dataset=${DATASET}, seq_len=${SEQ_LEN}, pred_len=${pred_len}, seed=${seed}"
        echo "=============================================="

        python -u ${SCRIPT} \
            --dataset ${DATASET} \
            --seed ${seed} \
            --device ${DEVICE} \
            --lrate ${lrate} \
            --patience ${PATIENCE} \
            --max_epoch ${MAX_EPOCHS} \
            --batch_size ${batch_size} \
            --seq_len ${SEQ_LEN} \
            --pred_len ${pred_len} \
            --use_revin ${USE_REVIN} \
            --loss_name ${LOSS_NAME} \
            --e_layers ${E_LAYERS} \
            --d_model ${D_MODEL} \
            --down_sampling_layers ${down_sampling_layers} \
            --down_sampling_window ${DOWN_SAMPLING_WINDOW} \
            --begin_order ${begin_order} \
            --mode train \
            --hessian_analysis True

        echo "Completed: pred_len=${pred_len}, seed=${seed}"
        echo ""
    done
done

echo "All experiments completed!"
