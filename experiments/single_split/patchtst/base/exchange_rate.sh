#!/bin/bash

# Usage: bash exchange_rate.sh

# Configuration

DATASET="exchange_rate"
SEQ_LEN=512
BATCH_SIZE=128
LRATE=0.0001
PATIENCE=10
MAX_EPOCHS=100
DEVICE="cuda:0"

# PatchTST specific

USE_REVIN=True
E_LAYERS=3
N_HEADS=4
D_MODEL=16
D_FF=128
DROPOUT=0.3
FC_DROPOUT=0.3
HEAD_DROPOUT=0
PATCH_LEN=16
STRIDE=8

# Prediction lengths to test

# PRED_LENS=(96 192 336 720)

PRED_LENS=(96 720)

# Seeds to run

SEEDS=(1)

# Script path

SCRIPT="experiments/single_split/patchtst/patchtst.py"

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
            --max_epoch ${MAX_EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --seq_len ${SEQ_LEN} \
            --pred_len ${pred_len} \
            --e_layers ${E_LAYERS} \
            --n_heads ${N_HEADS} \
            --d_model ${D_MODEL} \
            --d_ff ${D_FF} \
            --dropout ${DROPOUT} \
            --fc_dropout ${FC_DROPOUT} \
            --head_dropout ${HEAD_DROPOUT} \
            --patch_len ${PATCH_LEN} \
            --stride ${STRIDE} \
            --use_revin ${USE_REVIN} \
            --mode train \
            --hessian_analysis True

        echo "Completed: pred_len=${pred_len}, seed=${seed}"
        echo ""
    done
done

echo "All experiments completed!"
