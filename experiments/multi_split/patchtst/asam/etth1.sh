#!/bin/bash

# Usage: bash etth1.sh

# Configuration

DATASET="ETTh1"
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

# Multi-split specific

NUM_SPLITS=3
PRED_LEN=96
RHO=0.5

# Seeds to run

SEEDS=(1)

# Script path

SCRIPT="experiments/multi_split/patchtst/patchtst.py"

# Run experiments

for seed in "${SEEDS[@]}"; do
    echo "=============================================="
    echo "Running: Dataset=${DATASET}, seq_len=${SEQ_LEN}, pred_len=${PRED_LEN}, rho=${RHO}, num_splits=${NUM_SPLITS}, seed=${seed}"
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
        --pred_len ${PRED_LEN} \
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
        --shuffle_train_val False \
        --mode train \
        --num_splits ${NUM_SPLITS} \
        --sam True \
        --rho ${RHO} \
        --sam_adaptive True

    echo "Completed: pred_len=${PRED_LEN}, seed=${seed}"
    echo ""
done

echo "All experiments completed!"
