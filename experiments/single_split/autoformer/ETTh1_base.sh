#!/bin/bash

# Usage: bash run_autoformer_etth1.sh

# Configuration

DATASET="ETTh1"
SEQ_LEN=512
LABEL_LEN=48
E_LAYERS=2
D_LAYERS=1
FACTOR=3
D_MODEL=512
N_HEADS=8
D_FF=2048
MOVING_AVG=25
DROPOUT=0.05
BATCH_SIZE=32
LRATE=1e-4
PATIENCE=10
MAX_EPOCHS=100
DEVICE="cuda:0"

# Prediction lengths to test

PRED_LENS=(96 192 336 720)

# Seeds to run

# SEEDS=(1 2 3 4 5)
SEEDS=(1 2 3)

# Script path

SCRIPT="experiments/single_split/autoformer/autoformer.py"

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
            --label_len ${LABEL_LEN} \
            --pred_len ${pred_len} \
            --d_model ${D_MODEL} \
            --n_heads ${N_HEADS} \
            --e_layers ${E_LAYERS} \
            --d_layers ${D_LAYERS} \
            --d_ff ${D_FF} \
            --moving_avg ${MOVING_AVG} \
            --factor ${FACTOR} \
            --dropout ${DROPOUT} \
            --use_time_features True \
            --hessian_analysis True

        echo "Completed: pred_len=${pred_len}, seed=${seed}"
        echo ""
    done
done

echo "All experiments completed!"
