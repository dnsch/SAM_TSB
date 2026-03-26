#!/bin/bash

DATASET="ETTh1"
PRED_LEN=96
NUM_SPLITS=3
SEEDS=(1)
SCRIPT="experiments/multi_split/auto_mfles/auto_mfles.py"

for seed in "${SEEDS[@]}"; do
    echo "Running: seed=${seed}"
    python -u ${SCRIPT} \
        --dataset ${DATASET} \
        --seed ${seed} \
        --metrics mse \
        --pred_len ${PRED_LEN} \
        --n_jobs -1 \
        --num_splits ${NUM_SPLITS}
    echo "Completed: seed=${seed}"
done

echo "All experiments completed!"
