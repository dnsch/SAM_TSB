#!/bin/bash

DATASET="ETTh2"
PRED_LEN=96
NUM_SPLITS=3
SEASON_LENGTH=24
SEEDS=(1)
SCRIPT="experiments/multi_split/seasonal_naive/seasonal_naive.py"

for seed in "${SEEDS[@]}"; do
    echo "Running: seed=${seed}"
    python -u ${SCRIPT} \
        --dataset ${DATASET} \
        --seed ${seed} \
        --metrics mse \
        --pred_len ${PRED_LEN} \
        --num_splits ${NUM_SPLITS} \
        --season_length ${SEASON_LENGTH}
    echo "Completed: seed=${seed}"
done

echo "All experiments completed!"
