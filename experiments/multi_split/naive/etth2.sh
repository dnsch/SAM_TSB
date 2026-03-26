#!/bin/bash

DATASET="ETTh2"
SEQ_LEN=512
PRED_LEN=96
NUM_SPLITS=3
SEEDS=(1)
SCRIPT="experiments/multi_split/naive/naive.py"

for seed in "${SEEDS[@]}"; do
    echo "Running: seed=${seed}"
    python -u ${SCRIPT} \
        --dataset ${DATASET} \
        --seed ${seed} \
        --seq_len ${SEQ_LEN} \
        --pred_len ${PRED_LEN} \
        --num_splits ${NUM_SPLITS}
    echo "Completed: seed=${seed}"
done

echo "All experiments completed!"
