#!/bin/bash

# Master script to run all AutoTBATS multi-split experiments (quiet mode with logging and timing)

# Usage: bash run_all_tbats.sh

BASE_DIR="experiments/multi_split/auto_tbats"

# Datasets to run

DATASETS=("etth1" "etth2" "exchange_rate" "national_illness")

# Create logs directory

LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Counter for tracking progress

TOTAL=${#DATASETS[@]}
CURRENT=0

# Function to format seconds into human-readable time

format_duration() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))

    if [[ $hours -gt 0 ]]; then
        printf "%dh %dm %ds" $hours $minutes $seconds
    elif [[ $minutes -gt 0 ]]; then
        printf "%dm %ds" $minutes $seconds
    else
        printf "%ds" $seconds
    fi
}

# Record overall start time

OVERALL_START=$(date +%s)
OVERALL_START_FORMATTED=$(date "+%Y-%m-%d %H:%M:%S")

echo "=============================================="
echo "Starting AutoTBATS multi-split experiments"
echo "Total scripts to run: ${TOTAL}"
echo "Datasets: ${DATASETS[*]}"
echo "Logs will be saved to: ${LOG_DIR}"
echo "Started at: ${OVERALL_START_FORMATTED}"
echo "=============================================="
echo ""

for dataset in "${DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    SCRIPT_PATH="${BASE_DIR}/${dataset}.sh"
    LOG_FILE="${LOG_DIR}/${dataset}.log"

    if [[ -f "${SCRIPT_PATH}" ]]; then
        # Record script start time
        SCRIPT_START=$(date +%s)
        SCRIPT_START_FORMATTED=$(date "+%H:%M:%S")

        echo "[${CURRENT}/${TOTAL}] Running: ${dataset}.sh (started at ${SCRIPT_START_FORMATTED})"

        # Run script, redirect output to log file
        bash "${SCRIPT_PATH}" >"${LOG_FILE}" 2>&1
        EXIT_CODE=$?

        # Record script end time and calculate duration
        SCRIPT_END=$(date +%s)
        SCRIPT_DURATION=$((SCRIPT_END - SCRIPT_START))
        DURATION_FORMATTED=$(format_duration $SCRIPT_DURATION)

        if [[ $EXIT_CODE -eq 0 ]]; then
            echo "[${CURRENT}/${TOTAL}] Done: ${dataset}.sh (took ${DURATION_FORMATTED})"
        else
            echo "[${CURRENT}/${TOTAL}] FAILED: ${dataset}.sh (took ${DURATION_FORMATTED}, see ${LOG_FILE})"
        fi
        echo ""
    else
        echo "[${CURRENT}/${TOTAL}] NOT FOUND: ${SCRIPT_PATH}"
        echo ""
    fi
done

# Record overall end time and calculate total duration

OVERALL_END=$(date +%s)
OVERALL_END_FORMATTED=$(date "+%Y-%m-%d %H:%M:%S")
OVERALL_DURATION=$((OVERALL_END - OVERALL_START))
OVERALL_DURATION_FORMATTED=$(format_duration $OVERALL_DURATION)

echo "=============================================="
echo "AutoTBATS multi-split experiments finished!"
echo "Started at:  ${OVERALL_START_FORMATTED}"
echo "Finished at: ${OVERALL_END_FORMATTED}"
echo "Total time:  ${OVERALL_DURATION_FORMATTED}"
echo "Logs saved in: ${LOG_DIR}"
echo "=============================================="
