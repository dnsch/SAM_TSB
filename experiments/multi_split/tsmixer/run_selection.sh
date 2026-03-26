#!/bin/bash

# Master script to run TSMixer multi-split experiments (quiet mode with logging and timing)

# Automatically discovers which scripts exist in each method directory

# Usage: bash run_selection.sh

BASE_DIR="experiments/multi_split/tsmixer"

# Methods/variants to check (will only run if directory and scripts exist)

METHODS=("base" "sam" "asam" "fsam" "gsam")

# Create logs directory

LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${LOG_DIR}"

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

# First, discover all scripts that exist

declare -a SCRIPTS_TO_RUN
for method in "${METHODS[@]}"; do
    METHOD_DIR="${BASE_DIR}/${method}"
    if [[ -d "${METHOD_DIR}" ]]; then
        for script in "${METHOD_DIR}"/*.sh; do
            if [[ -f "${script}" ]]; then
                SCRIPTS_TO_RUN+=("${script}")
            fi
        done
    fi
done

TOTAL=${#SCRIPTS_TO_RUN[@]}
CURRENT=0

# Record overall start time

OVERALL_START=$(date +%s)
OVERALL_START_FORMATTED=$(date "+%Y-%m-%d %H:%M:%S")

echo "=============================================="
echo "Starting TSMixer multi-split experiments"
echo "Total scripts found: ${TOTAL}"
echo "Logs will be saved to: ${LOG_DIR}"
echo "Started at: ${OVERALL_START_FORMATTED}"
echo "=============================================="
echo ""

# List discovered scripts

echo "Scripts to run:"
for script in "${SCRIPTS_TO_RUN[@]}"; do
    # Extract method and dataset from path
    method=$(basename "$(dirname "${script}")")
    dataset=$(basename "${script}" .sh)
    echo "  - ${method}/${dataset}.sh"
done
echo ""

# Run each discovered script

for script in "${SCRIPTS_TO_RUN[@]}"; do
    CURRENT=$((CURRENT + 1))

    # Extract method and dataset from path
    method=$(basename "$(dirname "${script}")")
    dataset=$(basename "${script}" .sh)
    LOG_FILE="${LOG_DIR}/${method}_${dataset}.log"

    # Record script start time
    SCRIPT_START=$(date +%s)
    SCRIPT_START_FORMATTED=$(date "+%H:%M:%S")

    echo "[${CURRENT}/${TOTAL}] Running: ${method}/${dataset}.sh (started at ${SCRIPT_START_FORMATTED})"

    # Run script, redirect output to log file
    bash "${script}" >"${LOG_FILE}" 2>&1
    EXIT_CODE=$?

    # Record script end time and calculate duration
    SCRIPT_END=$(date +%s)
    SCRIPT_DURATION=$((SCRIPT_END - SCRIPT_START))
    DURATION_FORMATTED=$(format_duration $SCRIPT_DURATION)

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "[${CURRENT}/${TOTAL}] Done: ${method}/${dataset}.sh (took ${DURATION_FORMATTED})"
    else
        echo "[${CURRENT}/${TOTAL}] FAILED: ${method}/${dataset}.sh (took ${DURATION_FORMATTED}, see ${LOG_FILE})"
    fi
    echo ""
done

# Record overall end time and calculate total duration

OVERALL_END=$(date +%s)
OVERALL_END_FORMATTED=$(date "+%Y-%m-%d %H:%M:%S")
OVERALL_DURATION=$((OVERALL_END - OVERALL_START))
OVERALL_DURATION_FORMATTED=$(format_duration $OVERALL_DURATION)

echo "=============================================="
echo "TSMixer multi-split experiments finished!"
echo "Started at:  ${OVERALL_START_FORMATTED}"
echo "Finished at: ${OVERALL_END_FORMATTED}"
echo "Total time:  ${OVERALL_DURATION_FORMATTED}"
echo "Logs saved in: ${LOG_DIR}"
echo "=============================================="
