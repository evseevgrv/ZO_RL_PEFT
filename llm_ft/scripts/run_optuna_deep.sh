#!/bin/bash

# Legacy deep launcher kept as a convenience wrapper around optuna_runner.py.

set -e

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
N_TRIALS="${N_TRIALS:-50}"
STUDY_NAME="${STUDY_NAME:-zo_rl_deep_$(date +%Y%m%d_%H%M%S)}"
STORAGE="${STORAGE:-sqlite:///optuna_deep.db}"
CONFIG="${CONFIG:-}"

LOAD_ARGS=()
if [ "${LOAD_IF_EXISTS:-1}" = "1" ]; then
    LOAD_ARGS=(--load_if_exists)
fi

CONFIG_ARGS=()
if [ -n "$CONFIG" ]; then
    CONFIG_ARGS=(--config "$CONFIG")
fi

echo "Starting Optuna study: $STUDY_NAME"
echo "Trials: $N_TRIALS"
echo "Storage: $STORAGE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Results: result/optuna/$STUDY_NAME/best_params.json"

"$PYTHON_BIN" optuna_runner.py \
    --n_trials "$N_TRIALS" \
    --study_name "$STUDY_NAME" \
    --storage "$STORAGE" \
    "${LOAD_ARGS[@]}" \
    "${CONFIG_ARGS[@]}" \
    "$@"
