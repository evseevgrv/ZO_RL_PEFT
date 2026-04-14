#!/bin/bash

set -e

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
N_TRIALS="${N_TRIALS:-20}"
STUDY_NAME="${STUDY_NAME:-llm_ft_optuna_$(date +%Y%m%d_%H%M%S)}"

CONFIG_ARGS=()
if [ -n "${CONFIG:-}" ]; then
    CONFIG_ARGS=(--config "$CONFIG")
fi

STORAGE_ARGS=()
if [ -n "${STORAGE:-}" ]; then
    STORAGE_ARGS=(--storage "$STORAGE")
fi

LOAD_ARGS=()
if [ "${LOAD_IF_EXISTS:-0}" = "1" ]; then
    LOAD_ARGS=(--load_if_exists)
fi

"$PYTHON_BIN" optuna_runner.py \
    --n_trials "$N_TRIALS" \
    --study_name "$STUDY_NAME" \
    "${CONFIG_ARGS[@]}" \
    "${STORAGE_ARGS[@]}" \
    "${LOAD_ARGS[@]}" \
    "$@"
