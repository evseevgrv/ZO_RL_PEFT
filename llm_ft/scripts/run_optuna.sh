#!/bin/bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_DISABLED="false"
export WANDB_ENTITY="${WANDB_ENTITY:-andrey}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"

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

HAS_LOG_TRIALS_ARG=0
for arg in "$@"; do
    if [ "$arg" = "--log_trials" ] || [ "$arg" = "--no_log_trials" ]; then
        HAS_LOG_TRIALS_ARG=1
        break
    fi
done

LOG_TRIALS_ARGS=()
if [ "$HAS_LOG_TRIALS_ARG" = "0" ]; then
    case "$(printf '%s' "${LOG_TRIALS:-true}" | tr '[:upper:]' '[:lower:]')" in
        true|1|yes|y)
            LOG_TRIALS_ARGS=(--log_trials)
            ;;
        false|0|no|n)
            LOG_TRIALS_ARGS=(--no_log_trials)
            ;;
        *)
            echo "Invalid LOG_TRIALS=${LOG_TRIALS}. Use true or false." >&2
            exit 2
            ;;
    esac
fi

"$PYTHON_BIN" optuna_runner.py \
    --n_trials "$N_TRIALS" \
    --study_name "$STUDY_NAME" \
    "${CONFIG_ARGS[@]}" \
    "${STORAGE_ARGS[@]}" \
    "${LOAD_ARGS[@]}" \
    "${LOG_TRIALS_ARGS[@]}" \
    "$@"
