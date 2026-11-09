#!/bin/bash
# Cloud.ru (mlsub) entry point: one rebuttal run per job on one H100.
# usage (via mlsub): --entry llm_ft/scripts/mlsub_entry.sh --args "<run_rebuttal.sh arguments>"
# Needs WANDB_API_KEY in the job environment; image py3.10-torch2.1.2.
set -uo pipefail
cd "$(dirname "$0")/.."
pip install -q transformers==4.28.1 tokenizers==0.13.3 accelerate==0.25.0 datasets==2.16.0 \
    huggingface-hub==0.20.1 wandb==0.16.1 numpy==1.26.4 scikit-learn==1.3.2 protobuf==4.25.1 \
    sentencepiece safetensors==0.4.1 pyarrow==14.0.2 fsspec==2023.10.0 "setuptools<70"
nvidia-smi -L
NAME="${@: -1}"
bash scripts/run_rebuttal.sh "$@"
status=$?
tail -n 40 "$HOME/zorl_runs/$NAME/train.log" | tr '\r' '\n' | tail -n 40
exit $status
