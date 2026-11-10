#!/bin/bash
# One rebuttal run. All arms share the protocol of the paper (Appendix B);
# only the arguments below differ between them.
#
# usage: run_rebuttal.sh TRAINER K LR LR_MU STEPS SEED SELECTION MODEL MODE TASK NAME
#   TRAINER    zo_sgd | zo_adamm | zo_rl_sgd | zo_rl_adamm | zo_rl_jaguar | ...
#   K          candidates per step (zo_rl_*: K + 1 forwards; zo_sgd/zo_adamm with K=1: 2 forwards)
#   LR_MU      0 turns zo_rl_* into greedy Gaussian sampling (mu stays at 0)
#   SELECTION  best | random   (zo_rl_sgd only; ignored elsewhere)
#   MODE       ft | lora
set -euo pipefail
cd "$(dirname "$0")/.."

TRAINER=$1 K=$2 LR=$3 LR_MU=$4 STEPS=$5 SEED=$6 SELECTION=$7 MODEL=$8 MODE=$9 TASK=${10} NAME=${11}

export WANDB_BASE_URL="https://wandb-radfan.ru"
export WANDB_ENTITY="andrey"
export PYTHONHASHSEED="$SEED"
export WANDB_TAGS="${WANDB_TAGS:-rebuttal}"   # e.g. commit-68bb19d,blockA

LORA_FLAG=""
[ "$MODE" = "lora" ] && LORA_FLAG="--lora --lora_r=${ZORL_LORA_R:-8}"   # rank 8 as in the paper unless overridden
# Generation tasks (F1) train on the answer tokens, not as option classification.
CLS_FLAG="--train_as_classification"
case "$TASK" in SQuAD|DROP) CLS_FLAG="" ;; esac

# Evaluate 20 times per run whatever its length, so K=1 runs with 3x steps do not pay 3x eval time.
EVAL_STEPS=$(( STEPS / 20 > 500 ? STEPS / 20 : 500 ))
# transformers >= 4.41 renamed evaluation_strategy to eval_strategy (Qwen3 needs >= 4.51).
EVAL_FLAG="--evaluation_strategy"
"${PY:-python}" -c "import transformers, sys; from packaging import version; sys.exit(version.parse(transformers.__version__) < version.parse('4.41'))" && EVAL_FLAG="--eval_strategy"
OUT="${ZORL_OUT:-$HOME/zorl_runs/$NAME}"   # GigaLearn sets ZORL_OUT inside the run folder
mkdir -p "$OUT"

"${PY:-python}" run.py \
    --model_name="$MODEL" $LORA_FLAG \
    --task_name="$TASK" \
    --trainer="$TRAINER" \
    --output_dir="$OUT" \
    --report_to="wandb" \
    --project_name="zo-rl-rebuttal" \
    --tag="$NAME" \
    --run_name="$NAME" \
    --logging_steps=10 \
    --num_train_epochs=5 \
    --per_device_train_batch_size=16 \
    $EVAL_FLAG="steps" \
    --eval_steps="$EVAL_STEPS" \
    --save_strategy="no" \
    --max_steps="$STEPS" \
    --num_training_steps="$STEPS" \
    --num_eval=1000 \
    --num_train=1000 \
    --num_dev=500 \
    $CLS_FLAG \
    --seed="$SEED" \
    --train_set_seed="$SEED" \
    --perturbation_mode="two_side" \
    --zo_eps=1e-3 \
    --momentum=0.9 \
    --weight_decay=0.0 \
    --module_wise_perturbation=False \
    --overwrite_output_dir \
    --learning_rate="$LR" \
    --scheduler="cosine" \
    --warmup_steps=0 \
    --min_lr_ratio=0.1 \
    --scheduler_cycle_length=1 \
    --tensor_sampling_type="standard_normal" \
    --matrix_sampling_type="Random_baseline" \
    --zo_tau=1e-3 \
    --zo_beta=0.9 \
    --params_ratio=0.1 \
    --lr_mu="$LR_MU" \
    --k_value="$K" \
    --variance="${ZORL_VARIANCE:-1}" \
    ${ZORL_VARIANCE_FINAL:+--variance_final="$ZORL_VARIANCE_FINAL"} \
    --use_grad_first="${ZORL_GRAD_FIRST:-False}" \
    --mu_init_scale="${ZORL_MU_INIT_SCALE:-0}" \
    --zo_rl_sgd_candidate_selection_strategy="$SELECTION" \
    --beta1=0.9 \
    --beta2=0.999 \
    > "$OUT/train.log" 2>&1
echo "$NAME" > "$OUT/DONE"
