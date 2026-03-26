#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH}"

export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED="false"
export WANDB_ENTITY=""
export WANDB_API_KEY=""
export HF_TOKEN=""
export HF_HOME="/tmp/zorl_hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="/tmp/zorl_tmp"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR"

# List of learning_rate values to sweep
learning_rates=(
    1e-5 
)

for lr in "${learning_rates[@]}"; do
    # Build tag for output_dir and wandb (replace '.' and 'e' with valid characters)
    TAG="lr_${lr//./_}"        # Replace '.' with '_' (e.g.: 1e-3 → lr_1e-3, 0.01 → lr_0_01)
    TAG="${TAG//e/E}"          # Optionally replace 'e' with 'E' for readability

    command="PYTHONNOUSERSITE=1 PYTHONPATH=\"$SCRIPT_DIR\" python \"$SCRIPT_DIR/run.py\""

    # Model and Task Configuration
    command+=" --model_name=\"roberta-large\""
    # command+=" --lora"
    command+=" --task_name=\"SST2\""
    command+=" --trainer=\"zo_adamu\""

    # Logging and Reporting
    command+=" --output_dir=\"/tmp/zorl_runs/${TAG}\""
    command+=" --report_to=\"none\""
    command+=" --project_name=\"zo-rl\""
    command+=" --logging_steps=10"
    command+=" --run_name=\"${TAG}\""  # If run.py supports --run_name
    command+=" --log_dir=\"logs\""

    # Training Configuration
    command+=" --num_train_epochs=5"
    command+=" --per_device_train_batch_size=16"
    # command+=" --load_best_model_at_end"
    command+=" --evaluation_strategy=\"steps\""
    command+=" --save_strategy=\"no\""
    # command+=" --save_total_limit=1"
    command+=" --eval_steps=500"
    command+=" --max_steps=20000"
    # command+=" --save_steps=1000"

    # Dataset Settings
    command+=" --num_eval=1000"
    command+=" --num_train=1000"
    command+=" --num_dev=500"
    command+=" --train_as_classification"
    command+=" --train_set_seed=0"

    # Training Hyperparameters
    command+=" --perturbation_mode=\"two_side\""
    command+=" --zo_eps=1e-3"
    command+=" --momentum=0.0"
    command+=" --weight_decay=0.0"
    command+=" --module_wise_perturbation=False"

    # Miscellaneous
    command+=" --overwrite_output_dir"

    # Learning Rate Scheduler Settings
    command+=" --learning_rate=${lr}"
    command+=" --scheduler=\"cosine\""
    command+=" --num_training_steps=20000"
    command+=" --warmup_steps=0"
    command+=" --min_lr_ratio=0.1"
    command+=" --scheduler_cycle_length=1"

    # Sampling Methods
    command+=" --tensor_sampling_type=\"standard_normal\""
    command+=" --matrix_sampling_type=\"Random_baseline\"" 

    # Jaguar-Specific Parameters
    command+=" --zo_tau=1e-3"
    command+=" --zo_beta=0.0"

    # Sparse Jaguar-Specific Parameters
    command+=" --params_ratio=0.1"

    command+=" --lr_mu=4e-2"
    command+=" --k_value=5"
    command+=" --variance=1"
    command+=" --use_grad_first=False"

    command+=" --beta1=0.9"
    command+=" --beta2=0.999"

    command+=" --log_to_file"
    command+=" --no_use_wandb"

    # Run command
    eval "$command"
done
