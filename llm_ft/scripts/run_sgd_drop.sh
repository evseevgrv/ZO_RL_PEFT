#!/bin/bash
#
# ZO-SGD (MeZO) fine-tuning on DROP.
# DROP is a *generative* QA task (metric: token-level F1). Unlike the
# multiple-choice ARC recipe in run_sgd.sh, it must NOT set
# --train_as_classification (run.py raises for generative tasks) and it relies
# on the generation args below for evaluation. Otherwise the ZO-SGD
# hyperparameters mirror run_sgd.sh so the two scripts stay comparable.

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=1
export WANDB_DISABLED="false"
export WANDB_ENTITY="andrey"
export WANDB_API_KEY=""
export HF_TOKEN=""

learning_rates=(
    5e-5 1e-4 5e-4 1e-3 5e-3 1e-2
)

for lr in "${learning_rates[@]}"; do
    # Tag for output_dir / wandb (replace '.' and 'e' with safe characters)
    TAG="lr_${lr//./_}"        # e.g. 1e-3 -> lr_1e-3, 0.01 -> lr_0_01
    TAG="${TAG//e/E}"          # optional: 'e' -> 'E' for readability

    command="python run.py"

    # Model and Task Configuration
    command+=" --model_name=\"facebook/opt-13b\""   # swap to facebook/opt-1.3b for a lighter/faster run
    command+=" --lora"
    command+=" --task_name=\"DROP\""
    command+=" --template_ver=0"
    command+=" --trainer=\"zo_sgd\""

    # Logging and Reporting
    command+=" --output_dir=\"result/DROP-FT-${TAG}\""
    command+=" --report_to=\"wandb\""
    command+=" --project_name=\"zo-rl\""
    command+=" --logging_steps=10"
    command+=" --run_name=\"DROP-${TAG}\""

    # Training Configuration
    command+=" --num_train_epochs=5"
    command+=" --per_device_train_batch_size=16"
    command+=" --load_best_model_at_end"
    command+=" --evaluation_strategy=\"steps\""
    command+=" --save_strategy=\"steps\""
    command+=" --save_total_limit=1"
    # Generative eval is autoregressive (slow), so evaluate less often than the
    # classification recipe.
    command+=" --eval_steps=4000"
    command+=" --max_steps=20000"
    command+=" --save_steps=4000"

    # Dataset Settings
    # DROP: ~77k train / ~9.5k validation examples; subsample MeZO-style.
    # NOTE: DROP is generative — do NOT add --train_as_classification here.
    command+=" --num_train=1000"
    command+=" --num_dev=500"
    command+=" --num_eval=1000"
    command+=" --train_set_seed=0"
    command+=" --max_length=2048"   # DROP passages are long; keep the full context window

    # Generation (used to produce answers for F1 scoring)
    # Defaults already give greedy decoding (sampling off, num_beams=1) and stop
    # at the first newline (eos_token="\n"), which matches DROPTemplate.
    command+=" --max_new_tokens=20"
    command+=" --num_beams=1"

    # Training Hyperparameters
    command+=" --perturbation_mode=\"two_side\""
    command+=" --zo_eps=1e-3"
    command+=" --momentum=0.9"
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

    # Jaguar-Specific Parameters (ignored by zo_sgd; kept for parity with run_sgd.sh)
    command+=" --zo_tau=1e-3"
    command+=" --zo_beta=0.9"

    # Sparse Jaguar-Specific Parameters
    command+=" --params_ratio=0.1"

    command+=" --lr_mu=1e-3"
    command+=" --k_value=5"
    command+=" --variance=1"
    command+=" --use_grad_first=False"

    command+=" --beta1=0.9"
    command+=" --beta2=0.999"

    # command+=" --log_to_file"
    # command+=" --no_use_wandb"

    # Run
    eval "$command"
done
