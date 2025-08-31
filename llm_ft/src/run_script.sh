#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export WANDB_DISABLED="false"
# export WANDB_PROJECT="zo-lib-run"       
# export WANDB_ENTITY="andrey"   
export WANDB_API_KEY=""
export HF_TOKEN="" # for llama

command="python run.py"

# Model and Task Configuration
command+=" --model_name=\"facebook/opt-1.3b\""
command+="" # type of Fine-Tuning 
command+=" --task_name=\"SST2\""
command+=" --trainer=\"zo_sampling_muon\""

# Logging and Reporting
# TODO: output_dir is constructed in Python using args.tag, do we need it? 
command+=" --output_dir=\"result/SST2-FT-\$TAG\""
command+=" --report_to=\"wandb\""
command+=" --project_name=\"zo-bench\""
command+=" --logging_steps=10"

# Training Configuration
command+=" --num_train_epochs=5"
command+=" --per_device_train_batch_size=16"
command+=" --load_best_model_at_end"
command+=" --evaluation_strategy=\"steps\""
command+=" --save_strategy=\"steps\""
command+=" --save_total_limit=1"
command+=" --eval_steps=1000"
command+=" --max_steps=20000"
command+=" --save_steps=1000"

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
command+=" --lr_scheduler_type=\"constant\"" # FIXME: need to delete this 
command+=" --scheduler=\"cosine\""
command+=" --num_training_steps=20000"
command+=" --warmup_steps=0"
command+=" --min_lr_ratio=0.1"
command+=" --scheduler_cycle_length=1"

# Sampling Methods
command+=" --vector_sampling_type=\"standard_normal\""
command+=" --matrix_sampling_type=\"Householder_reflection\""

# Jaguar-Specific Parameters
command+=" --zo_tau=1e-3"
command+=" --zo_beta=0.9"
command+=" --zo_use_smoothing=true"

# ZO RL-Specific Parameters
command+=" --k_value=5"
command+=" --variance=1e-3"
command+=" --lr_mu=5e-3"

# Learning Rate Loop
for learning_rate in 1e-3; do
    full_command="$command --learning_rate=$learning_rate"
    eval "$full_command"
done
