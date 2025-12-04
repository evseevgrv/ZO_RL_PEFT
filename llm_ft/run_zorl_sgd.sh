#!/bin/bash

export CUDA_VISIBLE_DEVICES=6
export WANDB_DISABLED="false"
export WANDB_ENTITY="andrey"
export WANDB_API_KEY=""
export HF_TOKEN=""

# Список значений learning_rate для перебора
learning_rates=("5e-3")

for lr in "${learning_rates[@]}"; do
    # Формируем тег для output_dir и wandb (заменяем точку и 'e' на допустимые символы)
    TAG="lr_${lr//./_}" 
    TAG="${TAG//e/E}"

    command="python run.py"

    # Model and Task Configuration
    command+=" --model_name=\"roberta-large\""
    command+=" --lora"
    command+=" --task_name=\"SST2\""
    command+=" --trainer=\"zo_rl_sgd\""

    # Logging and Reporting
    command+=" --output_dir=\"result/SST2-FT-${TAG}\""
    command+=" --report_to=\"wandb\""
    command+=" --project_name=\"zo-rl\""
    command+=" --logging_steps=10"
    command+=" --run_name=\"${TAG}\""  # Если run.py поддерживает --run_name

    # Training Configuration
    command+=" --num_train_epochs=5"
    command+=" --per_device_train_batch_size=16"
    command+=" --load_best_model_at_end"
    command+=" --evaluation_strategy=\"steps\""
    command+=" --save_strategy=\"steps\""
    command+=" --save_total_limit=1"
    command+=" --eval_steps=500"
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
    command+=" --zo_beta=0.9"

    # Sparse Jaguar-Specific Parameters
    command+=" --params_ratio=0.1"

    command+=" --lr_mu=0"
    command+=" --k_value=1"
    command+=" --variance=1"
    command+=" --use_grad_first=False"

    # Запуск команды
    eval "$command"
done
