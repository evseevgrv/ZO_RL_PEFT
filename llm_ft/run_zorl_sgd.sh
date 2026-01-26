#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
export WANDB_DISABLED="false"
export WANDB_ENTITY="andrey"
export WANDB_API_KEY=""
export HF_TOKEN=""

# Список значений learning_rate для перебора
learning_rates=(
# "5e-5" "6e-5" "7e-5" "8e-5" "9e-5" "2e-4" "3e-4" "4e-4" "5e-4"
# 5e-3 6e-3 7e-3 9e-3 1e-2
# 2e-3 3e-3 8e-4 9e-4
5e-8 6e-8 7e-8
)

for mu_lr in "${learning_rates[@]}"; do
    # Формируем тег для output_dir и wandb (заменяем точку и 'e' на допустимые символы)
    TAG="lr_${lr//./_}"        # Заменяем '.' на '_' (например: 1e-3 → lr_1e-3, 0.01 → lr_0_01)
    TAG="${TAG//e/E}"          # Опционально: заменить 'e' на 'E' для читаемости

    command="python run.py"

    # Model and Task Configuration
    command+=" --model_name=\"facebook/opt-1.3b\""
    # command+=" --lora"
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
    command+=" --momentum=0.9"
    command+=" --weight_decay=0.0"
    command+=" --module_wise_perturbation=False"

    # Miscellaneous
    command+=" --overwrite_output_dir"

    # Learning Rate Scheduler Settings
    command+=" --learning_rate=${mu_lr}"
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

    command+=" --lr_mu=1e-3"
    command+=" --k_value=5"
    command+=" --variance=1"
    command+=" --use_grad_first=False"

    command+=" --beta1=0.9"
    command+=" --beta2=0.999"

    # command+=" --log_to_file"

    # command+=" --no_use_wandb"

    # Запуск команды
    eval "$command"
done
