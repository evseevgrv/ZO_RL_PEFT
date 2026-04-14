#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export WANDB_DISABLED="false"
export WANDB_ENTITY="andrey"
export WANDB_API_KEY=""
export HF_TOKEN=""

learning_rates=(
    5e-4
)

trainers=(
    # "zo_sgd"
    # "zo_sgd"
    # "zo_rl_sgd"
    # "zo_adamm"
    # "zo_adamm"
    # "zo_rl_adamm"
    "sparse_jaguar_signsgd"
    "sparse_jaguar_signsgd"
    "zo_rl_jaguar"
)

k_values=(
    # "1"
    # "5"
    # "5"
    # "1"
    # "5"
    # "5"
    "1"
    "5"
    "5"
)

run_names=(
    # "zo_sgd_k1"
    # "zo_sgd_k5"
    # "zo_rl_sgd_k5"
    # "zo_adamm_k1"
    # "zo_adamm_k5"
    # "zo_rl_adamm_k5"
    "sparse_jaguar_signsgd_k1"
    "sparse_jaguar_signsgd_k5"
    "zo_rl_jaguar_k5"
)

for lr in "${learning_rates[@]}"; do
    lr_tag="lr_${lr//./_}"
    lr_tag="${lr_tag//e/E}"

    for idx in "${!trainers[@]}"; do
        trainer="${trainers[$idx]}"
        k_value="${k_values[$idx]}"
        run_name="${run_names[$idx]}_${lr_tag}"

        command="python run.py"

        # Model and Task Configuration
        command+=" --model_name=\"roberta-large\""
        # command+=" --lora"
        command+=" --task_name=\"SST2\""
        command+=" --trainer=\"${trainer}\""

        # Logging and Reporting
        command+=" --output_dir=\"result/${run_name}\""
        command+=" --report_to=\"wandb\""
        command+=" --project_name=\"zo-rl\""
        command+=" --logging_steps=10"
        command+=" --run_name=\"${run_name}\""
        command+=" --tag=\"${run_name}\""

        # Training Configuration
        command+=" --num_train_epochs=5"
        command+=" --per_device_train_batch_size=16"
        # command+=" --load_best_model_at_end"
        command+=" --evaluation_strategy=\"steps\""
        command+=" --save_strategy=\"no\""
        # command+=" --save_total_limit=1"
        command+=" --eval_steps=500"
        command+=" --max_steps=10"
        # command+=" --save_steps=1000"

        command+=" --max_length 512"

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
        command+=" --learning_rate=${lr}"
        command+=" --scheduler=\"cosine\""
        command+=" --num_training_steps=10"
        command+=" --warmup_steps=0"
        command+=" --min_lr_ratio=0.1"
        command+=" --scheduler_cycle_length=1"

        # Sampling Methods
        command+=" --tensor_sampling_type=\"standard_normal\""
        command+=" --matrix_sampling_type=\"Random_baseline\""

        command+=" --evaluate_memory=True"

        # Jaguar-Specific Parameters
        command+=" --zo_tau=1e-3"
        command+=" --zo_beta=0.9"

        # Sparse Jaguar-Specific Parameters
        command+=" --params_ratio=0.1"

        command+=" --lr_mu=1e-2"
        command+=" --k_value=${k_value}"
        command+=" --variance=1"
        command+=" --use_grad_first=False"

        command+=" --beta1=0.9"
        command+=" --beta2=0.999"

        echo "Running ${trainer} with k=${k_value} and lr=${lr}"
        eval "$command"
    done
done
