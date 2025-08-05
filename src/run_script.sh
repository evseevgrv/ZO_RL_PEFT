#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED="false"          
# export WANDB_ENTITY="andrey"   
export WANDB_PROJECT="zo-lib-run"    
export WANDB_API_KEY=""
export HF_TOKEN="" # for llama 

python run.py --model_name="roberta-large" --lora \
            --task_name=SST2 --output_dir=result/SST2-FT-$TAG --num_train_epochs=5  --project_name="zo-rl" \
            --per_device_train_batch_size=16 --load_best_model_at_end --evaluation_strategy=steps  \
            --save_strategy=steps --save_total_limit=1 --eval_steps=1000 --max_steps=20000 --logging_steps=10  \
            --num_eval=1000 --num_train=1000 --num_dev=500 --train_as_classification --perturbation_mode=two_side  \
            --trainer=zorl_sgd --train_set_seed=0 --lr_scheduler_type=constant --save_steps=1000  \
            --learning_rate=1e-3 --zo_eps=1e-3 --momentum=0.0 --weight_decay=0 --module_wise_perturbation=False \
            --zo_tau=1e-3 --zo_use_smoothing=true --zo_beta=0.9 --overwrite_output_dir --report_to="wandb" \
            --scheduler="cosine" --num_training_steps=20000 --warmup_steps=0 --min_lr_ratio=0.1 --scheduler_cycle_length=1
