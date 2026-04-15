# Optuna configs

Use from `llm_ft`:

```bash
./scripts/run_optuna.sh --config configs/optuna/zo_rl_jaguar.json
```

Per-trial local logging is enabled by default. To keep only the final
`best_params.json` summary with the best Optuna parameters and their
`test_accuracy`, run:

```bash
LOG_TRIALS=false ./scripts/run_optuna.sh --config configs/optuna/zo_rl_jaguar.json
```

Each config is self-contained: `base_args` explicitly includes fixed launch
parameters such as `model_name`, `task_name`, dataset sizes, logging settings,
scheduler settings, `max_steps: 20000`, `num_training_steps: 20000`, and
`save_strategy: "no"`. Each trial logs to WandB via `use_wandb: true` and
`report_to: "wandb"`. Optuna configs also enable early stopping by default:
`early_stopping: true` monitors `test_acc` every `eval_steps` and stops a trial
after `early_stopping_patience` eval checks without improvement once
`early_stopping_min_steps` has been reached. `tau` is mapped to `zo_eps`
because `zo_tau` is currently not consumed by the LLM trainer.
Optuna optimizes `test_accuracy` by default.

Optuna trial tags, which are also used as WandB run names, include the study,
trainer, task, short model name, tuning mode, and trial number, for example:
`optuna/study/zo_rl_jaguar-SST2-roberta-large-lora/trial_0`.
