# Optuna configs

Use from `llm_ft`:

```bash
./scripts/run_optuna.sh --config configs/optuna/zo_rl_jaguar.json
```

Each config is self-contained: `base_args` explicitly includes fixed launch
parameters such as `model_name`, `task_name`, dataset sizes, logging settings,
scheduler settings, and `save_strategy: "no"`. `tau` is mapped to `zo_eps`
because `zo_tau` is currently not consumed by the LLM trainer.
