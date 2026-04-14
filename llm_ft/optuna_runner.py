#!/usr/bin/env python3
"""Generic Optuna runner for llm_ft/run.py.

This runner intentionally does not reuse the legacy optuna_tuning.py scripts:
it keeps the Optuna layer small, config-driven, and independent from a single
trainer/task setup.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


LOGGER = logging.getLogger("optuna_runner")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_METRIC_KEY = "val_accuracy"
DEFAULT_DIRECTION = "maximize"
DEFAULT_STUDY_PREFIX = "llm_ft_optuna"
BETA_TRAINERS = {
    "zo_rl_jaguar",
    "jaguar_signsgd",
    "jaguar_muon",
    "sparse_jaguar_signsgd",
    "sparse_jaguar_muon",
}


DEFAULT_BASE_ARGS: Dict[str, Any] = {
    "model_name": "roberta-large",
    "lora": True,
    "task_name": "SST2",
    "trainer": "zo_rl_jaguar",
    "project_name": "zo-rl-optuna",
    "report_to": "wandb",
    "logging_steps": 10,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 16,
    "load_best_model_at_end": False,
    "evaluation_strategy": "steps",
    "save_strategy": "no",
    "save_total_limit": 1,
    "eval_steps": 500,
    "max_steps": 20000,
    "save_steps": 1000,
    "num_eval": 1000,
    "num_train": 5000,
    "num_dev": 500,
    "train_as_classification": True,
    "train_set_seed": 0,
    "perturbation_mode": "two_side",
    "momentum": 0.0,
    "weight_decay": 0.0,
    "module_wise_perturbation": False,
    "output_dir": "result/optuna_trial",
    "overwrite_output_dir": True,
    "scheduler": "cosine",
    "num_training_steps": 20000,
    "warmup_steps": 0,
    "min_lr_ratio": 0.1,
    "scheduler_cycle_length": 1,
    "tensor_sampling_type": "standard_normal",
    "matrix_sampling_type": "Random_baseline",
    "params_ratio": 0.1,
    "k_value": 5,
    "variance": 1.0,
    "use_grad_first": False,
    "use_wandb": True,
}


DEFAULT_SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "learning_rate": {
        "type": "float",
        "low": 1e-8,
        "high": 1e-3,
        "log": True,
    },
    "tau": {
        "type": "float",
        "low": 1e-6,
        "high": 1e-2,
        "log": True,
        "target_arg": "zo_eps",
    },
    "beta": {
        "type": "float",
        "low": 0.0,
        "high": 0.999,
        "target_arg": "zo_beta",
    },
}


DEFAULT_CONFIG: Dict[str, Any] = {
    "base_args": DEFAULT_BASE_ARGS,
    "search_space": DEFAULT_SEARCH_SPACE,
    "metric_key": DEFAULT_METRIC_KEY,
    "metric_fallbacks": [],
    "direction": DEFAULT_DIRECTION,
    "env": {},
    "tag_prefix": "",
    "extend_default_search_space": False,
    "inherit_base_args": True,
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return data


def deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if (
            isinstance(value, Mapping)
            and isinstance(base.get(key), MutableMapping)
        ):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def normalize_config(user_config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if user_config is None:
        config = copy.deepcopy(DEFAULT_CONFIG)
        validate_config(config)
        return config

    config = copy.deepcopy(DEFAULT_CONFIG)
    user_config = copy.deepcopy(dict(user_config))

    inherit_base_args = user_config.pop("inherit_base_args", config["inherit_base_args"])
    extend_default_search_space = user_config.pop(
        "extend_default_search_space",
        config["extend_default_search_space"],
    )

    user_base_args = user_config.pop("base_args", None)
    if user_base_args is not None:
        if not isinstance(user_base_args, Mapping):
            raise ValueError("config.base_args must be a JSON object")
        config["base_args"] = copy.deepcopy(DEFAULT_BASE_ARGS) if inherit_base_args else {}
        deep_update(config["base_args"], user_base_args)

    user_search_space = user_config.pop("search_space", None)
    if user_search_space is not None:
        if not isinstance(user_search_space, Mapping):
            raise ValueError("config.search_space must be a JSON object")
        config["search_space"] = (
            copy.deepcopy(DEFAULT_SEARCH_SPACE)
            if extend_default_search_space
            else {}
        )
        deep_update(config["search_space"], user_search_space)

    for key, value in user_config.items():
        if isinstance(value, Mapping) and isinstance(config.get(key), MutableMapping):
            deep_update(config[key], value)
        else:
            config[key] = value

    config["inherit_base_args"] = inherit_base_args
    config["extend_default_search_space"] = extend_default_search_space
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    direction = config.get("direction", DEFAULT_DIRECTION)
    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be either 'maximize' or 'minimize'")

    search_space = config.get("search_space", {})
    if not isinstance(search_space, Mapping) or not search_space:
        raise ValueError("search_space must be a non-empty JSON object")

    seen_targets: Dict[str, str] = {}
    for name, spec in search_space.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"search_space.{name} must be a JSON object")
        validate_search_spec(name, spec)
        target = str(spec.get("target_arg", name))
        if target in seen_targets:
            other = seen_targets[target]
            raise ValueError(
                f"search_space entries '{other}' and '{name}' both target --{target}"
            )
        seen_targets[target] = name


def validate_search_spec(name: str, spec: Mapping[str, Any]) -> None:
    kind = infer_spec_type(spec)
    if kind == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"search_space.{name}.choices must be a non-empty list")
        return

    low, high = get_bounds(name, spec)
    if low > high:
        raise ValueError(f"search_space.{name} has low > high")

    if kind == "float" and spec.get("log", False) and spec.get("step") is not None:
        raise ValueError(f"search_space.{name} cannot use both log=true and step")


def infer_spec_type(spec: Mapping[str, Any]) -> str:
    kind = spec.get("type")
    if kind is None:
        if "choices" in spec:
            kind = "categorical"
        else:
            kind = "float"
    if kind not in {"float", "int", "categorical"}:
        raise ValueError(f"Unsupported search space type: {kind}")
    return str(kind)


def get_bounds(name: str, spec: Mapping[str, Any]) -> Tuple[Any, Any]:
    low = spec.get("low", spec.get("min"))
    high = spec.get("high", spec.get("max"))
    if low is None or high is None:
        raise ValueError(f"search_space.{name} must define low/high or min/max")
    return low, high


def suggest_value(trial: Any, name: str, spec: Mapping[str, Any]) -> Any:
    kind = infer_spec_type(spec)
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))

    low, high = get_bounds(name, spec)
    log = bool(spec.get("log", False))
    step = spec.get("step")
    if kind == "int":
        kwargs = {"log": log}
        if step is not None:
            kwargs["step"] = int(step)
        return trial.suggest_int(name, int(low), int(high), **kwargs)

    kwargs = {"log": log}
    if step is not None:
        kwargs["step"] = float(step)
    return trial.suggest_float(name, float(low), float(high), **kwargs)


def representative_value(name: str, spec: Mapping[str, Any]) -> Any:
    kind = infer_spec_type(spec)
    if kind == "categorical":
        return spec["choices"][0]

    low, high = get_bounds(name, spec)
    if kind == "int":
        return int(low)

    low = float(low)
    high = float(high)
    if spec.get("log", False):
        if low <= 0 or high <= 0:
            raise ValueError(f"Log-scaled search_space.{name} bounds must be positive")
        return math.sqrt(low * high)
    return (low + high) / 2.0


def sample_trial_params(trial: Any, search_space: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    return {name: suggest_value(trial, name, spec) for name, spec in search_space.items()}


def dry_run_params(search_space: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    return {name: representative_value(name, spec) for name, spec in search_space.items()}


def target_trial_args(
    trial_params: Mapping[str, Any],
    search_space: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    targeted: Dict[str, Any] = {}
    for name, value in trial_params.items():
        target = str(search_space[name].get("target_arg", name))
        targeted[target] = value
    return targeted


def sanitize_path_component(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return clean or "study"


def make_trial_tag(config: Mapping[str, Any], study_slug: str, trial_number: int) -> str:
    base_args = config.get("base_args", {})
    base_tag = str(base_args.get("tag") or config.get("tag_prefix") or "")
    tag_parts = ["optuna", study_slug]
    if base_tag:
        tag_parts.append(sanitize_path_component(base_tag))
    tag_parts.append(f"trial_{trial_number}")
    return "/".join(tag_parts)


def render_cli_arg(name: str, value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, bool):
        return [f"--{name}={str(value).lower()}"]
    if isinstance(value, (list, tuple)):
        rendered: List[str] = []
        for item in value:
            rendered.extend(render_cli_arg(name, item))
        return rendered
    return [f"--{name}={value}"]


def build_command(
    config: Mapping[str, Any],
    trial_params: Mapping[str, Any],
    study_slug: str,
    trial_number: int,
) -> Tuple[List[str], Dict[str, Any], str]:
    search_space = config["search_space"]
    base_args = copy.deepcopy(dict(config.get("base_args", {})))
    trial_args = target_trial_args(trial_params, search_space)
    tag = make_trial_tag(config, study_slug, trial_number)

    # run.py requires output_dir at parse time and then overwrites it from tag.
    base_args.pop("tag", None)
    base_args["tag"] = tag
    base_args["output_dir"] = f"result/{tag}"
    base_args.update(trial_args)

    command = [sys.executable, "run.py"]
    for name, value in base_args.items():
        command.extend(render_cli_arg(name, value))
    return command, trial_args, tag


def warn_about_noops(config: Mapping[str, Any]) -> None:
    search_space = config["search_space"]
    targets = {str(spec.get("target_arg", name)) for name, spec in search_space.items()}
    trainer = str(config.get("base_args", {}).get("trainer", ""))

    if "zo_tau" in targets:
        LOGGER.warning(
            "--zo_tau is currently parsed by run.py but not consumed by trainer.py; "
            "use target_arg='zo_eps' for perturbation-scale tuning."
        )
    if "zo_beta" in targets and trainer and trainer not in BETA_TRAINERS:
        LOGGER.warning(
            "--zo_beta is tuned, but trainer '%s' is not known to consume it.",
            trainer,
        )


def metric_candidates(config: Mapping[str, Any]) -> List[str]:
    metric_key = str(config.get("metric_key", DEFAULT_METRIC_KEY))
    fallbacks = config.get("metric_fallbacks", [])
    if not isinstance(fallbacks, list):
        raise ValueError("metric_fallbacks must be a list")
    keys = [metric_key]
    for key in fallbacks:
        key = str(key)
        if key not in keys:
            keys.append(key)
    return keys


def extract_metric(results: Mapping[str, Any], config: Mapping[str, Any]) -> Tuple[str, float]:
    for key in metric_candidates(config):
        value = results.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    raise KeyError(
        "None of the metric keys were found as numeric values: "
        + ", ".join(metric_candidates(config))
    )


def failure_value(direction: str) -> float:
    return 1e30 if direction == "minimize" else -1e30


def read_log_tail(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"<could not read {path}: {exc}>"
    return "\n".join(lines[-max_lines:])


def log_trial_failure_details(trial_number: int, trial_dir: Path) -> None:
    stderr_tail = read_log_tail(trial_dir / "stderr.log")
    stdout_tail = read_log_tail(trial_dir / "stdout.log")

    if stderr_tail:
        LOGGER.error("Trial %s stderr tail:\n%s", trial_number, stderr_tail)
    if stdout_tail:
        LOGGER.error("Trial %s stdout tail:\n%s", trial_number, stdout_tail)


def run_training_trial(
    trial: Any,
    config: Mapping[str, Any],
    study_slug: str,
    results_root: Path,
    log_trials: bool,
) -> float:
    trial_params = sample_trial_params(trial, config["search_space"])
    command, trial_args, tag = build_command(config, trial_params, study_slug, trial.number)
    trial_dir = results_root / f"trial_{trial.number}"
    if log_trials:
        trial_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting trial %s with params: %s", trial.number, trial_params)
    LOGGER.info("Trial %s command: %s", trial.number, shlex.join(command))

    if log_trials:
        write_json(trial_dir / "params.json", {
            "trial_params": trial_params,
            "target_args": trial_args,
            "tag": tag,
        })
        write_json(trial_dir / "command.json", command)
        (trial_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in config.get("env", {}).items()})

    if log_trials:
        stdout_target = (trial_dir / "stdout.log").open("w", encoding="utf-8")
        stderr_target = (trial_dir / "stderr.log").open("w", encoding="utf-8")
    else:
        stdout_target = subprocess.DEVNULL
        stderr_target = subprocess.DEVNULL

    try:
        result = subprocess.run(
            command,
            cwd=str(SCRIPT_DIR),
            env=env,
            stdout=stdout_target,
            stderr=stderr_target,
            text=True,
            check=False,
        )
    finally:
        if log_trials:
            stdout_target.close()
            stderr_target.close()

    if result.returncode != 0:
        if log_trials:
            LOGGER.error(
                "Trial %s failed with return code %s. See %s",
                trial.number,
                result.returncode,
                trial_dir,
            )
            log_trial_failure_details(trial.number, trial_dir)
        else:
            LOGGER.error(
                "Trial %s failed with return code %s. Re-run with LOG_TRIALS=true for local logs.",
                trial.number,
                result.returncode,
            )
            shutil.rmtree(trial_dir, ignore_errors=True)
        return failure_value(str(config.get("direction", DEFAULT_DIRECTION)))

    result_file = SCRIPT_DIR / "result" / tag / "results.json"
    if not result_file.exists():
        LOGGER.error("Trial %s result file not found: %s", trial.number, result_file)
        if not log_trials:
            shutil.rmtree(trial_dir, ignore_errors=True)
        return failure_value(str(config.get("direction", DEFAULT_DIRECTION)))

    results = load_json(result_file)
    if log_trials:
        write_json(trial_dir / "metrics.json", results)

    try:
        metric_key, metric_value = extract_metric(results, config)
    except KeyError as exc:
        LOGGER.error("Trial %s has no target metric: %s", trial.number, exc)
        if not log_trials:
            shutil.rmtree(trial_dir, ignore_errors=True)
        return failure_value(str(config.get("direction", DEFAULT_DIRECTION)))

    trial.set_user_attr("metric_key", metric_key)
    for key, value in results.items():
        if isinstance(value, (int, float)):
            trial.set_user_attr(key, value)

    LOGGER.info(
        "Trial %s completed: %s=%s",
        trial.number,
        metric_key,
        metric_value,
    )
    if not log_trials:
        shutil.rmtree(trial_dir, ignore_errors=True)
    return metric_value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def make_study_summary(
    study: Any,
    study_name: str,
    study_slug: str,
    config: Mapping[str, Any],
    log_trials: bool,
) -> Dict[str, Any]:
    try:
        best_trial = study.best_trial
        best_test_accuracy = best_trial.user_attrs.get("test_accuracy")
        if best_test_accuracy is None:
            best_test_accuracy = best_trial.user_attrs.get("accuracy")
        best_summary = {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "target_args": target_trial_args(best_trial.params, config["search_space"]),
            "test_accuracy": best_test_accuracy,
            "user_attrs": best_trial.user_attrs,
        }
    except ValueError:
        best_summary = None

    if not log_trials:
        summary = {
            "study_name": study_name,
            "study_slug": study_slug,
            "optimized_metric_key": config.get("metric_key", DEFAULT_METRIC_KEY),
        }
        if best_summary is not None:
            summary.update({
                "best_trial_number": best_summary["number"],
                "best_params": best_summary["params"],
                "best_target_args": best_summary["target_args"],
                "optimized_metric_value": best_summary["value"],
                "test_accuracy": best_summary["test_accuracy"],
            })
        return summary

    summary = {
        "study_name": study_name,
        "study_slug": study_slug,
        "direction": config.get("direction", DEFAULT_DIRECTION),
        "metric_key": config.get("metric_key", DEFAULT_METRIC_KEY),
        "best_trial": best_summary,
    }

    if best_summary is not None:
        summary["best_params"] = best_summary["params"]
        summary["best_target_args"] = best_summary["target_args"]
        summary["best_value"] = best_summary["value"]
        summary["test_accuracy"] = best_summary["test_accuracy"]

    if log_trials:
        summary["trials"] = [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
                "user_attrs": trial.user_attrs,
            }
            for trial in study.trials
        ]

    return summary


def run_study(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError as exc:
        raise SystemExit(
            "Optuna is not installed. Install dependencies with "
            "`pip install -r llm_ft/requirements.txt` or `pip install 'optuna>=4,<5'`."
        ) from exc

    study_name = args.study_name or f"{DEFAULT_STUDY_PREFIX}_{int(time.time())}"
    study_slug = sanitize_path_component(study_name)
    results_root = SCRIPT_DIR / "result" / "optuna" / study_slug
    results_root.mkdir(parents=True, exist_ok=True)

    if args.gpu_id is not None:
        config = copy.deepcopy(dict(config))
        env = dict(config.get("env", {}))
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        config["env"] = env

    warn_about_noops(config)
    if args.log_trials:
        write_json(results_root / "config.json", config)

    LOGGER.info("Starting Optuna study '%s'", study_name)
    LOGGER.info("Results root: %s", results_root)
    LOGGER.info("Metric key: %s", config.get("metric_key", DEFAULT_METRIC_KEY))
    LOGGER.info("Local trial logging: %s", args.log_trials)

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        load_if_exists=args.load_if_exists,
        direction=str(config.get("direction", DEFAULT_DIRECTION)),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=args.pruner_startup_trials,
            n_warmup_steps=args.pruner_warmup_steps,
        ),
    )
    study.optimize(
        lambda trial: run_training_trial(
            trial,
            config,
            study_slug,
            results_root,
            log_trials=args.log_trials,
        ),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
    )

    summary = make_study_summary(
        study,
        study_name,
        study_slug,
        config,
        log_trials=args.log_trials,
    )
    write_json(results_root / "best_params.json", summary)

    completed = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    LOGGER.info("Optimization finished: %s completed trials", len(completed))
    if "best_trial" in summary and summary["best_trial"] is not None:
        LOGGER.info("Best value: %s", summary["best_trial"]["value"])
        LOGGER.info("Best params: %s", summary["best_trial"]["params"])
        LOGGER.info("Best target args: %s", summary["best_trial"]["target_args"])
        LOGGER.info("Best test_accuracy: %s", summary["best_trial"]["test_accuracy"])
    elif "best_params" in summary:
        LOGGER.info("Best value: %s", summary["optimized_metric_value"])
        LOGGER.info("Best params: %s", summary["best_params"])
        LOGGER.info("Best target args: %s", summary["best_target_args"])
        LOGGER.info("Best test_accuracy: %s", summary["test_accuracy"])
    LOGGER.info("Summary saved to: %s", results_root / "best_params.json")


def run_dry_run(config: Mapping[str, Any], args: argparse.Namespace) -> None:
    study_name = args.study_name or f"{DEFAULT_STUDY_PREFIX}_dry_run"
    study_slug = sanitize_path_component(study_name)
    trial_params = dry_run_params(config["search_space"])
    command, target_args, tag = build_command(config, trial_params, study_slug, 0)
    payload = {
        "study_name": study_name,
        "tag": tag,
        "metric_key": config.get("metric_key", DEFAULT_METRIC_KEY),
        "log_trials": args.log_trials,
        "trial_params": trial_params,
        "target_args": target_args,
        "command": command,
        "command_text": shlex.join(command),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic Optuna runner for llm_ft/run.py")
    parser.add_argument("--config", type=Path, default=None, help="Path to JSON config")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--study_name", type=str, default=None, help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL")
    parser.add_argument("--load_if_exists", action="store_true", help="Resume existing study")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel Optuna jobs")
    parser.add_argument("--timeout", type=float, default=None, help="Study timeout in seconds")
    parser.add_argument("--gpu_id", type=str, default=None, help="Set CUDA_VISIBLE_DEVICES")
    parser.add_argument("--dry_run", action="store_true", help="Print one generated command and exit")
    parser.add_argument("--print_default_config", action="store_true", help="Print the default JSON config and exit")
    parser.add_argument("--pruner_startup_trials", type=int, default=5, help="MedianPruner startup trials")
    parser.add_argument("--pruner_warmup_steps", type=int, default=5, help="MedianPruner warmup steps")
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        "--log_trials",
        dest="log_trials",
        action="store_true",
        default=True,
        help="Write per-trial command/stdout/stderr/metrics files",
    )
    log_group.add_argument(
        "--no_log_trials",
        dest="log_trials",
        action="store_false",
        help="Only write the final best_params.json summary",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    configure_logging()
    args = parse_args(argv)

    if args.print_default_config:
        print(json.dumps(DEFAULT_CONFIG, indent=2, sort_keys=True))
        return

    user_config = load_json(args.config) if args.config else None
    config = normalize_config(user_config)

    if args.dry_run:
        warn_about_noops(config)
        run_dry_run(config, args)
        return

    run_study(args, config)


if __name__ == "__main__":
    main()
