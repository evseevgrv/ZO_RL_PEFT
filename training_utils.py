import json
import logging
import math
import os
from collections import defaultdict
from datetime import datetime
from functools import partial

import numpy as np
import torch
import transformers
from torch.optim.lr_scheduler import LambdaLR

_local_file_logging_logger = logging.getLogger(__name__)
_local_run_logger = None


class LocalRunLogger:
    def __init__(self, log_dir, run_tag, primary_eval_metric=None):
        self.run_tag = run_tag
        self.run_dir = os.path.join(log_dir, run_tag)
        self.log_path = os.path.join(self.run_dir, f"{run_tag}.jsonl")
        self.primary_eval_metric = primary_eval_metric
        self.series_history = defaultdict(list)
        self.test_series_key = None
        os.makedirs(self.run_dir, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8"):
            pass
        for filename in os.listdir(self.run_dir):
            if filename.startswith(f"{self.run_tag}__") and filename.endswith(".pdf"):
                os.remove(os.path.join(self.run_dir, filename))

    def log_metrics(self, metrics_dict, step=None, phase="metrics"):
        metrics = {
            key: self._convert_to_python_types(value)
            for key, value in metrics_dict.items()
        }
        event = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": step,
            "phase": phase,
            "metrics": metrics,
        }
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

        for series_key, value in self._extract_plot_updates(metrics, phase).items():
            if not isinstance(value, (int, float)):
                continue
            history = self.series_history[series_key]
            series_step = step if step is not None else len(history)
            history.append((series_step, float(value)))
            self._render_series_plot(series_key)

    def _extract_plot_updates(self, metrics, phase):
        updates = {}

        if phase == "train" and "train_loss" in metrics:
            updates["train_loss"] = metrics["train_loss"]
        elif phase == "train" and "loss" in metrics:
            updates["train_loss"] = metrics["loss"]

        if phase == "train" and "peak_memory_gb" in metrics:
            updates["peak_memory_gb"] = metrics["peak_memory_gb"]
        elif phase == "train" and "peak_mem" in metrics:
            updates["peak_memory_gb"] = metrics["peak_mem"]

        test_series_key, test_metric_value = self._resolve_test_metric(metrics, phase)
        if test_series_key is not None and test_metric_value is not None:
            updates[test_series_key] = test_metric_value

        return updates

    def _resolve_test_metric(self, metrics, phase):
        if phase not in {"eval", "icl_eval"}:
            return None, None

        if "test_accuracy" in metrics:
            self.test_series_key = "test_accuracy"
            return self.test_series_key, metrics["test_accuracy"]

        if "test_acc" in metrics:
            self.test_series_key = "test_accuracy"
            return self.test_series_key, metrics["test_acc"]

        if phase in {"eval", "icl_eval"} and "accuracy" in metrics:
            self.test_series_key = "test_accuracy"
            return self.test_series_key, metrics["accuracy"]

        preferred_key = None
        if self.primary_eval_metric:
            preferred_key = f"test_{self.primary_eval_metric}"
            if preferred_key in metrics:
                self.test_series_key = preferred_key
                return self.test_series_key, metrics[preferred_key]
            if phase in {"eval", "icl_eval"} and self.primary_eval_metric in metrics:
                self.test_series_key = preferred_key
                return self.test_series_key, metrics[self.primary_eval_metric]

        test_keys = [key for key in metrics if key.startswith("test_")]
        if test_keys:
            chosen_key = sorted(test_keys)[0]
            self.test_series_key = chosen_key
            return chosen_key, metrics[chosen_key]

        return None, None

    def _render_series_plot(self, series_key):
        history = self.series_history.get(series_key)
        if not history:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            _local_file_logging_logger.warning(
                "matplotlib is not available, skipping local PDF chart generation"
            )
            return
        except Exception as exc:
            _local_file_logging_logger.warning(
                f"Failed to import matplotlib for local logging: {exc}"
            )
            return

        steps, values = zip(*history)
        colors = {
            "train_loss": "#1f77b4",
            "peak_memory_gb": "#2ca02c",
        }
        color = colors.get(series_key, "#d62728")

        fig, ax = plt.subplots(figsize=(8.5, 5.0), facecolor="white")
        ax.set_facecolor("white")
        ax.plot(steps, values, color=color, linewidth=2.0)
        ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel(self._axis_label(series_key), fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        fig.tight_layout()
        fig.savefig(self._plot_path(series_key), format="pdf", bbox_inches="tight")
        plt.close(fig)

    def _plot_path(self, series_key):
        return os.path.join(self.run_dir, f"{self.run_tag}__{series_key}.pdf")

    def _axis_label(self, series_key):
        if series_key == "train_loss":
            return "Train Loss"
        if series_key == "peak_memory_gb":
            return "Peak Memory (GB)"
        if series_key == "test_accuracy":
            return "Test Accuracy"
        return series_key.replace("_", " ").title()

    def _convert_to_python_types(self, value):
        if torch.is_tensor(value):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.item() if value.size == 1 else value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value


def init_local_run_logger(log_dir, run_tag, primary_eval_metric=None):
    global _local_run_logger
    _local_run_logger = LocalRunLogger(
        log_dir=log_dir,
        run_tag=run_tag,
        primary_eval_metric=primary_eval_metric,
    )
    _local_file_logging_logger.info(
        "Local file logging enabled. Artifacts will be written to %s",
        _local_run_logger.run_dir,
    )
    return _local_run_logger.run_dir


def log_local_metrics(metrics_dict, step=None, phase="metrics"):
    if _local_run_logger is None:
        return
    _local_run_logger.log_metrics(metrics_dict, step=step, phase=phase)


def infer_local_log_phase(metrics_dict):
    metric_keys = set(metrics_dict.keys())
    if any(key.startswith("test_") or key.startswith("val_") for key in metric_keys):
        return "eval"
    if metric_keys & {
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "total_flos",
        "train_loss",
    }:
        return "summary"
    return "train"

def get_scheduler(
    optimizer,
    *,
    scheduler_type,
    num_training_steps,
    warmup_steps,
    min_lr_ratio,
    cycle_length=None,
    restart_warmup_steps=None,
    adjust_step=0,
    last_epoch=-1,
):
    if adjust_step != 0 and scheduler_type != "cosine_restarts":
        raise ValueError("adjust_step is only supported for cosine_restarts scheduler")

    if scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
    if scheduler_type == "constant":
        return transformers.get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            last_epoch=last_epoch,
        )
    if scheduler_type == "cosine":
        return get_cyclical_cosine_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            cycle_length=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
        )
    if scheduler_type == "cosine_restarts":
        assert restart_warmup_steps is not None, "restart_warmup_steps must be specified for cosine_restarts scheduler"
        return get_cosine_schedule_with_multiple_warmups(
            optimizer,
            num_training_steps=num_training_steps,
            first_warmup_steps=warmup_steps,
            restart_warmup_steps=restart_warmup_steps,
            restart_every=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
            adjust_step=adjust_step,
        )

    raise NotImplementedError(f"Scheduler {scheduler_type} is not implemented")


def get_cyclical_cosine_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, cycle_length, min_lr_ratio=0.1, last_epoch=-1):
    assert cycle_length is not None or num_training_steps is not None, "You must specify either cycle_length or num_training_steps"
    
    if cycle_length is None:
        cycle_length = num_training_steps

    if num_training_steps % cycle_length != 0:
        raise ValueError(f"num_training_steps ({num_training_steps}) must be divisible by cycle_length ({cycle_length})")

    lr_lambda = partial(
        _get_cyclical_cosine_schedule_with_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        cycle_length=cycle_length,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_multiple_warmups(
    optimizer,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio=0.1,
    adjust_step=0,
    last_epoch=-1,
):
    if restart_every is None:
        raise ValueError("restart_every must be specified for cosine_restarts scheduler")

    if num_training_steps % restart_every != 0:
        raise ValueError(f"num_training_steps ({num_training_steps}) must be divisible by restart_every ({restart_every})")

    lr_lambda = partial(
        _get_cosine_schedule_with_multiple_warmups_lambda,
        num_training_steps=num_training_steps,
        first_warmup_steps=first_warmup_steps,
        restart_warmup_steps=restart_warmup_steps,
        restart_every=restart_every,
        min_lr_ratio=min_lr_ratio,
        adjust_step=adjust_step,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cyclical_cosine_schedule_with_min_lr_lambda(current_step, *, num_warmup_steps, cycle_length, min_lr_ratio):
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"

    # compute where we are in the current cycle
    cycle_step = current_step % cycle_length

    if cycle_step < num_warmup_steps:
        if current_step != cycle_step:
            if cycle_step < 2:
                return 1e-7
        return float(cycle_step) / float(max(1, num_warmup_steps))

    progress = float(cycle_step - num_warmup_steps) / float(max(1, cycle_length - num_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def _get_cosine_schedule_with_multiple_warmups_lambda(
    current_step,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio,
    adjust_step,
):
    """
    Args:
        adjust_step: useful when continuing training from a warmed up checkpoint,
            it allows to sync the resets by reducing the number of steps
            after the first warmup and before the first reset.
            Thus, your ReLoRA resets can be synced with the optimizer resets.
    """
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"
    assert restart_every > 0, "restart_every must be positive"
    assert adjust_step + first_warmup_steps < num_training_steps, "warmup + adjust_step is more than full training steps"
    assert adjust_step + first_warmup_steps < restart_every, "the first reset will happen before the warmup is done"

    if current_step < first_warmup_steps:
        return float(current_step) / float(max(1, first_warmup_steps))

    _current_step = current_step + adjust_step

    restart_step = _current_step % restart_every
    restart_number = _current_step // restart_every

    if restart_step < restart_warmup_steps:
        # get expected lr multipler at the end of the warmup
        end_of_warmup_progress = (
            float(restart_number * restart_every) /
            float(max(1, num_training_steps - first_warmup_steps))
        )

        _cosine_decay = 0.5 * (1.0 + math.cos(math.pi * end_of_warmup_progress))
        warmup_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * _cosine_decay
    
        return float(restart_step) / float(max(1, restart_warmup_steps)) * warmup_lr_multiplier

    progress = float(_current_step - first_warmup_steps) / float(max(1, num_training_steps - first_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
