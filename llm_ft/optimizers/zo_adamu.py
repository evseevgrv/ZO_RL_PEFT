import math
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import torch

from .base import ZeroOrderOptimizer


class ZO_AdaMU(ZeroOrderOptimizer):
    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        weight_decay: float = 0.0,
        tensor_sampling_type: str = "standard_normal",
        matrix_sampling_type: str = None,
        perturbation_mode: str = "two_side",
        sigma: float = 1e-8,
        max_steps: Optional[int] = None,
        t1: Optional[int] = None,
        t2: Optional[int] = None,
        t3: Optional[int] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            tensor_sampling_type=tensor_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
        )

        self.sigma = sigma
        self.max_steps = max(
            1,
            max_steps if max_steps is not None else (t3 if t3 is not None else 20_000),
        )
        self.warmup_1st_steps = max(
            1,
            t1 if t1 is not None else min(1024, self.max_steps),
        )
        default_t2 = max(self.warmup_1st_steps + 1, int(self.max_steps * 0.8))
        self.warmup_2nd_steps = max(
            self.warmup_1st_steps + 1,
            t2 if t2 is not None else default_t2,
        )
        self.limit_steps = float(self.max_steps)
        self.perturb_history_coeff = 0.9
        self.perturb_current_coeff = 0.1

        self.global_step = 0
        self.projected_grad = None

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0
                state["hist_perturbation"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

    def _ema_weight(self, varphi: float = 1.0) -> float:
        if self.global_step < self.warmup_1st_steps:
            return 1.0

        if self.global_step < self.warmup_2nd_steps:
            weight = 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * ((self.global_step - self.warmup_1st_steps) / self.limit_steps)
                )
            )
            delta = (
                varphi
                * (self.max_steps - self.warmup_2nd_steps)
                / max(1, self.warmup_2nd_steps - self.warmup_1st_steps)
            )
            self.limit_steps = max(1.0, self.limit_steps - delta)
            return weight

        return 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (
                    (self.warmup_2nd_steps - self.warmup_1st_steps)
                    / self.limit_steps
                )
            )
        )

    def _sample_standard_normal_like(self, param: torch.Tensor) -> torch.Tensor:
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=param.shape,
            device=param.device,
            generator=self.generator,
        )

    def _apply_author_perturbation(self, scaling_factor: float) -> None:
        std_current_step = self._ema_weight()
        hist_std = math.sqrt(max(0.0, 1.0 - std_current_step))
        curr_std = math.sqrt(max(0.0, std_current_step))

        for group in self.param_groups:
            eps = group["eps"]
            for p in group["params"]:
                state = self.state[p]
                z_history = self.perturb_history_coeff * (
                    state["hist_perturbation"]
                    + self._sample_standard_normal_like(p) * hist_std
                )
                z_current = (
                    self.perturb_current_coeff
                    * self._sample_standard_normal_like(p)
                    * curr_std
                )
                z = z_history + z_current
                p.data.add_(z, alpha=eps * scaling_factor)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("ZO_AdaMU requires a closure")

        self.global_step += 1
        self.zo_random_seed = np.random.randint(1_000_000_000)

        self.generator.manual_seed(self.zo_random_seed)
        self._apply_author_perturbation(scaling_factor=1.0)
        loss_plus = closure()

        if self.perturbation_mode == "one_side":
            self.generator.manual_seed(self.zo_random_seed)
            self._apply_author_perturbation(scaling_factor=-1.0)
            loss_minus = closure()
        elif self.perturbation_mode == "two_side":
            self.generator.manual_seed(self.zo_random_seed)
            self._apply_author_perturbation(scaling_factor=-2.0)
            loss_minus = closure()
            self.generator.manual_seed(self.zo_random_seed)
            self._apply_author_perturbation(scaling_factor=1.0)
        else:
            raise ValueError(f"Unknown perturbation mode: {self.perturbation_mode}")

        self.projected_grad = self.grad_approx(
            loss_plus=loss_plus,
            loss_minus=loss_minus,
            perturbation_mode=self.perturbation_mode,
        )

        self.generator.manual_seed(self.zo_random_seed)
        std_current_step = self._ema_weight(1.0)
        beta1 = self._ema_weight(0.1)
        beta2 = self._ema_weight(1.5)
        hist_std = math.sqrt(max(0.0, 1.0 - std_current_step))
        curr_std = math.sqrt(max(0.0, std_current_step))
        grad_sign = -1.0 if self.projected_grad < 0 else 1.0

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                state = self.state[p]
                state["step"] = self.global_step
                z_history = (
                    state["hist_perturbation"]
                    + self._sample_standard_normal_like(p) * hist_std
                )
                z_current = self._sample_standard_normal_like(p) * curr_std

                m = beta1 * z_current + (1.0 - beta1) * z_history
                v = beta2 * z_current.square() + (1.0 - beta2) * z_history.square()
                update = self.projected_grad * m / torch.sqrt(v + self.sigma)

                if weight_decay is not None and weight_decay != 0:
                    update = update + weight_decay * p.data

                p.data.add_(update, alpha=-lr)
                state["hist_perturbation"] = m.mul(grad_sign)

        return loss_plus
