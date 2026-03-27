from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import torch

from .base import ZeroOrderOptimizer


class HiZOO(ZeroOrderOptimizer):
    _HESSIAN_SMOOTH_CONSTANTS = {
        "constant0": 0.0,
        "constant1e-12": 1e-12,
        "constant1e-10": 1e-10,
        "constant1e-8": 1e-8,
        "constant1e-6": 1e-6,
        "constant1e-4": 1e-4,
        "constant1e-2": 1e-2,
    }

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        weight_decay: float = 0.0,
        tensor_sampling_type: str = "standard_normal",
        matrix_sampling_type: str = None,
        perturbation_mode: str = "two_side",
        hessian_smooth_type: str = "constant1e-8",
        min_curvature: float = 1e-12,
    ):
        if perturbation_mode != "two_side":
            raise ValueError("HiZOO requires two_side perturbations")

        super().__init__(
            params,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            tensor_sampling_type=tensor_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
        )

        self.hessian_smooth_type = hessian_smooth_type
        self.min_curvature = min_curvature
        self.global_step = 0
        self.projected_grad = None

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0
                state["hessian_diag"] = torch.ones_like(
                    p, memory_format=torch.preserve_format
                )

    def _get_hessian_smooth(self) -> float:
        if self.hessian_smooth_type == "constant_decay1":
            return 1e-6 if self.global_step < 9800 else 1e-8

        if self.hessian_smooth_type in self._HESSIAN_SMOOTH_CONSTANTS:
            return self._HESSIAN_SMOOTH_CONSTANTS[self.hessian_smooth_type]

        try:
            return float(self.hessian_smooth_type)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported HiZOO hessian_smooth_type: {self.hessian_smooth_type}"
            ) from exc

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        tensor_sampling_type = self.state[param]["tensor_sampling_type"]
        z = self.tensor_sampler.sample(
            param.shape,
            generator=self.generator,
            sampler_type=tensor_sampling_type,
        )
        return z.to(param.device, dtype=param.dtype)

    def _hizoo_perturb_parameters(self, scaling_factor: float = 1.0) -> None:
        for group in self.param_groups:
            eps = group["eps"]
            for p in group["params"]:
                state = self.state[p]
                z = self._sample_direction(p)
                inv_sqrt_hessian = torch.rsqrt(
                    state["hessian_diag"].clamp_min(self.min_curvature)
                )
                p.data.add_(z * inv_sqrt_hessian, alpha=eps * scaling_factor)

    @staticmethod
    def _loss_to_float(loss) -> float:
        if torch.is_tensor(loss):
            return loss.detach().float().item()
        return float(loss)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("HiZOO requires a closure")

        self.global_step += 1
        self.zo_random_seed = np.random.randint(1_000_000_000)
        hessian_smooth = self._get_hessian_smooth()

        loss_base = closure()
        loss_base_value = self._loss_to_float(loss_base)

        self.generator.manual_seed(self.zo_random_seed)
        self._hizoo_perturb_parameters(scaling_factor=1.0)
        loss_plus = closure()
        loss_plus_value = self._loss_to_float(loss_plus)

        self.generator.manual_seed(self.zo_random_seed)
        self._hizoo_perturb_parameters(scaling_factor=-2.0)
        loss_minus = closure()
        loss_minus_value = self._loss_to_float(loss_minus)

        self.generator.manual_seed(self.zo_random_seed)
        self._hizoo_perturb_parameters(scaling_factor=1.0)

        curvature_scale = abs(loss_plus_value + loss_minus_value - 2.0 * loss_base_value)

        self.generator.manual_seed(self.zo_random_seed)
        for group in self.param_groups:
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            grad_scale = (loss_plus_value - loss_minus_value) / (2.0 * eps)

            self.projected_grad = grad_scale

            for p in group["params"]:
                state = self.state[p]
                state["step"] += 1

                z = self._sample_direction(p)
                hessian_diag = state["hessian_diag"]
                hessian_estimator = (
                    curvature_scale * hessian_diag * z.square() / (2.0 * eps * eps)
                )
                hessian_diag.mul_(1.0 - hessian_smooth).add_(
                    hessian_estimator, alpha=hessian_smooth
                )
                hessian_diag.clamp_(min=self.min_curvature)

                grad = grad_scale * z / torch.sqrt(hessian_diag)
                if weight_decay is not None and weight_decay != 0:
                    grad = grad + weight_decay * p.data

                p.data.add_(grad, alpha=-lr)

        return loss_base
