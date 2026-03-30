from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import torch
import wandb

from .hizoo import HiZOO


class HiZOO_RL(HiZOO):
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
        variance: float = 1.0,
        lr_mu: Optional[float] = None,
        use_grad_first: bool = False,
        k: int = 10,
    ):
        super().__init__(
            params=params,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            tensor_sampling_type=tensor_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
            hessian_smooth_type=hessian_smooth_type,
            min_curvature=min_curvature,
        )

        self.variance = variance
        self.lr_mu = lr if lr_mu is None else lr_mu
        self.use_grad_first = use_grad_first
        self.k = max(1, k)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if self.use_grad_first and p.grad is not None:
                    mu = p.grad.detach().clone()
                else:
                    mu = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["mu"] = mu
                state["mu_old"] = mu.detach().clone()

    def _sample_mu_direction(
        self, param: torch.Tensor, mean_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        mean = self.state[param]["mu_old"] if mean_tensor is None else mean_tensor
        z = torch.normal(mean=mean, std=self.variance, generator=self.generator)
        return z.to(param.device, dtype=param.dtype)

    def _apply_hizoo_rl_perturbation(self, scaling_factor: float = 1.0) -> None:
        for group in self.param_groups:
            eps = group["eps"]
            for p in group["params"]:
                state = self.state[p]
                z = self._sample_mu_direction(p)
                inv_sqrt_hessian = torch.rsqrt(
                    state["hessian_diag"].clamp_min(self.min_curvature)
                )
                p.data.add_(z * inv_sqrt_hessian, alpha=eps * scaling_factor)

    def _log_metric(self, metric_name: str, metric_value: float, step: Optional[int]) -> None:
        if wandb.run is not None:
            wandb.log({metric_name: metric_value})

        try:
            from trainer import _optimizer_log_func

            if _optimizer_log_func is not None and step is not None:
                _optimizer_log_func({metric_name: metric_value}, step=step)
        except (ImportError, AttributeError):
            pass

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("HiZOO_RL requires a closure")

        if self.variance <= 0:
            raise ValueError("HiZOO_RL requires variance > 0")

        self.global_step += 1
        hessian_smooth = self._get_hessian_smooth()

        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["mu_old"] = self.state[p]["mu"].detach().clone()

        candidate_seeds = []
        candidate_losses = []
        for _ in range(self.k):
            seed = np.random.randint(1_000_000_000)
            candidate_seeds.append(seed)

            self.generator.manual_seed(seed)
            self._apply_hizoo_rl_perturbation(scaling_factor=1.0)
            candidate_losses.append(self._loss_to_float(closure()))

            self.generator.manual_seed(seed)
            self._apply_hizoo_rl_perturbation(scaling_factor=-1.0)

        optimal_idx = min(range(len(candidate_losses)), key=candidate_losses.__getitem__)
        self.zo_random_seed = candidate_seeds[optimal_idx]

        loss_base = closure()
        loss_base_value = self._loss_to_float(loss_base)

        self.generator.manual_seed(self.zo_random_seed)
        self._apply_hizoo_rl_perturbation(scaling_factor=1.0)
        loss_plus = closure()
        loss_plus_value = self._loss_to_float(loss_plus)

        self.generator.manual_seed(self.zo_random_seed)
        self._apply_hizoo_rl_perturbation(scaling_factor=-2.0)
        loss_minus = closure()
        loss_minus_value = self._loss_to_float(loss_minus)

        self.generator.manual_seed(self.zo_random_seed)
        self._apply_hizoo_rl_perturbation(scaling_factor=1.0)

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

                z = self._sample_mu_direction(p)
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

        loss_tensor = torch.tensor(candidate_losses, dtype=torch.float32)
        if self.k > 1:
            coeffs = (loss_tensor * self.k - loss_tensor.sum()) / (self.k - 1)
        else:
            coeffs = torch.zeros_like(loss_tensor)

        for coeff, seed in zip(coeffs.tolist(), candidate_seeds):
            self.generator.manual_seed(seed)
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[p]
                    mu_old = state["mu_old"]
                    z = self._sample_mu_direction(p, mean_tensor=mu_old)
                    update = (mu_old - z) * coeff / (self.k * (self.variance ** 2))
                    state["mu"].add_(update, alpha=self.lr_mu)

        mu_norms = []
        mu_norm_diff_sq = 0.0
        mu_grad_norm_sq = 0.0
        step = None

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                mu = state["mu"]
                mu_old = state["mu_old"]
                mu_norms.append(torch.norm(mu).item())
                mu_norm_diff_sq += torch.linalg.norm(mu - mu_old).item() ** 2
                if self.lr_mu != 0:
                    g_mu = (mu_old - mu) / self.lr_mu
                    mu_grad_norm_sq += torch.linalg.norm(g_mu).item() ** 2

                if step is None:
                    step = state.get("step")

        if mu_norms:
            self._log_metric("avg_mu_norm", sum(mu_norms) / len(mu_norms), step)

            avg_mu_norm_diff = (mu_norm_diff_sq ** 0.5) / len(mu_norms)
            avg_mu_grad_norm = (mu_grad_norm_sq ** 0.5) / len(mu_norms)
            self._log_metric("avg_mu_norm_diff", avg_mu_norm_diff, step)
            self._log_metric("avg_mu_grad_norm", avg_mu_grad_norm, step)

        return loss_base
