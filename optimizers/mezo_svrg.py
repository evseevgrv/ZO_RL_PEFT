from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import torch

from .base import ZeroOrderOptimizer


class MeZO_SVRG(ZeroOrderOptimizer):
    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        weight_decay: float = 0.0,
        tensor_sampling_type: str = "standard_normal",
        matrix_sampling_type: str = None,
        perturbation_mode: str = "two_side",
        q: int = 2,
        full_lr: Optional[float] = None,
    ):
        if perturbation_mode != "two_side":
            raise ValueError("MeZO-SVRG requires two_side perturbations")

        super().__init__(
            params,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            tensor_sampling_type=tensor_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
        )

        self.q = max(1, q)
        self.full_lr = full_lr
        self.global_step = 0
        self.projected_grad = None

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0
                state["snapshot_param"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["full_grad"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

    @staticmethod
    def _loss_to_float(loss) -> float:
        if torch.is_tensor(loss):
            return loss.detach().float().item()
        return float(loss)

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        tensor_sampling_type = self.state[param]["tensor_sampling_type"]
        z = self.tensor_sampler.sample(
            param.shape,
            generator=self.generator,
            sampler_type=tensor_sampling_type,
        )
        return z.to(param.device, dtype=param.dtype)

    def _perturb_parameters_with_seed(self, scaling_factor: float = 1.0) -> None:
        for group in self.param_groups:
            eps = group["eps"]
            for p in group["params"]:
                z = self._sample_direction(p)
                p.data.add_(z, alpha=eps * scaling_factor)

    def _estimate_gradient(self, closure) -> tuple[Dict[torch.Tensor, torch.Tensor], Any]:
        self.zo_random_seed = np.random.randint(1_000_000_000)

        self.generator.manual_seed(self.zo_random_seed)
        self._perturb_parameters_with_seed(scaling_factor=1.0)
        loss_plus = closure()

        self.generator.manual_seed(self.zo_random_seed)
        self._perturb_parameters_with_seed(scaling_factor=-2.0)
        loss_minus = closure()

        self.generator.manual_seed(self.zo_random_seed)
        self._perturb_parameters_with_seed(scaling_factor=1.0)

        loss_plus_value = self._loss_to_float(loss_plus)
        loss_minus_value = self._loss_to_float(loss_minus)
        grad_scale = (loss_plus_value - loss_minus_value) / 2.0
        self.projected_grad = grad_scale

        estimator = {}
        self.generator.manual_seed(self.zo_random_seed)
        for group in self.param_groups:
            eps = group["eps"]
            for p in group["params"]:
                z = self._sample_direction(p)
                estimator[p] = z * (grad_scale / eps)

        return estimator, loss_plus

    def _capture_current_parameters(self) -> Dict[torch.Tensor, torch.Tensor]:
        current_params = {}
        for group in self.param_groups:
            for p in group["params"]:
                current_params[p] = p.data.detach().clone(
                    memory_format=torch.preserve_format
                )
        return current_params

    def _load_parameter_values(self, parameter_values: Dict[torch.Tensor, torch.Tensor]) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                p.data.copy_(parameter_values[p])

    def _load_snapshot_parameters(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                p.data.copy_(self.state[p]["snapshot_param"])

    def _store_snapshot_and_full_grad(
        self, full_grad: Dict[torch.Tensor, torch.Tensor]
    ) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["snapshot_param"].copy_(p.data)
                state["full_grad"].copy_(full_grad[p])

    def _apply_full_batch_update(self, full_grad: Dict[torch.Tensor, torch.Tensor]) -> None:
        for group in self.param_groups:
            lr = group["lr"] if self.full_lr is None else self.full_lr
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                state = self.state[p]
                state["step"] += 1
                update = full_grad[p]
                if weight_decay is not None and weight_decay != 0:
                    update = update + weight_decay * p.data
                p.data.add_(update, alpha=-lr)

    def _apply_minibatch_update(
        self,
        curr_grad: Dict[torch.Tensor, torch.Tensor],
        snapshot_grad: Dict[torch.Tensor, torch.Tensor],
    ) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                state = self.state[p]
                state["step"] += 1
                update = curr_grad[p] - snapshot_grad[p] + state["full_grad"]
                if weight_decay is not None and weight_decay != 0:
                    update = update + weight_decay * p.data
                p.data.add_(update, alpha=-lr)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("MeZO-SVRG requires closure information")

        if callable(closure):
            mini_closure = closure
            full_closure = None
        else:
            mini_closure = closure.get("mini")
            full_closure = closure.get("full")

        if mini_closure is None:
            raise ValueError("MeZO-SVRG requires a mini-batch closure")

        self.global_step += 1
        is_full_step = (self.global_step - 1) % self.q == 0

        if is_full_step:
            if full_closure is None:
                raise ValueError("MeZO-SVRG full steps require a full-batch closure")
            full_grad, loss = self._estimate_gradient(full_closure)
            self._store_snapshot_and_full_grad(full_grad)
            self._apply_full_batch_update(full_grad)
            return loss

        curr_grad, loss = self._estimate_gradient(mini_closure)
        current_params = self._capture_current_parameters()
        self._load_snapshot_parameters()
        snapshot_grad, _ = self._estimate_gradient(mini_closure)
        self._load_parameter_values(current_params)
        self._apply_minibatch_update(curr_grad, snapshot_grad)
        return loss
