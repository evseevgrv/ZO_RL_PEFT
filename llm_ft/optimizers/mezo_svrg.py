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
                state["snapshot_param_cpu"] = torch.zeros_like(
                    p, device="cpu", memory_format=torch.preserve_format
                )
                state["full_grad_cpu"] = torch.zeros_like(
                    p, device="cpu", memory_format=torch.preserve_format
                )

    @staticmethod
    def _loss_to_float(loss) -> float:
        if torch.is_tensor(loss):
            return loss.detach().float().item()
        return float(loss)

    def _make_local_generator(self, seed: int) -> torch.Generator:
        generator_device = getattr(
            self.generator,
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        local_generator = torch.Generator(device=generator_device)
        local_generator.manual_seed(seed)
        return local_generator

    def _sample_direction(
        self, param: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        tensor_sampling_type = self.state[param]["tensor_sampling_type"]
        z = self.tensor_sampler.sample(
            param.shape,
            generator=self.generator if generator is None else generator,
            sampler_type=tensor_sampling_type,
        )
        return z.to(param.device, dtype=param.dtype)

    def _perturb_parameters_with_seed(self, scaling_factor: float = 1.0) -> None:
        for group in self.param_groups:
            eps = group["eps"]
            for p in group["params"]:
                z = self._sample_direction(p)
                p.data.add_(z, alpha=eps * scaling_factor)

    def _estimate_projected_grad(self, closure) -> tuple[float, int, Any]:
        random_seed = np.random.randint(1_000_000_000)

        self.generator.manual_seed(random_seed)
        self._perturb_parameters_with_seed(scaling_factor=1.0)
        loss_plus = closure()

        self.generator.manual_seed(random_seed)
        self._perturb_parameters_with_seed(scaling_factor=-2.0)
        loss_minus = closure()

        self.generator.manual_seed(random_seed)
        self._perturb_parameters_with_seed(scaling_factor=1.0)

        loss_plus_value = self._loss_to_float(loss_plus)
        loss_minus_value = self._loss_to_float(loss_minus)
        projected_grad = (loss_plus_value - loss_minus_value) / 2.0
        self.projected_grad = projected_grad
        return projected_grad, random_seed, loss_plus

    def _capture_current_parameters_cpu(self) -> Dict[torch.Tensor, torch.Tensor]:
        current_params_cpu = {}
        for group in self.param_groups:
            for p in group["params"]:
                current_params_cpu[p] = p.data.detach().to(
                    device="cpu", copy=True, memory_format=torch.preserve_format
                )
        return current_params_cpu

    def _load_parameter_values(
        self, parameter_values: Dict[torch.Tensor, torch.Tensor]
    ) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                p.data.copy_(parameter_values[p].to(device=p.device, dtype=p.dtype))

    def _load_snapshot_parameters(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                p.data.copy_(
                    self.state[p]["snapshot_param_cpu"].to(
                        device=p.device, dtype=p.dtype
                    )
                )

    def _store_snapshot_and_full_grad_cpu(
        self, projected_grad: float, random_seed: int
    ) -> None:
        local_generator = self._make_local_generator(random_seed)
        for group in self.param_groups:
            eps = group["eps"]
            for p in group["params"]:
                state = self.state[p]
                state["snapshot_param_cpu"].copy_(p.data.detach().to("cpu"))
                z = self._sample_direction(p, generator=local_generator)
                grad = z * (projected_grad / eps)
                state["full_grad_cpu"].copy_(grad.detach().to("cpu"))

    def _apply_full_batch_update(self, projected_grad: float, random_seed: int) -> None:
        local_generator = self._make_local_generator(random_seed)
        for group in self.param_groups:
            lr = group["lr"] if self.full_lr is None else self.full_lr
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                state = self.state[p]
                state["step"] += 1
                z = self._sample_direction(p, generator=local_generator)
                update = z * (projected_grad / eps)
                if weight_decay is not None and weight_decay != 0:
                    update = update + weight_decay * p.data
                p.data.add_(update, alpha=-lr)

    def _apply_minibatch_update(
        self,
        projected_grad_curr: float,
        current_seed: int,
        projected_grad_snapshot: float,
        snapshot_seed: int,
    ) -> None:
        current_generator = self._make_local_generator(current_seed)
        snapshot_generator = self._make_local_generator(snapshot_seed)
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                state = self.state[p]
                state["step"] += 1
                z_curr = self._sample_direction(p, generator=current_generator)
                z_snapshot = self._sample_direction(p, generator=snapshot_generator)
                full_grad = state["full_grad_cpu"].to(device=p.device, dtype=p.dtype)
                update = (
                    z_curr * (projected_grad_curr / eps)
                    - z_snapshot * (projected_grad_snapshot / eps)
                    + full_grad
                )
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
            projected_grad, random_seed, loss = self._estimate_projected_grad(
                full_closure
            )
            self._store_snapshot_and_full_grad_cpu(projected_grad, random_seed)
            self._apply_full_batch_update(projected_grad, random_seed)
            return loss

        projected_grad_curr, current_seed, loss = self._estimate_projected_grad(
            mini_closure
        )
        current_params = self._capture_current_parameters_cpu()
        self._load_snapshot_parameters()
        projected_grad_snapshot, snapshot_seed, _ = self._estimate_projected_grad(
            mini_closure
        )
        self._load_parameter_values(current_params)
        self._apply_minibatch_update(
            projected_grad_curr,
            current_seed,
            projected_grad_snapshot,
            snapshot_seed,
        )
        return loss
