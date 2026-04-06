from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import torch
import wandb

from .mezo_svrg import MeZO_SVRG


class MeZO_SVRG_RL(MeZO_SVRG):
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
        variance: float = 1.0,
        lr_mu: Optional[float] = None,
        use_grad_first: bool = False,
        k: int = 10,
        evaluate_memory: bool = False,
    ):
        super().__init__(
            params=params,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            tensor_sampling_type=tensor_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
            q=q,
            full_lr=full_lr,
        )

        self.variance = variance
        self.lr_mu = lr if lr_mu is None else lr_mu
        self.use_grad_first = use_grad_first
        self.k = max(1, k)
        self.evaluate_memory = evaluate_memory

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if self.use_grad_first:
                    if p.grad is None:
                        raise ValueError(
                            "param.grad is None, but use_grad_first is True"
                        )
                    mu = p.grad.detach().clone()
                else:
                    mu = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["mu"] = mu
                if not self.evaluate_memory:
                    state["mu_old"] = mu.detach().clone()

    def _sample_mu_directions(
        self,
        generator: torch.Generator,
        mean_tensors: Optional[Dict[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[torch.Tensor, torch.Tensor]:
        directions = {}
        for group in self.param_groups:
            for p in group["params"]:
                mean = self.state[p]["mu"] if mean_tensors is None else mean_tensors[p]
                tensor_sampling_type = self.state[p]["tensor_sampling_type"]
                noise = self.tensor_sampler.sample(
                    p.shape,
                    generator=generator,
                    sampler_type=tensor_sampling_type,
                )
                directions[p] = mean + self.variance * noise.to(
                    device=p.device, dtype=p.dtype
                )
        return directions

    def _apply_directions(
        self,
        directions: Dict[torch.Tensor, torch.Tensor],
        scaling_factor: float = 1.0,
    ) -> None:
        for group in self.param_groups:
            eps = group["eps"]
            for p in group["params"]:
                p.data.add_(directions[p], alpha=eps * scaling_factor)

    def _candidate_coefficients(self, candidate_losses: list[float]) -> torch.Tensor:
        loss_tensor = torch.tensor(candidate_losses, dtype=torch.float32)
        if self.k > 1:
            return (loss_tensor * self.k - loss_tensor.sum()) / (self.k - 1)
        return torch.zeros_like(loss_tensor)

    def _estimate_projected_grad(self, closure) -> Dict[str, Any]:
        candidate_seeds = []
        candidate_losses = []
        best_loss = None
        best_seed = None

        for _ in range(self.k):
            seed = np.random.randint(1_000_000_000)
            candidate_seeds.append(seed)
            directions = self._sample_mu_directions(self._make_local_generator(seed))

            self._apply_directions(directions, scaling_factor=1.0)
            loss_value = self._loss_to_float(closure())
            candidate_losses.append(loss_value)
            self._apply_directions(directions, scaling_factor=-1.0)

            if best_loss is None or loss_value < best_loss:
                best_loss = loss_value
                best_seed = seed

        selected_directions = self._sample_mu_directions(
            self._make_local_generator(best_seed)
        )

        self._apply_directions(selected_directions, scaling_factor=1.0)
        loss_plus = closure()
        loss_plus_value = self._loss_to_float(loss_plus)

        self._apply_directions(selected_directions, scaling_factor=-2.0)
        loss_minus = closure()
        loss_minus_value = self._loss_to_float(loss_minus)

        self._apply_directions(selected_directions, scaling_factor=1.0)

        projected_grad = (loss_plus_value - loss_minus_value) / 2.0
        self.projected_grad = projected_grad

        return {
            "projected_grad": projected_grad,
            "loss": loss_plus,
            "candidate_seeds": candidate_seeds,
            "candidate_losses": candidate_losses,
            "selected_directions": selected_directions,
        }

    def _store_snapshot_and_full_grad_cpu_from_directions(
        self,
        projected_grad: float,
        directions: Dict[torch.Tensor, torch.Tensor],
    ) -> None:
        for group in self.param_groups:
            eps = group["eps"]
            for p in group["params"]:
                state = self.state[p]
                state["snapshot_param_cpu"].copy_(p.data.detach().to("cpu"))
                grad = directions[p] * (projected_grad / eps)
                state["full_grad_cpu"].copy_(grad.detach().to("cpu"))

    def _apply_full_batch_update_from_directions(
        self,
        projected_grad: float,
        directions: Dict[torch.Tensor, torch.Tensor],
    ) -> None:
        for group in self.param_groups:
            lr = group["lr"] if self.full_lr is None else self.full_lr
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                state = self.state[p]
                state["step"] += 1
                update = directions[p] * (projected_grad / eps)
                if weight_decay is not None and weight_decay != 0:
                    update = update + weight_decay * p.data
                p.data.add_(update, alpha=-lr)

    def _apply_minibatch_update_from_directions(
        self,
        projected_grad_curr: float,
        directions_curr: Dict[torch.Tensor, torch.Tensor],
        projected_grad_snapshot: float,
        directions_snapshot: Dict[torch.Tensor, torch.Tensor],
    ) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                state = self.state[p]
                state["step"] += 1
                full_grad = state["full_grad_cpu"].to(device=p.device, dtype=p.dtype)
                update = (
                    directions_curr[p] * (projected_grad_curr / eps)
                    - directions_snapshot[p] * (projected_grad_snapshot / eps)
                    + full_grad
                )
                if weight_decay is not None and weight_decay != 0:
                    update = update + weight_decay * p.data
                p.data.add_(update, alpha=-lr)

    def _compute_mu_update(
        self,
        candidate_seeds: list[int],
        candidate_losses: list[float],
    ) -> Dict[torch.Tensor, torch.Tensor]:
        updates = {}
        for group in self.param_groups:
            for p in group["params"]:
                updates[p] = torch.zeros_like(
                    self.state[p]["mu"], memory_format=torch.preserve_format
                )

        if self.k == 1:
            return updates

        coeffs = self._candidate_coefficients(candidate_losses)
        denom = self.k * (self.variance**2)

        for coeff, seed in zip(coeffs.tolist(), candidate_seeds):
            directions = self._sample_mu_directions(self._make_local_generator(seed))
            for group in self.param_groups:
                for p in group["params"]:
                    mu = self.state[p]["mu"]
                    updates[p].add_(mu - directions[p], alpha=coeff / denom)

        return updates

    def _log_metric(self, metric_name: str, metric_value: float, step: Optional[int]) -> None:
        if wandb.run is not None:
            wandb.log({metric_name: metric_value})

        try:
            from trainer import _optimizer_log_func

            if _optimizer_log_func is not None and step is not None:
                _optimizer_log_func({metric_name: metric_value}, step=step)
        except (ImportError, AttributeError):
            pass

    def _prepare_mu_tracking(self) -> None:
        if self.evaluate_memory:
            return

        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["mu_old"] = self.state[p]["mu"].detach().clone()

    def _apply_mu_updates(self, update_dicts: list[Dict[torch.Tensor, torch.Tensor]]) -> None:
        if not update_dicts:
            return

        mu_norms = []
        mu_norm_diff_sq = 0.0
        mu_grad_norm_sq = 0.0
        step = None
        num_updates = float(len(update_dicts))

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                avg_update = torch.zeros_like(
                    state["mu"], memory_format=torch.preserve_format
                )
                for update_dict in update_dicts:
                    avg_update.add_(update_dict[p], alpha=1.0 / num_updates)

                state["mu"].add_(avg_update, alpha=self.lr_mu)
                mu_norms.append(torch.norm(state["mu"]).item())

                if not self.evaluate_memory:
                    mu_old = state["mu_old"]
                    mu_norm_diff_sq += torch.linalg.norm(state["mu"] - mu_old).item() ** 2
                    if self.lr_mu != 0:
                        mu_grad_norm_sq += torch.linalg.norm(avg_update).item() ** 2

                if step is None:
                    step = state.get("step")

        if mu_norms:
            self._log_metric("avg_mu_norm", sum(mu_norms) / len(mu_norms), step)

        if not self.evaluate_memory and mu_norms:
            avg_mu_norm_diff = (mu_norm_diff_sq**0.5) / len(mu_norms)
            avg_mu_grad_norm = (mu_grad_norm_sq**0.5) / len(mu_norms)
            self._log_metric("avg_mu_norm_diff", avg_mu_norm_diff, step)
            self._log_metric("avg_mu_grad_norm", avg_mu_grad_norm, step)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("MeZO-SVRG-RL requires closure information")

        if self.variance <= 0:
            raise ValueError("MeZO-SVRG-RL requires variance > 0")

        if callable(closure):
            mini_closure = closure
            full_closure = None
        else:
            mini_closure = closure.get("mini")
            full_closure = closure.get("full")

        if mini_closure is None:
            raise ValueError("MeZO-SVRG-RL requires a mini-batch closure")

        self.global_step += 1
        is_full_step = (self.global_step - 1) % self.q == 0
        self._prepare_mu_tracking()

        if is_full_step:
            if full_closure is None:
                raise ValueError("MeZO-SVRG-RL full steps require a full-batch closure")

            estimate = self._estimate_projected_grad(full_closure)
            self._store_snapshot_and_full_grad_cpu_from_directions(
                estimate["projected_grad"], estimate["selected_directions"]
            )
            self._apply_full_batch_update_from_directions(
                estimate["projected_grad"], estimate["selected_directions"]
            )
            self._apply_mu_updates(
                [
                    self._compute_mu_update(
                        estimate["candidate_seeds"], estimate["candidate_losses"]
                    )
                ]
            )
            return estimate["loss"]

        estimate_curr = self._estimate_projected_grad(mini_closure)
        current_params = self._capture_current_parameters_cpu()

        self._load_snapshot_parameters()
        estimate_snapshot = self._estimate_projected_grad(mini_closure)
        self._load_parameter_values(current_params)

        self._apply_minibatch_update_from_directions(
            estimate_curr["projected_grad"],
            estimate_curr["selected_directions"],
            estimate_snapshot["projected_grad"],
            estimate_snapshot["selected_directions"],
        )
        self._apply_mu_updates(
            [
                self._compute_mu_update(
                    estimate_curr["candidate_seeds"], estimate_curr["candidate_losses"]
                ),
                self._compute_mu_update(
                    estimate_snapshot["candidate_seeds"],
                    estimate_snapshot["candidate_losses"],
                ),
            ]
        )
        return estimate_curr["loss"]
