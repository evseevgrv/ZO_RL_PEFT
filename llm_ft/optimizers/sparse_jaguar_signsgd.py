from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
import time

from .opt_utils import *

class Sparse_Jaguar_SignSGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            beta: float = 0.9,
            lr: float = 0.01,
            eps: float = 1e-3,
            tensor_sampling_type: str = "standard_normal", 
            matrix_sampling_type: str = None,  
            perturbation_mode: str = "two_side",
            params_ratio: float = 0.1,
            k: int = 1,
            evaluate_memory: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            tensor_sampling_type=tensor_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
        )

        for group in self.param_groups:
            group['beta'] = beta
        self.k = max(1, k)
        self.params_ratio = params_ratio
        self.evaluate_memory = evaluate_memory
        self.all_params = [p for group in self.param_groups for p in group['params']]
        for group in self.param_groups:
            for param in group['params']:    
                state = self.state[param]
                if 'step' not in state:
                    state['step'] = 0
                    state['grad_accum'] = torch.zeros_like(
                        param, 
                        memory_format=torch.preserve_format
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss_plus_values = []
        projected_grads = []
        grad_sums = {}

        for group in self.param_groups:
            for param in group['params']:    
                state = self.state[param]
                state['step'] += 1

        # Keep the sparse parameter subset fixed across the k probes in this step.
        selection_seed = np.random.randint(1_000_000_000)
        self.generator.manual_seed(selection_seed)
        selected_param_ids = self._sample_selected_param_ids(params_ratio=self.params_ratio)
        for group in self.param_groups:
            for param in group['params']:
                if id(param) in selected_param_ids:
                    grad_sums[param] = torch.zeros_like(
                        param,
                        memory_format=torch.preserve_format,
                    )

        for _ in range(self.k):
            seed = np.random.randint(1_000_000_000)
            self.zo_random_seed = seed

            self.generator.manual_seed(seed)
            self._sparse_indices_perturb(
                scaling_factor=1.0,
                selected_param_ids=selected_param_ids,
            )
            loss_plus = closure()
            loss_plus_values.append(loss_plus)

            self.generator.manual_seed(seed)
            self._sparse_indices_perturb(
                scaling_factor=-2.0,
                selected_param_ids=selected_param_ids,
            )
            loss_minus = closure()

            projected_grad = self.grad_approx(
                loss_plus=loss_plus,
                loss_minus=loss_minus,
                perturbation_mode="two_side",
            )
            projected_grads.append(projected_grad)

            self.generator.manual_seed(seed)
            self._sparse_indices_perturb(
                scaling_factor=1.0,
                selected_param_ids=selected_param_ids,
            )

            # Replay the same generator stream: selected params get different sequential z,
            # while each param sees the same z here as it saw during perturb.
            self.generator.manual_seed(seed)
            for group in self.param_groups:
                eps = group['eps']
                for param in group['params']:
                    if id(param) not in selected_param_ids:
                        continue

                    z = self._sample_direction(param)
                    grad_sums[param].add_(z * (projected_grad / (eps * self.k)))

        self.projected_grad = sum(projected_grads) / len(projected_grads)

        for group in self.param_groups:
            lr = group['lr']  
            beta = group['beta']

            for param in group['params']:
                state = self.state[param]
                grad = grad_sums.get(param)
                if grad is not None:
                    state['grad_accum'].mul_(beta).add_(grad, alpha=(1.0 - beta))

                update_direction = torch.sign(state['grad_accum'])
                param.data.add_(update_direction, alpha=-lr)

        return torch.stack(loss_plus_values).mean()
    
    def _sample_selected_param_ids(self, params_ratio=0.1):
        n = max(1, int(len(self.all_params) * params_ratio))
        param_indices = torch.randperm(
            len(self.all_params),
            device=self.all_params[0].device,
            generator=self.generator,
        )[:n]
        return {id(self.all_params[int(idx)]) for idx in param_indices}

    def _sample_direction(self, param):
        tensor_sampling_type = self.state[param]['tensor_sampling_type']
        return self.tensor_sampler.sample(
            param.shape,
            generator=self.generator,
            sampler_type=tensor_sampling_type,
        ).to(param.device)

    def _sparse_indices_perturb(self, scaling_factor = 1.0, params_ratio = 0.1, selected_param_ids=None):
        if selected_param_ids is None:
            selected_param_ids = self._sample_selected_param_ids(params_ratio=params_ratio)
            self.generator.manual_seed(self.zo_random_seed)

        for group in self.param_groups:
            eps = group['eps']
            
            for param in group['params']:                     
                if id(param) in selected_param_ids:
                    z = self._sample_direction(param)
                    param.data += scaling_factor * eps * z
