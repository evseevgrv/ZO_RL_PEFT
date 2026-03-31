from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
import time

from .opt_utils import *

class Jaguar_SignSGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            beta: float = 0.9,
            lr: float = 0.01,
            eps: float = 1e-3,
            tensor_sampling_type: str = "standard_normal", 
            matrix_sampling_type: str = None, 
            perturbation_mode: str = "two_side",
            k: int = 1,
    ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            tensor_sampling_type=tensor_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
        )
        self.k = max(1, k)
        
        for group in self.param_groups:
            group['beta'] = beta

    @torch.no_grad()
    def step(self, closure=None):
        loss_plus_values = []
        projected_grads = []
        grad_sums = {}
        touched_masks = {}

        for group in self.param_groups:
            for param in group['params']:    
                state = self.state[param]
                if 'step' not in state:
                    state['step'] = 0
                    state['grad_accum'] = torch.zeros_like(
                        param, 
                        memory_format=torch.preserve_format
                    )
                state['step'] += 1
                grad_sums[param] = torch.zeros_like(
                    param,
                    memory_format=torch.preserve_format,
                )
                touched_masks[param] = torch.zeros_like(param, dtype=torch.bool)

        for _ in range(self.k):
            seed = np.random.randint(1_000_000_000)
            self.zo_random_seed = seed
            self.generator.manual_seed(seed)

            self._indices_perturb(scaling_factor = 1.0)
            if closure is not None:
                loss_plus = closure()
                loss_plus_values.append(loss_plus)
            self.generator.manual_seed(seed)

            self._indices_perturb(scaling_factor = -2.0)
            if closure is not None:
                loss_minus = closure()
            self.generator.manual_seed(seed)

            self._indices_perturb(scaling_factor = 1.0)
            self.generator.manual_seed(seed)

            grad_update = self.grad_approx(loss_plus=loss_plus, loss_minus=loss_minus, perturbation_mode="two_side")
            projected_grads.append(grad_update)

            for group in self.param_groups:
                eps = group['eps']
                grad_final = grad_update / (eps * self.k)
                for p in group['params']:
                    indices = self._select_indices(param_shape=p.shape, device=p.device)

                    if isinstance(indices, torch.Tensor):
                        grad_sums[p][indices] += grad_final
                        touched_masks[p][indices] = True
                    else:
                        rows, cols = indices
                        grad_sums[p][rows[:, None], cols] += grad_final
                        touched_masks[p][rows[:, None], cols] = True

        self.projected_grad = sum(projected_grads) / len(projected_grads)

        for group in self.param_groups:
            lr = group['lr']  
            beta = group['beta']

            for p in group['params']:
                state = self.state[p]
                touched = touched_masks[p]
                if torch.any(touched):
                    state['grad_accum'][touched] = (
                        beta * state['grad_accum'][touched] +
                        (1 - beta) * grad_sums[p][touched]
                    )
                
                update_direction = torch.sign(state['grad_accum'])
                p.data.add_(update_direction, alpha=-lr)
                
        return torch.stack(loss_plus_values).mean()
