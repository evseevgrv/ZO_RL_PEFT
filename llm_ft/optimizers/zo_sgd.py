from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
from gradient_pruning import fast_random_mask_like
from .opt_utils import *
from collections import defaultdict

class ZO_SGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = None,
            weight_decay: float = 0.0,
            tensor_sampling_type: str = "standard_normal",
            matrix_sampling_type: str = None, 
            perturbation_mode: str = "two_side",
            k: int = 1,
            evaluate_memory: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            weight_decay=weight_decay,
            tensor_sampling_type=tensor_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
        )
        self.k = max(1, k)
        self.evaluate_memory = evaluate_memory

        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['step'] = 0
        
    @torch.no_grad()
    def step(self, closure=None):
        loss_plus_values = []
        projected_grads = []
        probe_seeds = []
        grad_sums = {}

        for group in self.param_groups:
            for param in group['params']:
                grad_sums[param] = torch.zeros_like(
                    param,
                    memory_format=torch.preserve_format,
                )

        for _ in range(self.k):
            seed = np.random.randint(1_000_000_000)
            probe_seeds.append(seed)
            self.zo_random_seed = seed

            self.generator.manual_seed(seed)
            self._mu_pertrub(scaling_factor=1)
            loss_plus = closure()
            loss_plus_values.append(loss_plus)

            self.generator.manual_seed(seed)
            self._mu_pertrub(scaling_factor=-2)
            loss_minus = closure()

            projected_grads.append((loss_plus - loss_minus) / 2)

            self.generator.manual_seed(seed)
            self._mu_pertrub(scaling_factor=1)

        self.projected_grad = torch.stack(projected_grads).mean()

        for seed, projected_grad in zip(probe_seeds, projected_grads):
            self.generator.manual_seed(seed)
            for group in self.param_groups:
                eps = group['eps']
                for param in group['params']:
                    z = torch.normal(mean=0, std=1, size=param.shape, device=param.device, generator=self.generator)
                    grad_sums[param].add_(z * (projected_grad / (eps * self.k)))

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            
            for param in group['params']:
                state = self.state[param]
                state['step'] += 1

                grad = grad_sums[param]
                if momentum is not None and momentum != 0:
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(grad)
                    update = buf
                else:
                    update = grad    
                param.data.add_(update, alpha=-lr)
                
        return torch.stack(loss_plus_values).mean()
    
    def _mu_pertrub(self, scaling_factor: float = 1.0):
        for group in self.param_groups:
            eps = group['eps']
            for param in group['params']:
                state = self.state[param]
                z = torch.normal(mean=0, std=1, size=param.shape, device=param.device, generator=self.generator)
                param.data.add_(z * eps * scaling_factor)
