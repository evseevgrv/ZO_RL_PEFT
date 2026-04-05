import torch
from torch.optim import Optimizer
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable, Tuple
from .base import ZeroOrderOptimizer

class ZO_AdaMM(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            weight_decay: float = 0.0,
            tensor_sampling_type: str = "standard_normal",
            matrix_sampling_type: str = None, 
            perturbation_mode: str = "two_side",
            betas: Tuple[float, float] = (0.9, 0.999),
            k: int = 1,
            evaluate_memory: bool = False,
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
        self.k = max(1, k)
        self.evaluate_memory = evaluate_memory
        
        for group in self.param_groups:
            group['betas'] = betas
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        loss_plus_values = []
        projected_grads = []
        probe_seeds = []
        grad_sums = {}

        for group in self.param_groups:
            for p in group['params']:
                grad_sums[p] = torch.zeros_like(p, memory_format=torch.preserve_format)

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
                for p in group['params']:
                    z = torch.normal(mean=0, std=1, size=p.shape, device=p.device, generator=self.generator)
                    grad_sums[p].add_(z * (projected_grad / (eps * self.k)))

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            for p in group['params']:
                state = self.state[p]
                state['step'] += 1
                grad = grad_sums[p]
    
                # Do the AdaMM updates
                state['exp_avg'].mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2))
                state['max_exp_avg_sq'] = torch.maximum(state['max_exp_avg_sq'],
                                                        state['exp_avg_sq'])

                # Use max_exp_avg_sq for normalization as per Algorithm 1
                # Add small epsilon for numerical stability (separate from perturbation eps)
                p.data.addcdiv_(state['exp_avg'], state['max_exp_avg_sq'].sqrt().add_(1e-10), value=(-lr))

        return torch.stack(loss_plus_values).mean()
    
    def _mu_pertrub(self, scaling_factor: float = 1.0):
        for group in self.param_groups:
            eps = group['eps']
            for param in group['params']:
                # Use the generator directly without resetting seed per parameter
                # This ensures all parameters use the same random sequence
                # self.generator.manual_seed(self.zo_random_seed)' 
                state = self.state[param]
                # if 'seed' not in state:
                #     state['seed'] = np.random.randint(1_000_000_000)
                # seed = state['seed'] 
                # self.generator.manual_seed(seed)
                z = torch.normal(mean=0, std=1, size=param.shape, device=param.device, generator=self.generator)
                param.data.add_(z * eps * scaling_factor)
