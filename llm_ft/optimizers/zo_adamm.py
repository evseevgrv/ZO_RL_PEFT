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
        loss1, loss2 = None, None 
        
        self.zo_random_seed = np.random.randint(1_000_000_000)

        self.generator.manual_seed(self.zo_random_seed)
        self._mu_pertrub(scaling_factor=1)
        loss1 = closure()

        self.generator.manual_seed(self.zo_random_seed)
        self._mu_pertrub(scaling_factor=-2)
        loss2 = closure()
        
        self.projected_grad = (loss1 - loss2) / 2

        # Restore parameters to original state (undo perturbations)
        self.generator.manual_seed(self.zo_random_seed)
        self._mu_pertrub(scaling_factor=1)

        self.generator.manual_seed(self.zo_random_seed)
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            for p in group['params']:
                state = self.state[p]
                state['step'] += 1
                
                # z = torch.normal(mean=0, std=1, size=p.shape, device=p.device, generator=self.generator)
                # z = state['z']
                
                # seed = state['seed'] 
                # self.generator.manual_seed(seed)
                z = torch.normal(mean=0, std=1, size=p.shape, device=p.device, generator=self.generator)
                grad = (z * self.projected_grad) / eps
    
                # Do the AdaMM updates
                state['exp_avg'].mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2))
                state['max_exp_avg_sq'] = torch.maximum(state['max_exp_avg_sq'],
                                                        state['exp_avg_sq'])

                # Use max_exp_avg_sq for normalization as per Algorithm 1
                # Add small epsilon for numerical stability (separate from perturbation eps)
                p.data.addcdiv_(state['exp_avg'], state['max_exp_avg_sq'].sqrt().add_(1e-10), value=(-lr))

        return loss1
    
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
