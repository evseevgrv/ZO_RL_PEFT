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
    ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            tensor_sampling_type=tensor_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
        )
        
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
        
        self.projected_grad = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="two_side")

        self.generator.manual_seed(self.zo_random_seed)
        self._mu_pertrub(scaling_factor=1)       

        self.generator.manual_seed(self.zo_random_seed)
        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            eps = group['eps']
            momentum = group['momentum']
            
            for param in group['params']:
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                device = param.device
                z = torch.normal(mean=0, std=1, size=param.shape, device=param.device, generator=self.generator)
                grad = (z * self.projected_grad) / eps
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

        self.generator.manual_seed(self.zo_random_seed)
        return loss1 
    
    def _mu_pertrub(self, scaling_factor: float = 1.0):
        for group in self.param_groups:
            eps = group['eps']
            for param in group['params']:
                state = self.state[param]
                z = torch.normal(mean=0, std=1, size=param.shape, device=param.device, generator=self.generator)
                param.data.add_(z * eps * scaling_factor)

