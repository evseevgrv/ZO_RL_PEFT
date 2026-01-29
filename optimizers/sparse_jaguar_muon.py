from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
import time

from .opt_utils import *

class Sparse_Jaguar_MUON(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            beta: float = 0.9,
            lr: float = 0.01,
            eps: float = 1e-3,
            tensor_sampling_type: str = "standard_normal", 
            matrix_sampling_type: str = None,  
            perturbation_mode: str = "two_side",
            params_ratio: float = 0.1
    ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            tensor_sampling_type=tensor_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
        )

        self.params_ratio = params_ratio
        self.grad_norm = None 
        
        for group in self.param_groups:
            group['beta'] = beta

        self.all_params = [p for group in self.param_groups for p in group['params']]
        total_params = sum(p.numel() for p in self.all_params)
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
        loss1, loss2 = None, None 

        for group in self.param_groups:
            for param in group['params']:    
                state = self.state[param]
                state['step'] += 1

        self.zo_random_seed = np.random.randint(1_000_000_000)
        self.generator.manual_seed(self.zo_random_seed)

        self._sparse_indices_perturb(scaling_factor = 1.0, params_ratio = self.params_ratio)
        if closure is not None:
            loss1 = closure()
        self.generator.manual_seed(self.zo_random_seed)

        self._sparse_indices_perturb(scaling_factor = -2.0, params_ratio = self.params_ratio)
        if closure is not None:
            loss2 = closure()
        self.generator.manual_seed(self.zo_random_seed)

        self._sparse_indices_perturb(scaling_factor = 1.0, params_ratio = self.params_ratio)
        self.generator.manual_seed(self.zo_random_seed)

        grad_update = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="two_side")

        n = max(1, int(len(self.all_params) * self.params_ratio))
        param_indices =  torch.randperm(len(self.all_params), device=self.all_params[0].device, generator=self.generator)[:n]
        self.generator.manual_seed(self.zo_random_seed)
        selected_param_ids = {id(self.all_params[idx]) for idx in param_indices}
        for group in self.param_groups:
            lr = group['lr']  
            beta = group['beta']
            eps = group['eps']
            
            for param in group['params']:     

                state = self.state[param]                
                device = param.device
                
                if id(param) in selected_param_ids:
                    z = self.tensor_sampler.sample(param.shape, generator=self.generator).to(device)
                    grad_final = z * grad_update / eps 
                    state['grad_accum'].mul_(beta).add_(grad_final, alpha=(1.0 - beta))
                
                if param.ndim >= 2:
                    update_direction = zeropower_via_newtonschulz5(state['grad_accum'], steps=5)
                else:
                    update_direction = torch.sign(state['grad_accum'])
                param.data.add_(update_direction, alpha=-lr)

        return loss1
    
    def _sparse_indices_perturb(self, scaling_factor = 1.0, params_ratio = 0.1):
        n = max(1, int(len(self.all_params) * params_ratio))
        param_indices =  torch.randperm(len(self.all_params), device=self.all_params[0].device, generator=self.generator)[:n]
        self.generator.manual_seed(self.zo_random_seed)
        selected_param_ids = {id(self.all_params[idx]) for idx in param_indices}
        for group in self.param_groups:
            eps = group['eps']
            
            for param in group['params']:                     
                if id(param) in selected_param_ids:
                    device = param.device
                    z = self.tensor_sampler.sample(param.shape, generator=self.generator).to(device)
                    param.data += scaling_factor * eps * z
