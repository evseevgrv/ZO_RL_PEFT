from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
from gradient_pruning import fast_random_mask_like
from .opt_utils import *
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class ZO_RL(ZeroOrderOptimizer):
    def __init__(
            self,
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            tensor_sampling_type: str = "standard_normal",
            perturbation_mode: str = "two_side",
            k: Optional[int] = 10,
            variance: Optional[float] = 1e-3,
            lr_mu: Optional[float] = None,
            use_grad_first: bool = False,
    ):
        super().__init__(
            params=params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            tensor_sampling_type=tensor_sampling_type,
            gradient_sparsity=gradient_sparsity,
        )
        self.lr = lr 
        self.lr_mu = lr_mu if lr_mu is not None else lr 
        self.perturbation_mode = perturbation_mode 
        self.k = k
        self.variance = variance
        self.use_grad_first = use_grad_first
        self.tracked_param_id = 0  # Track parameter with id=0 for logging
        for group in self.param_groups:
            for param in group['params']:    
                state = self.state[param]
                if 'step' not in state:
                    state['step'] = 0
                    if param.requires_grad and self.use_grad_first:
                        if param.grad is None:
                            raise ValueError("param.grad is None, but use_grad_first is True")
                            
                        state['mu'] = param.grad.clone()
                        # state['mu'] /= torch.linalg.norm(state['mu'])
                    else:
                        state['mu'] = torch.zeros_like(
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

        e_values = {}

        for idx in range(self.k):
            self.zo_random_seed = np.random.randint(1_000_000_000)
            self.generator.manual_seed(self.zo_random_seed)
            for group in self.param_groups:
                eps = group['eps']
                for p in group['params']:   
                    state = self.state[p]
                    z = torch.normal(mean=state["mu"], std=self.variance, generator=self.generator)
                    # z /= torch.linalg.norm(z)
                    p.data.add_(z * eps)

            loss1 = closure()
            e_values[self.zo_random_seed] = loss1 

            self.generator.manual_seed(self.zo_random_seed)

            for group in self.param_groups:
                eps = group['eps']
                for p in group['params']:
                    state = self.state[p]
                    z = torch.normal(mean=state["mu"], std=self.variance, generator=self.generator)
                    # z /= torch.linalg.norm(z)
                    p.data.add_(-z * eps)
        
        optimal_seed = min(e_values, key=e_values.get)

        loss1 = e_values[optimal_seed]
        
        self.zo_random_seed = optimal_seed
        self.generator.manual_seed(self.zo_random_seed)

        for group in self.param_groups:
            eps = group['eps']
            for p in group['params']:
                state = self.state[p]
                z = torch.normal(mean=state["mu"], std=self.variance, generator=self.generator)
                # z /= torch.linalg.norm(z)
                p.data.add_(-z * eps)

        loss2 = closure()

        self.generator.manual_seed(self.zo_random_seed)

        for group in self.param_groups:
            eps = group['eps']
            for p in group['params']:
                state = self.state[p]
                z = torch.normal(mean=state["mu"], std=self.variance, generator=self.generator)
                #   z /= torch.linalg.norm(z)
                p.data.add_(z * eps)

        projected_grad = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="two_side")
        self.generator.manual_seed(self.zo_random_seed)

        seeds = list(e_values.keys())
        f_tensor = torch.tensor(list(e_values.values()))
        f_sum = torch.sum(f_tensor)
        coeff = (f_tensor * self.k - f_sum) / (self.k - 1)

        self.generator.manual_seed(self.zo_random_seed)

        param_id = 0
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                mu = state['mu']

                # OPTIMIZE X
                z = torch.normal(mean=mu, std=self.variance, generator=self.generator)
                g_x = projected_grad * z 

                p.data.add_(g_x, alpha=-self.lr)
                
                # OPTIMIZE MU
                e_samples_list = []
                for seed in seeds:
                    self.generator.manual_seed(seed)
                    z = torch.normal(mean=mu, std=self.variance, generator=self.generator)
                    # z /= torch.linalg.norm(z)
                    e_samples_list.append(z)
                e_samples = torch.stack(e_samples_list, dim=0)  # shape (k, *p.shape)
                
                mu_diff = (mu.unsqueeze(0) - e_samples).to(p.device)  # broadcast mu to (1, *p.shape) -> (k, *p.shape)
                
                # Broadcast coeff to (k, 1, 1, ...) matching e_samples dims
                expanded_coeff = (coeff.view(self.k, *([1] * len(p.shape)))).to(p.device)
                term = expanded_coeff * mu_diff  # shape (k, *p.shape)
                sum_term = torch.sum(term, dim=0)  # shape (*p.shape)
                
                g_mu = -sum_term / (self.k * (self.variance ** 2))

                state["mu"].add_(g_mu, alpha=-self.lr_mu)
                # state['mu'] /= torch.linalg.norm(state['mu'])
                
                # Log g_x and mu for tracked parameter
                if param_id == self.tracked_param_id:
                    print(f"\nStep {state['step']}, Param ID {param_id}:")
                    print(f"\ng_x stats: {g_x}")
                    print(f"\nmu stats: {mu}")
                    print(f"\ng_mu stats: {g_mu}")
                
                param_id += 1

        return loss1  
