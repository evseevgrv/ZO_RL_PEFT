from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
from gradient_pruning import fast_random_mask_like
from .opt_utils import *
from collections import defaultdict

class ZO_RL(ZeroOrderOptimizer):
    def __init__(
            self,
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            vector_sampling_type: str = "standard_normal",
            perturbation_mode: str = "two_side",
            k: Optional[int] = 10,
            variance: Optional[float] = 1e-3,
            lr_mu: Optional[float] = None
    ):
        super().__init__(
            params=params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            vector_sampling_type=vector_sampling_type,
            gradient_sparsity=gradient_sparsity,
        )
        self.lr = lr 
        self.lr_mu = lr_mu if lr_mu is not None else lr 
        self.perturbation_mode = perturbation_mode 
        self.k = k
        self.variance = variance

    @torch.no_grad()
    def step(self, closure=None):
        loss1, loss2 = None, None
        self._prepare_parameters()

        for group in self.param_groups:
            for param in group['params']:    
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['mu'] = torch.zeros_like(
                        param, 
                        memory_format=torch.preserve_format
                    )
                state['step'] += 1

        e_values = {}

        for idx in range(self.k):
            self.zo_random_seed = np.random.randint(1_000_000_000)
            self.generator.manual_seed(self.zo_random_seed)
            for group in self.param_groups:
                for p in group['params']:
                    z = torch.normal(mean=self.state[p]["mu"], std=self.variance, generator=self.generator)
                    perturb = z * self.zo_eps
                    perturb = perturb.to(p.device)
                    p.data.add_(perturb)

            loss1 = closure()
            e_values[self.zo_random_seed] = loss1 

            self.generator.manual_seed(self.zo_random_seed)

            for group in self.param_groups:
                for p in group['params']:
                    z = torch.normal(mean=self.state[p]["mu"], std=self.variance, generator=self.generator)
                    perturb = z * self.zo_eps
                    perturb = perturb.to(p.device)
                    p.data.add_(-perturb)
        
        optimal_seed = min(e_values, key=e_values.get)

        loss1 = e_values[optimal_seed]
        
        self.zo_random_seed = optimal_seed
        self.generator.manual_seed(self.zo_random_seed)

        for group in self.param_groups:
            for p in group['params']:
                z = torch.normal(mean=self.state[p]["mu"], std=self.variance, generator=self.generator)
                perturb = z * self.zo_eps
                perturb = perturb.to(p.device)
                p.data.add_(-perturb)

        loss2 = closure()

        self.generator.manual_seed(self.zo_random_seed)

        for group in self.param_groups:
            for p in group['params']:
                z = torch.normal(mean=self.state[p]["mu"], std=self.variance, generator=self.generator)
                perturb = z * self.zo_eps
                perturb = perturb.to(p.device)
                p.data.add_(perturb)

        projected_grad = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="two_side")
        self.generator.manual_seed(self.zo_random_seed)

        seeds = list(e_values.keys())
        f_tensor = torch.tensor(list(e_values.values()), device=self.device)
        f_sum = torch.sum(f_tensor)
        coeff = (f_tensor * self.k - f_sum) / (self.k - 1)

        self.generator.manual_seed(self.zo_random_seed)

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
                    e_samples_list.append(z)
                e_samples = torch.stack(e_samples_list, dim=0)  # shape (k, *p.shape)
                
                mu_diff = mu.unsqueeze(0) - e_samples  # broadcast mu to (1, *p.shape) -> (k, *p.shape)
                
                # Broadcast coeff to (k, 1, 1, ...) matching e_samples dims
                expanded_coeff = coeff.view(self.k, *([1] * len(p.shape)))
                
                term = expanded_coeff * mu_diff  # shape (k, *p.shape)
                
                sum_term = torch.sum(term, dim=0)  # shape (*p.shape)
                
                g_mu = -sum_term / (self.k * (self.variance ** 2))
                self.state[p]["mu"].add_(g_mu, alpha=-self.lr_mu)

        return loss1  
