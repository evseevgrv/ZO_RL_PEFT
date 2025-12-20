from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
from gradient_pruning import fast_random_mask_like
from .opt_utils import *
from collections import defaultdict
import wandb

class ZO_RL_SGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = None,
            weight_decay: float = 0.0,
            tensor_sampling_type: str = "standard_normal",
            matrix_sampling_type: str = None, 
            perturbation_mode: str = "two_side",
            k: Optional[int] = 10,
            variance: Optional[float] = 1e-3,
            lr_mu: Optional[float] = None,
            use_grad_first: bool = False,
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
        # if torch.cuda.is_available():
        #     self.generator = torch.Generator(device='cuda')
        # else:
        #     self.generator = torch.Generator(device='cpu')
        self.k = k
        self.variance = variance
        self.lr_mu = lr_mu if lr_mu is not None else lr
        self.use_grad_first = use_grad_first
        self.mu0 = True
        for group in self.param_groups:
            for param in group['params']:    
                state = self.state[param]
                if 'step' not in state:
                    state['step'] = 0
                    if param.requires_grad and self.use_grad_first:
                        if param.grad is None:
                            raise ValueError("param.grad is None, but use_grad_first is True")
                            
                        state['mu'] = param.grad.clone()
                    else:
                        state['mu'] = torch.randn_like( # N(0,1)
                            param, 
                            memory_format=torch.preserve_format
                        )
                        state['mu'] /= torch.linalg.norm(state['mu'])

                        state['z'] = {}
                        # state['mu'] = torch.zeros_like(
                        #     param, 
                        #     memory_format=torch.preserve_format
                        # )
                        
    @torch.no_grad()
    def step(self, closure=None):
        loss1, loss2 = None, None 
        
        for group in self.param_groups:
            for param in group['params']:    
                state = self.state[param]
                state['step'] += 1
                state['z'] = {}
                state['perturbation_z'] = None

        if self.mu0:
            self.zo_random_seed = np.random.randint(1_000_000_000)
            self.generator.manual_seed(self.zo_random_seed)
            self._zo_pertrub(scaling_factor=1.0)

            if closure is not None:
                loss1 = closure()
                        
            self.generator.manual_seed(self.zo_random_seed)
            self._zo_pertrub(scaling_factor=-2.0)
                        
            if closure is not None:
                loss2 = closure()
                        
            self.generator.manual_seed(self.zo_random_seed)
            self._zo_pertrub(scaling_factor=1.0)

            self.projected_grad = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="two_side")
            self.generator.manual_seed(self.zo_random_seed)
            for group in self.param_groups:
                eps = group['eps']
                for param in group['params']:
                    state = self.state[param]
                    z = torch.normal(mean=0, std=1, size=param.shape, device=param.device, generator=self.generator)
                    state['mu'] = self.projected_grad * z / eps
                    state['mu'] /= torch.linalg.norm(state['mu'])
                    
            self.mu0 = False
        
        loss_values = {}
       
        for _ in range(self.k):
            self.zo_random_seed = np.random.randint(1_000_000_000)

            self.generator.manual_seed(self.zo_random_seed)
            self._mu_pertrub(scaling_factor=1.0)
            
            if closure is not None:
                loss = closure()
                loss_values[self.zo_random_seed] = loss
            
            self.generator.manual_seed(self.zo_random_seed)
            self._mu_pertrub(scaling_factor=-1.0)

        optimal_seed = min(loss_values, key=loss_values.get)
        self.zo_random_seed = optimal_seed
        # self.generator.manual_seed(self.zo_random_seed)

        # for group_idx, group in enumerate(self.param_groups):
        #     lr = group['lr']
        #     eps = group['eps']

        #     for param in group['params']:
        #         state = self.state[param]
        #         mu = state['mu']
        #         state['perturbation_z'] = torch.normal(mean=mu, std=self.variance, generator=self.generator).to(param.device)

        self.generator.manual_seed(self.zo_random_seed)
        self._mu_pertrub(scaling_factor=1.0)
        
        if closure is not None:
            loss1 = closure()
        
        self.generator.manual_seed(self.zo_random_seed)
        self._mu_pertrub(scaling_factor=-2.0)
        
        if closure is not None:
            loss2 = closure()
        
        self.generator.manual_seed(self.zo_random_seed)
        self._mu_pertrub(scaling_factor=1.0)
        
        # Step 4: Compute gradient approximation
        self.projected_grad = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="two_side")
        
        # Step 5: Update both x and mu
        seeds = list(loss_values.keys())
        f_tensor = torch.tensor(list(loss_values.values()))
        f_sum = torch.sum(f_tensor)
        # Handle k=1 case to avoid division by zero
        if self.k > 1:
            coeff = (f_tensor * self.k - f_sum) / (self.k - 1)
        else:
            coeff = torch.zeros_like(f_tensor)  # When k=1, mu update term is zero

        # self.generator.manual_seed(self.zo_random_seed)
        # self._mu_pertrub(scaling_factor=-1.0)

        # if closure is not None:
        #     loss2 = closure()

        # self.projected_grad = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="two_side") 

        # self.generator.manual_seed(self.zo_random_seed)
        # self._mu_pertrub(scaling_factor=1.0)

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
                mu = state['mu']
                # z = torch.normal(mean=mu, std=self.variance, generator=self.generator)
                z = state['z'][self.zo_random_seed]
                grad = (z * self.projected_grad) / eps    
                if momentum is not None and momentum != 0:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        state['momentum_buffer'].mul_(momentum).add_(grad)
                    update = state['momentum_buffer']
                else:
                    update = grad    
                param.data.add_(update, alpha=-lr)

                # OPTIMIZE MU
                mu_diff = torch.zeros_like(mu).to(device)
                for i, seed in enumerate(seeds):
                    # z = torch.normal(mean=mu, std=self.variance, generator=self.generator)
                    z = state['z'][seed]
                    # self.generator.manual_seed(seed)
                    # z = torch.normal(mean=mu, std=self.variance, generator=self.generator).to(device)
                    mu_diff += (mu - z) * coeff[i]
                    
                g_mu = -mu_diff / (self.k * (self.variance ** 2))
                state['mu'].add_(g_mu, alpha=-self.lr_mu)


         # Calculate and log average norm of mu across all parameters
        mu_norms = []
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                if 'mu' in state:
                    mu_norm = torch.norm(state['mu']).item()
                    mu_norms.append(mu_norm)
        
        if mu_norms:
            avg_mu_norm = sum(mu_norms) / len(mu_norms)
            wandb.log({"avg_mu_norm": avg_mu_norm})
        self.generator.manual_seed(self.zo_random_seed)
        return loss1     

    def _mu_pertrub(self, scaling_factor: float = 1.0):
        for group in self.param_groups:
            eps = group['eps']
            for param in group['params']:
                state = self.state[param]
                mu = state['mu']
                if self.zo_random_seed not in state['z']:
                    z = torch.normal(mean=mu, std=self.variance, generator=self.generator).to(param.device)
                    state['z'][self.zo_random_seed] = z.clone()
                else:
                    z = state['z'][self.zo_random_seed]

                param.data.add_(z * eps * scaling_factor)

    def _zo_pertrub(self, scaling_factor: float = 1.0):
        for group in self.param_groups:
            eps = group['eps']
            for param in group['params']:
                state = self.state[param]
                z = torch.normal(mean=0, std=1, size=param.shape, device=param.device, generator=self.generator)
          
                param.data.add_(z * eps * scaling_factor)
                
    # def _sparse_mu_perturb_with_saved_z(self, scaling_factor=1.0, selected_param_ids=None):
    #     """
    #     Sparse perturbation using pre-saved z values (for two-sided finite difference).
    #     This ensures the same z is used for perturbation and gradient accumulation.
    #     """
    #     for group in self.param_groups:
    #         eps = group['eps']
    #         for param in group['params']:
    #             state = self.state[param]
    #             # if 'perturbation_z' in state:
    #             if 'z' in state:
    #                 z = state['z'][self.zo_random_seed]
    #                 param.data.add_(z * eps * scaling_factor)
