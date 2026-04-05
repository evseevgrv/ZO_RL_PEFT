from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
import time
import wandb
import math
from .opt_utils import *

class ZO_RL(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            beta: float = 0.9,
            lr: float = 0.01,
            eps: float = 1e-3,
            tensor_sampling_type: str = "standard_normal", 
            matrix_sampling_type: str = None,  
            perturbation_mode: str = "two_side",
            params_ratio: float = 0.1,
            k: Optional[int] = 10,
            variance: Optional[float] = 1e-3,
            lr_mu: Optional[float] = None,
            use_grad_first: bool = False,
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
        self.k = k
        self.variance = variance
        self.lr = lr
        self.lr_mu = lr_mu if lr_mu is not None else lr
        self.use_grad_first = use_grad_first
        
        for group in self.param_groups:
            group['beta'] = beta

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
                    if param.requires_grad and self.use_grad_first:
                        if param.grad is None:
                            raise ValueError("param.grad is None, but use_grad_first is True")

                        state['mu'] = param.grad.clone()
                    else:
                        state['mu'] = torch.randn_like(
                            param,
                            memory_format=torch.preserve_format
                        )
                        state['mu'] /= torch.linalg.norm(state['mu'])
                    state['mu_old'] = state['mu'].detach().clone()
                    state['mu_old_norm'] = torch.norm(state['mu_old']).item()**2
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

        # Step 1: Try k random seeds and evaluate each
        e_values = {}

        for idx in range(self.k):
            self.zo_random_seed = np.random.randint(1_000_000_000)
            self.generator.manual_seed(self.zo_random_seed)
            
            # Sparse perturbation forward
            self._sparse_mu_perturb(scaling_factor=1.0, params_ratio=self.params_ratio)
            
            if closure is not None:
                loss = closure()
                e_values[self.zo_random_seed] = loss
            
            # Reset state
            self.generator.manual_seed(self.zo_random_seed)
            self._sparse_mu_perturb(scaling_factor=-1.0, params_ratio=self.params_ratio)
        
        # Step 2: Select optimal seed
        optimal_seed = min(e_values, key=e_values.get)
        loss1 = e_values[optimal_seed]
        self.zo_random_seed = optimal_seed
        
        # Step 3: Use optimal seed for two-sided finite difference
        # First, save the z values used for perturbation (same as sparse_jaguar_signsgd)
        self.generator.manual_seed(self.zo_random_seed)
        # Select parameters for perturbation (same logic as sparse_jaguar_signsgd)
        n = max(1, int(len(self.all_params) * self.params_ratio))
        param_indices = torch.randperm(len(self.all_params), device=self.all_params[0].device, generator=self.generator)[:n]
        self.generator.manual_seed(self.zo_random_seed)
        selected_param_ids = {id(self.all_params[idx]) for idx in param_indices}
        
        # Save z values for selected parameters (will be reused for gradient accumulation)
        for group in self.param_groups:
            for param in group['params']:
                if id(param) in selected_param_ids:
                    state = self.state[param]
                    mu = state['mu']
                    device = param.device
                    # Sample z and save it (same z will be used for gradient accumulation)
                    self.generator.manual_seed(self.zo_random_seed)
                    z = torch.normal(mean=mu, std=self.variance, generator=self.generator).to(device)
                    state['perturbation_z'] = z.clone()
        
        # Now apply perturbation with saved z
        self.generator.manual_seed(self.zo_random_seed)
        self._sparse_mu_perturb_with_saved_z(scaling_factor=1.0, selected_param_ids=selected_param_ids)
        
        if closure is not None:
            loss1 = closure()
        
        self.generator.manual_seed(self.zo_random_seed)
        self._sparse_mu_perturb_with_saved_z(scaling_factor=-2.0, selected_param_ids=selected_param_ids)
        
        if closure is not None:
            loss2 = closure()
        
        self.generator.manual_seed(self.zo_random_seed)
        self._sparse_mu_perturb_with_saved_z(scaling_factor=1.0, selected_param_ids=selected_param_ids)
        
        # Step 4: Compute gradient approximation
        projected_grad = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="two_side")
        
        # Step 5: Update both x and mu
        seeds = list(e_values.keys())
        f_tensor = torch.tensor(list(e_values.values()))
        f_sum = torch.sum(f_tensor)
        # Handle k=1 case to avoid division by zero
        if self.k > 1:
            coeff = (f_tensor * self.k - f_sum) / (self.k - 1)
        else:
            coeff = torch.zeros_like(f_tensor)  # When k=1, mu update term is zero

        
        dot_product = 0 
        old_mu_norms = 0
        new_mu_norms = 0
        
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            eps = group['eps']
            
            for param in group['params']:
                state = self.state[param]
                device = param.device
                
                # OPTIMIZE X and MU only for selected parameters
                if id(param) in selected_param_ids:
                    mu = state['mu']
                    
                    # OPTIMIZE X - use the SAME z that was used for perturbation (like sparse_jaguar_signsgd)
                    self.generator.manual_seed(self.zo_random_seed)
                    z = state['perturbation_z']  # Reuse the same z from perturbation
                    grad_final = z * projected_grad / eps
                    state['grad_accum'].mul_(beta).add_(grad_final, alpha=(1.0 - beta))
                    
                    # OPTIMIZE MU
                    mu_diff = torch.zeros_like(mu).to(device)
                    for i, seed in enumerate(seeds):
                        self.generator.manual_seed(seed)
                        z = torch.normal(mean=mu, std=self.variance, generator=self.generator).to(device)
                        mu_diff += (mu - z) * coeff[i]
                    
                    g_mu = -mu_diff / (self.k * (self.variance ** 2))
                    state["mu"].add_(g_mu, alpha=-self.lr_mu)
                    dot_product += torch.sum(state['mu_old'] * state["mu"]).item()
                    new_mu_norms += torch.norm(state["mu"]).item()**2
                    old_mu_norms += state['mu_old_norm']
                    # state["mu"] /= torch.linalg.norm(state["mu"])
                
                # SignSGD update (for all parameters)
                update_direction = torch.sign(state['grad_accum'])
                param.data.add_(update_direction, alpha=-lr)

        # Calculate and log average norm of mu across all parameters
        mu_norms = []
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                if 'mu' in state:
                    mu_norm = torch.norm(state['mu']).item()
                    mu_norms.append(mu_norm)
        
        # Get step from first parameter
        step = None
        try:
            first_param = list(self.param_groups[0]['params'])[0]
            step = self.state[first_param].get('step', None)
        except (KeyError, IndexError, AttributeError):
            pass
        
        if mu_norms:
            avg_mu_norm = sum(mu_norms) / len(mu_norms)
            if wandb.run is not None:
                wandb.log({"avg_mu_norm": avg_mu_norm})
            # Log to file if enabled
            try:
                from trainer import _optimizer_log_func
                if _optimizer_log_func is not None and step is not None:
                    _optimizer_log_func({"avg_mu_norm": avg_mu_norm}, step=step)
            except (ImportError, AttributeError):
                pass

        mu_degree = dot_product / (math.sqrt(new_mu_norms) * math.sqrt(old_mu_norms))
        if wandb.run is not None:
            wandb.log({"mu_degree": mu_degree})
        # Log to file if enabled
        try:
            from trainer import _optimizer_log_func
            if _optimizer_log_func is not None and step is not None:
                _optimizer_log_func({"mu_degree": mu_degree}, step=step)
        except (ImportError, AttributeError):
            pass
        return loss1
    
    def _sparse_mu_perturb(self, scaling_factor=1.0, params_ratio=0.1):
        """
        Sparse perturbation using mu distribution (mean=mu, std=variance) for each parameter.
        This is the key method that combines sparse selection with mu-based perturbations.
        """
        n = max(1, int(len(self.all_params) * params_ratio))
        param_indices = torch.randperm(len(self.all_params), device=self.all_params[0].device, generator=self.generator)[:n]
        self.generator.manual_seed(self.zo_random_seed)
        selected_param_ids = {id(self.all_params[idx]) for idx in param_indices}
        
        for group in self.param_groups:
            eps = group['eps']
            
            for param in group['params']:
                if id(param) in selected_param_ids:
                    state = self.state[param]
                    mu = state['mu']
                    device = param.device
                    
                    # Sample from normal distribution with mean=mu and std=variance
                    self.generator.manual_seed(self.zo_random_seed)
                    z = torch.normal(mean=mu, std=self.variance, generator=self.generator).to(device)
                    param.data.add_(z * eps * scaling_factor)
    
    def _sparse_mu_perturb_with_saved_z(self, scaling_factor=1.0, selected_param_ids=None):
        """
        Sparse perturbation using pre-saved z values (for two-sided finite difference).
        This ensures the same z is used for perturbation and gradient accumulation.
        """
        for group in self.param_groups:
            eps = group['eps']
            
            for param in group['params']:
                if id(param) in selected_param_ids:
                    state = self.state[param]
                    if 'perturbation_z' in state:
                        z = state['perturbation_z']
                        self.generator.manual_seed(self.zo_random_seed)
                        param.data.add_(z * eps * scaling_factor)
