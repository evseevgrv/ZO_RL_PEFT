from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
import time
import wandb
import math
from .opt_utils import *

class ZO_RL_Jaguar(ZeroOrderOptimizer):
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
            evaluate_memory: bool = False,
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
        self.k = max(1, k or 1)
        self.variance = variance
        self.lr = lr
        self.lr_mu = lr_mu if lr_mu is not None else lr
        self.use_grad_first = use_grad_first
        self.evaluate_memory = evaluate_memory
        
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
                    if not self.evaluate_memory:
                        state['mu_old'] = state['mu'].detach().clone()
                        state['mu_old_norm'] = torch.norm(state['mu_old']).item()**2
                        # state['mu'] = torch.zeros_like(
                        #     param, 
                        #     memory_format=torch.preserve_format
                        # )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("ZO_RL_Jaguar requires a closure")
        if self.variance is None or self.variance <= 0:
            raise ValueError("ZO_RL_Jaguar requires variance > 0")

        loss1, loss2 = None, None

        for group in self.param_groups:
            for param in group['params']:    
                state = self.state[param]
                state['step'] += 1
                if 'perturbation_z' in state:
                    del state['perturbation_z']

        if not self.evaluate_memory:
            for group in self.param_groups:
                for param in group['params']:
                    state = self.state[param]
                    state['mu_old'] = state['mu'].detach().clone()
                    state['mu_old_norm'] = torch.norm(state['mu_old']).item()**2

        selection_seed = np.random.randint(1_000_000_000)
        self.generator.manual_seed(selection_seed)
        selected_param_ids = self._sample_selected_param_ids(params_ratio=self.params_ratio)

        # Step 1: Try k random seeds on the same sparse parameter subset.
        e_values = {}

        for idx in range(self.k):
            self.zo_random_seed = np.random.randint(1_000_000_000)
            self.generator.manual_seed(self.zo_random_seed)
            
            # Sparse perturbation forward
            self._sparse_mu_perturb(
                scaling_factor=1.0,
                selected_param_ids=selected_param_ids,
            )
            
            loss = closure()
            e_values[self.zo_random_seed] = self._loss_to_float(loss)
            
            # Reset state
            self.generator.manual_seed(self.zo_random_seed)
            self._sparse_mu_perturb(
                scaling_factor=-1.0,
                selected_param_ids=selected_param_ids,
            )
        
        # Step 2: Select optimal seed
        optimal_seed = min(e_values, key=e_values.get)
        self.zo_random_seed = optimal_seed
        
        # Step 3: Use optimal seed for two-sided finite difference
        self.generator.manual_seed(self.zo_random_seed)
        self._sparse_mu_perturb(
            scaling_factor=1.0,
            selected_param_ids=selected_param_ids,
        )
        
        loss1 = closure()
        
        self.generator.manual_seed(self.zo_random_seed)
        self._sparse_mu_perturb(
            scaling_factor=-2.0,
            selected_param_ids=selected_param_ids,
        )
        
        loss2 = closure()
        
        self.generator.manual_seed(self.zo_random_seed)
        self._sparse_mu_perturb(
            scaling_factor=1.0,
            selected_param_ids=selected_param_ids,
        )
        
        # Step 4: Compute gradient approximation
        projected_grad = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="two_side")
        
        # Step 5: Update both x and mu
        seeds = list(e_values.keys())
        f_tensor = torch.tensor(list(e_values.values()), dtype=torch.float32)
        f_sum = torch.sum(f_tensor)
        # Handle k=1 case to avoid division by zero
        if self.k > 1:
            coeff = (f_tensor * self.k - f_sum) / (self.k - 1)
        else:
            coeff = torch.zeros_like(f_tensor)  # When k=1, mu update term is zero

        
        track_mu_degree = not self.evaluate_memory
        dot_product = 0 
        old_mu_norms = 0
        new_mu_norms = 0

        self.generator.manual_seed(self.zo_random_seed)
        
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            eps = group['eps']
            
            for param in group['params']:
                state = self.state[param]
                
                # OPTIMIZE X and MU only for selected parameters
                if id(param) in selected_param_ids:
                    # OPTIMIZE X - use the SAME z that was used for perturbation (like sparse_jaguar_signsgd)
                    z = self._sample_mu_direction(param)
                    grad_final = z * projected_grad / eps
                    state['grad_accum'].mul_(beta).add_(grad_final, alpha=(1.0 - beta))
                
                # SignSGD update (for all parameters)
                update_direction = torch.sign(state['grad_accum'])
                param.data.add_(update_direction, alpha=-lr)

        mu_diffs = {}
        for group in self.param_groups:
            for param in group['params']:
                if id(param) in selected_param_ids:
                    state = self.state[param]
                    mu_diffs[param] = torch.zeros_like(state['mu'])

        for coeff_value, seed in zip(coeff.tolist(), seeds):
            self.generator.manual_seed(seed)
            for group in self.param_groups:
                for param in group['params']:
                    if id(param) not in selected_param_ids:
                        continue

                    state = self.state[param]
                    mu = state['mu']
                    z = self._sample_mu_direction(param)
                    mu_diffs[param].add_(mu - z, alpha=coeff_value)

        for param, mu_diff in mu_diffs.items():
            state = self.state[param]
            g_mu = -mu_diff / (self.k * (self.variance ** 2))
            state["mu"].add_(g_mu, alpha=-self.lr_mu)
            if track_mu_degree:
                dot_product += torch.sum(state['mu_old'] * state["mu"]).item()
                new_mu_norms += torch.norm(state["mu"]).item()**2
                old_mu_norms += state['mu_old_norm']
            # state["mu"] /= torch.linalg.norm(state["mu"])

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

        if track_mu_degree and new_mu_norms > 0 and old_mu_norms > 0:
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

    @staticmethod
    def _loss_to_float(loss) -> float:
        if torch.is_tensor(loss):
            return loss.detach().float().item()
        return float(loss)

    def _sample_selected_param_ids(self, params_ratio=0.1):
        n = max(1, int(len(self.all_params) * params_ratio))
        param_indices = torch.randperm(
            len(self.all_params),
            device=self.all_params[0].device,
            generator=self.generator,
        )[:n]
        return {id(self.all_params[int(idx)]) for idx in param_indices}

    def _sample_mu_direction(self, param):
        state = self.state[param]
        tensor_sampling_type = state['tensor_sampling_type']
        noise = self.tensor_sampler.sample(
            param.shape,
            generator=self.generator,
            sampler_type=tensor_sampling_type,
        )
        return state['mu'] + self.variance * noise.to(device=param.device, dtype=param.dtype)
    
    def _sparse_mu_perturb(self, scaling_factor=1.0, params_ratio=0.1, selected_param_ids=None):
        """
        Sparse perturbation using mu distribution (mean=mu, std=variance) for each parameter.
        This is the key method that combines sparse selection with mu-based perturbations.
        """
        if selected_param_ids is None:
            selected_param_ids = self._sample_selected_param_ids(params_ratio=params_ratio)
            self.generator.manual_seed(self.zo_random_seed)
        
        for group in self.param_groups:
            eps = group['eps']
            
            for param in group['params']:
                if id(param) in selected_param_ids:
                    # Sample from normal distribution with mean=mu and std=variance
                    z = self._sample_mu_direction(param)
                    param.data.add_(z * eps * scaling_factor)

    def _sparse_mu_perturb_with_seed(self, scaling_factor=1.0, selected_param_ids=None, seed: Optional[int] = None):
        """
        Sparse perturbation with deterministically re-sampled z (for two-sided finite difference).
        This ensures the same z is used for perturbation and gradient accumulation.
        """
        if selected_param_ids is None:
            return
        if seed is None:
            seed = self.zo_random_seed
        self.generator.manual_seed(seed)
        self._sparse_mu_perturb(
            scaling_factor=scaling_factor,
            selected_param_ids=selected_param_ids,
        )
