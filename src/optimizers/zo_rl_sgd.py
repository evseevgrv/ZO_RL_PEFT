from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Callable, Dict, Any, Union, List, Iterable

class ZORL_SGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            tau: float = 0.01,
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            perturbation_mode: str = "two_side",  # Ignored in new algorithm
            rl_directions: int = 5,
            rl_temperature: float = 1.0,         # Ignored in new algorithm
            vector_sampling_type: str = "lp_sphere"  # Ignored in new algorithm
    ):
        """
        Zero-Order RL-guided Stochastic Gradient Descent optimizer implementing:
        sample {e^i ~ N(mu^t, eps^2*I)}_{i=1}^K
        g_x^t = 1/(tau*K) * sum_i [(f_i - avg_{j≠i} f_j) * e^i]
        g_mu^t = 1/K * sum_i [(f_i - avg_{j≠i} f_j) * (mu^t - e^i)/eps^2]
        x^{t+1} = x^t - lr * g_x^t
        mu^{t+1} = mu^t - lr * g_mu^t
        
        Args:
            params: Parameters to optimize
            tau: Perturbation magnitude (default: 0.01)
            lr: Learning rate (must be specified per group)
            eps: Noise scale (must be specified per group)
            momentum: Ignored (not used in this algorithm)
            gradient_sparsity: Ignored (not used in this algorithm)
            perturbation_mode: Ignored (always uses single-side perturbation)
            rl_directions: Number of candidate directions K (must be >=2)
            rl_temperature: Ignored
            vector_sampling_type: Ignored
        """
        super().__init__(
            params=params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            gradient_sparsity=gradient_sparsity,
            vector_sampling_type=vector_sampling_type
        )
        self.tau = tau
        self.rl_directions = rl_directions
        
        # Initialize mu buffers for all parameters
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['mu'] = torch.zeros_like(p.data)
     
    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """
        Performs a single optimization step following the specified algorithm.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss tensor after the update
        """
        # Collect all parameters requiring gradients and their groups
        params_list = []
        groups_list = []
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    params_list.append(p)
                    groups_list.append(group)
        
        if not params_list:
            return closure()  # No parameters to optimize
        
        # Validate K (must be >=2 for baseline subtraction)
        K = self.rl_directions
        if K < 2:
            raise ValueError("rl_directions must be at least 2 for baseline subtraction")
        
        # Save current state: main parameters (x) and mu buffers
        original_params = [p.data.clone() for p in params_list]
        mu_current = [self.state[p]['mu'].clone() for p in params_list]
        
        # Evaluate K candidates
        losses = []
        candidate_perturbations = []  # Structure: [candidate][param] = perturbation tensor
        
        for _ in range(K):
            perturbations = []
            # Apply perturbation to each parameter
            for j, p in enumerate(params_list):
                group = groups_list[j]
                eps_val = group['eps']
                device = p.device  # Get the device of the parameter
                
                # Sample e ~ N(mu, eps^2 * I) using torch.normal
                noise = torch.normal(
                    mean=0.0, 
                    std=1.0, 
                    size=p.data.shape
                    # generator=self.generator
                ).to(device)  # Ensure noise is on the same device as parameter
                
                e = mu_current[j] + eps_val * noise
                perturbations.append(e)
                
                # Perturb main parameters: x + tau * e
                p.data = original_params[j] + self.tau * e
            
            # Evaluate loss at perturbed point
            loss_val = closure()
            losses.append(loss_val.item())
            candidate_perturbations.append(perturbations)
        
        # Compute baseline-subtracted differences
        total_loss = sum(losses)
        diffs = []
        for i in range(K):
            # Baseline: average of all other losses
            baseline = (total_loss - losses[i]) / (K - 1)
            diffs.append(losses[i] - baseline)
        
        # Initialize gradient accumulators
        g_x_accum = [torch.zeros_like(p.data) for p in params_list]
        g_mu_accum = [torch.zeros_like(p.data) for p in params_list]
        
        # Accumulate gradients for all candidates
        for i in range(K):
            diff = diffs[i]
            for j in range(len(params_list)):
                e = candidate_perturbations[i][j]
                mu = mu_current[j]
                eps_val = groups_list[j]['eps']
                
                # Accumulate g_x component
                g_x_accum[j] += diff * e
                
                # Accumulate g_mu component
                g_mu_accum[j] += diff * (mu - e) / (eps_val ** 2)
        
        # Normalize gradients
        for j in range(len(params_list)):
            g_x_accum[j] /= (self.tau * K)
            g_mu_accum[j] /= K
        
        # Update parameters and mu buffers
        for j, p in enumerate(params_list):
            lr = groups_list[j]['lr']
            
            # Update main parameters: x = x - lr * g_x
            p.data = original_params[j] - lr * g_x_accum[j]
            
            # Update mu buffer: mu = mu - lr * g_mu
            self.state[p]['mu'] = mu_current[j] - lr * g_mu_accum[j]
        
        # Return loss at updated state
        return closure()
