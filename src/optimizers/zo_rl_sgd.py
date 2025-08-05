from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Callable, Dict, Any, Union, List, Iterable, Tuple
from gradient_pruning import fast_random_mask_like

class ZORL_SGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            tau: float = 0.01,  # Default value for tau
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            perturbation_mode: str = "two_side",  # Should be "two_side" for your algorithm
            rl_directions: int = 5,
            rl_temperature: float = 1.0,
            vector_sampling_type: str = "lp_sphere"  # Using lp_sphere with p=2 for unit sphere
    ):
        """
        Zero-Order RL-guided Stochastic Gradient Descent optimizer.
        Uses RL-inspired approach to select directions that minimize f(x+tau*u).
        
        Args:
            params: Parameters to optimize (can specify per-group hyperparameters)
            tau: Perturbation magnitude (default: 0.01)
            lr: Optional base learning rate (must be specified in groups if None)
            eps: Optional base perturbation (must be specified in groups if None)
            momentum: Momentum factor (default: 0.0)
            gradient_sparsity: Gradient sparsity (float or dict per parameter)
            perturbation_mode: 'one_side' or 'two_side' (default: 'two_side')
            rl_directions: Number of candidate directions to evaluate for RL selection (default: 5)
            rl_temperature: Temperature for softmax selection (higher = more exploration)
            vector_sampling_type: Type of vector sampling ('lp_sphere' with p=2 for unit sphere)
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
        self.rl_temperature = rl_temperature
        self.perturbation_mode = perturbation_mode
    
    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """
        RL-guided Zero-Order step following the algorithm:
        1. Sample {u_k}_{i=1}^N ~ S_2^1(0) (unit sphere)
        2. u^t = argmin_{u in {u_k}} f(x^t + tau*u)
        3. ∇̂f(x^t) = (f(x^t + tau*u^t) - f(x^t - tau*u^t))/(2*tau)
        4. x^{t+1} = x^t - gamma^t * ∇̂f(x^t)
        
        Args:
            closure: Callable that returns the loss
            
        Returns:
            Loss tensor after the update
        """
        self._prepare_parameters()
        
        # Store original parameters to reset after candidate evaluation
        original_params = self._get_flat_params()
        
        # Generate candidate directions directly from the unit sphere
        candidate_directions = []
        candidate_losses = []
        
        for _ in range(self.rl_directions):
            # Sample a direction vector from the unit sphere
            direction = self._sample_direction()
            candidate_directions.append(direction)
            
            # Perturb in the candidate direction: x + tau*u
            self._apply_direction(direction, scaling_factor=self.tau)
            
            # Evaluate loss after perturbation
            loss_val = closure()
            candidate_losses.append(loss_val.item())
            
            # Reset to original parameters
            self._set_flat_params(original_params)
        
        # Select the best direction (with minimum loss)
        best_idx = np.argmin(candidate_losses)
        best_direction = candidate_directions[best_idx]
        
        # Compute f(x + tau*u^t)
        self._apply_direction(best_direction, scaling_factor=self.tau)
        loss_plus = closure()
        
        # Compute f(x - tau*u^t)
        self._apply_direction(best_direction, scaling_factor=-2*self.tau)
        loss_minus = closure()
        
        # Reset back to original parameters
        self._set_flat_params(original_params)
        
        # Estimate the gradient: (f(x+tau*u) - f(x-tau*u))/(2*tau)
        # Note: In ZO_SGD, the gradient approximation uses eps, but here we use tau directly
        # We need to store this for _apply_gradients
        self.projected_grad = (loss_plus - loss_minus) / (2 * self.tau)
        
        # Use the best direction for gradient estimation and update
        self._apply_gradients(direction=best_direction)
        
        return loss_plus
    
    @torch.no_grad()
    def _sample_direction(self) -> List[torch.Tensor]:
        """
        Sample a direction vector from the unit sphere for all parameters.
        Returns:
            List of direction vectors (one per parameter tensor)
        """
        directions = []
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    # Sample a direction vector from the unit sphere (p=2 for L2 norm)
                    direction = self.vector_sampler.sample(p.shape, generator=self.generator)
                    directions.append(direction)
        return directions
    
    @torch.no_grad()
    def _apply_direction(self, direction: List[torch.Tensor], scaling_factor: float = 1.0) -> None:
        """
        Apply a specific direction to the parameters.
        
        Args:
            direction: List of direction vectors (one per parameter tensor)
            scaling_factor: Scaling factor for the perturbation
        """
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    # Apply the direction with scaling factor
                    p.data.add_(scaling_factor * direction[idx])
                    idx += 1
    
    @torch.no_grad()
    def _apply_gradients(self, direction: List[torch.Tensor]) -> None:
        """
        Apply the gradient update using the estimated gradient.
        This method correctly uses _inner_optimizers as in ZO_SGD.
        
        Args:
            direction: The best direction selected by the RL process
        """
        # Prepare parameters for gradient application
        self.named_parameters_to_optim = [
            (name, param) for name, param in self.named_parameters_all 
            if param.requires_grad
        ]
        
        # We need to set the gradient for each parameter
        idx = 0
        for group_idx, group in enumerate(self.param_groups):
            for param in group['params']:
                if not param.requires_grad:
                    continue
                
                # Calculate the gradient contribution for this parameter
                # In ZO_SGD: grad = (projected_grad * z * eps) / len(random_seeds)
                # Here: projected_grad is already (f(x+tau*u)-f(x-tau*u))/(2*tau)
                # But we need to multiply by direction and eps to get the actual gradient
                eps = group['eps']
                grad = self.projected_grad * direction[idx] * eps
                
                # Set the gradient for this parameter
                param.grad = grad
                
                idx += 1
        
        # Use the inner optimizers to apply the gradients
        for group_idx, _ in enumerate(self.param_groups):
            self._inner_optimizers[group_idx].step()
            
        # Clear gradients after update
        for _, param in self.named_parameters_to_optim:
            param.grad = None
