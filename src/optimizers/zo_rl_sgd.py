from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Callable, Dict, Any, Union, List, Iterable, Tuple
from gradient_pruning import fast_random_mask_like
from .opt_utils import *

class ZORL_SGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            vector_sampling_type: str = "lp_sphere",
            perturbation_mode: str = "two_side",
            q: int = 1,
            module_wise_perturbation: bool = False,
            coordinate_perturbation: bool = False,
            rl_directions: int = 5,
            rl_temperature: float = 1.0
    ):
        """
        Zero-Order RL-guided Stochastic Gradient Descent optimizer.
        Uses RL-inspired approach to select directions that minimize f(x+tau*e).
        
        Args:
            params: Parameters to optimize (can specify per-group hyperparameters)
            lr: Optional base learning rate (must be specified in groups if None)
            eps: Optional base perturbation (must be specified in groups if None)
            momentum: Momentum factor (default: 0.0)
            gradient_sparsity: Gradient sparsity (float or dict per parameter)
            vector_sampling_type: 'standard_normal' or 'lp_sphere' (default: 'standard_normal')
            perturbation_mode: 'one_side' or 'two_side' (default: 'two_side')
            q: Number of gradient estimates to average (default: 1)
            module_wise_perturbation: Whether to perturb modules separately
            coordinate_perturbation: Whether to update immediately after perturbation
            rl_directions: Number of candidate directions to evaluate for RL selection (default: 5)
            rl_temperature: Temperature for softmax selection (higher = more exploration)
        """
        super().__init__(
            params=params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            gradient_sparsity=gradient_sparsity,
            vector_sampling_type=vector_sampling_type
        )
        self.perturbation_mode = perturbation_mode
        self.q = q
        self.module_wise_perturbation = module_wise_perturbation
        self.coordinate_perturbation = coordinate_perturbation
        self.rl_directions = rl_directions
        self.rl_temperature = rl_temperature
        
        # Inner optimizers for each parameter group
        self._inner_optimizers = []
        for group in self.param_groups:
            self._inner_optimizers.append(
                SGD(group['params'], lr=group['lr'], momentum=group['momentum'])
            )
        
        self.projected_grad: Optional[float] = None
        self.zo_random_seed: Optional[int] = None
    
    @torch.no_grad()
    def step(self, closure):
        """ 
        Performs a single optimization step with RL-guided direction selection.
        Args:
            closure: Callable that returns the loss and recomputes gradients.
        Returns:
            Loss tensor
        """
        if self.module_wise_perturbation:
            assert self.q == 1, "module-wise perturbation only supports q=1"
            if self.coordinate_perturbation:
                return self.zo_step_with_module_wise_perturbation_coordinate(closure)
            return self.zo_step_with_module_wise_perturbation(closure)
        elif self.q >= 1:
            return self.zorl_step(closure)
        else:
            raise ValueError(f"q={self.q} is not supported.")
    
    @torch.no_grad()
    def zorl_step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """
        RL-guided Zero-Order step: selects direction that minimizes f(x+tau*e).
        For each gradient estimate, evaluates multiple candidate directions and
        selects the one that gives the best improvement.
        """
        self._prepare_parameters()
        sum_projected_grads = 0
        direction_list = []
        
        for i_q in range(self.q):
            # Store original parameters to reset after candidate evaluation
            original_params = self._get_flat_params()
            
            # Evaluate multiple candidate directions
            candidate_directions = []
            candidate_losses = []
            
            # Generate candidate directions directly from the unit sphere
            for _ in range(self.rl_directions):
                # Sample a direction vector from the unit sphere
                direction = self._sample_direction()
                candidate_directions.append(direction)
                
                # Perturb in the candidate direction
                self._apply_direction(direction, scaling_factor=1)
                
                # Evaluate loss after perturbation
                loss_val = closure()
                candidate_losses.append(loss_val.item())
                
                # Reset to original parameters
                self._set_flat_params(original_params)
            
            # Select the best direction (with minimum loss)
            best_idx = np.argmin(candidate_losses)
            best_direction = candidate_directions[best_idx]
            direction_list.append(best_direction)
            
            # Use the best direction for gradient estimation
            self._apply_direction(best_direction, scaling_factor=1)
            loss1 = closure()
            
            # Second function evaluation for gradient estimation
            if self.perturbation_mode == "one_side":
                self._apply_direction(best_direction, scaling_factor=-1)
                loss2 = closure()
                grad = self.grad_approx(loss_original=loss1, loss_perturbed=loss2, 
                                      perturbation_mode="one_side")
            else:  # two-side perturbation
                self._apply_direction(best_direction, scaling_factor=-2)
                loss2 = closure()
                grad = self.grad_approx(loss_original=loss1, loss_perturbed=loss2, 
                                      perturbation_mode="two_side")
                # Reset back to parameters at start of step
                self._apply_direction(best_direction, scaling_factor=1)
            
            sum_projected_grads += grad
            
            # Record loss from first iteration
            if i_q == 0:
                first_loss = loss1
            
            # Clean up - reset to original parameters
            self._set_flat_params(original_params)
        
        self.projected_grad = sum_projected_grads / self.q
        self._apply_gradients(directions=direction_list)
        
        return first_loss
    
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
                    # Sample a direction vector from the unit sphere
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
            eps = group['eps']
            for p in group['params']:
                if p.requires_grad:
                    # Apply the direction with scaling factor and eps
                    p.data.add_(scaling_factor * eps * direction[idx])
                    idx += 1
    
    def _apply_gradients(self, directions: Optional[List[List[torch.Tensor]]] = None) -> None:
        """
        Applies gradients using per-group hyperparameters.
        Args:
            directions: List of direction vectors for perturbation (q > 1)
        """
        if directions is None:
            # Fallback to standard ZO_SGD behavior if no directions provided
            super()._apply_gradients()
            return
            
        for group_idx, group in enumerate(self.param_groups):
            eps = group['eps']
            for param_idx, param in enumerate(group['params']):
                if not param.requires_grad:
                    continue
                
                # Average the gradient contributions from all directions
                grad = torch.zeros_like(param)
                for direction in directions:
                    # The direction index corresponds to the parameter index
                    direction_idx = 0
                    for g in self.param_groups:
                        for p in g['params']:
                            if p.requires_grad:
                                if p is param:
                                    break
                                direction_idx += 1
                    
                    # Calculate the gradient contribution from this direction
                    grad += (self.projected_grad * direction[direction_idx] * eps) / len(directions)
                
                param.grad = grad
                self._inner_optimizers[group_idx].step()
                param.grad = None
