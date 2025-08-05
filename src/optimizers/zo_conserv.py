from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Callable, List, Dict, Any, Union, Iterable
from gradient_pruning import fast_random_mask_like
from .opt_utils import *

class ZO_Conserv(ZeroOrderOptimizer):
    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        momentum: float = 0.0,
        gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
        perturbation_mode: str = "two_side"
    ):
        """
        Zero-Order Conservative Optimizer (MeZO with conservative update).
        
        Args:
            params: Parameters to optimize
            lr: Learning rate (required)
            eps: Perturbation magnitude (required)
            momentum: Must be 0 (not supported)
            gradient_sparsity: Gradient sparsity (float or dict per parameter)
            perturbation_mode: 'one_side' or 'two_side' gradient estimation
        """
        super().__init__(
            params=params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            gradient_sparsity=gradient_sparsity
        )
        
        # Validate momentum=0
        for group in self.param_groups:
            if group['momentum'] != 0:
                raise ValueError("ZO_Conserv does not support momentum")
                
        self.perturbation_mode = perturbation_mode
        self.projected_grad: Optional[float] = None
        self.zo_random_seed: Optional[int] = None

    @torch.no_grad()
    def _apply_update(self, projected_grad: float, sign: float = 1.0) -> None:
        """Apply parameter update using re-generated perturbation vectors."""
        torch.manual_seed(self.zo_random_seed)
        self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
        
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                    
                param_id = id(p)
                name = next((name for name, param in self.named_parameters_all 
                            if id(param) == param_id), None)
                
                z = torch.randn_like(p)
                if name:
                    sparsity = self.get_grad_sparsity_by_name(name)
                    if sparsity is not None:
                        mask = fast_random_mask_like(
                            z, sparsity, generator=self.sparse_grad_rng
                        )
                        z[mask] = 0
                
                p.data.add_(sign * lr * projected_grad * z)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """
        Performs a single conservative optimization step.
        
        Args:
            closure: Callable that returns the loss tensor
        Returns:
            Loss value of the final selected parameters
        """
        self._prepare_parameters()
        
        loss0 = closure()
        
        self.zo_random_seed = np.random.randint(1000000000)
        
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = closure()
        
        if self.perturbation_mode == "one_side":
            self.zo_perturb_parameters(scaling_factor=-1)
            loss2 = closure()
            self.projected_grad = self.grad_approx(loss_original=loss1, loss_perturbed=loss2, perturbation_mode="one_side")
        else:  # two_side
            self.zo_perturb_parameters(scaling_factor=-2)
            loss2 = closure()
            self.projected_grad = self.grad_approx(loss_original=loss1, loss_perturbed=loss2, perturbation_mode="two_side")
            self.zo_perturb_parameters(scaling_factor=1)

        # Trial update 1: positive direction (theta + update)
        self._apply_update(self.projected_grad, sign=1.0)
        loss1_after = closure()
        
        # Trial update 2: negative direction (theta - update)
        self._apply_update(self.projected_grad, sign=-2.0)
        loss2_after = closure()
        
        # Select best parameter state
        if loss1_after > loss0 and loss0 < loss2_after:
            self._apply_update(self.projected_grad, sign=1.0)
            final_loss = loss0
        elif loss1_after <= loss0 and loss1_after < loss2_after:
            self._apply_update(self.projected_grad, sign=2.0)
            final_loss = loss1_after
        else:
            final_loss = loss2_after
            
        return final_loss