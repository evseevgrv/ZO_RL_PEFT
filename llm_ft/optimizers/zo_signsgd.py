from .zo_sgd import ZO_SGD
import torch
from torch.optim import SGD
import numpy as np
from typing import Optional, Callable, Dict, Any, Union, List, Iterable, Tuple
from .opt_utils import *

class ZO_SignSGD(ZO_SGD):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = None,
            weight_decay: float = 0.0,
            vector_sampling_type: str = "standard_normal",
            perturbation_mode: str = "two_side",
    ):
        """
        Zero-Order Stochastic Gradient Descent optimizer (MeZO implementation).
        
        Args:
            params: Parameters to optimize (can specify per-group hyperparameters)
            lr: Optional base learning rate (must be specified in groups if None)
            eps: Optional base perturbation (must be specified in groups if None)
            momentum: Momentum factor (default: 0.0)
            gradient_sparsity: Gradient sparsity (float or dict per parameter)
            perturbation_mode: 'one_side' or 'two_side' (default: 'two_side')
            q: Number of random directions (default: 1)
            module_wise_perturbation: Whether to perturb modules separately
            coordinate_perturbation: Whether to update immediately after perturbation
        """
        # self._inner_optimizer = SGD(params, lr=args.learning_rate, momentum=args.momentum)
        super().__init__(
            params = params, 
            lr = lr, 
            eps = eps,
            momentum = momentum,
            perturbation_mode = perturbation_mode,
            )
        
    def grad_approx(
        self,
        loss_plus: torch.Tensor,
        loss_minus: torch.Tensor,
        perturbation_mode: str = "two_side"
    ) -> float:
        if perturbation_mode == "one_side":
            return torch.sign(loss_plus - loss_minus)
        elif perturbation_mode == "two_side":
            return  torch.sign((loss_plus - loss_minus) / 2)
        else:
            raise ValueError(f"Unknown perturbation mode: {perturbation_mode}")
