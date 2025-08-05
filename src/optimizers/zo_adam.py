from .zo_sgd import ZO_SGD
import torch
from torch.optim import Adam
from typing import Optional, Callable, Dict, Any, Union, List, Iterable, Tuple
from .opt_utils import *

class ZO_Adam(ZO_SGD):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            perturbation_mode: str = "two_side",
            q: int = 1,
            module_wise_perturbation: bool = False,
            coordinate_perturbation: bool = False
    ):
        """
        Zero-Order Adam.
        
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
        # params = list(params)
        # self._inner_optimizer = Adam(params, lr=args.learning_rate)
        super().__init__(
            params = params, 
            lr = lr, 
            eps = eps,
            momentum = momentum,
            gradient_sparsity = gradient_sparsity,
            perturbation_mode = perturbation_mode,
            q = q,
            module_wise_perturbation = module_wise_perturbation,
            coordinate_perturbation = coordinate_perturbation
            )

        self._inner_optimizers = []
        for group in self.param_groups:
            self._inner_optimizers.append(
                Adam(group['params'], lr=group['lr'])
            )