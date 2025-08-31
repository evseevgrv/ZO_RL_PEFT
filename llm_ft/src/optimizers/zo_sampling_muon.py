from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Callable, Dict, Any, Union, List, Iterable, Tuple
from .opt_utils import *
from gradient_pruning import fast_random_mask_like

class ZO_SamplingMUON(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            matrix_sampling_type: str = "Householder_reflection",
            vector_sampling_type: str = "standard_normal",
            perturbation_mode: str = "two_side"
        ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            vector_sampling_type=vector_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss1, loss2 = None, None 
        self._prepare_parameters()  

        self.zo_random_seed = np.random.randint(1_000_000_000) 
        self.generator.manual_seed(self.zo_random_seed)

        self.matrix_perturb_parameters(scaling_factor=1)
        self.generator.manual_seed(self.zo_random_seed)
        if closure is not None:
            loss1 = closure()

        if self.perturbation_mode == "one_side":
            self.matrix_perturb_parameters(scaling_factor=-1)
            self.generator.manual_seed(self.zo_random_seed)
            if closure is not None:
                loss2 = closure()
            projected_grad = torch.sign(loss1 - loss2).item()
        else:  
            self.matrix_perturb_parameters(scaling_factor=-2)
            self.generator.manual_seed(self.zo_random_seed)
            if closure is not None:
                loss2 = closure()
            projected_grad = torch.sign(loss1 - loss2).item()
            self.matrix_perturb_parameters(scaling_factor=1)
            self.generator.manual_seed(self.zo_random_seed)
        
        self.generator.manual_seed(self.zo_random_seed)
        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            eps = group['eps']
            for param in group['params']:
                device = param.device

                if param.ndim >= 2:
                    z = self.matrix_sampler.sample_single_matrix(param_shape=param.shape, generator=self.generator)
                else:
                    z = self.vector_sampler.sample(param.shape, generator=self.generator).to(device)

                grad_final = projected_grad * z / eps

                param.data.add_(grad_final.to(device), alpha=-lr) 
        return loss1
