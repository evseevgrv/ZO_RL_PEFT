from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Callable, Dict, Any, Union, List, Iterable, Tuple

from .opt_utils import *

class ZO_MUON(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            matrix_sampling_type: str = "Random_baseline",
            vector_sampling_type: str = "standard_normal",
            perturbation_mode: str = "two_side"
        ):
        super().__init__(
            params=params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            vector_sampling_type=vector_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            gradient_sparsity=gradient_sparsity,
        )
        self.lr = lr 
        self.perturbation_mode = perturbation_mode

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
            self.projected_grad = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="one_side")
        else:  
            self.matrix_perturb_parameters(scaling_factor=-2)
            self.generator.manual_seed(self.zo_random_seed)
            if closure is not None:
                loss2 = closure()
            self.projected_grad = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="two_side")
            self.matrix_perturb_parameters(scaling_factor=1)
            self.generator.manual_seed(self.zo_random_seed)
        
        self.generator.manual_seed(self.zo_random_seed)
        for group_idx, group in enumerate(self.param_groups):
            for param in group['params']:
                device = param.device

                if param.ndim >= 2:
                    z = self.matrix_sampler.sample_single_matrix(param_shape=param.shape, generator=self.generator)
                else:
                    z = self.vector_sampler.sample(param.shape, generator=self.generator).to(device)
                
                self.generator.manual_seed(self.zo_random_seed)

                grad_update = self.projected_grad * z

                if param.ndim >= 2:
                    grad_update = zeropower_via_newtonschulz5(grad_update, steps=5)
                else:
                    grad_update = torch.sign(grad_update)

                grad_update_final = grad_update.to(device)
                
                param.data.add_(grad_update_final, alpha=-self.lr) 

        return loss1
