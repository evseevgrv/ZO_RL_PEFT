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
            tensor_sampling_type: str = "standard_normal",
            matrix_sampling_type: str = None, 
            perturbation_mode: str = "two_side"
        ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            tensor_sampling_type=tensor_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss1, loss2 = None, None 

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
            projected_grad = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="one_side")
        else:  
            self.matrix_perturb_parameters(scaling_factor=-2)
            self.generator.manual_seed(self.zo_random_seed)
            if closure is not None:
                loss2 = closure()
            projected_grad = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="two_side")
            self.matrix_perturb_parameters(scaling_factor=1)
            self.generator.manual_seed(self.zo_random_seed)
        
        self.generator.manual_seed(self.zo_random_seed)
        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            eps = group['eps']
            tensor_sampling_type = group['tensor_sampling_type']
            for p in group['params']:
                device = p.device
                state = self.state[p]
                tensor_sampling_type = state["tensor_sampling_type"]

                z = self.tensor_sampler.sample(p.shape, generator=self.generator, sampler_type=tensor_sampling_type).to(device)
                self.generator.manual_seed(self.zo_random_seed)

                grad_update = projected_grad * z / eps 

                if p.ndim >= 2:
                    grad_final = zeropower_via_newtonschulz5(grad_update, steps=5)
                else:
                    grad_final = torch.sign(grad_update)

                p.data.add_(grad_final.to(device), alpha=-lr) 
        return loss1
