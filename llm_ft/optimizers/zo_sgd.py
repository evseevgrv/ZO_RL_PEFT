from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
from gradient_pruning import fast_random_mask_like
from .opt_utils import *
from collections import defaultdict

class ZO_SGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = None,
            weight_decay: float = 0.0,
            tensor_sampling_type: str = "standard_normal",
            matrix_sampling_type: str = None, 
            perturbation_mode: str = "two_side",
    ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            tensor_sampling_type=tensor_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
        )
        
    @torch.no_grad()
    def step(self, closure=None):
        loss1, loss2 = None, None 
        
        self.zo_random_seed = np.random.randint(1_000_000_000)
        self.generator.manual_seed(self.zo_random_seed)
        
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = closure()
        self.generator.manual_seed(self.zo_random_seed)

        if self.perturbation_mode == "one_side":
            self.zo_perturb_parameters(scaling_factor=-1)
            self.generator.manual_seed(self.zo_random_seed)
            loss2 = closure()
            self.projected_grad = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="one_side")
        else:
            self.zo_perturb_parameters(scaling_factor=-2)
            loss2 = closure()
            self.projected_grad = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="two_side")
            self.generator.manual_seed(self.zo_random_seed)
            self.zo_perturb_parameters(scaling_factor=1)
            self.generator.manual_seed(self.zo_random_seed)
            
        self._apply_gradients()
        self.generator.manual_seed(self.zo_random_seed)
        return loss1 
    
    @torch.no_grad()
    def _apply_gradients(self) -> None:
        self.generator.manual_seed(self.zo_random_seed)
        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            eps = group['eps']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            

            for param in group['params']:
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                tensor_sampling_type = state["tensor_sampling_type"]
                device = param.device
                z = self.tensor_sampler.sample(param.shape, generator=self.generator, sampler_type=tensor_sampling_type).to(device)
                grad = (z * self.projected_grad) / eps        

                grad.add_(param, alpha=weight_decay) # decay

                # Apply momentum if applicable
                if momentum is not None and momentum != 0:
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(grad)
                    update = buf
                else:
                    update = grad

                param.data.add_(update, alpha=-lr)
