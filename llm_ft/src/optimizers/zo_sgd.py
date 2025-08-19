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
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            vector_sampling_type: str = "standard_normal",
            perturbation_mode: str = "two_side",
            q: int = 1,
            module_wise_perturbation: bool = False,
            coordinate_perturbation: bool = False
    ):
        defaults = dict(
            lr=lr,
            eps=eps,
            momentum=momentum,
            vector_sampling_type=vector_sampling_type,
            gradient_sparsity=gradient_sparsity
        )
        super().__init__(params, defaults)
        
        self.state = defaultdict(dict)
        self.perturbation_mode = perturbation_mode
        self.lr = lr 
        self.module_wise_perturbation = module_wise_perturbation
        self.coordinate_perturbation = coordinate_perturbation
        self.projected_grad = None
        self.zo_random_seed = None

    @torch.no_grad()
    def step(self, closure=None):
        loss1, loss2 = None, None 
        self._prepare_parameters()
        
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
        for group_idx, group in enumerate(self.param_groups):
            for param in group['params']:
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                
                device = param.device
                z = self.vector_sampler.sample(param.shape, generator=self.generator).to(device)
                grad = (z * self.projected_grad * self.zo_eps)
                
                param.data.add_(grad, alpha=-self.lr)

    def _get_module_parameters(self):
        return [("all", [(name, p) for name, p in self.named_parameters_all if p.requires_grad])]
