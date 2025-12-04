from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
from gradient_pruning import fast_random_mask_like
from .opt_utils import *
from collections import defaultdict
import wandb

class ZO_RL_SGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = None,
            weight_decay: float = 0.0,
            tensor_sampling_type: str = "standard_normal",
            matrix_sampling_type: str = None, 
            perturbation_mode: str = "two_side",
            k: Optional[int] = 10,
            variance: Optional[float] = 1e-3,
            lr_mu: Optional[float] = None,
            use_grad_first: bool = False,
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
        self.k = k
        self.variance = variance
        self.lr_mu = lr_mu if lr_mu is not None else lr
        self.use_grad_first = use_grad_first
        for group in self.param_groups:
            for param in group['params']:    
                state = self.state[param]
                if 'step' not in state:
                    state['step'] = 0
                    if param.requires_grad and self.use_grad_first:
                        if param.grad is None:
                            raise ValueError("param.grad is None, but use_grad_first is True")
                            
                        state['mu'] = param.grad.clone()
                    else:
                        # state['mu'] = torch.randn_like( # N(0,1)
                        #     param, 
                        #     memory_format=torch.preserve_format
                        # )
                        # state['mu'] /= torch.linalg.norm(state['mu'])
                        state['mu'] = torch.zeros_like(
                            param, 
                            memory_format=torch.preserve_format
                        )

    @torch.no_grad()
    def step(self, closure=None):
        loss1, loss2 = None, None 
        
        for group in self.param_groups:
            for param in group['params']:    
                state = self.state[param]
                state['step'] += 1
        
        e_values = {}
       
        for idx in range(self.k):
            self.zo_random_seed = np.random.randint(1_000_000_000)
            self.generator.manual_seed(self.zo_random_seed)
            self.zo_perturb_parameters(scaling_factor=1.0)
            
            if closure is not None:
                loss = closure()
                e_values[self.zo_random_seed] = loss
            
            self.generator.manual_seed(self.zo_random_seed)
            self.zo_perturb_parameters(scaling_factor=-1.0)

        optimal_seed = min(e_values, key=e_values.get)
        loss1 = e_values[optimal_seed]
        self.zo_random_seed = optimal_seed
        self.generator.manual_seed(self.zo_random_seed)

        self.zo_perturb_parameters(scaling_factor=-1.0)
        if closure is not None:
            loss2 = closure()
        self.generator.manual_seed(self.zo_random_seed)
        
        self.zo_perturb_parameters(scaling_factor=1.0)
        self.generator.manual_seed(self.zo_random_seed)
        
        self.projected_grad = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="two_side")            
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
                self.generator.manual_seed(self.zo_random_seed)
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

         # Calculate and log average norm of mu across all parameters
        mu_norms = []
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                if 'mu' in state:
                    mu_norm = torch.norm(state['mu']).item()
                    mu_norms.append(mu_norm)
        
        if mu_norms:
            avg_mu_norm = sum(mu_norms) / len(mu_norms)
            wandb.log({"avg_mu_norm": avg_mu_norm})
