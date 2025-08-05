from .base import ZeroOrderOptimizer
import torch
# from torch.optim import SGD
import numpy as np
from typing import Optional, Callable, Dict, Any, Union, List, Iterable, Tuple

from .opt_utils import *

class Jaguar_MUON(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            tau: Optional[float] = None,
            beta: Optional[float] = None, 
            use_smoothing: Optional[bool] = None,
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            perturbation_mode: str = "two_side",
        ):
        super().__init__(
            params=params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            gradient_sparsity=gradient_sparsity
        )
        self.tau = tau 
        self.beta = beta
        self.use_smoothing = use_smoothing

        # self._inner_optimizers = []
        # for group in self.param_groups:
        #     self._inner_optimizers.append(
        #         SGD(group['params'], lr=group['lr'], momentum=group['momentum'])
        #     )
       
        # self._lr_schedulers = None
            
    @torch.no_grad()
    def step(self, closure=None):
        loss1, loss2 = None, None
        tau = self.tau
        beta = self.beta
        use_smoothing = self.use_smoothing

        self._prepare_parameters()   
                
        self.zo_random_seed = np.random.randint(1_000_000_000)
        torch.manual_seed(self.zo_random_seed)

        for group_idx, _ in enumerate(self.param_groups):
            self._inner_optimizers[group_idx].zero_grad()

        original_grads = {}
        for name, param in self.named_parameters_to_optim:
            original_grads[name] = param.grad.clone() if param.grad is not None else None


        selected_indices = {}
        original_values = {}

        for name, param in self.named_parameters_to_optim:
            if param not in self._inner_optimizers[group_idx].state:
                self._inner_optimizers[group_idx].state[param] = {}
                self._inner_optimizers[group_idx].state[param]['grad_accum'] = torch.zeros_like(param.data)
            if len(param.data.shape) == 1:
                n_elements = param.data.shape[0]
                k = max(1, n_elements // 10)
                indices = torch.randperm(n_elements, device=param.device)[:k]
                selected_indices[name] = indices
                original_values[name] = param.data[indices].clone()
            else:
                n_rows, n_cols = param.data.shape
                k = max(1, n_rows // 10)
                m = max(1, n_cols // 10)
                selected_rows = torch.randperm(n_rows, device=param.device)[:k]
                selected_cols = torch.randperm(n_cols, device=param.device)[:m]
                selected_indices[name] = (selected_rows, selected_cols)
                original_values[name] = param.data[selected_rows[:, None], selected_cols].clone()

        for name, param in self.named_parameters_to_optim:
            if len(param.data.shape) == 1:
                indices = selected_indices[name]
                param.data[indices] += tau
            else:
                selected_rows, selected_cols = selected_indices[name]
                param.data[selected_rows[:, None], selected_cols] += tau
        if closure is not None:
            with torch.enable_grad():
                loss1 = closure()

        for name, param in self.named_parameters_to_optim:
            if len(param.data.shape) == 1:
                indices = selected_indices[name]
                param.data[indices] = original_values[name] - tau
            else:
                selected_rows, selected_cols = selected_indices[name]
                param.data[selected_rows[:, None], selected_cols] = original_values[name] - tau
        if closure is not None:
            with torch.enable_grad():
                loss2 = closure()

        for name, param in self.named_parameters_to_optim:
            if len(param.data.shape) == 1:
                indices = selected_indices[name]
                param.data[indices] = original_values[name]
            else:
                selected_rows, selected_cols = selected_indices[name]
                param.data[selected_rows[:, None], selected_cols] = original_values[name]

        grad_update = (loss1 - loss2).item() / (2 * tau) 

        for group_idx, group in enumerate(self.param_groups):
            for param in group['params']:
                name = next(name for name, p in self.named_parameters_to_optim if p is param)
                # if param.grad is None:
                #     param.grad = torch.zeros_like(param)
                    
                # grad_update = rho
                state = self._inner_optimizers[group_idx].state[param]
                if len(param.data.shape) == 1:
                    indices = selected_indices[name]
                    if use_smoothing:
                        state['grad_accum'][indices] = beta * state['grad_accum'][indices] + (1 - beta) * grad_update
                    else:
                        state['grad_accum'][indices] = grad_update
                    accum = state['grad_accum']
                    accum[indices] = torch.sign(accum[indices])
                else:
                    selected_rows, selected_cols = selected_indices[name]
                    if use_smoothing:
                        state['grad_accum'][selected_rows[:, None], selected_cols] = beta * state['grad_accum'][selected_rows[:, None], selected_cols] + (1 - beta) * grad_update
                    else:
                        state['grad_accum'][selected_rows[:, None], selected_cols] = grad_update

                    accum = zeropower_via_newtonschulz5(state['grad_accum'], steps=5).to(param.data.dtype)

                param.grad = accum.clone()

                self._inner_optimizers[group_idx].step()
                # param.grad = None

        for name, param in self.named_parameters_to_optim:
            param.grad = original_grads[name]

        for group_idx, _ in enumerate(self.param_groups):
            self._lr_schedulers[group_idx].step()

        return loss1
