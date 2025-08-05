from .base import ZeroOrderOptimizer
import torch
from torch.optim import SGD
import numpy as np
from typing import Optional, Callable, Dict, Any, Union, List, Iterable, Tuple

from .opt_utils import *

class Jaguar_SignSGD(ZeroOrderOptimizer):
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
            q: int = 1,
            module_wise_perturbation: bool = False,
            coordinate_perturbation: bool = False
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
        self.perturbation_mode = perturbation_mode
        self.q = q
        self.module_wise_perturbation = module_wise_perturbation
        self.coordinate_perturbation = coordinate_perturbation

        # inner optimizer for each param_gropus
        self._inner_optimizers = []
        for group in self.param_groups:
            # print(f"GROUP: {group.keys()}")
            # print(f"LR: {group['lr']}")
            # print(f"EPS: {group['eps']}")
            self._inner_optimizers.append(
                SGD(group['params'], lr=group['lr'], momentum=group['momentum'])
            )
        # TODO: Maybe we need a random seed in the input
        self.projected_grad: Optional[float] = None
        self.zo_random_seed: Optional[int] = None

    def step(self, closure):
        tau = self.tau
        beta = self.beta 
        use_smoothing = self.use_smoothing

        self._prepare_parameters()  
                
        self.zo_random_seed = np.random.randint(1_000_000_000)
        torch.manual_seed(self.zo_random_seed)

        selected_indices = {}
        original_values = {}

        for name, param in self.named_parameters_to_optim:
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
        with torch.no_grad():
            for name, param in self.named_parameters_to_optim:
                if len(param.data.shape) == 1:
                    indices = selected_indices[name]
                    param.data[indices] += tau
                else:
                    selected_rows, selected_cols = selected_indices[name]
                    param.data[selected_rows[:, None], selected_cols] += tau
        loss1 = closure()

        with torch.no_grad():
            for name, param in self.named_parameters_to_optim:
                if len(param.data.shape) == 1:
                    indices = selected_indices[name]
                    param.data[indices] = original_values[name] - tau
                else:
                    selected_rows, selected_cols = selected_indices[name]
                    param.data[selected_rows[:, None], selected_cols] = original_values[name] - tau
        loss2 = closure()

        with torch.no_grad():
            for name, param in self.named_parameters_to_optim:
                if len(param.data.shape) == 1:
                    indices = selected_indices[name]
                    param.data[indices] = original_values[name]
                else:
                    selected_rows, selected_cols = selected_indices[name]
                    param.data[selected_rows[:, None], selected_cols] = original_values[name]

        # rho = sign(f(z_+) - f(z_-))
        rho = (loss1 - loss2).item() / (2 * tau)

        for group_idx, group in enumerate(self.param_groups):
            for param in group['params']:
                name = next(name for name, p in self.named_parameters_to_optim if p is param)
                if param.grad is None:
                    param.grad = torch.zeros_like(param)

                grad_update = rho
                if len(param.data.shape) == 1:
                    indices = selected_indices[name]
                    if use_smoothing:
                        param.grad[indices] = beta * param.grad[indices] + (1 - beta) * grad_update
                    else:
                        param.grad[indices] = grad_update
                else:
                    selected_rows, selected_cols = selected_indices[name]
                    if use_smoothing:
                        param.grad[selected_rows[:, None], selected_cols] = beta * param.grad[selected_rows[:, None], selected_cols] + (1 - beta) * grad_update
                    else:
                        param.grad[selected_rows[:, None], selected_cols] = grad_update

                param.grad = torch.sign(param.grad)

                self._inner_optimizers[group_idx].step()
                param.grad = None
        for _, param in self.named_parameters_to_optim:
            param.grad = None

        return loss1
