from torch.optim import Optimizer
from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Dict, Any, Tuple, Union, Iterable
import torch
import numpy as np
from .opt_utils import *
from gradient_pruning import fast_random_mask_like
from torch.optim import SGD
from collections import defaultdict

class ZeroOrderOptimizer(Optimizer, ABC):
    def __init__(self,
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = 0.0,
            weight_decay: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            tensor_sampling_type: str = "standard_normal",
            matrix_sampling_type: str = None, 
            perturbation_mode: str = "two_side",
            device: str = "cuda",
    ):
        if lr is not None or eps is not None: 
            defaults = {
                'lr': lr,
                'eps': eps,
                'momentum': momentum,
                'weight_decay': weight_decay
            }
        else:
            defaults = {'momentum': momentum, 'weight_decay': weight_decay}

        super().__init__(params, defaults)

        self._validate_hyperparameters()
        self.gradient_sparsity = gradient_sparsity

        self.state = defaultdict(dict)

        self.generator = torch.Generator(device=device)

        self.tensor_sampler = TensorSampler(tensor_sampling_type, device=device)

        self.perturbation_mode = perturbation_mode

        for group in self.param_groups:
            for p in group['params']:
                if p.ndim == 1:
                    self.state[p]['tensor_sampling_type'] = tensor_sampling_type
                elif p.ndim >= 2:
                     self.state[p]['tensor_sampling_type'] = matrix_sampling_type if matrix_sampling_type is not None else tensor_sampling_type


    def _validate_hyperparameters(self):
        """Obligatory hyperparameters check"""
        required = ['lr', 'eps']
        for group in self.param_groups:
            for key in required:
                if key not in group:
                    raise ValueError(f"Missing required hyperparameter: {key}")
    
    @abstractmethod
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        pass
    
    def get_grad_sparsity_by_name(self, name: str) -> Optional[float]:
        if self.gradient_sparsity is None:
            return None
        elif isinstance(self.gradient_sparsity, float):
            return self.gradient_sparsity
        elif isinstance(self.gradient_sparsity, dict):
            return self.gradient_sparsity[name]
        
    def zo_perturb_parameters(
        self, 
        scaling_factor: float = 1.0,
    ) -> None:
        for group in self.param_groups:
            eps = group["eps"]
            tensor_sampling_type = group["tensor_sampling_type"]
            for p in group['params']:
                z = self.tensor_sampler.sample(p.shape, generator=self.generator, sampler_type=tensor_sampling_type)
                perturb = z * eps
                p.data.add_(scaling_factor * perturb.to(p.device))

    def grad_approx(
        self,
        loss_plus: torch.Tensor,
        loss_minus: torch.Tensor,
        perturbation_mode: str = "two_side"
    ) -> float:
        if perturbation_mode == "one_side":
            return ((loss_plus - loss_minus)).item()
        elif perturbation_mode == "two_side":
            return ((loss_plus - loss_minus) / 2).item()
        else:
            raise ValueError(f"Unknown perturbation mode: {perturbation_mode}")
                    
    def _select_indices(self, param_shape, rows_ratio = 0.1, cols_ratio = 0.1, device='cuda'):
        if len(param_shape) == 1:
            n_elems = param_shape[0]
            k = max(1, int(n_elems * rows_ratio))
            indices = torch.randperm(n_elems, device=device, generator=self.generator)[:k]
            return indices
        n_rows, n_cols = param_shape
        k = max(1, int(n_rows * rows_ratio))
        m = max(1, int(n_cols * cols_ratio))

        selected_rows = torch.randperm(n_rows, device=device, generator=self.generator)[:k]
        selected_cols = torch.randperm(n_cols, device=device, generator=self.generator)[:m]
        return (selected_rows, selected_cols)
    
    def _indices_perturb(self, scaling_factor = 1.0):
        for group in self.param_groups:
            eps = group["eps"]
            for p in group['params']:
                indices = self._select_indices(param_shape=p.shape, device=p.device)
                if isinstance(indices, torch.Tensor):
                    p.data[indices] += scaling_factor * eps
                else:
                    rows, cols = indices
                    p.data[rows[:, None], cols] += scaling_factor * eps

    def matrix_perturb_parameters(
        self, 
        scaling_factor: float = 1.0,
    ) -> None:
        for group in self.param_groups:
            eps = group["eps"]
            tensor_sampling_type = group["tensor_sampling_type"]
            for p in group['params']:
                z = self.tensor_sampler.sample(p.shape, generator=self.generator, sampler_type=tensor_sampling_type)

                perturb = z * eps
                p.data.add_(scaling_factor * perturb.to(p.device))
