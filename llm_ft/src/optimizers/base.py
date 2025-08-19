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
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            vector_sampling_type: str = "standard_normal",
            matrix_sampling_type: str = None,
            perturbation_mode: str = "two_side",
            device: str = "cuda", # FIXME: maybe change it
    ):
        """
        Base class for zero-order optimizers.

        Args:
            params: Model parameters to optimize:
                - Iterable[Tensor] (all parameters)
                - Iterable[Dict] (parameter gruops with different hyperparameters)
            lr: Learning rate, if None, then it has to be in parameter groups
            eps: Perturbation magnitude, if None, then it has to be in parameter groups
            momentum: Momentum factor, zero by default
            gradient_sparsity: Gradient sparsity (float for global or dict per parameter)
        """
        if lr is not None or eps is not None:
            defaults = {
                'lr': lr,
                'eps': eps,
                'momentum': momentum,
            }
        else:
            defaults = {'momentum': momentum}

        super().__init__(params, defaults)

        self._validate_hyperparameters()
        self.gradient_sparsity = gradient_sparsity

        self.state = defaultdict(dict)

        self.generator = torch.Generator(device=device)

        self.vector_sampler = VectorSampler(vector_sampling_type, device=device)
        if matrix_sampling_type is not None:
            self.matrix_sampler = MatrixSampler(matrix_sampling_type, device=device)
        self.perturbation_mode = perturbation_mode

        self.named_parameters_all = []
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group['params']):
                self.device = param.device
                param_name = f"group_{group_idx}.param_{param_idx}"
                self.named_parameters_all.append((param_name, param))
    
        self.zo_eps = self._calculate_zo_eps(eps=eps)

        self._inner_optimizers = None
        self._lr_schedulers = None

    def _prepare_parameters(self) -> None:
        """Prepares parameters for optimization. Common for all optimizer's steps"""
        self.named_parameters_to_optim = [
            (name, param) for name, param in self.named_parameters_all 
            if param.requires_grad
        ]
        for _, param in self.named_parameters_to_optim:
            param.grad = None

    def _calculate_zo_eps(self, eps: Optional[float] = None):
        """"Estimates zo_eps for accurate grad approx as a weighted sum of all epsilons"""
        total_params = 0
        eps_sum = 0.0
        
        for group in self.param_groups:
            group_eps = group['eps']
            if group_eps is not None:
                group_params = sum(p.numel() for p in group['params'] if p.requires_grad)
                eps_sum += group_eps * group_params
                total_params += group_params
        
        return eps_sum / total_params if total_params > 0 else (eps if eps is not None else 1e-3)

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
            for p in group['params']:
                z = self.vector_sampler.sample(p.shape, generator=self.generator)
                # print(self.zo_eps)
                perturb = z * self.zo_eps
                perturb = perturb.to(p.device)
                p.data.add_(scaling_factor * perturb)

    def grad_approx(
        self,
        loss_plus: torch.Tensor,
        loss_minus: torch.Tensor,
        perturbation_mode: str = "two_side"
    ) -> float:
        if perturbation_mode == "one_side":
            return ((loss_plus - loss_minus) / self.zo_eps).item()
        elif perturbation_mode == "two_side":
            return ((loss_plus - loss_minus) / (2 * self.zo_eps)).item()
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
        for name, param in self.named_parameters_to_optim:
            indices = self._select_indices(param_shape=param.shape, device=param.device)
            if isinstance(indices, torch.Tensor):
                param.data[indices] += scaling_factor * self.zo_eps
            else:
                rows, cols = indices
                param.data[rows[:, None], cols] += scaling_factor * self.zo_eps

    def matrix_perturb_parameters(
        self, 
        scaling_factor: float = 1.0,
    ) -> None:
        for group in self.param_groups:
            for p in group['params']:
                if len(p.shape) == 1:
                    z = self.vector_sampler.sample(p.shape, generator=self.generator)
                else:
                    z = self.matrix_sampler.sample_single_matrix(p.shape, generator=self.generator)

                perturb = z *self.zo_eps
                perturb = perturb.to(p.device)
                p.data.add_(scaling_factor * perturb)

