from .base import ZeroOrderOptimizer
import torch
from torch.optim import SGD
import numpy as np
from typing import Optional, Callable, Dict, Any, Union, List, Iterable, Tuple
from .opt_utils import *

class ZO_SamplingMUON(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            tau: Optional[float] = None,
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            matrix_sampling_type: str = "Householder_reflection",
            vector_sampling_type: str = "standard_normal",
            device: str = "cuda", # FIXME: maybe change it
        ):
        super().__init__(
            params=params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            gradient_sparsity=gradient_sparsity,
            vector_sampling_type=vector_sampling_type,
            device=device
        )
        self.tau = tau 
        self.eps = eps
        self._inner_optimizers = []
        for group in self.param_groups:
            self._inner_optimizers.append(
                SGD(group['params'], lr=group['lr'], momentum=group['momentum'])
            )
        self.matrix_sampler = MatrixSampler(matrix_sampling_type, device=device)

    @torch.no_grad()
    def step(self, closure):
        tau = self.tau
        self.zo_eps = self._calculate_zo_eps(eps=self.eps)

        self._prepare_parameters()  

        seed = np.random.randint(1_000_000_000)
        torch.manual_seed(seed)
        if hasattr(self, 'sparse_grad_rng'):
            self.sparse_grad_rng.manual_seed(seed)
        
        shapes = [(name, tuple(param.shape)) for name, param in self.named_parameters_to_optim if param.ndim >= 2 and param.size(0) < 10000]
        E_dict = self.matrix_sampler.sample(shapes)

        # Perturb
        # FIXME: what to do with this one?
        def zo_muon_perturb_parameters(scaling_factor=1):
            for name, param in self.named_parameters_to_optim:
                # if name in E_dict:
                if param.ndim >= 2 and param.size(0) < 10000:
                    E = E_dict[name][0]
                    param.data.add_(tau * scaling_factor * E)
                else:
                    z = torch.randn_like(param)
                    mask = getattr(self, 'get_grad_sparsity_by_name', lambda x: None)(name)
                    if mask is not None:
                        z[fast_random_mask_like(z, mask, generator=self.sparse_grad_rng)] = 0
                    param.data.add_(tau * scaling_factor * z)

        # f(z_+)
        zo_muon_perturb_parameters(1)
        loss1 = closure()
        # f(z_-)
        zo_muon_perturb_parameters(-2)
        loss2 = closure()
        zo_muon_perturb_parameters(1)

        delta = (loss1 - loss2) / (2 * self.zo_eps)
        # rho = sign(f(z_+) - f(z_-))
        rho = torch.sign(loss1 - loss2)

        for group_idx, group in enumerate(self.param_groups):
            for param in group['params']:
                name = next(name for name, p in self.named_parameters_to_optim if p is param)
                # if name in E_dict:
                if param.ndim >= 2 and param.size(0) < 10000:
                    E, U, S, V = E_dict[name]
                    grad = rho * (U @ V.T)
                else:
                    z = torch.randn_like(param)
                    mask = getattr(self, 'get_grad_sparsity_by_name', lambda x: None)(name)
                    if mask is not None:
                        z[fast_random_mask_like(z, mask, generator=self.sparse_grad_rng)] = 0
                    # grad = (rho if args.trainer=='zo_signsgd' else delta) * z
                    grad = rho * z
                param.grad = grad.to(param.dtype)

                self._inner_optimizers[group_idx].step()
                param.grad = None

        for _, param in self.named_parameters_to_optim:
                param.grad = None
        return loss1
