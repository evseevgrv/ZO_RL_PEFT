from .base import ZeroOrderOptimizer
import torch
from torch.optim import SGD
import numpy as np
from typing import Optional, Callable, Dict, Any, Union, List, Iterable, Tuple
from gradient_pruning import fast_random_mask_like
from .opt_utils import *

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
        """
        Zero-Order Stochastic Gradient Descent optimizer (MeZO implementation).
        
        Args:
            params: Parameters to optimize (can specify per-group hyperparameters)
            lr: Optional base learning rate (must be specified in groups if None)
            eps: Optional base perturbation (must be specified in groups if None)
            momentum: Momentum factor (default: 0.0)
            gradient_sparsity: Gradient sparsity (float or dict per parameter)
            perturbation_mode: 'one_side' or 'two_side' (default: 'two_side')
            q: Number of random directions (default: 1)
            module_wise_perturbation: Whether to perturb modules separately
            coordinate_perturbation: Whether to update immediately after perturbation
        """
        super().__init__(
            params=params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            gradient_sparsity=gradient_sparsity,
            vector_sampling_type=vector_sampling_type
        )
        self.perturbation_mode = perturbation_mode
        self.q = q
        self.module_wise_perturbation = module_wise_perturbation
        self.coordinate_perturbation = coordinate_perturbation

        # inner optimizer for each param_gropus
        self._inner_optimizers = []
        for group in self.param_groups:
            self._inner_optimizers.append(
                SGD(group['params'], lr=group['lr'], momentum=group['momentum'])
            )
        # TODO: Maybe we need a random seed in the input
        self.projected_grad: Optional[float] = None
        self.zo_random_seed: Optional[int] = None

    @torch.no_grad()
    def step(self, closure):
        """ 
        Performs a single optimization step.

        Args:
            closure: Callable that returns the loss and recomputes gradients.
        Returns:
            Loss tensor or None
        """
        # NOTE: In the original code, the zo_step_v2 isn't used anywhere
        # args = self.args
        if self.module_wise_perturbation:
            assert self.q == 1, "module-wise perturbation only supports q=1"
            if self.coordinate_perturbation:
                return self.zo_step_with_module_wise_perturbation_coordinate(closure)
            return self.zo_step_with_module_wise_perturbation(closure)
        # elif self.q == 1:
        elif self.q >= 1:
            return self.zo_step(closure)
        # elif self.q > 1:
        #     return self.zo_step_v1(closure)
        else:
            raise ValueError(f"q={self.q} is not supported.")
    
    @torch.no_grad()
    def zo_step_with_module_wise_perturbation_coordinate(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """
        Update the parameters right after perturbing the parameters.
        Module-wise perturbation with immediate updates.
        """

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)
        # torch.manual_seed(self.zo_random_seed)
        self.generator.manual_seed(self.zo_random_seed)

        losses = []

        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.

        for _, module_params in self._get_module_parameters():
            self.named_parameters_to_optim = module_params
            loss = self._module_perturbation_step(closure)
            # torch.manual_seed(self.zo_random_seed)
            self.generator.manual_seed(self.zo_random_seed)
            losses.append(loss)
            self._apply_gradients()  # therefore the changes are here
            # torch.manual_seed(self.zo_random_seed)
            self.generator.manual_seed(self.zo_random_seed)
        
        return torch.stack(losses).mean()

    @torch.no_grad()
    def zo_step_with_module_wise_perturbation(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Update all parameters once after perturbing all the parameters."""
    
        losses = []
        grad_dict = {}

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)
        # torch.manual_seed(self.zo_random_seed)
        self.generator.manual_seed(self.zo_random_seed)


        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.

        for module_name, module_params in self._get_module_parameters():
            self.named_parameters_to_optim = module_params
            loss = self._module_perturbation_step(closure)
            # torch.manual_seed(self.zo_random_seed)
            self.generator.manual_seed(self.zo_random_seed)
            losses.append(loss)
            grad_dict[module_name] = self.projected_grad
        
        for module_name, module_params in self._get_module_parameters():
            self.named_parameters_to_optim = module_params
            self.projected_grad = grad_dict[module_name]
            self._apply_gradients()
            # torch.manual_seed(self.zo_random_seed)
            self.generator.manual_seed(self.zo_random_seed)

        return torch.stack(losses).mean()

    
    @torch.no_grad()
    def zo_step(self, closure):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        Multi-direction gradient estimation (q > 1).
        """

        self._prepare_parameters()
        sum_projected_grads = 0
        random_seeds = []
        
        for i_q in range(self.q):  # TODO: shall we change the seed?
            # Sample the random seed for sampling z

            seed = np.random.randint(1000000000) # FIXME: is it ok?
            random_seeds.append(seed)
            # set the torch state
            # torch.manual_seed(seed)
            self.generator.manual_seed(seed)

            # First function evaluation
            self.zo_perturb_parameters(scaling_factor=1, random_seed=seed)
            loss1 = closure()
            # torch.manual_seed(seed)  # return to the same state
            self.generator.manual_seed(seed)

            # Second function evaluation
            if self.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1, random_seed=seed)
                # torch.manual_seed(seed)  # return to the same state
                self.generator.manual_seed(seed)
                loss2 = closure()
                grad = self.grad_approx(loss_original=loss1, loss_perturbed=loss2, perturbation_mode="one_side")
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2, random_seed=seed)
                loss2 = closure()
                grad = self.grad_approx(loss_original=loss1, loss_perturbed=loss2, perturbation_mode="two_side")
                # torch.manual_seed(seed)  # return to the same state
                self.generator.manual_seed(seed)

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1, random_seed=seed)
                # torch.manual_seed(seed)  # return to the same state
                self.generator.manual_seed(seed)
            
            sum_projected_grads = sum_projected_grads + grad

            # we alwayas return loss1, now need to return loss1 from the first itersstion
            if i_q == 0:
                first_loss = loss1
            
        self.projected_grad = sum_projected_grads / self.q
        self._apply_gradients(random_seeds=random_seeds)
        return first_loss
    
    def _apply_gradients(self, random_seeds: Optional[List[int]] = None) -> None:
        """
        Applies gradients using per-group hyperparameters.
        
        Args:
            random_seeds: List of seeds for perturbation vectors (q > 1)
        """
        if random_seeds is None:
            random_seeds = [self.zo_random_seed]
            
        for group_idx, group in enumerate(self.param_groups):
            for param in group['params']:
                if not any(name for name, p in self.named_parameters_to_optim if p is param):
                    continue
                
                grad = torch.zeros_like(param)
                eps = group['eps']
                
                for seed in random_seeds:
                    # torch.manual_seed(seed)
                    self.generator.manual_seed(seed)
                    # self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
                    
                    z = self.vector_sampler.sample(param.shape, generator=self.generator)
                    # print("Update vector:\n", z)
                    name = next(name for name, p in self.named_parameters_to_optim if p is param)
                    sparsity = self.get_grad_sparsity_by_name(name)
                    if sparsity is not None:
                        z[fast_random_mask_like(z, sparsity, generator=self.sparse_grad_rng)] = 0
                        self.generator.manual_seed(seed)

                    grad += (self.projected_grad * z * eps) / len(random_seeds)
                
                param.grad = grad
                self._inner_optimizers[group_idx].step()
                param.grad = None
    
    def _module_perturbation_step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Performs single module perturbation step."""
        self.zo_random_seed = np.random.randint(1000000000)
        # torch.manual_seed(self.zo_random_seed)
        self.generator.manual_seed(self.zo_random_seed)
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = closure()
        # torch.manual_seed(self.zo_random_seed)
        self.generator.manual_seed(self.zo_random_seed)
        
        if self.perturbation_mode == "one_side":
            self.zo_perturb_parameters(scaling_factor=-1)
            self.generator.manual_seed(self.zo_random_seed)
            loss2 = closure()
            self.projected_grad = self.grad_approx(loss_original=loss1, loss_perturbed=loss2, perturbation_mode="one_side")
        else:  # two_side
            self.zo_perturb_parameters(scaling_factor=-2)
            # torch.manual_seed(self.zo_random_seed)
            self.generator.manual_seed(self.zo_random_seed)
            loss2 = closure()
            self.projected_grad = self.grad_approx(loss_original=loss1, loss_perturbed=loss2, perturbation_mode="two_side")
            self.zo_perturb_parameters(scaling_factor=1)
            # torch.manual_seed(self.zo_random_seed)
            self.generator.manual_seed(self.zo_random_seed)
        
        return loss1

    def _get_module_parameters(self) -> List[Tuple[str, List[Tuple[str, torch.Tensor]]]]:
        """
        Groups parameters by module (simplified implementation).
        
        Returns:
            List of (module_name, parameters) tuples
        """
        # NOTE: maybe need to find another replacement for modules
        return [("all", [(name, p) for name, p in self.named_parameters_all if p.requires_grad])]
