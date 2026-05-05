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
            k: int = 10,
            variance: float = 1.0,
            lr_mu: Optional[float] = None,
            use_grad_first: bool = False,
            evaluate_memory: bool = False,
            candidate_selection_strategy: str = "best",
            candidate_average_count: int = 3,
    ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            weight_decay=weight_decay,
            tensor_sampling_type=tensor_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
        )
        self.k = max(1, k)
        self.variance = variance
        self.lr_mu = lr_mu if lr_mu is not None else lr
        self.use_grad_first = use_grad_first
        self.evaluate_memory = evaluate_memory
        self.candidate_selection_strategy = candidate_selection_strategy
        self.candidate_average_count = max(1, candidate_average_count)
        if self.candidate_selection_strategy not in {"best", "random", "mean"}:
            raise ValueError(
                "candidate_selection_strategy must be one of: best, random, mean"
            )
        
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['step'] = 0

                if self.use_grad_first:
                    if param.grad is None:
                        raise ValueError("param.grad is None, but use_grad_first is True")
                    state['mu'] = param.grad.detach().clone()
                else:
                    state['mu'] = torch.zeros_like(
                        param,
                        memory_format=torch.preserve_format
                    )
                # state['mu'] = torch.randn_like(
                #     param, 
                #     memory_format=torch.preserve_format
                # )
                # state['mu'] /= torch.linalg.norm(state['mu'])
                if not self.evaluate_memory:
                    state['mu_old'] = state['mu'].detach().clone()
                    state['mu_old_norm'] = torch.norm(state['mu_old']).item()**2

        
    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("ZO_RL_SGD requires a closure")
        if self.variance <= 0:
            raise ValueError("ZO_RL_SGD requires variance > 0")

        candidate_seeds = []
        candidate_losses = []

        if not self.evaluate_memory:
            for group in self.param_groups:
                for param in group['params']:
                    state = self.state[param]
                    state['mu_old'] = state['mu'].detach().clone()
                    state['mu_old_norm'] = torch.norm(state['mu_old']).item()**2

        for _ in range(self.k):
            self.zo_random_seed = np.random.randint(1_000_000_000)
            candidate_seeds.append(self.zo_random_seed)
            self.generator.manual_seed(self.zo_random_seed)
            self._mu_pertrub(scaling_factor=1)
            loss = closure()
            candidate_losses.append(self._loss_to_float(loss))
            self.generator.manual_seed(self.zo_random_seed)
            self._mu_pertrub(scaling_factor=-1)

        selected_indices = self._select_candidate_indices(candidate_losses)
        selected_seeds = [candidate_seeds[idx] for idx in selected_indices]
        self.selected_zo_random_seeds = selected_seeds
        self.zo_random_seed = selected_seeds[0]

        selected_projected_grads = []
        selected_loss_plus_values = []
        selected_loss_refs = []

        for seed in selected_seeds:
            self.generator.manual_seed(seed)
            self._mu_pertrub(scaling_factor=1)
            loss1 = closure()
            loss1_value = self._loss_to_float(loss1)

            self.generator.manual_seed(seed)
            self._mu_pertrub(scaling_factor=-2)
            loss2 = closure()
            loss2_value = self._loss_to_float(loss2)

            selected_projected_grads.append((loss1_value - loss2_value) / 2.0)
            selected_loss_plus_values.append(loss1_value)
            selected_loss_refs.append(loss1)

            self.generator.manual_seed(seed)
            self._mu_pertrub(scaling_factor=1)

        self.projected_grad = sum(selected_projected_grads) / len(selected_projected_grads)

        f_tensor = torch.tensor(candidate_losses, dtype=torch.float32)
        f_sum = torch.sum(f_tensor)
        # Handle k=1 case to avoid division by zero
        if self.k > 1:
            coeff = (f_tensor * self.k - f_sum) / (self.k - 1)
        else:
            coeff = torch.zeros_like(f_tensor)  # When k=1, mu update term is zero

        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            
            for param in group['params']:
                state = self.state[param]
                state['step'] += 1

                grad = torch.zeros_like(param, memory_format=torch.preserve_format)
                for projected_grad, seed in zip(selected_projected_grads, selected_seeds):
                    self.generator.manual_seed(seed)
                    z = self._sample_mu_direction(param)
                    grad.add_(z, alpha=projected_grad / (eps * len(selected_seeds)))
                if weight_decay is not None and weight_decay != 0:
                    grad = grad.add(param.data, alpha=weight_decay)
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

        mu_diffs = {}
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                mu_diffs[p] = torch.zeros_like(state['mu'])

        for coeff_value, seed in zip(coeff.tolist(), candidate_seeds):
            self.generator.manual_seed(seed)
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    mu = state['mu']
                    z = self._sample_mu_direction(p)
                    mu_diffs[p].add_(mu - z, alpha=coeff_value)

        mu_norm_diff = None
        mu_grad_norm = None
        track_mu_stats = not self.evaluate_memory
        for group in self.param_groups:
            for p in group['params']:    
                state = self.state[p]
                g_mu = -mu_diffs[p] / (self.k * (self.variance ** 2))
                state['mu'].add_(g_mu, alpha=-self.lr_mu)
                if track_mu_stats:
                    if mu_norm_diff is None:
                        mu_norm_diff = torch.linalg.norm(state['mu'] - state['mu_old'])**2 
                    else:
                        mu_norm_diff += torch.linalg.norm(state['mu'] - state['mu_old'])**2 
                    if mu_grad_norm is None:
                        mu_grad_norm = torch.linalg.norm(g_mu)**2 
                    else:
                        mu_grad_norm += torch.linalg.norm(g_mu)**2 
        mu_norms = []
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                if 'mu' in state:
                    mu_norm = torch.norm(state['mu']).item()
                    mu_norms.append(mu_norm)
        
        # Get step from first parameter
        step = None
        try:
            first_param = list(self.param_groups[0]['params'])[0]
            step = self.state[first_param].get('step', None)
        except (KeyError, IndexError, AttributeError):
            pass
        
        if mu_norms:
            avg_mu_norm = sum(mu_norms) / len(mu_norms)
            if wandb.run is not None:
                wandb.log({"avg_mu_norm": avg_mu_norm})
            # Log to file if enabled
            try:
                from trainer import _optimizer_log_func
                if _optimizer_log_func is not None and step is not None:
                    _optimizer_log_func({"avg_mu_norm": avg_mu_norm}, step=step)
            except (ImportError, AttributeError):
                pass
        self.generator.manual_seed(self.zo_random_seed)
        
        if track_mu_stats:
            avg_mu_norm_diff = torch.sqrt(mu_norm_diff) / len(mu_norms)
            avg_mu_grad_norm = torch.sqrt(mu_grad_norm) / len(mu_norms)
            # Convert tensors to Python numbers
            avg_mu_norm_diff_val = avg_mu_norm_diff.item() if torch.is_tensor(avg_mu_norm_diff) else float(avg_mu_norm_diff)
            avg_mu_grad_norm_val = avg_mu_grad_norm.item() if torch.is_tensor(avg_mu_grad_norm) else float(avg_mu_grad_norm)
            if wandb.run is not None:
                wandb.log({"avg_mu_norm_diff": avg_mu_norm_diff_val, "avg_mu_grad_norm": avg_mu_grad_norm_val})
            # Log to file if enabled
            try:
                from trainer import _optimizer_log_func
                if _optimizer_log_func is not None and step is not None:
                    _optimizer_log_func({"avg_mu_norm_diff": avg_mu_norm_diff_val, "avg_mu_grad_norm": avg_mu_grad_norm_val}, step=step)
            except (ImportError, AttributeError):
                pass       

        return self._average_selected_losses(selected_loss_refs, selected_loss_plus_values)

    @staticmethod
    def _loss_to_float(loss) -> float:
        if torch.is_tensor(loss):
            return loss.detach().float().item()
        return float(loss)

    def _select_candidate_indices(self, candidate_losses):
        if not candidate_losses:
            raise ValueError("candidate_losses must not be empty")

        if self.candidate_selection_strategy == "best":
            return [min(range(len(candidate_losses)), key=candidate_losses.__getitem__)]

        if self.candidate_selection_strategy == "random":
            random_idx = int(np.random.randint(len(candidate_losses)))
            return [random_idx]

        selected_count = min(self.candidate_average_count, len(candidate_losses))
        ranked_indices = sorted(
            range(len(candidate_losses)),
            key=candidate_losses.__getitem__,
        )
        return ranked_indices[:selected_count]

    def _average_selected_losses(self, losses, loss_values):
        average_loss = sum(loss_values) / len(loss_values)
        first_loss = losses[0]
        if torch.is_tensor(first_loss):
            return first_loss.detach().new_tensor(average_loss)
        return average_loss

    def _sample_mu_direction(self, param):
        state = self.state[param]
        tensor_sampling_type = state['tensor_sampling_type']
        noise = self.tensor_sampler.sample(
            param.shape,
            generator=self.generator,
            sampler_type=tensor_sampling_type,
        )
        return state['mu'] + self.variance * noise.to(device=param.device, dtype=param.dtype)
    
    def _mu_pertrub(self, scaling_factor: float = 1.0):
        for group in self.param_groups:
            eps = group['eps']
            for param in group['params']:
                # Use the generator directly without resetting seed per parameter
                # This ensures all parameters use the same random sequence
                # self.generator.manual_seed(self.zo_random_seed)' 
                state = self.state[param]
                # if 'seed' not in state:
                #     state['seed'] = np.random.randint(1_000_000_000)
                # seed = state['seed'] 
                # self.generator.manual_seed(seed)
                z = self._sample_mu_direction(param)
                param.data.add_(z * eps * scaling_factor)
