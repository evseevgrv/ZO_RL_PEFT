import torch

class VectorSampler:
    def __init__(self, sampler_type, p=2.0, device=None):
        """
        Initialize a vector sampler.
        
        Args:
            sampler_type: The type of sampling to use ("standard_normal" or "lp_sphere")
            device: The device to place tensors on (default: None, uses current default device)
            p (float of 'inf'): The p-norm value for "lp_sphere" sampler (default: 2.0)
        """
        self.p = p
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if sampler_type == "standard_normal":
            self._sample_func = self._standard_normal
        elif sampler_type == "lp_sphere":
            self._sample_func = self._sample_lp_sphere
        else:
            raise NotImplementedError(f"Sampling {sampler_type} is not implemented")
    
    def sample(self, param_shape, generator=None):
        return  self._sample_func(param_shape, generator)

    def _standard_normal(self, param_shape, generator=None):
        return torch.normal(mean=0, std=1, size=param_shape, device=self.device, generator=generator)
    
    def _sample_lp_sphere(self, param_shape, generator=None):
        return self._lp_uniform_sphere(param_shape=param_shape, p=self.p, generator=generator)
    
    def _lp_uniform_sphere(self, param_shape, p=2.0, device=None, generator=None):
        if p == 'inf':
            # For L_infinity norm, sample from {-1, 1}^d uniformly
            return torch.randint(0, 2, param_shape, device=device, generator=generator) * 2 - 1

        if p == 2.0:
            # For L2 norm, we can use the standard Gaussian method
            x = torch.randn(param_shape, device=device, generator=generator)
            norm = torch.norm(x, p=2, dim=-1, keepdim=True)
            return x / norm

        elif p == 1.0:
            # For L1 norm, we can use the Dirichlet distribution
            exp_samples = torch.empty(param_shape, device=device).exponential_(generator=generator)
            l1_norm = torch.sum(exp_samples, dim=-1, keepdim=True)
            samples = exp_samples / l1_norm
            signs = torch.randint(0, 2, param_shape, device=device, generator=generator) * 2 - 1
            return samples * signs

        else:
            # General case for any p-norm
            gamma_shape = 1.0 / p
            exp_samples = torch.empty(param_shape, device=device).exponential_(generator=generator)
            gamma_samples = exp_samples.pow(gamma_shape)
            p_norm = torch.norm(gamma_samples, p=p, dim=-1, keepdim=True)
            samples = gamma_samples / p_norm
            signs = torch.randint(0, 2, param_shape, device=device, generator=generator) * 2 - 1
            return samples * signs

