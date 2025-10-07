import torch

class TensorSampler:
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
        self.create_sampler(sampler_type)

    def create_sampler(self, sampler_type):
        self.sampler_type = sampler_type
        if sampler_type == "standard_normal":
            self._sample_func = self._standard_normal
        elif sampler_type == "lp_sphere":
            self._sample_func = self._sample_lp_sphere
        elif self.sampler_type == 'GS':                      
            self.sampler = self._GS_matrix
        elif self.sampler_type == 'GS_v2':                 
            self.sampler = self._GS_matrix_v2
        elif self.sampler_type == 'Householder_reflection':
            self.sampler = self._householder_matrix
        elif self.sampler_type == 'Rotation':               
            self.sampler = self._rotation_matrix
        elif self.sampler_type == 'Reflection':            
            self.sampler = self._reflection_matrix
        elif self.sampler_type == 'Random_baseline':       
            self.sampler = self._random_baseline
        elif self.sampler_type == 'Torch_QR':       
            self.sampler = self._torch_qr
        else:
            raise NotImplementedError(f"Sampling {sampler_type} is not implemented")
    
    def sample(self, param_shape, generator=None, sampler_type=None):
        if sampler_type is not None:
            self.create_sampler(sampler_type)

        if self.sampler_type in ["standard_normal", "lp_sphere"]:
            return self._sample_func(param_shape, generator)

        assert len(param_shape) > 1, f"Sample only matrices, current shape: {param_shape}"
        n, m = param_shape
        if self.sampler_type == 'Torch_QR':
            return self.sampler(n, m, generator=generator)
        elif n > m:
            return self.sampler(n, generator=generator)[:, :m]
        return self.sampler(m, generator=generator)[:n, :]

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

    # Matrix methods 
    def _householder_matrix(self, d, generator = None):
        
        u = torch.randn(d, device=self.device, generator=generator)
        H = torch.eye(d, device=self.device) - 2*(u*u.unsqueeze(1))/(u.norm()**2)

        return H

    def _rotation_matrix(self, d, num_rotations=None, generator=None):
        return self._rotation_matrix_sequential(d, num_rotations, generator)
    
    def _rotation_matrix_sequential(self, d, num_rotations=None, generator=None):
        if num_rotations is None:
            num_rotations = d
        
        if generator is None:
            generator = torch.Generator(device=self.device)
        
        Q = torch.eye(d, device=self.device, dtype=torch.float32)
        
        pairs = torch.randint(0, d, (num_rotations, 2), device=self.device, generator=generator)
        mask = pairs[:, 0] != pairs[:, 1]
        valid_pairs = pairs[mask]
        
        while len(valid_pairs) < num_rotations:
            new_pairs = torch.randint(0, d, (num_rotations, 2), device=self.device, generator=generator)
            mask = new_pairs[:, 0] != new_pairs[:, 1]
            valid_pairs = torch.cat([valid_pairs, new_pairs[mask]])
        
        valid_pairs = valid_pairs[:num_rotations]
        
        thetas = torch.rand(num_rotations, device=self.device, generator=generator) * 2 * math.pi
        cos_thetas = torch.cos(thetas)
        sin_thetas = torch.sin(thetas)
        
        for idx in range(num_rotations):
            i = valid_pairs[idx, 0]
            j = valid_pairs[idx, 1]
            c = cos_thetas[idx]
            s = sin_thetas[idx]
            
            col_i = Q[:, i].clone()
            col_j = Q[:, j].clone()
            Q[:, i].mul_(c).add_(col_j, alpha=-s)
            Q[:, j].mul_(c).add_(col_i, alpha=s)
        
        return Q
    
    def _reflection_matrix(self, d, generator=None):
        if generator is None:
            generator = torch.Generator(device=self.device)
        
        num_reflections = torch.randint(0, d, (1,), device=self.device, generator=generator).item()
        
        if num_reflections == 0:
            return torch.eye(d, device=self.device, dtype=torch.float32)
        
        diag = torch.ones(d, device=self.device, dtype=torch.float32)
        indices = torch.randperm(d, device=self.device, generator=generator)[:num_reflections]
        diag[indices] = -1
        
        Q = torch.diag(diag)
        return Q


    def _random_baseline(self, d, generator = None):
        return torch.randn((d,d), device=self.device, generator=generator)

    def _GS_matrix(
        self,
        dim,
        num_blocks=10,
        use_PL=True,
        use_P=True,
        use_PR=True,
        generator=None,
    ):
        if generator is None:
            generator = torch.Generator(device=self.device)
        
        base = dim // num_blocks
        rem = dim % num_blocks
        blocks = [base + (1 if i < rem else 0) for i in range(num_blocks)]
        
        L = torch.zeros(dim, dim, device=self.device, dtype=torch.float32)
        R = torch.zeros(dim, dim, device=self.device, dtype=torch.float32)
        
        offset = 0
        for b in blocks:
            X_L = torch.randn(b, b, device=self.device, generator=generator)
            Q_L, R_L = torch.linalg.qr(X_L)
            sign_L = torch.sign(torch.diagonal(R_L, 0))
            Q_L = Q_L * sign_L.unsqueeze(0)
            if torch.det(Q_L) < 0:
                Q_L[:, 0] *= -1
            
            X_R = torch.randn(b, b, device=self.device, generator=generator)
            Q_R, R_R = torch.linalg.qr(X_R)
            sign_R = torch.sign(torch.diagonal(R_R, 0))
            Q_R = Q_R * sign_R.unsqueeze(0)
            if torch.det(Q_R) < 0:
                Q_R[:, 0] *= -1
            
            L[offset:offset+b, offset:offset+b] = Q_L
            R[offset:offset+b, offset:offset+b] = Q_R
            offset += b
        
        identity = torch.arange(dim, device=self.device)
        idx_PR = torch.randperm(dim, device=self.device, generator=generator) if use_PR else identity
        idx_P = torch.randperm(dim, device=self.device, generator=generator) if use_P else identity
        idx_PL = torch.randperm(dim, device=self.device, generator=generator) if use_PL else identity
        
        A = R[idx_P, :][:, idx_PR]
        A = L @ A
        A = A[idx_PL, :]
        
        return A
    
    def _GS_matrix_v2(
        self,
        dim,
        num_blocks=None,
        use_PL=True,
        use_P=True,
        use_PR=True,
        generator=None
    ):
        if generator is None:
            generator = torch.Generator(device=self.device)
        
        if num_blocks is None:
            num_blocks = max(1, int(np.sqrt(dim)))
        
        base = dim // num_blocks
        rem = dim % num_blocks
        blocks = [base + (1 if i < rem else 0) for i in range(num_blocks)]
        
        L = torch.zeros(dim, dim, device=self.device, dtype=torch.float32)
        R = torch.zeros(dim, dim, device=self.device, dtype=torch.float32)
        
        offset = 0
        for b in blocks:
            X_L = torch.randn(b, b, device=self.device, generator=generator)
            Q_L, R_L = torch.linalg.qr(X_L)
            sign_L = torch.sign(torch.diagonal(R_L, 0))
            Q_L = Q_L * sign_L.unsqueeze(0)
            if torch.det(Q_L) < 0:
                Q_L[:, 0] *= -1
            
            X_R = torch.randn(b, b, device=self.device, generator=generator)
            Q_R, R_R = torch.linalg.qr(X_R)
            sign_R = torch.sign(torch.diagonal(R_R, 0))
            Q_R = Q_R * sign_R.unsqueeze(0)
            if torch.det(Q_R) < 0:
                Q_R[:, 0] *= -1
            
            L[offset:offset+b, offset:offset+b] = Q_L
            R[offset:offset+b, offset:offset+b] = Q_R
            offset += b
        
        identity = torch.arange(dim, device=self.device)
        idx_PR = torch.randperm(dim, device=self.device, generator=generator) if use_PR else identity
        idx_P = torch.randperm(dim, device=self.device, generator=generator) if use_P else identity
        idx_PL = torch.randperm(dim, device=self.device, generator=generator) if use_PL else identity
        
        A = R[idx_P, :][:, idx_PR]
        A = L @ A
        A = A[idx_PL, :]
        
        return A
    
    def _torch_qr(self, n, m, generator=None):
        return torch.nn.init.orthogonal_(torch.empty((n, m), device=self.device), generator=generator)
