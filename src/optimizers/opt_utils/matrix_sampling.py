import torch
import math
import numpy as np
import random
import scipy.stats as sps
from collections import defaultdict

class MatrixSampler:
    
    def __init__(self, sampler_type, device='cuda'):
        
        self.sampler_type = sampler_type
        self.device = device

    def sample(self, param_shapes):

        if self.sampler_type == 'GS':                       # + +
            self.sampler = self._GS_matrix
        elif self.sampler_type == 'GS_v2':                  # + +
            self.sampler = self._GS_matrix_v2
        elif self.sampler_type == 'Householder_reflection': # + +
            self.sampler = self._householder_matrix
        elif self.sampler_type == 'Rotation':               # + + Reflection \subset Rotation
            self.sampler = self._rotation_matrix
        elif self.sampler_type == 'Reflection':             # + +
            self.sampler = self._reflection_matrix
        elif self.sampler_type == 'Random_baseline':        # + +
            self.sampler = self._random_baseline
        else:
            raise NotImplementedError(f"Sampling {self.sampler_type} is not implemented")

        shape_to_names = defaultdict(list)
        for name, shape in param_shapes:
            shape_to_names[shape].append(name)

        E_dict = {}
        for (n, m), names in shape_to_names.items():
            
            k = min(n, m)
            S = self.Sigma(n, m)
            U = self.sampler(n)
            V = self.sampler(m)
            S_k = S[:k, :k]
            U_k = U[:, :k]
            V_k = V[:, :k]
            E_k = U_k @ S_k @ V_k.T
            # E = U @ S @ V.T
            for name in names:
                E_dict[name] = (E_k.clone(), U_k.clone(), S_k.clone(), V_k.clone())
                # E_dict[name] = (E.clone(), U.clone(), S.clone(), V.clone())
        return E_dict

    def Sigma(self, n, m, dtype=torch.float32):

        p = min(n, m)

        sigma = torch.zeros((n, m), device=self.device, dtype=dtype)
        sigma[torch.arange(p), torch.arange(p)] = torch.diag(self._rotation_matrix(p))

        return sigma

    def _householder_matrix(self, d):
        
        u = torch.randn(d, device=self.device)
        H = torch.eye(d, device=self.device) - 2*(u*u.unsqueeze(1))/(u.norm()**2)

        return H


    def _rotation_matrix(self, d, num_rotations=None):
        if num_rotations is None:
            num_rotations = d
        Q = torch.eye(d, device=self.device)
        for _ in range(num_rotations):
            i, j = torch.randint(0, d, (2,), device=self.device)
            while i == j:
                j = torch.randint(0, d, (1,), device=self.device)
                j = j.item()
            theta = torch.rand(1, device=self.device) * 2 * math.pi
            c = torch.cos(theta)
            s = torch.sin(theta)
            col_i = Q[:, i].clone()
            col_j = Q[:, j].clone()
            Q[:, i] = c * col_i - s * col_j
            Q[:, j] = s * col_i + c * col_j
        return Q


    def _reflection_matrix(self, d):
        Q = torch.eye(d, device=self.device)
        idx = torch.randint(0, d - 1, (torch.randint(0,d-1, (1,)), ))
        Q[idx, idx] = -1
        return Q


    def _random_baseline(self, d):
        return torch.randn((d,d), device=self.device)


# def create_random_matrix(n, m, device='cpu'):
#     """Создает случайную матрицу в стиле SVD: U @ diag(S) @ V.T"""
#     U = generate_rotation_matrix(n, device=device)
#     V = generate_rotation_matrix(m, device=device)
    
#     S = generate_semi_diagonal(n, m, device=device)
    
#     return U @ S @ V.T, U, V, S

# def torch_ortho_rvs(dim: int, device="cpu", dtype=torch.float32):
#     """
#     Возвращает ортогональную матрицу из O(N), равномерно распределённую
#     по мере Хаара (аналог scipy.stats.ortho_group.rvs(dim)).

#     Parameters:
#         dim (int): размерность матрицы
#         device (str or torch.device): 'cpu' или 'cuda'
#         dtype (torch.dtype): тип данных (например, float32 или float64)

#     Returns:
#         torch.Tensor: ортогональная матрица размерности (dim, dim)
#     """
#     z = torch.randn(dim, dim, device=device, dtype=dtype)

#     q, r = torch.linalg.qr(z)

#     d = torch.diagonal(r, 0)
#     ph = d.sign()
#     q *= ph.unsqueeze(0)

#     return q

    def _GS_matrix(
        self,
        dim,
        num_blocks: int = 10,
        use_PL: bool = True,
        use_P:  bool = True,
        use_PR: bool = True,
    ):
        # num_blocks is a variable
        # base = dim*num_blocks = sqrt(min(n,m)) => num_blocks = dim // sqrt(min(n,m))
        base = dim // num_blocks
        rem  = dim %  num_blocks
        blocks = [base + (1 if i < rem else 0) for i in range(num_blocks)]
        block_sizes_L = blocks
        block_sizes_R = blocks
    
    
        if block_sizes_L:
            bsL = block_sizes_L
            maxL = max(bsL)
            L_blocks = []
            for b in bsL:
                X = torch.randn(b, b, device=self.device)
                Qb, Rb = torch.linalg.qr(X)
                sign = torch.sign(torch.diagonal(Rb, 0))
                Qb *= sign
                if torch.det(Qb) < 0:
                    Qb[:, 0] *= -1
                L_blocks.append(Qb)
            L = torch.block_diag(*L_blocks).to(self.device)
        else:
            L = torch.eye(dim, device=self.device)
    
        if block_sizes_R:
            bsR = block_sizes_R
            R_blocks = []
            for b in bsR:
                X = torch.randn(b, b, device=self.device)
                Qb, Rb = torch.linalg.qr(X)
                sign = torch.sign(torch.diagonal(Rb, 0))
                Qb *= sign
                if torch.det(Qb) < 0:
                    Qb[:, 0] *= -1
                R_blocks.append(Qb)
            R = torch.block_diag(*R_blocks).to(self.device)
        else:
            R = torch.eye(dim, device=self.device)
    
        idx_PL = torch.randperm(dim, device=self.device) if use_PL else torch.arange(dim, device=self.device)
        idx_P  = torch.randperm(dim, device=self.device) if use_P  else torch.arange(dim, device=self.device)
        idx_PR = torch.randperm(dim, device=self.device) if use_PR else torch.arange(dim, device=self.device)
    
        M1 = R[:, idx_PR]
        M2 = M1[idx_P, :]
        M3 = L @ M2
        A  = M3[idx_PL, :]
    
        return A
    
    def _GS_matrix_v2(
        self,
        dim,
        num_blocks: int = 10,
        use_PL: bool = True,
        use_P:  bool = True,
        use_PR: bool = True,
    ):
        # num_blocks is a variable
        # base*num_blocks = n
        # base = sqrt(n) => num_blocks = n / sqrt(n)
        num_blocks = int(dim // np.sqrt(dim))
        base = dim // num_blocks
        rem  = dim % num_blocks
        blocks = [base + (1 if i < rem else 0) for i in range(num_blocks)]
        block_sizes_L = blocks
        block_sizes_R = blocks
    
    
        if block_sizes_L:
            bsL = block_sizes_L
            maxL = max(bsL)
            L_blocks = []
            for b in bsL:
                X = torch.randn(b, b, device=self.device)
                Qb, Rb = torch.linalg.qr(X)
                sign = torch.sign(torch.diagonal(Rb, 0))
                Qb *= sign
                if torch.det(Qb) < 0:
                    Qb[:, 0] *= -1
                L_blocks.append(Qb)
            L = torch.block_diag(*L_blocks).to(self.device)
        else:
            L = torch.eye(dim, device=self.device)
    
        if block_sizes_R:
            bsR = block_sizes_R
            R_blocks = []
            for b in bsR:
                X = torch.randn(b, b, device=self.device)
                Qb, Rb = torch.linalg.qr(X)
                sign = torch.sign(torch.diagonal(Rb, 0))
                Qb *= sign
                if torch.det(Qb) < 0:
                    Qb[:, 0] *= -1
                R_blocks.append(Qb)
            R = torch.block_diag(*R_blocks).to(self.device)
        else:
            R = torch.eye(dim, device=self.device)
    
        idx_PL = torch.randperm(dim, device=self.device) if use_PL else torch.arange(dim, device=self.device)
        idx_P  = torch.randperm(dim, device=self.device) if use_P  else torch.arange(dim, device=self.device)
        idx_PR = torch.randperm(dim, device=self.device) if use_PR else torch.arange(dim, device=self.device)
    
        M1 = R[:, idx_PR]
        M2 = M1[idx_P, :]
        M3 = L @ M2
        A  = M3[idx_PL, :]
    
        return A

# def generate_reflection_matrix(d, device='cpu'):
#     Q = torch.eye(d, device=device)
#     i = torch.randint(0, d, (1,), device=device).item()
#     Q[i, i] = -1
#     return Q

# def generate_rotation_via_householders(d, device='cpu'):
#     v1 = torch.randn(d, device=device)
#     v1 /= v1.norm()

#     v2 = torch.randn(d, device=device)
#     v2 -= (v1 * v2).sum() * v1
#     v2 /= v2.norm()

#     I = torch.eye(d, device=device)
#     H1 = I - 2.0 * v1.unsqueeze(1) @ v1.unsqueeze(0)
#     H2 = I - 2.0 * v2.unsqueeze(1) @ v2.unsqueeze(0)
#     Q = H2 @ H1
#     return Q

# def generate_housholder_and_reflection_matrix(d, device='cpu'):
#     Q = generate_rotation_via_householders(d, device=device)
#     if torch.rand(1, device=device) < 0.5:
#         Q[0, :] = -Q[0, :]
#     return Q

######### FAST ONE ###########

# def create_random_matrix_fast_big_matrix(param_shapes, device='cpu'):
#     max_n = max(n for _, (n, m) in param_shapes)
#     max_m = max(m for _, (n, m) in param_shapes)

#     # U_big = generate_orthogonal_matrix(max_n, device=device)  107 sec / iter
#     # V_big = generate_orthogonal_matrix(max_m, device=device)


#     # U_big = generate_reflection_matrix(max_n, device=device)  0.1 sec / iter
#     # V_big = generate_reflection_matrix(max_m, device=device)

#     # U_big = torch.FloatTensor(ortho_group.rvs(max_n)).to(device)  8 sec / iter
#     # V_big = torch.FloatTensor(ortho_group.rvs(max_m)).to(device)


#     U_big = torch_ortho_rvs(max_n, device=device)  # 0.2 sec / iter ----> GREAT
#     V_big = torch_ortho_rvs(max_n, device=device)

#     E_dict = {}
#     for name, (n, m) in param_shapes:
#         k = min(n, m)
#         U_slice = U_big[:n, :k]      # n x k
#         V_slice = V_big[:m, :k]      # m x k
#         # E = U @ I_nm @ V.T
#         E = U_slice @ V_slice.T
#         E_dict[name] = (E, U_slice, V_slice)
#     return E_dict

# def create_random_matrix_fast_per_param(param_shapes, device='cpu'):

#     # U_big = generate_orthogonal_matrix(max_n, device=device)  107 sec / iter
#     # V_big = generate_orthogonal_matrix(max_m, device=device)


#     # U_big = generate_reflection_matrix(max_n, device=device)  0.1 sec / iter
#     # V_big = generate_reflection_matrix(max_m, device=device)

#     # U_big = torch.FloatTensor(ortho_group.rvs(max_n)).to(device)  8 sec / iter
#     # V_big = torch.FloatTensor(ortho_group.rvs(max_m)).to(device)


#     # U_big = torch_ortho_rvs(max_n, device=device)  # 0.2 sec / iter ----> GREAT
#     # V_big = torch_ortho_rvs(max_n, device=device)

#     E_dict = {}
#     for name, (n, m) in param_shapes:
#         k = min(n, m)

#         U = torch_ortho_rvs(n, device=device)
#         V = torch_ortho_rvs(m, device=device)

#         U_slice = U[:, :k]  # (n x k)
#         V_slice = V[:, :k]  # (m x k)

#         E = U_slice @ V_slice.T  # (n x m)
#         E_dict[name] = (E, U_slice, V_slice)

#     return E_dict

# def create_random_matrix_fast(param_shapes, device='cpu'):
#     shape_to_names = defaultdict(list)
#     for name, shape in param_shapes:
#         shape_to_names[shape].append(name)

#     E_dict = {}
#     for (n, m), names in shape_to_names.items():
#         k = min(n, m)
#         # U = torch_ortho_rvs(n, device=device)
#         # V = torch_ortho_rvs(m, device=device)
#         U = generate_micola_matrix(n, device=device)
#         V = generate_micola_matrix(m, device=device)
#         U_k = U[:, :k]
#         V_k = V[:, :k]
#         E = U_k @ V_k.T
#         for name in names:
#             E_dict[name] = (E.clone(), U_k.clone(), V_k.clone())
#     return E_dict

# def create_random_matrix_fast_v2(param_shapes, device='cpu'):
#     shape_to_names = defaultdict(list)
#     for name, shape in param_shapes:
#         shape_to_names[shape].append(name)

#     E_dict = {}
#     for (n, m), names in shape_to_names.items():
#         k = min(n, m)
#         # U = torch_ortho_rvs(n, device=device)
#         # V = torch_ortho_rvs(m, device=device)
#         if args.sampler == 'Micola':
#             U = generate_micola_matrix(n, device=device)
#             V = generate_micola_matrix(m, device=device)
#         elif args.sampler == 'Micola_v2':
#             U = generate_micola_matrix_v2(n, device=device)
#             V = generate_micola_matrix_v2(m, device=device)
#         elif args.sampler == 'Householder':
#             U = (n, device=device)
#             V = (m, device=device)
            
#         U_k = U[:, :k]
#         V_k = V[:, :k]
#         E = U_k @ V_k.T
#         for name in names:
#             E_dict[name] = (E.clone(), U_k.clone(), V_k.clone())
#     return E_dict


# ############################

# def generate_orthogonal_approx(G, device='cpu'):
#     assert len(G.shape) == 2, "Input must be a 2D tensor"
#     m, n = G.shape
    
#     # U_e m x m
#     U_e = generate_orthogonal_matrix(m, device=device)
    
#     #V_e n x n
#     V_e = generate_orthogonal_matrix(n, device=device)
    
#     # U_e * V_e^T --- СЕЙЧАС РАБОТАЕТ ТОЛЬКО В СЛУЧАЕ m == n
#     approx = U_e @ V_e.T
    
#     return approx

# # ТО ЖЕ САМОЕ ДЛЯ ПОЛУОРТОГОНАЛЬНЫХ

# def generate_semi_rotation_matrix(m, n, num_rotations=None, device='cpu'):
#     if num_rotations is None:
#         num_rotations = min(m, n)

#     k = min(m, n)
#     V = torch.zeros((m, n), device=device)
#     for i in range(min(m, k)):
#         V[i, i] = 1.0

#     for _ in range(num_rotations):
#         i, j = torch.randint(0, k, (2,), device=device)
#         while i == j:
#             j = torch.randint(0, k, (1,), device=device)
#             j = j.item()

#         theta = torch.rand(1, device=device) * 2 * math.pi
#         c = torch.cos(theta)
#         s = torch.sin(theta)

#         col_i = V[:, i].clone()
#         col_j = V[:, j].clone()
#         V[:, i] = c * col_i - s * col_j
#         V[:, j] = s * col_i + c * col_j

#     return V

# def generate_semi_orthogonal_matrix(m, n, num_rotations=None, device='cpu'):
#     V = generate_semi_rotation_matrix(m, n, num_rotations, device)
#     if torch.rand(1, device=device) < 0.5:
#         V[0, :] = -V[0, :]
#     return V

# def generate_semi_orthogonal_approx(G, device='cpu'):
#     assert len(G.shape) == 2, "Input must be a 2D tensor"
#     m, n = G.shape
#     V_e = generate_semi_orthogonal_matrix(m, n, device=device)
#     return V_e

# ######## ФУНКЦИЯ ДЛЯ СОЗДАНИЯ ОДНОЙ "ОРТОГОНАЛЬНОЙ" МАТРИЦЫ ЗАДАННОГО РАЗМЕРА #########

# def sample_ortho_approx(shape, device='cpu'):
#     assert len(shape) == 2, "Input dimension must be 2D"
#     m, n = shape
#     p = max(m, n)
#     E = torch_ortho_rvs(p, device=device)
#     return E[:m, :n]