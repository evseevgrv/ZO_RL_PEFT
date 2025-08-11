import numpy as np
from .utils import *
from .base import BaseOptimizer

class ZORL_Optimizer(BaseOptimizer):
    def __init__(
        self, 
        f, 
        eps=0.1, 
        tau=0.1, 
        K=50, 
        gamma=0.01,
        x0=None, 
        mu0=None, 
        d=2,
        eps_scheduler=None
    ):
        super().__init__(f, d)
        if K <= 1:
            raise ValueError("K must be greater than 1")
        
        self.eps0 = eps
        self.eps = eps
        self.has_mu = True

        if eps_scheduler is None:
            self.eps_scheduler = EpsilonScheduler(
                initial_eps=eps,
                schedule_type='constant'
            )
        else:
            self.eps_scheduler = eps_scheduler

        self.tau = tau
        self.K = K
        self.gamma = gamma
        
        self.x = np.array(x0) if x0 is not None else np.array([1.0, 1.0])
        self.mu = np.array(mu0) if mu0 is not None else np.array([0.0, 0.0])

    def step(self):
        self.eps_scheduler.step()
        self.eps = self.eps_scheduler.get_eps()

        e_samples = np.random.normal(
            loc=self.mu, 
            scale=self.eps, 
            size=(self.K, self.d)
        )
        
        f_vals = np.array([
            self.f(self.x + self.tau * e) for e in e_samples
        ])
        
        total_sum = np.sum(f_vals)

        # term_x = (total_sum - f_vals) / total_sum
        # term_i = term_x

        term_i = (self.K * f_vals - total_sum) / self.K
        
        g_x = np.sum(
            term_i[:, np.newaxis] * e_samples, 
            axis=0
        ) / (self.tau * self.K)
        
        
        g_mu = np.sum(
            term_i[:, np.newaxis] * (self.mu - e_samples), 
            axis=0
        ) / (self.K * self.eps**2)
        
        self.x = self.x - self.gamma * g_x
        self.mu = self.mu + self.gamma * g_mu

        # self.trajectory_e.append(e_samples.copy())
        
        return {
            'x': self.x.copy(),
            'mu': self.mu.copy(),
            'f_value': self.f(self.x),
            'grad_x_norm': np.linalg.norm(g_x),
            'grad_mu_norm': np.linalg.norm(g_mu),
            'grad_x': g_x.copy()
        }
