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
        gamma_mu=None,
        beta=0.99,
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
        self.gamma_mu = gamma_mu if gamma_mu is not None else gamma
        self.beta = beta
        
        self.x = np.array(x0) if x0 is not None else np.array([1.0, 1.0])
        self.v_x = np.zeros_like(self.x)
        self.mu = np.array(mu0) if mu0 is not None else np.array([0.0, 0.0])

    def step(self):
        # self.mu /= np.linalg.norm(self.mu)
        
        self.eps_scheduler.step()
        self.eps = self.eps_scheduler.get_eps()

        e_samples = np.random.normal(
            loc=self.mu, 
            scale=self.eps, 
            size=(self.K, self.d)
        )

        e_norms = np.linalg.norm(e_samples, axis=1, keepdims=True)  
        e_samples = e_samples / e_norms
        
        f_vals = np.array([
            self.f(self.x + self.tau * e) for e in e_samples
        ])
        
        total_sum = np.sum(f_vals)

        # term_i = (self.K * f_vals - total_sum) / self.K
        term_x = (total_sum - f_vals) / total_sum

        # f_current = self.f(self.x)

        # term_x = (f_vals - f_current) 
        term_i = term_x
        
        g_x = np.sum(
            term_i[:, np.newaxis] * e_samples, 
            axis=0
        ) / (self.tau * self.K)

        print(g_x)

        # e_ind = np.argmin(f_vals)
        # e_argmin = e_samples[e_ind]
        # f_plus = f_vals[e_ind]
        # f_minus = self.f(self.x - self.tau * e_argmin)
        # g_x = (f_plus - f_minus) * e_argmin / (2 * self.tau)

        # print(self.x, f_plus, f_minus, g_x)
        
        term_i = (self.K * f_vals - total_sum) / self.K
        
        g_mu = np.sum(
            term_i[:, np.newaxis] * (self.mu - e_samples), 
            axis=0
        ) / (self.K * self.eps**2)
                
        self.v_x = self.beta * self.v_x + (1 - self.beta) * g_x

        self.x = self.x - self.gamma * self.v_x
        self.mu = self.mu + self.gamma_mu * g_mu
        
        return {
            'x': self.x.copy(),
            'mu': self.mu.copy(),
            'f_value': self.f(self.x),
            'grad_x_norm': np.linalg.norm(g_x),
            'grad_mu_norm': np.linalg.norm(g_mu),
            'grad_x': g_x.copy()
        }
