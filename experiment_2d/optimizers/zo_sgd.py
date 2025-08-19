import numpy as np
from .utils import *
from .base import BaseOptimizer

class ZOSGD_Optimizer(BaseOptimizer):
    def __init__(
        self, 
        f, 
        tau=0.1, 
        gamma=0.01,
        x0=None, 
        d=2,
        eps_scheduler=None
    ):
        super().__init__(f, d)
        self.tau = tau
        self.gamma = gamma
        
        self.x = np.array(x0) if x0 is not None else np.array([1.0, 1.0])

        if eps_scheduler is None:
            self.eps_scheduler = EpsilonScheduler(
                initial_eps=1,
                schedule_type='constant'
            )
        else:
            self.eps_scheduler = eps_scheduler

    def step(self):
        e = np.random.normal(loc=0.0, scale=1.0, size=self.d)

        f_val_1 = self.f(self.x + self.tau * e)
        f_val_2 = self.f(self.x - self.tau * e)

        g = (f_val_1 - f_val_2) / (2 * self.tau) * e
        
        self.x = self.x - self.gamma * g
        
        return {
            'x': self.x.copy(),
            'f_value': self.f(self.x),
            'grad_x_norm': np.linalg.norm(g),
            'grad_x': g.copy()
        }
