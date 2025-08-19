import numpy as np
from .utils import *
from .base import BaseOptimizer
from autograd import grad

class SGD_Optimizer(BaseOptimizer):
    def __init__(self, f, gamma=0.01, x0=None, d=2, eps_scheduler=None):
        super().__init__(f, d)
        self.grad_f = grad(f)
        self.gamma = gamma
        self.x = np.array(x0) if x0 is not None else np.ones(d)

        self.tau = 0

        if eps_scheduler is None:
            self.eps_scheduler = EpsilonScheduler(
                initial_eps=1,
                schedule_type='constant'
            )
        else:
            self.eps_scheduler = eps_scheduler
    
    def step(self):
        g = self.grad_f(self.x)
        self.x = self.x - self.gamma * g
        return {
            'x': self.x.copy(),
            'f_value': self.f(self.x),
            'grad_x_norm': np.linalg.norm(g),
            'grad_x': g.copy()
        }
