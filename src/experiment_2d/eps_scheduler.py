import numpy as np


class EpsilonScheduler:
    def __init__(self, initial_eps: float, schedule_type: str = 'constant', schedule_params: dict = None):
        self.initial_eps = initial_eps
        self.current_eps = initial_eps
        self.schedule_type = schedule_type
        self.schedule_params = schedule_params or {}
        self.step_count = 0
        
        self.default_params = {
            'decay_rate': 0.001,
            'min_eps': 1e-6,
            'milestones': [5000, 10000, 15000],
            'decay_factor': 0.5,
            'power': 2.0,
            'total_steps': 20000,
            'cycle_length': 2000
        }
        
    def step(self):
        self.step_count += 1
        params = {**self.default_params, **self.schedule_params}
        
        if self.schedule_type == 'constant':
            return
            
        elif self.schedule_type == 'exponential':
            decay_rate = params['decay_rate']
            min_eps = params['min_eps']
            self.current_eps = max(min_eps, self.initial_eps * np.exp(-decay_rate * self.step_count))
            
        elif self.schedule_type == 'step':
            milestones = params['milestones']
            decay_factor = params['decay_factor']
            min_eps = params['min_eps']
            
            for milestone in milestones:
                if self.step_count == milestone:
                    self.current_eps = max(min_eps, self.current_eps * decay_factor)
                    
        elif self.schedule_type == 'cosine':
            min_eps = params['min_eps']
            total_steps = params['total_steps']
            ratio = min(1.0, self.step_count / total_steps)
            self.current_eps = min_eps + 0.5 * (self.initial_eps - min_eps) * (1 + np.cos(np.pi * ratio))
            
        elif self.schedule_type == 'linear':
            min_eps = params['min_eps']
            total_steps = params['total_steps']
            ratio = min(1.0, self.step_count / total_steps)
            self.current_eps = self.initial_eps - (self.initial_eps - min_eps) * ratio
            
        elif self.schedule_type == 'polynomial':
            min_eps = params['min_eps']
            total_steps = params['total_steps']
            power = params['power']
            ratio = min(1.0, self.step_count / total_steps)
            self.current_eps = self.initial_eps - (self.initial_eps - min_eps) * (ratio ** power)
            
        elif self.schedule_type == 'cyclic':
            min_eps = params['min_eps']
            cycle_length = params['cycle_length']
            cycle_progress = self.step_count % cycle_length
            self.current_eps = max(min_eps, self.initial_eps - 
                                  (self.initial_eps - min_eps) * (cycle_progress / cycle_length))
            
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def get_eps(self) -> float:
        return self.current_eps
    
    def reset(self):
        self.step_count = 0
        self.current_eps = self.initial_eps
