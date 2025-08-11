class BaseOptimizer:
    def __init__(self, f, d=2):
        self.f = f
        self.d = d

    def step(self):
        pass
