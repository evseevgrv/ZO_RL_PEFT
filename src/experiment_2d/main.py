from optimizator import * 
from functions import (
    ackley_function, 
    levy_function, 
    quadratic_function, 
    rosenbrock_function
)


if __name__ == "__main__":
    d = 2
    optimizer = Experiment2d(
        f=rosenbrock_function,
        eps=1e-1,
        tau=1e-3,
        K=10,
        gamma=5e-2,
        x0=np.random.normal(
            loc=0, 
            scale=1, 
            size=(d,)
        ),
        mu0=np.random.uniform(
            low=-1, 
            high=1, 
            size=(d,)
        ),
        d=d
    )
    optimizer.run(
        num_steps=20000,
        project_name="zo-rl-2d",
        config={"function": "rosenbrock_function"}
    )
