from optimizator import * 
from functions import (
    ackley_function, 
    levy_function, 
    quadratic_function, 
    rosenbrock_function
)


if __name__ == "__main__":
    d = 2
    for gamma in [1e-2]:
        for eps in [1e-2]:
            print(f"RUN FOR gamma={gamma}, eps={eps}")
            optimizer = Experiment2d(
                f=rosenbrock_function,
                eps=eps,
                tau=1e-3,
                K=10,
                gamma=gamma,
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
                d=d,
                plot_mu_vectors=True,
                plot_e_vectors=False,
                plot_gx_vectors=False,
                plot_contour=True
            )
            optimizer.run(
                num_steps=20000,
                project_name="zo-rl-2d",
                config={"function": "rosenbrock_function"}
            )
