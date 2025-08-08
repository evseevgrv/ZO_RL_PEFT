from optimizator import * 
from eps_scheduler import * 
from functions import (
    ackley_function, 
    levy_function, 
    quadratic_function, 
    rosenbrock_function
)


if __name__ == "__main__":
    d = 2
    for gamma in [1e-2]:
        for eps in [1e-1]:
            print(f"RUN FOR gamma={gamma}, eps={eps}")
            eps_scheduler = EpsilonScheduler(
                initial_eps=eps,
                schedule_type='cosine',
                schedule_params={
                    'total_steps': 20000,
                    'min_eps': 1e-6
                }
            )

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
                plot_contour=True,
                eps_scheduler=eps_scheduler
            )
            optimizer.run(
                num_steps=20000,
                project_name="zo-rl-2d",
                config={"function": "rosenbrock_function"}
            )
