from experiment import * 
from optimizers import * 
from functions import *
from plot import CombinedPlotter

if __name__ == "__main__":
    d = 2
    log_files = []
    for gamma in [1e-3]:
        for eps in [1e-3]:
            print(f"RUN FOR gamma={gamma}, eps={eps}")
            eps_scheduler = None
            # eps_scheduler = EpsilonScheduler(
            #     initial_eps=eps,
            #     schedule_type='cosine',
            #     schedule_params={
            #         'total_steps': 20000,
            #         'min_eps': 1e-6,
            #         'decay_rate': 0.001,
            #         'min_eps': 1e-6,
            #         'milestones': [5000, 10000, 15000],
            #         'decay_factor': 0.5,
            #         'power': 2.0,
            #         'total_steps': 20000,
            #         'cycle_length': 2000
            #     }
            # )
            # print(rosenbrock_function([1,1]))
            # print(rosenbrock_function([0,0]))
            optimizer = ZORL_Optimizer(
                f=rosenbrock_function,
                eps=eps,
                tau=1e-2,
                K=10,
                gamma=gamma,
                gamma_mu=1e-2,
                beta=0.0,
                x0=np.random.normal(
                    loc=1, 
                    scale=1, 
                    size=(d,)
                ),
                mu0=np.random.uniform(
                    low=-1, 
                    high=1, 
                    size=(d,)
                ),
                d=d,
                eps_scheduler=eps_scheduler
            )

            experiment = Experiment2d(
                optimizer=optimizer,
                use_wandb=False,
                plot_mu_vectors=True,
                plot_e_vectors=False,
                plot_gx_vectors=False,
                plot_contour=True
            )

            experiment.run(
                num_steps=20000,
                project_name="zo-rl-2d",
                config={"function": "rosenbrock_function"}
            )

            log_path = experiment.log_path 
            log_files.append(log_path)


            # optimizer = ZOSGD_Optimizer(
            #     f=rosenbrock_function,
            #     tau=1e-2,
            #     gamma=1e-4,
            #     x0=np.random.normal(
            #         loc=0, 
            #         scale=1, 
            #         size=(d,)
            #     ),
            #     d=d,
            #     eps_scheduler=eps_scheduler
            # )

            # experiment = Experiment2d(
            #     optimizer=optimizer,
            #     use_wandb=False,
            #     plot_mu_vectors=False,
            #     plot_e_vectors=False,
            #     plot_gx_vectors=False,
            #     plot_contour=True
            # )

            # experiment.run(
            #     num_steps=20000,
            #     project_name="zo-rl-2d",
            #     config={"function": "rosenbrock_function"}
            # )

            # log_path = experiment.log_path 
            # log_files.append(log_path)

            # optimizer = SGD_Optimizer(
            #     f=rosenbrock_function,
            #     gamma=1e-3,
            #     x0=np.random.normal(
            #         loc=0, 
            #         scale=1, 
            #         size=(d,)
            #     ),
            #     d=d,
            #     eps_scheduler=eps_scheduler
            # )

            # experiment = Experiment2d(
            #     optimizer=optimizer,
            #     use_wandb=False,
            #     plot_mu_vectors=False,
            #     plot_e_vectors=False,
            #     plot_gx_vectors=False,
            #     plot_contour=True
            # )

            # experiment.run(
            #     num_steps=20000,
            #     project_name="zo-rl-2d",
            #     config={"function": "rosenbrock_function"}
            # )

            # log_path = experiment.log_path 
            # log_files.append(log_path)


            # log_files = [
            #     "experiment_2d/logs/rosenbrock_function_SGD_Optimizer_gamma0p001_eps1_tau0_K1_schedconstant.log",
            #     "experiment_2d/logs/rosenbrock_function_ZO_SGD_Optimizer_gamma0p001_eps1_tau0p01_K1_schedconstant.log"
            # ]
            
    
            plotter = CombinedPlotter(
                log_files=log_files,
                plot_mu_vectors=True,
                plot_gx_vectors=False,
                plot_contour=True,
                f=rosenbrock_function,
                function_name="rosenbrock_function",
                mu_methods=["ZO_RL"],  
                gx_methods=None,      
                step_interval=1000
            )
            
            plotter.plot(
                output_path="experiment_2d/plots/combined_trajectory.pdf",
                title="Comparison of Optimization Trajectories"
            )
