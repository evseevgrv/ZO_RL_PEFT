import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.lines import Line2D
from eps_scheduler import * 


class Experiment2d:
    def __init__(
            self, f, eps=0.1, tau=0.1, K=50, gamma=0.01,
            x0=None, mu0=None, d=2,
            plot_mu_vectors=True,
            plot_e_vectors=True,
            plot_gx_vectors=False,
            plot_contour=False,
            eps_scheduler=None
        ):  
        if K <= 1:
            raise ValueError("K must be greater than 1")
        
        self.d = d
        self.eps0 = eps
        self.eps = eps

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
        self.f = f
        
        self.x = np.array(x0) if x0 is not None else np.array([1.0, 1.0])
        self.mu = np.array(mu0) if mu0 is not None else np.array([0.0, 0.0])
        
        self.trajectory_e = []
        self.plot_mu_vectors = plot_mu_vectors
        self.plot_e_vectors = plot_e_vectors
        self.plot_gx_vectors = plot_gx_vectors
        self.plot_contour = plot_contour

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
        term_i = (self.K * f_vals - total_sum) / (self.K - 1)
        
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

        metrics = {
            'x': self.x.copy(),
            'mu': self.mu.copy(),
            'f_value': self.f(self.x),
            'grad_x_norm': np.linalg.norm(g_x),
            'grad_mu_norm': np.linalg.norm(g_mu),
            'grad_x': g_x.copy()
        }
        
        self.trajectory_e.append(e_samples.copy())
        
        return metrics

    def run(self, num_steps, project_name="optimization_experiment", config=None):
        scheduler_config = {
            'schedule_type': self.eps_scheduler.schedule_type,
            'schedule_params': self.eps_scheduler.schedule_params
        }
        run_config = {
            "eps": self.eps_scheduler.initial_eps,
            "tau": self.tau,
            "K": self.K,
            "gamma": self.gamma,
            "d": self.d,
            "num_steps": num_steps,
            "eps_scheduler": scheduler_config
        }
        if config:
            run_config.update(config)
        
        wandb.init(
            project=project_name,
            name=config["function"] + f"-gamma={self.gamma}-eps={self.eps}-eps_scheduler={self.eps_scheduler.get_schedule_type()}",
            config=run_config,
            notes="d=2 experiment"
        )
        
        self.trajectory_e = []
        
        trajectory_x = [self.x.copy()]
        trajectory_mu = [self.mu.copy()]
        trajectory_gx = []

        init_metrics = {
            "step": 0,
            "f_value": self.f(self.x),
            "x_0": self.x[0],
            "x_1": self.x[1],
            "mu_0": self.mu[0],
            "mu_1": self.mu[1],
            "grad_x_norm": 0,
            "grad_mu_norm": 0
        }
        wandb.log(init_metrics)
        
        for step in range(1, num_steps + 1):
            metrics = self.step()
            
            trajectory_x.append(metrics['x'].copy())
            trajectory_mu.append(metrics['mu'].copy())
            trajectory_gx.append(metrics['grad_x'])

            log_data = {
                "step": step,
                "f_value": metrics['f_value'],
                "x_0": metrics['x'][0],
                "x_1": metrics['x'][1],
                "mu_0": metrics['mu'][0],
                "mu_1": metrics['mu'][1],
                "grad_x_norm": metrics['grad_x_norm'],
                "grad_mu_norm": metrics['grad_mu_norm']
            }
            wandb.log(log_data)
            
            if step % 1000 == 0:
                print(f"Step {step}: f(x) = {metrics['f_value']:.6f}, "
                      f"x = [{metrics['x'][0]:.4f}, {metrics['x'][1]:.4f}]")

        trajectory_x = np.array(trajectory_x)
        trajectory_mu = np.array(trajectory_mu)
        trajectory_gx = np.array(trajectory_gx)

        os.makedirs("experiment_2d/plots/x_plots", exist_ok=True)
        os.makedirs("experiment_2d/plots/mu_plots", exist_ok=True)

        sns.set_style("whitegrid")

        step_indices_x = list(range(0, len(trajectory_x), 500)) + [len(trajectory_x) - 1]
        step_indices_mu = list(range(0, len(trajectory_mu), 500)) + [len(trajectory_mu) - 1]

        plt.figure(figsize=(6, 6))
        plt.plot(trajectory_x[:, 0], trajectory_x[:, 1], color='blue', linewidth=1.5, alpha=0.7)
        plt.plot(trajectory_x[step_indices_x, 0], trajectory_x[step_indices_x, 1], 
                 marker='>', linestyle='None', color='blue', markersize=5, markeredgewidth=0)
        plt.plot(trajectory_x[-1, 0], trajectory_x[-1, 1], 
                 marker='*', color='red', markersize=10, markeredgecolor='black', markeredgewidth=0.5)
        plt.title("Optimization Trajectory of $\\mathbf{x}$", fontsize=14, pad=20)
        plt.xlabel("$x_1$", fontsize=12)
        plt.ylabel("$x_2$", fontsize=12)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"experiment_2d/plots/x_plots/trajectory_x-function={config['function']}-gamma={self.gamma}-eps={self.eps}-eps_scheduler={self.eps_scheduler.get_schedule_type()}.pdf", 
                    bbox_inches='tight', dpi=300)
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.plot(trajectory_mu[:, 0], trajectory_mu[:, 1], color='red', linewidth=1.5, alpha=0.7)
        plt.plot(trajectory_mu[step_indices_mu, 0], trajectory_mu[step_indices_mu, 1], 
                 marker='>', linestyle='None', color='red', markersize=5, markeredgewidth=0)
        plt.plot(trajectory_mu[-1, 0], trajectory_mu[-1, 1], 
                 marker='*', color='blue', markersize=10, markeredgecolor='black', markeredgewidth=0.5)
        plt.title("Distribution Mean Trajectory $\\boldsymbol{\\mu}$", fontsize=14, pad=20)
        plt.xlabel("$\\mu_1$", fontsize=12)
        plt.ylabel("$\\mu_2$", fontsize=12)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"experiment_2d/plots/mu_plots/trajectory_mu-function={config['function']}-gamma={self.gamma}-eps={self.eps}-eps_scheduler={self.eps_scheduler.get_schedule_type()}.pdf", 
                    bbox_inches='tight', dpi=300)
        plt.close()

        os.makedirs("experiment_2d/plots/x_mu_vectors", exist_ok=True)
        
        step_indices_vector = list(range(0, len(trajectory_x), 2000)) + [len(trajectory_x) - 1]
        step_indices_vector = sorted(set(step_indices_vector))

        plt.figure(figsize=(8, 6))
        
        x_min, x_max = min(0, min(trajectory_x[:, 0])) - 0.15, max(0, max(trajectory_x[:, 0])) + 0.15
        y_min, y_max = min(0, min(trajectory_x[:, 1])) - 0.15, max(0, max(trajectory_x[:, 1])) + 0.15
        
        if self.plot_contour:
            X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = self.f(np.array([X[i, j], Y[i, j]]))
            contourf = plt.contourf(X, Y, Z, levels=20, cmap='vlag', alpha=0.7, zorder=0)
            plt.colorbar(contourf, label='f(x)')

        plt.plot(trajectory_x[:, 0], trajectory_x[:, 1], 
                 color='black', linewidth=1.5, alpha=0.5, zorder=1)
        
        plt.plot(trajectory_x[step_indices_vector, 0], trajectory_x[step_indices_vector, 1], 
                 'o',color='black', markersize=3, markeredgewidth=0, zorder=2)
        
        plt.plot(trajectory_x[-1, 0], trajectory_x[-1, 1], 
                 marker='*', color='red', markersize=10, 
                 markeredgecolor='black', markeredgewidth=0.5, zorder=4)
        plt.plot(0, 0,  
                 marker='*', color='gold', markersize=10, 
                 markeredgecolor='black', markeredgewidth=0.5, zorder=4)

        scale_factor_mu = 0.01
        scale_factor_e = 1.0
        scale_factor_gx = 0.01
        arrow_width_mu = 0.0002
        head_width_mu = 0.005
        head_length_mu = 0.01
        arrow_width_e = 0.0001
        head_width_e = 0.03
        head_length_e = 0.05
        arrow_width_gx = 0.00015
        head_width_gx = 0.004
        head_length_gx = 0.008

        for idx in step_indices_vector:
            if idx == 0:
                continue
            if idx - 1 >= len(self.trajectory_e):
                continue

            x_point = trajectory_x[idx]
            
            if self.plot_mu_vectors:
                mu_vec = trajectory_mu[idx]
                plt.arrow(
                    x_point[0], x_point[1],
                    scale_factor_mu * mu_vec[0], scale_factor_mu * mu_vec[1],
                    width=arrow_width_mu,
                    head_width=head_width_mu,
                    head_length=head_length_mu,
                    fc='red', ec='darkred',
                    length_includes_head=True,
                    zorder=3
                )
            
            if self.plot_e_vectors:
                e_samples = self.trajectory_e[idx - 1]
                for e in e_samples:
                    dx, dy = self.tau * e
                    plt.arrow(
                        x_point[0], x_point[1],
                        dx * scale_factor_e, dy * scale_factor_e,
                        width=arrow_width_e,
                        head_width=head_width_e,
                        head_length=head_length_e,
                        fc='gray', ec='gray',
                        alpha=0.3,
                        length_includes_head=True,
                        zorder=2
                    )
            
            if self.plot_gx_vectors:
                gx_vec = trajectory_gx[idx - 1]
                plt.arrow(
                    x_point[0], x_point[1],
                    -scale_factor_gx * gx_vec[0], -scale_factor_gx * gx_vec[1],
                    width=arrow_width_gx,
                    head_width=head_width_gx,
                    head_length=head_length_gx,
                    fc='blue', ec='darkblue',
                    length_includes_head=True,
                    zorder=3
                )

        plt.title("Trajectory of $\\mathbf{x}$", fontsize=14, pad=20)
        plt.xlabel("$x_1$", fontsize=12)
        plt.ylabel("$x_2$", fontsize=12)
        # plt.axis('equal')

        legend_elements = [
            Line2D([0], [0], color='black', lw=1.5, label='Trajectory $\\mathbf{x}$'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                   markersize=8, label='Sampled steps'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                   markersize=10, markeredgecolor='black', label='Final point'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                   markersize=10, markeredgecolor='black', label='(0,0)')
        ]
        
        if self.plot_mu_vectors:
            legend_elements.append(Line2D([0], [0], color='red', lw=2, 
                                label=f'$\\boldsymbol{{\\mu}}$ (×{scale_factor_mu})'))
        if self.plot_e_vectors:
            legend_elements.append(Line2D([0], [0], color='gray', lw=2, 
                                label=r'$\tau \mathbf{e}$: perturbation directions'))
        if self.plot_gx_vectors:
            legend_elements.append(Line2D([0], [0], color='blue', lw=2, 
                                label=f'$-\\nabla f(\\mathbf{{x}})$ (×{scale_factor_gx})'))
        
        plt.legend(handles=legend_elements, loc='best', fontsize=9)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.tight_layout()
        plt.savefig(
            f"experiment_2d/plots/x_mu_vectors/trajectory_x_mu-function={config['function']}-gamma={self.gamma}-eps={self.eps0}-eps_scheduler={self.eps_scheduler.get_schedule_type()}.pdf",
            bbox_inches='tight',
            dpi=300
        )
       
        plt.close()

        print("Plots are saved.")
        wandb.finish()
