import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.lines import Line2D
from optimizers import *

class Experiment2d:
    def __init__(
        self,
        optimizer,
        use_wandb=True,
        plot_mu_vectors=True,
        plot_e_vectors=True,
        plot_gx_vectors=False,
        plot_contour=False
    ):
        self.optimizer = optimizer
        self.use_wandb = use_wandb
        self.plot_mu_vectors = plot_mu_vectors
        self.plot_e_vectors = plot_e_vectors
        self.plot_gx_vectors = plot_gx_vectors
        self.plot_contour = plot_contour

    def _flatten_metrics(self, metrics, step):
        flat = {'step': step}
        for key, value in metrics.items():
            if isinstance(value, (list, np.ndarray)):
                for i, v in enumerate(value):
                    flat[f"{key}_{i}"] = v
            else:
                flat[key] = value
        return flat

    def run(self, num_steps, project_name="optimization_experiment", config=None):
        scheduler_config = {
            'schedule_type': self.optimizer.eps_scheduler.schedule_type,
            'schedule_params': self.optimizer.eps_scheduler.schedule_params
        }
        run_config = {
            "eps": self.optimizer.eps_scheduler.initial_eps,
            "tau": self.optimizer.tau,
            "gamma": self.optimizer.gamma,
            "d": self.optimizer.d,
            "num_steps": num_steps,
            "eps_scheduler": scheduler_config,
            "optimizer_type": self.optimizer.__class__.__name__
        }
        if hasattr(self.optimizer, 'K'):
            run_config["K"] = self.optimizer.K
        if config:
            run_config.update(config)
        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=config["function"] + f"-gamma={self.optimizer.gamma}-eps={self.optimizer.eps_scheduler.get_eps()}-eps_scheduler={self.optimizer.eps_scheduler.get_schedule_type()}",
                config=run_config,
                notes="d=2 experiment"
            )
        
        os.makedirs("experiment_2d/logs", exist_ok=True)
        
        gamma_str = str(self.optimizer.gamma).replace('.', 'p')
        eps_str = str(self.optimizer.eps_scheduler.get_eps()).replace('.', 'p')
        tau_str = str(self.optimizer.tau).replace('.', 'p')
        K_val = self.optimizer.K if hasattr(self.optimizer, 'K') else '1'
        scheduler_type = self.optimizer.eps_scheduler.schedule_type
        optimizer_name = self.optimizer.__class__.__name__
        function_name = config['function']
        
        file_name = (
            f"{function_name}_{optimizer_name}_"
            f"gamma{gamma_str}_eps{eps_str}_"
            f"tau{tau_str}_K{K_val}_"
            f"sched{scheduler_type}.log"
        )
        log_path = os.path.join("experiment_2d/logs", file_name)
        self.log_path = log_path
        
        initial_metrics = {
            'x': self.optimizer.x.copy(),
            'f_value': self.optimizer.f(self.optimizer.x),
            'grad_x_norm': 0.0,
            'grad_x': np.zeros(self.optimizer.d)
        }
        if hasattr(self.optimizer, 'has_mu') and self.optimizer.has_mu:
            initial_metrics['mu'] = self.optimizer.mu.copy()
            initial_metrics['grad_mu_norm'] = 0.0
        
        with open(log_path, 'w') as log_file:
            flat_init = self._flatten_metrics(initial_metrics, 0)
            
            header = ",".join(flat_init.keys()) + "\n"
            log_file.write(header)
            
            init_line = ",".join(str(flat_init[key]) for key in flat_init.keys()) + "\n"
            log_file.write(init_line)
            if self.use_wandb:
                wandb.log(flat_init)
            
            trajectory_x = [self.optimizer.x.copy()]
            trajectory_gx = [initial_metrics['grad_x'].copy()]
            trajectory_mu = []
            if hasattr(self.optimizer, 'has_mu') and self.optimizer.has_mu:
                trajectory_mu = [self.optimizer.mu.copy()]

            for step in range(1, num_steps + 1):
                metrics = self.optimizer.step()
                
                trajectory_x.append(metrics['x'].copy())
                trajectory_gx.append(metrics['grad_x'].copy())
                if hasattr(self.optimizer, 'has_mu') and self.optimizer.has_mu:
                    trajectory_mu.append(metrics['mu'].copy())

                flat_metrics = self._flatten_metrics(metrics, step)
                
                line = ",".join(str(flat_metrics[key]) for key in flat_metrics.keys()) + "\n"
                log_file.write(line)
                
                if self.use_wandb:
                    wandb.log(flat_metrics)
                
                if step % 1000 == 0:
                    x_str = ", ".join(f"{val:.4f}" for val in metrics['x'])
                    print(f"Step {step}: f(x) = {metrics['f_value']:.6f}, x = [{x_str}]")

        trajectory_x = np.array(trajectory_x)
        # trajectory_gx = np.array(trajectory_gx)
        if trajectory_mu:
            trajectory_mu = np.array(trajectory_mu)

        os.makedirs("experiment_2d/plots/x_plots", exist_ok=True)
        os.makedirs("experiment_2d/plots/mu_plots", exist_ok=True)

        sns.set_style("whitegrid")

        step_indices_x = list(range(0, len(trajectory_x), 500)) + [len(trajectory_x) - 1]
        
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
        plt.savefig(f"experiment_2d/plots/x_plots/trajectory_x-function={config['function']}-gamma={self.optimizer.gamma}-eps={self.optimizer.eps_scheduler.get_eps()}-eps_scheduler={self.optimizer.eps_scheduler.get_schedule_type()}.pdf", 
                    bbox_inches='tight', dpi=300)
        plt.close()

        if hasattr(self.optimizer, 'has_mu') and self.optimizer.has_mu:
            step_indices_mu = list(range(0, len(trajectory_mu), 500)) + [len(trajectory_mu) - 1]
            
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
            plt.savefig(f"experiment_2d/plots/mu_plots/trajectory_mu-function={config['function']}-gamma={self.optimizer.gamma}-eps={self.optimizer.eps_scheduler.get_eps()}-eps_scheduler={self.optimizer.eps_scheduler.get_schedule_type()}.pdf", 
                        bbox_inches='tight', dpi=300)
            plt.close()

        os.makedirs("experiment_2d/plots/x_mu_vectors", exist_ok=True)
        
        step_indices_vector = list(range(0, len(trajectory_x), 2000)) + [len(trajectory_x) - 1]
        step_indices_vector = sorted(set(step_indices_vector))

        plt.figure(figsize=(8, 6))
        
        x_min, x_max = min(1, min(trajectory_x[:, 0])) - 0.15, max(1, max(trajectory_x[:, 0])) + 0.15
        y_min, y_max = min(1, min(trajectory_x[:, 1])) - 0.15, max(1, max(trajectory_x[:, 1])) + 0.15
        
        if self.plot_contour:
            X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = self.optimizer.f(np.array([X[i, j], Y[i, j]]))
            contourf = plt.contourf(X, Y, Z, levels=20, cmap='vlag', alpha=0.7, zorder=0)
            plt.colorbar(contourf, label='f(x)')

        plt.plot(trajectory_x[:, 0], trajectory_x[:, 1], 
                 color='black', linewidth=1.5, alpha=0.5, zorder=1)
        
        plt.plot(trajectory_x[step_indices_vector, 0], trajectory_x[step_indices_vector, 1], 
                 'o',color='black', markersize=3, markeredgewidth=0, zorder=2)
        
        plt.plot(trajectory_x[-1, 0], trajectory_x[-1, 1], 
                 marker='*', color='red', markersize=10, 
                 markeredgecolor='black', markeredgewidth=0.5, zorder=4)
        plt.plot(1, 1,  
                 marker='*', color='gold', markersize=10, 
                 markeredgecolor='black', markeredgewidth=0.5, zorder=4)

        scale_factor_mu = 0.05
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
            if idx - 1 >= len(trajectory_x) - 1:
                continue

            x_point = trajectory_x[idx]
            
            if self.plot_mu_vectors and hasattr(self.optimizer, 'has_mu') and self.optimizer.has_mu:
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
            
            if self.plot_e_vectors and hasattr(self.optimizer, 'trajectory_e') and (idx - 1) < len(self.optimizer.trajectory_e):
                e_samples = self.optimizer.trajectory_e[idx - 1]
                for e in e_samples:
                    dx, dy = self.optimizer.tau * e
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
            
            if self.plot_gx_vectors and idx - 1 < len(trajectory_gx):
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

        plt_title = "Trajectory of $\\mathbf{x}$ (" + self.optimizer.__class__.__name__  + ")"           
        plt.title(plt_title, fontsize=14, pad=20)
        plt.xlabel("$x_1$", fontsize=12)
        plt.ylabel("$x_2$", fontsize=12)

        legend_elements = [
            Line2D([0], [0], color='black', lw=1.5, label='Trajectory $\\mathbf{x}$'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                   markersize=8, label='Sampled steps'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                   markersize=10, markeredgecolor='black', label='Final point'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                   markersize=10, markeredgecolor='black', label='Theoretical solution')
        ]
        
        if self.plot_mu_vectors and hasattr(self.optimizer, 'has_mu') and self.optimizer.has_mu:
            legend_elements.append(Line2D([0], [0], color='red', lw=2, 
                                label=f'$\\boldsymbol{{\\mu}}$ (×{scale_factor_mu})'))
        if self.plot_e_vectors and hasattr(self.optimizer, 'trajectory_e') and self.optimizer.trajectory_e:
            legend_elements.append(Line2D([0], [0], color='gray', lw=2, 
                                label=r'$\tau \mathbf{e}$: perturbation directions'))
        if self.plot_gx_vectors and trajectory_gx.size > 0:
            legend_elements.append(Line2D([0], [0], color='blue', lw=2, 
                                label=f'$-\\nabla f(\\mathbf{{x}})$ (×{scale_factor_gx})'))
        
        plt.legend(handles=legend_elements, loc='best', fontsize=9)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.tight_layout()
        
        optimizer_type = self.optimizer.__class__.__name__
        plt.savefig(
            f"experiment_2d/plots/x_mu_vectors/trajectory_{optimizer_type}-function={config['function']}-gamma={self.optimizer.gamma}-eps={self.optimizer.eps_scheduler.get_eps()}-eps_scheduler={self.optimizer.eps_scheduler.get_schedule_type()}.pdf",
            bbox_inches='tight',
            dpi=300
        )
       
        plt.close()

        print("Plots are saved.")
        print(f"experiment_2d/plots/x_mu_vectors/trajectory_{optimizer_type}-function={config['function']}-gamma={self.optimizer.gamma}-eps={self.optimizer.eps_scheduler.get_eps()}-eps_scheduler={self.optimizer.eps_scheduler.get_schedule_type()}.pdf")
        if self.use_wandb:
            wandb.finish()
