import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Experiment2d:
    def __init__(self, f, eps=0.1, tau=0.1, K=50, gamma=0.01,
                 x0=None, mu0=None, d=2):
        if K <= 1:
            raise ValueError("K must be greater than 1")
        
        self.d = d
        self.eps = eps
        self.tau = tau
        self.K = K
        self.gamma = gamma
        self.f = f
        
        self.x = np.array(x0) if x0 is not None else np.array([1.0, 1.0])
        self.mu = np.array(mu0) if mu0 is not None else np.array([0.0, 0.0])
        

    def step(self):
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
        self.mu = self.mu - self.gamma * g_mu

        metrics = {
            'x': self.x.copy(),
            'mu': self.mu.copy(),
            'f_value': self.f(self.x),
            'grad_x_norm': np.linalg.norm(g_x),
            'grad_mu_norm': np.linalg.norm(g_mu)
        }
        
        return metrics

    def run(self, num_steps, project_name="optimization_experiment", config=None):
        run_config = {
            "eps": self.eps,
            "tau": self.tau,
            "K": self.K,
            "gamma": self.gamma,
            "d": self.d,
            "num_steps": num_steps
        }
        if config:
            run_config.update(config)
        
        wandb.init(
            project=project_name,
            name=config["function"],
            config=run_config,
            notes="d=2 experiment"
        )
        
        trajectory_x = [self.x.copy()]
        trajectory_mu = [self.mu.copy()]

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

        os.makedirs("experiment_2d/plots/x_plots", exist_ok=True)
        os.makedirs("experiment_2d/plots/mu_plots", exist_ok=True)

        sns.set_style("whitegrid")

        step_indices_x = list(range(0, len(trajectory_x), 500)) + [len(trajectory_x) - 1]
        step_indices_mu = list(range(0, len(trajectory_mu), 500)) + [len(trajectory_mu) - 1]

        plt.figure(figsize=(6, 6))
        plt.plot(trajectory_x[:, 0], trajectory_x[:, 1], color='blue', linewidth=1.5, alpha=0.7)
        plt.plot(trajectory_x[step_indices_x, 0], trajectory_x[step_indices_x, 1], 
                 marker='>', linestyle='None', color='blue', markersize=5, markeredgewidth=0)
        plt.title("Optimization Trajectory of $\\mathbf{x}$", fontsize=14, pad=20)
        plt.xlabel("$x_1$", fontsize=12)
        plt.ylabel("$x_2$", fontsize=12)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"experiment_2d/plots/x_plots/trajectory_x-function={config['function']}-gamma={self.gamma}-eps={self.eps}.pdf", bbox_inches='tight', dpi=300)
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.plot(trajectory_mu[:, 0], trajectory_mu[:, 1], color='red', linewidth=1.5, alpha=0.7)
        plt.plot(trajectory_mu[step_indices_mu, 0], trajectory_mu[step_indices_mu, 1], 
                 marker='>', linestyle='None', color='red', markersize=5, markeredgewidth=0)
        plt.title("Distribution Mean Trajectory $\\boldsymbol{\\mu}$", fontsize=14, pad=20)
        plt.xlabel("$\\mu_1$", fontsize=12)
        plt.ylabel("$\\mu_2$", fontsize=12)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"experiment_2d/plots/mu_plots/trajectory_mu-function={config['function']}-gamma={self.gamma}-eps={self.eps}.pdf", bbox_inches='tight', dpi=300)
        plt.close()

        wandb.finish()
