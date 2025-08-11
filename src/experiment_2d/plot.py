import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from matplotlib.lines import Line2D
from collections import defaultdict

class CombinedPlotter:
    def __init__(
        self,
        log_files,
        plot_mu_vectors=False,
        plot_e_vectors=False,
        plot_gx_vectors=False,
        plot_contour=True,
        mu_methods=None,
        e_methods=None,
        gx_methods=None,
        f=None,
        function_name=None,
        step_interval=2000
    ):
        """
        Класс для построения объединенных графиков из нескольких лог-файлов экспериментов.
        
        Параметры:
        log_files : list
            Список путей к .log файлам с метриками
        plot_mu_vectors : bool
            Рисовать векторы mu (среднее распределения)
        plot_e_vectors : bool
            Рисовать векторы возмущений e
        plot_gx_vectors : bool
            Рисовать векторы градиента -∇f(x)
        plot_contour : bool
            Рисовать контур функции
        mu_methods : list or None
            Список методов, для которых рисовать mu_vectors (None = все)
        e_methods : list or None
            Список методов, для которых рисовать e_vectors (None = все)
        gx_methods : list or None
            Список методов, для которых рисовать gx_vectors (None = все)
        f : callable or None
            Функция для построения контура (если plot_contour=True)
        function_name : str or None
            Название функции для подписи графика
        step_interval : int
            Интервал шагов для отображения векторов
        """
        self.log_files = log_files
        self.plot_mu_vectors = plot_mu_vectors
        self.plot_e_vectors = plot_e_vectors
        self.plot_gx_vectors = plot_gx_vectors
        self.plot_contour = plot_contour
        self.mu_methods = mu_methods
        self.e_methods = e_methods
        self.gx_methods = gx_methods
        self.f = f
        self.function_name = function_name
        self.step_interval = step_interval
        
        self.optimizer_colors = {
            'SGD': 'green',
            'ZOSGD': 'purple',
            'ZORL': 'black'
        }
        self.default_color = 'gray'
        
        self.scale_factors = {
            'mu': 0.01,
            'e': 1.0,
            'gx': 0.01
        }
        
        self.arrow_params = {
            'mu': {
                'width': 0.0002,
                'head_width': 0.005,
                'head_length': 0.01,
                'fc': 'red',
                'ec': 'darkred'
            },
            'e': {
                'width': 0.0001,
                'head_width': 0.03,
                'head_length': 0.05,
                'fc': 'gray',
                'ec': 'gray',
                'alpha': 0.3
            },
            'gx': {
                'width': 0.00015,
                'head_width': 0.004,
                'head_length': 0.008,
                'fc': 'blue',
                'ec': 'darkblue'
            }
        }
        
        if not self.function_name and log_files:
            self.function_name = self._extract_function_name(log_files[0])
        
        self.optimizer_data = self._parse_log_files()

    def _extract_function_name(self, log_file):
        base = os.path.basename(log_file)
        name = os.path.splitext(base)[0]
        return name.split('_')[0] if '_' in name else "unknown"

    def _extract_optimizer_name(self, log_file):
        base = os.path.basename(log_file)
        name = os.path.splitext(base)[0]
        parts = name.split('_')
        
        for part in parts:
            if part in self.optimizer_colors or part.replace('Optimizer', '') in self.optimizer_colors:
                return part
        
        return "Unknown"

    def _parse_log_file(self, log_file):
        optimizer_name = self._extract_optimizer_name(log_file)
        data = {
            'x_trajectory': [],
            'mu_trajectory': [],
            'grad_x_trajectory': [],
            'steps': [],
            'f_values': []
        }
        
        try:
            with open(log_file, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                step_idx = headers.index('step')
                
                x_indices = []
                i = 0
                while True:
                    col_name = f'x_{i}'
                    if col_name in headers:
                        x_indices.append(headers.index(col_name))
                        i += 1
                    else:
                        break
                
                mu_indices = []
                i = 0
                while True:
                    col_name = f'mu_{i}'
                    if col_name in headers:
                        mu_indices.append(headers.index(col_name))
                        i += 1
                    else:
                        break
                
                grad_x_indices = []
                i = 0
                while True:
                    col_name = f'grad_x_{i}'
                    if col_name in headers:
                        grad_x_indices.append(headers.index(col_name))
                        i += 1
                    else:
                        break
                
                f_value_idx = headers.index('f_value') if 'f_value' in headers else None
                
                for row in reader:
                    step = int(float(row[step_idx]))
                    data['steps'].append(step)
                    
                    if x_indices:
                        x = [float(row[i]) for i in x_indices]
                        data['x_trajectory'].append(x)
                    
                    if mu_indices:
                        mu = [float(row[i]) for i in mu_indices]
                        data['mu_trajectory'].append(mu)
                    
                    if grad_x_indices:
                        grad_x = [float(row[i]) for i in grad_x_indices]
                        data['grad_x_trajectory'].append(grad_x)
                    
                    if f_value_idx is not None:
                        f_value = float(row[f_value_idx])
                        data['f_values'].append(f_value)
                    
        except Exception as e:
            print(f"Error parsing {log_file}: {str(e)}")
        
        return optimizer_name, data

    def _parse_log_files(self):
        optimizer_data = {}
        
        for log_file in self.log_files:
            optimizer_name, data = self._parse_log_file(log_file)
            if data['x_trajectory']:
                optimizer_data[optimizer_name] = data
        
        return optimizer_data

    def _get_color(self, optimizer_name):
        if optimizer_name in self.optimizer_colors:
            return self.optimizer_colors[optimizer_name]
        
        base_name = optimizer_name.replace('Optimizer', '')
        if base_name in self.optimizer_colors:
            return self.optimizer_colors[base_name]
        
        return self.default_color

    def _plot_contour(self, ax, x_min, x_max, y_min, y_max):
        if not self.plot_contour or self.f is None:
            return
            
        X, Y = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.f(np.array([X[i, j], Y[i, j]]))
        
        contourf = ax.contourf(X, Y, Z, levels=20, cmap='vlag', alpha=0.7, zorder=0)
        plt.colorbar(contourf, ax=ax, label='f(x)')

    def _plot_vectors(self, ax, optimizer_name, data, step_indices):
        if self.plot_mu_vectors and data['mu_trajectory'] and len(data['mu_trajectory']) > 0:
            should_plot = (self.mu_methods is None) or (optimizer_name in self.mu_methods)
            if should_plot:
                for idx in step_indices:
                    if idx >= len(data['x_trajectory']) or idx >= len(data['mu_trajectory']):
                        continue
                        
                    x = data['x_trajectory'][idx]
                    mu = data['mu_trajectory'][idx]
                    
                    scale = self.scale_factors['mu']
                    params = self.arrow_params['mu']
                    
                    ax.arrow(
                        x[0], x[1],
                        scale * mu[0], scale * mu[1],
                        width=params['width'],
                        head_width=params['head_width'],
                        head_length=params['head_length'],
                        fc=params['fc'],
                        ec=params['ec'],
                        length_includes_head=True,
                        zorder=3
                    )
        
        if self.plot_gx_vectors and data['grad_x_trajectory'] and len(data['grad_x_trajectory']) > 0:
            should_plot = (self.gx_methods is None) or (optimizer_name in self.gx_methods)
            if should_plot:
                for idx in step_indices:
                    if idx >= len(data['x_trajectory']) or (idx-1) >= len(data['grad_x_trajectory']):
                        continue
                        
                    x = data['x_trajectory'][idx]
                    grad_x = data['grad_x_trajectory'][idx-1]  
                    
                    scale = self.scale_factors['gx']
                    params = self.arrow_params['gx']
                    
                    ax.arrow(
                        x[0], x[1],
                        -scale * grad_x[0], -scale * grad_x[1],
                        width=params['width'],
                        head_width=params['head_width'],
                        head_length=params['head_length'],
                        fc=params['fc'],
                        ec=params['ec'],
                        length_includes_head=True,
                        zorder=3
                    )

    def plot(self, output_path, title=None):
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        all_x = []
        for data in self.optimizer_data.values():
            if data['x_trajectory']:
                all_x.extend(data['x_trajectory'])
        
        if not all_x:
            raise ValueError("No trajectory data found in log files")
        
        all_x = np.array(all_x)
        x_min, x_max = min(0,all_x[:, 0].min()), max(0,all_x[:, 0].max())
        y_min, y_max = min(0,all_x[:, 1].min()), max(0,all_x[:, 1].max())

        x_min -= 0.15 
        x_max += 0.15 
        y_min -= 0.15 
        y_max += 0.15 
        
        self._plot_contour(ax, x_min, x_max, y_min, y_max)
        
        max_steps = max(len(data['steps']) for data in self.optimizer_data.values())
        step_indices = list(range(0, max_steps, self.step_interval)) + [max_steps - 1]
        step_indices = sorted(set(step_indices))
        
        legend_elements = []
        
        for optimizer_name, data in self.optimizer_data.items():
            color = self._get_color(optimizer_name)
            x_trajectory = np.array(data['x_trajectory'])
            
            ax.plot(
                x_trajectory[:, 0], x_trajectory[:, 1],
                color=color,
                linewidth=2.0,
                alpha=0.7,
                zorder=1,
                label=optimizer_name
            )
            
            valid_indices = [i for i in step_indices if i < len(x_trajectory)]
            if valid_indices:
                ax.plot(
                    x_trajectory[valid_indices, 0],
                    x_trajectory[valid_indices, 1],
                    'o',
                    color=color,
                    markersize=5,
                    markeredgewidth=0,
                    zorder=2
                )
            
            if len(x_trajectory) > 0:
                ax.plot(
                    x_trajectory[-1, 0], x_trajectory[-1, 1],
                    marker='*', color='red', markersize=12,
                    markeredgecolor='black', markeredgewidth=0.5,
                    zorder=4
                )
            
            legend_elements.append(
                Line2D([0], [0], color=color, lw=2, label=optimizer_name)
            )
            
            self._plot_vectors(ax, optimizer_name, data, valid_indices)
        
        if x_min <= 0 <= x_max and y_min <= 0 <= y_max:
            ax.plot(
                0, 0,
                marker='*', color='gold', markersize=12,
                markeredgecolor='black', markeredgewidth=0.5,
                zorder=4
            )
            legend_elements.append(
                Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                       markersize=12, markeredgecolor='black', label='(0,0)')
            )
        
        if self.plot_mu_vectors:
            params = self.arrow_params['mu']
            legend_elements.append(
                Line2D([0], [0], color=params['fc'], lw=2,
                       label=f'$\\boldsymbol{{\\mu}}$ (×{self.scale_factors["mu"]})')
            )
        
        if self.plot_gx_vectors:
            params = self.arrow_params['gx']
            legend_elements.append(
                Line2D([0], [0], color=params['fc'], lw=2,
                       label=f'$-\\nabla f(\\mathbf{{x}})$ (×{self.scale_factors["gx"]})')
            )
        
        plt_title = title or f"Combined Trajectories ({self.function_name})"
        ax.set_title(plt_title, fontsize=16, pad=20)
        ax.set_xlabel("$x_1$", fontsize=14)
        ax.set_ylabel("$x_2$", fontsize=14)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        # ax.axis('equal')
        
        ax.legend(
            handles=legend_elements,
            loc='best',
            fontsize=10,
            frameon=True,
            framealpha=0.9
        )
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Combined plot saved to: {output_path}")
        return output_path
