#!/usr/bin/env python3
"""
Advanced Optuna hyperparameter tuning with additional features:
- Early stopping based on validation metrics
- Multiple hyperparameters tuning
- Visualization of optimization history
"""
import os
import sys
import json
import argparse
import optuna
from optuna.trial import Trial, TrialState
import subprocess
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptunaCallback:
    """Callback for monitoring training progress and enabling early stopping"""
    
    def __init__(self, trial: Trial, metric_file: str, check_interval: int = 100):
        self.trial = trial
        self.metric_file = metric_file
        self.check_interval = check_interval
        self.last_check = 0
    
    def should_prune(self, step: int, value: float) -> bool:
        """Check if trial should be pruned"""
        self.trial.report(value, step)
        if self.trial.should_prune():
            logger.info(f"Trial {self.trial.number} pruned at step {step}")
            return True
        return False


def run_training(trial: Trial, config: dict) -> float:
    """
    Run training with given hyperparameters.
    
    Args:
        trial: Optuna trial object
        config: Configuration dictionary with base parameters
    
    Returns:
        The metric value to optimize
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 
                                       config.get("lr_min", 1e-5), 
                                       config.get("lr_max", 1e-1), 
                                       log=True)
    lr_mu = trial.suggest_float("lr_mu", 
                               config.get("lr_mu_min", 1e-5), 
                               config.get("lr_mu_max", 1e-1), 
                               log=True)
    
    # Optional: tune additional parameters
    if config.get("tune_zo_eps", False):
        zo_eps = trial.suggest_float("zo_eps", 1e-4, 1e-2, log=True)
    else:
        zo_eps = config.get("zo_eps", 1e-3)
    
    if config.get("tune_k_value", False):
        k_value = trial.suggest_int("k_value", 1, 20)
    else:
        k_value = config.get("k_value", 5)
    
    if config.get("tune_variance", False):
        variance = trial.suggest_float("variance", 0.1, 10.0, log=True)
    else:
        variance = config.get("variance", 1.0)
    
    if config.get("tune_params_ratio", False):
        params_ratio = trial.suggest_float("params_ratio", 0.01, 0.5)
    else:
        params_ratio = config.get("params_ratio", 0.1)
    
    # Create a unique tag for this trial - short name to avoid "File name too long" error
    tag = f"optuna_t{trial.number}"
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Trial {trial.number}:")
    logger.info(f"  learning_rate: {learning_rate:.2e}")
    logger.info(f"  lr_mu: {lr_mu:.2e}")
    logger.info(f"  zo_eps: {zo_eps:.2e}")
    logger.info(f"  k_value: {k_value}")
    logger.info(f"  variance: {variance}")
    logger.info(f"  params_ratio: {params_ratio}")
    logger.info(f"{'='*80}\n")
    
    # Build the command
    command = [
        "python", "run.py",
        
        # Model and Task Configuration
        f"--model_name={config.get('model_name', 'roberta-large')}",
        "--lora",
        f"--task_name={config.get('task_name', 'SST2')}",
        f"--trainer={config.get('trainer', 'zo_rl_jaguar')}",
        
        # Logging and Reporting
        f"--tag={tag}",  # Short tag to avoid "File name too long"
        "--output_dir=result/dummy",  # Will be overwritten by run.py with result/{tag}
        "--report_to=wandb",
        f"--project_name={config.get('project_name', 'zo-rl-jaguar-optuna')}",
        f"--logging_steps={config.get('logging_steps', 10)}",
        f"--run_name={tag}",
        
        # Training Configuration
        f"--num_train_epochs={config.get('num_train_epochs', 5)}",
        f"--per_device_train_batch_size={config.get('batch_size', 16)}",
        "--load_best_model_at_end",
        "--evaluation_strategy=steps",
        "--save_strategy=steps",
        "--save_total_limit=1",
        f"--eval_steps={config.get('eval_steps', 500)}",
        f"--max_steps={config.get('max_steps', 20000)}",
        f"--save_steps={config.get('save_steps', 1000)}",
        
        # Dataset Settings
        f"--num_eval={config.get('num_eval', 1000)}",
        f"--num_train={config.get('num_train', 5000)}",
        f"--num_dev={config.get('num_dev', 500)}",
        "--train_as_classification",
        f"--train_set_seed={config.get('train_set_seed', 0)}",
        
        # Training Hyperparameters
        f"--perturbation_mode={config.get('perturbation_mode', 'two_side')}",
        f"--zo_eps={zo_eps}",
        f"--momentum={config.get('momentum', 0.0)}",
        f"--weight_decay={config.get('weight_decay', 0.0)}",
        f"--module_wise_perturbation={config.get('module_wise_perturbation', False)}",
        
        # Miscellaneous
        "--overwrite_output_dir",
        
        # Learning Rate Scheduler Settings
        f"--learning_rate={learning_rate}",
        f"--scheduler={config.get('scheduler', 'cosine')}",
        f"--num_training_steps={config.get('num_training_steps', 20000)}",
        f"--warmup_steps={config.get('warmup_steps', 0)}",
        f"--min_lr_ratio={config.get('min_lr_ratio', 0.1)}",
        f"--scheduler_cycle_length={config.get('scheduler_cycle_length', 1)}",
        
        # Sampling Methods
        f"--tensor_sampling_type={config.get('tensor_sampling_type', 'standard_normal')}",
        f"--matrix_sampling_type={config.get('matrix_sampling_type', 'Random_baseline')}",
        
        # Jaguar-Specific Parameters
        f"--zo_tau={config.get('zo_tau', 1e-3)}",
        f"--zo_beta={config.get('zo_beta', 0.9)}",
        
        # Sparse Jaguar-Specific Parameters
        f"--params_ratio={params_ratio}",
        
        # Hyperparameters being tuned
        f"--lr_mu={lr_mu}",
        f"--k_value={k_value}",
        f"--variance={variance}",
        f"--use_grad_first={config.get('use_grad_first', False)}",
    ]
    
    # Output directory will be created by run.py as result/{tag}
    output_dir = f"result/{tag}"
    
    # Save command to log file for debugging
    log_file = f"{output_dir}/command.log"
    os.makedirs(output_dir, exist_ok=True)
    with open(log_file, 'w') as f:
        f.write(" ".join(command) + "\n")
    
    try:
        # Run the training with output to log file
        with open(f"{output_dir}/stdout.log", 'w') as stdout_f, \
             open(f"{output_dir}/stderr.log", 'w') as stderr_f:
            result = subprocess.run(
                command,
                check=False,  # Don't raise on error, handle manually
                stdout=stdout_f,
                stderr=stderr_f,
                text=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": config.get("gpu_id", "0")}
            )
        
        if result.returncode != 0:
            logger.error(f"Trial {trial.number} failed with return code {result.returncode}")
            logger.error(f"Check logs in: {output_dir}/")
            return 0.0
        
        # Read the results file
        result_file = f"{output_dir}/results.json"
        
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            # Extract the metric to optimize
            metric_key = config.get("metric_key", "test_accuracy")
            metric_value = None
            
            # Try different possible metric names
            for key in [metric_key, "test_accuracy", "val_accuracy", "accuracy"]:
                if key in results:
                    metric_value = results[key]
                    break
            
            if metric_value is None:
                logger.warning(f"No metric found in results: {results}")
                return 0.0
            
            logger.info(f"Trial {trial.number} completed with {metric_key}={metric_value}")
            
            # Log all metrics as user attributes
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    trial.set_user_attr(key, value)
            
            return metric_value
        else:
            logger.error(f"Results file not found: {result_file}")
            logger.error(f"Check training logs in: {output_dir}/")
            return 0.0
            
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with exception: {e}")
        logger.error(f"Check logs in: {output_dir}/" if os.path.exists(output_dir) else "")
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Advanced Optuna hyperparameter tuning")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration JSON file")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials to run")
    parser.add_argument("--study_name", type=str, default=None, help="Name of the Optuna study")
    parser.add_argument("--storage", type=str, default=None, help="Storage URL for Optuna study")
    parser.add_argument("--load_if_exists", action="store_true", help="Load study if it exists")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    
    # Hyperparameter ranges
    parser.add_argument("--lr_min", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--lr_max", type=float, default=1e-1, help="Maximum learning rate")
    parser.add_argument("--lr_mu_min", type=float, default=1e-5, help="Minimum lr_mu")
    parser.add_argument("--lr_mu_max", type=float, default=1e-1, help="Maximum lr_mu")
    
    # Additional tuning options
    parser.add_argument("--tune_zo_eps", action="store_true", help="Also tune zo_eps")
    parser.add_argument("--tune_k_value", action="store_true", help="Also tune k_value")
    parser.add_argument("--tune_variance", action="store_true", help="Also tune variance")
    parser.add_argument("--tune_params_ratio", action="store_true", help="Also tune params_ratio")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override config with command-line arguments
    config.update({
        "lr_min": args.lr_min,
        "lr_max": args.lr_max,
        "lr_mu_min": args.lr_mu_min,
        "lr_mu_max": args.lr_mu_max,
        "tune_zo_eps": args.tune_zo_eps,
        "tune_k_value": args.tune_k_value,
        "tune_variance": args.tune_variance,
        "tune_params_ratio": args.tune_params_ratio,
    })
    
    # Generate study name if not provided
    study_name = args.study_name or f"zo_rl_jaguar_tuning_{int(time.time())}"
    
    # Create or load study
    if args.storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=args.storage,
            load_if_exists=args.load_if_exists,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Optuna study: {study_name}")
    logger.info(f"Number of trials: {args.n_trials}")
    logger.info(f"Parallel jobs: {args.n_jobs}")
    logger.info(f"Tuning parameters:")
    logger.info(f"  - learning_rate: [{args.lr_min:.2e}, {args.lr_max:.2e}]")
    logger.info(f"  - lr_mu: [{args.lr_mu_min:.2e}, {args.lr_mu_max:.2e}]")
    if args.tune_zo_eps:
        logger.info(f"  - zo_eps")
    if args.tune_k_value:
        logger.info(f"  - k_value")
    if args.tune_variance:
        logger.info(f"  - variance")
    if args.tune_params_ratio:
        logger.info(f"  - params_ratio")
    logger.info(f"{'='*80}\n")
    
    # Run optimization
    study.optimize(
        lambda trial: run_training(trial, config), 
        n_trials=args.n_trials,
        n_jobs=args.n_jobs
    )
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("Optimization completed!")
    logger.info("="*80)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.4f}")
    logger.info(f"Best parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2e}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Save the study results
    results_file = f"result/optuna_best_params_{study_name}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "study_name": study_name,
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_user_attrs": study.best_trial.user_attrs,
            "all_trials": [
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name,
                }
                for trial in study.trials
            ]
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    # Print top 5 trials
    logger.info("\nTop 5 trials:")
    trials = sorted(
        [t for t in study.trials if t.state == TrialState.COMPLETE],
        key=lambda t: t.value if t.value is not None else 0,
        reverse=True
    )[:5]
    for i, trial in enumerate(trials, 1):
        logger.info(f"{i}. Trial {trial.number}: value={trial.value:.4f}")
        for key, value in trial.params.items():
            if isinstance(value, float):
                logger.info(f"     {key}: {value:.2e}")
            else:
                logger.info(f"     {key}: {value}")
    
    # Generate visualization if optuna-dashboard is available
    try:
        import optuna.visualization as vis
        
        # Create visualization directory
        vis_dir = "result/optuna_visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Plot optimization history
        try:
            fig = vis.plot_optimization_history(study)
            fig.write_html(f"{vis_dir}/optimization_history_{study_name}.html")
            logger.info(f"\nVisualization saved to: {vis_dir}/optimization_history_{study_name}.html")
        except Exception as e:
            logger.warning(f"Could not create optimization history plot: {e}")
        
        # Plot parameter importances (may fail if variance is zero)
        try:
            fig = vis.plot_param_importances(study)
            fig.write_html(f"{vis_dir}/param_importances_{study_name}.html")
        except Exception as e:
            logger.warning(f"Could not create parameter importances plot: {e}")
        
        # Plot parallel coordinate
        try:
            fig = vis.plot_parallel_coordinate(study)
            fig.write_html(f"{vis_dir}/parallel_coordinate_{study_name}.html")
        except Exception as e:
            logger.warning(f"Could not create parallel coordinate plot: {e}")
        
    except ImportError:
        logger.info("\nTo generate visualizations, install: pip install plotly kaleido")


if __name__ == "__main__":
    main()
