#!/usr/bin/env python3
"""
Optuna hyperparameter tuning script for learning_rate and lr_mu
"""
import os
import sys
import json
import argparse
import optuna
from optuna.trial import Trial
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def objective(trial: Trial) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
    
    Returns:
        The metric value to optimize (we maximize accuracy)
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    lr_mu = trial.suggest_float("lr_mu", 1e-5, 1e-1, log=True)
    
    # Create a unique tag for this trial - short name to avoid "File name too long" error
    tag = f"optuna_t{trial.number}"
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Trial {trial.number}: learning_rate={learning_rate:.2e}, lr_mu={lr_mu:.2e}")
    logger.info(f"{'='*80}\n")
    
    # Build the command
    command = [
        "python", "run.py",
        
        # Model and Task Configuration
        "--model_name=roberta-large",
        "--lora",
        "--task_name=SST2",
        "--trainer=zo_rl_jaguar",
        
        # Logging and Reporting
        f"--tag={tag}",  # Short tag to avoid "File name too long"
        "--output_dir=result/dummy",  # Will be overwritten by run.py with result/{tag}
        "--report_to=wandb",
        "--project_name=zo-rl-jaguar-optuna",
        "--logging_steps=10",
        f"--run_name={tag}",
        
        # Training Configuration
        "--num_train_epochs=5",
        "--per_device_train_batch_size=16",
        "--load_best_model_at_end",
        "--evaluation_strategy=steps",
        "--save_strategy=steps",
        "--save_total_limit=1",
        "--eval_steps=500",
        "--max_steps=20000",
        "--save_steps=1000",
        
        # Dataset Settings
        "--num_eval=1000",
        "--num_train=5000",
        "--num_dev=500",
        "--train_as_classification",
        "--train_set_seed=0",
        
        # Training Hyperparameters
        "--perturbation_mode=two_side",
        "--zo_eps=1e-3",
        "--momentum=0.0",
        "--weight_decay=0.0",
        "--module_wise_perturbation=False",
        
        # Miscellaneous
        "--overwrite_output_dir",
        
        # Learning Rate Scheduler Settings
        f"--learning_rate={learning_rate}",
        "--scheduler=cosine",
        "--num_training_steps=20000",
        "--warmup_steps=0",
        "--min_lr_ratio=0.1",
        "--scheduler_cycle_length=1",
        
        # Sampling Methods
        "--tensor_sampling_type=standard_normal",
        "--matrix_sampling_type=Random_baseline",
        
        # Jaguar-Specific Parameters
        "--zo_tau=1e-3",
        "--zo_beta=0.9",
        
        # Sparse Jaguar-Specific Parameters
        "--params_ratio=0.1",
        
        # Hyperparameters to tune
        f"--lr_mu={lr_mu}",
        "--k_value=5",
        "--variance=1",
        "--use_grad_first=False",
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
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
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
            # Prioritize test accuracy, then validation accuracy
            if "test_accuracy" in results:
                metric_value = results["test_accuracy"]
            elif "val_accuracy" in results:
                metric_value = results["val_accuracy"]
            elif "accuracy" in results:
                metric_value = results["accuracy"]
            else:
                logger.warning(f"No accuracy metric found in results: {results}")
                # Return a low value if no metric is found
                return 0.0
            
            logger.info(f"Trial {trial.number} completed with metric value: {metric_value}")
            
            # Log additional metrics if available
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
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials to run")
    parser.add_argument("--study_name", type=str, default="zo_rl_jaguar_tuning", help="Name of the Optuna study")
    parser.add_argument("--storage", type=str, default=None, help="Storage URL for Optuna study (e.g., sqlite:///optuna.db)")
    parser.add_argument("--load_if_exists", action="store_true", help="Load study if it exists")
    args = parser.parse_args()
    
    # Create output directory for trials
    os.makedirs("result/optuna_trials", exist_ok=True)
    
    # Create or load study
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=args.load_if_exists,
            direction="maximize",  # We want to maximize accuracy
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )
    
    logger.info(f"Starting Optuna study: {args.study_name}")
    logger.info(f"Number of trials: {args.n_trials}")
    
    # Run optimization
    study.optimize(objective, n_trials=args.n_trials)
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("Optimization completed!")
    logger.info("="*80)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value}")
    logger.info(f"Best parameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    # Save the study results
    results_file = "result/optuna_best_params.json"
    with open(results_file, 'w') as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_user_attrs": study.best_trial.user_attrs,
        }, f, indent=2)
    
    logger.info(f"\nBest parameters saved to: {results_file}")
    
    # Print top 5 trials
    logger.info("\nTop 5 trials:")
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else 0, reverse=True)[:5]
    for i, trial in enumerate(trials, 1):
        logger.info(f"{i}. Trial {trial.number}: value={trial.value}, params={trial.params}")


if __name__ == "__main__":
    main()
