"""
Hyperparameter optimization utilities for TAuto.

This module provides tools for optimizing hyperparameters of machine learning
models, including integration with Optuna, Bayesian optimization, and
search space definition with constraints.
"""

import os
import torch
import time
import copy
import logging
import warnings
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Tuple, Set

import numpy as np
from tauto.utils import get_logger

# Try to import Optuna, but provide fallbacks if not available
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner, NopPruner
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Install it with `pip install optuna` to use hyperparameter optimization.")

logger = get_logger(__name__)


class HyperparameterOptimization:
    """
    Hyperparameter optimization utilities.
    
    This class provides utilities for optimizing hyperparameters using Optuna,
    with support for various search strategies and search space definitions.
    """
    
    def __init__(
        self,
        search_space: Dict[str, Any],
        objective_fn: Callable,
        direction: str = "minimize",
        search_algorithm: str = "tpe",
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        load_if_exists: bool = False,
        seed: Optional[int] = None,
        pruner_cls: Optional[str] = "median",
        pruner_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize hyperparameter optimization.
        
        Args:
            search_space: Search space definition for hyperparameters
                         (see create_search_space docs for format)
            objective_fn: Function to optimize, should take a trial object
                         and return the objective value
            direction: Optimization direction, 'minimize' or 'maximize'
            search_algorithm: Search algorithm ('tpe', 'random', 'grid', 'cmaes')
            n_trials: Number of trials to run
            timeout: Time limit in seconds for the optimization
            n_jobs: Number of parallel jobs
            study_name: Name of the study
            storage: Database URL for persistent storage
            load_if_exists: Whether to load an existing study
            seed: Random seed for reproducibility
            pruner_cls: Pruner class name ('median', 'percentile', 'hyperband', 'threshold', 'nop')
            pruner_kwargs: Additional arguments for the pruner
        """
        if not _OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter optimization. "
                             "Install it with `pip install optuna`.")
        
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.direction = direction
        self.search_algorithm = search_algorithm
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.study_name = study_name
        self.storage = storage
        self.load_if_exists = load_if_exists
        self.seed = seed
        
        # Set up pruner
        self.pruner_cls = pruner_cls
        self.pruner_kwargs = pruner_kwargs or {}
        
        # Initialize study
        self.study = self._setup_study()
        
        logger.info(f"Initialized hyperparameter optimization with {search_algorithm} algorithm")
    
    def _setup_study(self) -> "optuna.Study":
        """
        Set up Optuna study with the specified configuration.
        
        Returns:
            optuna.Study: Configured Optuna study
        """
        # Set up pruner
        pruner = self._setup_pruner()
        
        # Set up sampler
        sampler = self._setup_sampler()
        
        # Set up direction
        direction = (
            optuna.study.StudyDirection.MINIMIZE
            if self.direction == "minimize"
            else optuna.study.StudyDirection.MAXIMIZE
        )
        
        # Create or load the study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=self.load_if_exists,
            sampler=sampler,
            pruner=pruner,
            direction=direction,
        )
        
        return study
    
    def _setup_pruner(self) -> "optuna.pruners.BasePruner":
        """
        Set up Optuna pruner.
        
        Returns:
            optuna.pruners.BasePruner: Configured pruner
        """
        if self.pruner_cls == "median":
            return MedianPruner(**self.pruner_kwargs)
        elif self.pruner_cls == "percentile":
            return optuna.pruners.PercentilePruner(**self.pruner_kwargs)
        elif self.pruner_cls == "hyperband":
            return optuna.pruners.HyperbandPruner(**self.pruner_kwargs)
        elif self.pruner_cls == "threshold":
            return optuna.pruners.ThresholdPruner(**self.pruner_kwargs)
        elif self.pruner_cls == "nop":
            return NopPruner()
        else:
            logger.warning(f"Unknown pruner: {self.pruner_cls}. Using median pruner.")
            return MedianPruner()
    
    def _setup_sampler(self) -> "optuna.samplers.BaseSampler":
        """
        Set up Optuna sampler.
        
        Returns:
            optuna.samplers.BaseSampler: Configured sampler
        """
        if self.search_algorithm == "tpe":
            return TPESampler(seed=self.seed)
        elif self.search_algorithm == "random":
            return RandomSampler(seed=self.seed)
        elif self.search_algorithm == "grid":
            return optuna.samplers.GridSampler(self._create_grid())
        elif self.search_algorithm == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=self.seed)
        else:
            logger.warning(f"Unknown search algorithm: {self.search_algorithm}. Using TPE.")
            return TPESampler(seed=self.seed)
    
    def _create_grid(self) -> Dict[str, List[Any]]:
        """
        Create a grid for grid search based on the search space.
        
        Returns:
            Dict[str, List[Any]]: Grid for grid sampling
        """
        grid = {}
        
        for param_name, param_config in self.search_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get("type", "float")
                
                if param_type == "categorical":
                    grid[param_name] = param_config.get("choices", [])
                elif param_type in ["float", "int"]:
                    low = param_config.get("low", 0)
                    high = param_config.get("high", 1)
                    step = param_config.get("step", (high - low) / 10)
                    
                    if param_type == "int":
                        grid[param_name] = list(range(low, high + 1, max(1, int(step))))
                    else:
                        grid[param_name] = list(np.arange(low, high + step / 2, step))
            else:
                # If directly a list, assume it's a categorical parameter
                grid[param_name] = param_config if isinstance(param_config, list) else [param_config]
        
        return grid
    
    def _wrapped_objective(self, trial: "optuna.Trial") -> float:
        """
        Wrapped objective function that handles search space configuration.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            float: Objective value
        """
        # Suggest parameters from search space
        params = self.suggest_params(trial, self.search_space)
        
        # Call user-defined objective function with parameters
        return self.objective_fn(trial, params)
    
    def suggest_params(
        self,
        trial: "optuna.Trial",
        search_space: Dict[str, Any],
        prefix: str = "",
    ) -> Dict[str, Any]:
        """
        Suggest hyperparameters based on the search space definition.
        
        Args:
            trial: Optuna trial object
            search_space: Search space definition
            prefix: Prefix for parameter names (for nested parameters)
            
        Returns:
            Dict[str, Any]: Suggested hyperparameters
        """
        params = {}
        
        for param_name, param_config in search_space.items():
            full_param_name = f"{prefix}{param_name}" if prefix else param_name
            
            # Handle nested parameters
            if isinstance(param_config, dict) and "search_space" in param_config:
                nested_params = self.suggest_params(
                    trial, param_config["search_space"], prefix=f"{full_param_name}."
                )
                params[param_name] = nested_params
                continue
            
            # Handle normal parameters
            if isinstance(param_config, dict):
                param_type = param_config.get("type", "float")
                
                # Handle conditional parameters
                condition = param_config.get("condition", None)
                if condition is not None:
                    condition_param, condition_value = condition
                    if prefix:
                        condition_param = f"{prefix}{condition_param}"
                    
                    # Skip suggestion if condition not met
                    if condition_param not in trial.params or trial.params[condition_param] != condition_value:
                        continue
                
                if param_type == "categorical":
                    choices = param_config.get("choices", [])
                    params[param_name] = trial.suggest_categorical(full_param_name, choices)
                    
                elif param_type == "float":
                    low = param_config.get("low", 0.0)
                    high = param_config.get("high", 1.0)
                    log = param_config.get("log", False)
                    step = param_config.get("step", None)
                    
                    if step is not None:
                        # Discrete float suggestion
                        choices = list(np.arange(low, high + step / 2, step))
                        params[param_name] = trial.suggest_categorical(full_param_name, choices)
                    else:
                        # Continuous float suggestion
                        params[param_name] = trial.suggest_float(
                            full_param_name, low, high, log=log
                        )
                
                elif param_type == "int":
                    low = param_config.get("low", 0)
                    high = param_config.get("high", 10)
                    log = param_config.get("log", False)
                    step = param_config.get("step", 1)
                    
                    params[param_name] = trial.suggest_int(
                        full_param_name, low, high, step=step, log=log
                    )
                
                elif param_type == "bool":
                    params[param_name] = trial.suggest_categorical(full_param_name, [True, False])
            
            # Handle direct parameter values
            elif isinstance(param_config, list):
                # If a list is provided, treat it as categorical choices
                params[param_name] = trial.suggest_categorical(full_param_name, param_config)
            else:
                # If a single value is provided, use it directly
                params[param_name] = param_config
        
        return params
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Returns:
            Dict[str, Any]: Best parameters and optimization results
        """
        # Run optimization
        self.study.optimize(
            self._wrapped_objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )
        
        # Get best parameters and value
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        logger.info(f"Optimization completed. Best value: {best_value}")
        logger.info(f"Best parameters: {best_params}")
        
        return {
            "best_params": best_params,
            "best_value": best_value,
            "best_trial": self.study.best_trial,
            "study": self.study,
        }
    
    def save_study(self, path: Union[str, Path]) -> None:
        """
        Save the study to a file.
        
        Args:
            path: Path to save the study
        """
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        
        # Use pickle to save the study
        with open(path, "wb") as f:
            pickle.dump(self.study, f)
        
        logger.info(f"Study saved to {path}")
    
    @classmethod
    def load_study(cls, path: Union[str, Path]) -> "optuna.Study":
        """
        Load a study from a file.
        
        Args:
            path: Path to the saved study
            
        Returns:
            optuna.Study: Loaded study
        """
        path = Path(path)
        
        # Load the study from pickle
        with open(path, "rb") as f:
            study = pickle.load(f)
        
        logger.info(f"Study loaded from {path}")
        return study
    
    def plot_optimization_history(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot optimization history.
        
        Args:
            path: Path to save the plot (if None, plot is displayed)
        """
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization import plot_optimization_history
            
            fig = plot_optimization_history(self.study)
            
            if path is not None:
                path = Path(path)
                os.makedirs(path.parent, exist_ok=True)
                fig.write_image(str(path))
                logger.info(f"Optimization history plot saved to {path}")
            else:
                fig.show()
                
        except ImportError:
            logger.warning("Plotting requires matplotlib and plotly. Install them with "
                          "`pip install matplotlib plotly`.")
    
    def plot_param_importances(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot parameter importances.
        
        Args:
            path: Path to save the plot (if None, plot is displayed)
        """
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization import plot_param_importances
            
            fig = plot_param_importances(self.study)
            
            if path is not None:
                path = Path(path)
                os.makedirs(path.parent, exist_ok=True)
                fig.write_image(str(path))
                logger.info(f"Parameter importances plot saved to {path}")
            else:
                fig.show()
                
        except ImportError:
            logger.warning("Plotting requires matplotlib and plotly. Install them with "
                          "`pip install matplotlib plotly`.")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the optimization results.
        
        Returns:
            Dict[str, Any]: Optimization summary
        """
        return {
            "best_params": self.study.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials),
            "n_completed_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "direction": self.direction,
            "search_algorithm": self.search_algorithm,
        }


def create_search_space(
    parameter_ranges: Dict[str, Any],
    categorical_params: Optional[Dict[str, List[Any]]] = None,
    conditional_params: Optional[Dict[str, Tuple[str, Any]]] = None,
    nested_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Create a search space definition for hyperparameter optimization.
    
    Args:
        parameter_ranges: Dictionary mapping parameter names to (low, high) tuples
                         for numerical parameters, or single values for fixed parameters
        categorical_params: Dictionary mapping parameter names to lists of choices
        conditional_params: Dictionary mapping parameter names to (condition_param, condition_value) tuples
        nested_params: Dictionary mapping parameter names to nested parameter dictionaries
        
    Returns:
        Dict[str, Any]: Search space definition
    """
    search_space = {}
    
    # Process numerical parameters
    for param_name, param_range in parameter_ranges.items():
        if isinstance(param_range, tuple) and len(param_range) >= 2:
            # Numerical parameter with range
            low, high = param_range[:2]
            
            # Get parameter type and additional options
            param_type = "int" if isinstance(low, int) and isinstance(high, int) else "float"
            log = False
            step = None
            
            # Check for additional options
            if len(param_range) > 2:
                # Third element can be log scale flag or step size
                if isinstance(param_range[2], bool):
                    log = param_range[2]
                else:
                    step = param_range[2]
            
            if len(param_range) > 3:
                # Fourth element is step size if third was log flag
                step = param_range[3]
            
            search_space[param_name] = {
                "type": param_type,
                "low": low,
                "high": high,
                "log": log,
            }
            
            if step is not None:
                search_space[param_name]["step"] = step
        else:
            # Fixed parameter
            search_space[param_name] = param_range
    
    # Process categorical parameters
    if categorical_params:
        for param_name, choices in categorical_params.items():
            search_space[param_name] = {
                "type": "categorical",
                "choices": choices,
            }
    
    # Process conditional parameters
    if conditional_params:
        for param_name, condition in conditional_params.items():
            # If param name is not in search space yet, add it with default settings
            if param_name not in search_space:
                # Add parameter with default range
                search_space[param_name] = {
                    "type": "float",
                    "low": 0.0,
                    "high": 0.5,
                }
            
            # Add the condition to the parameter
            if isinstance(search_space[param_name], dict):
                search_space[param_name]["condition"] = condition
            else:
                # Convert fixed value to a dictionary with condition
                search_space[param_name] = {
                    "type": "categorical",
                    "choices": [search_space[param_name]],
                    "condition": condition,
                }
    
    # Process nested parameters
    if nested_params:
        for param_name, nested_space in nested_params.items():
            search_space[param_name] = {
                "search_space": nested_space,
            }
    
    return search_space


def optimize_hyperparameters(
    train_fn: Callable,
    eval_fn: Callable,
    search_space: Dict[str, Any],
    direction: str = "minimize",
    n_trials: int = 100,
    timeout: Optional[int] = None,
    search_algorithm: str = "tpe",
    pruner: str = "median",
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    n_jobs: int = 1,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Optimize hyperparameters for a model training process.
    
    Args:
        train_fn: Function that trains a model with given hyperparameters.
                 Should take hyperparameters as input and return a trained model.
        eval_fn: Function that evaluates a trained model and returns a metric.
                 Should take a model as input and return a float metric.
        search_space: Search space definition for hyperparameters.
        direction: Optimization direction, 'minimize' or 'maximize'.
        n_trials: Number of trials to run.
        timeout: Time limit in seconds for the optimization.
        search_algorithm: Search algorithm ('tpe', 'random', 'grid', 'cmaes').
        pruner: Pruner to use ('median', 'percentile', 'hyperband', 'threshold', 'nop').
        storage: Database URL for persistent storage.
        study_name: Name of the study.
        save_path: Path to save the optimization results.
        n_jobs: Number of parallel jobs.
        seed: Random seed for reproducibility.
        verbose: Whether to print progress and results.
        
    Returns:
        Dict[str, Any]: Optimization results, including best parameters and trained model.
    """
    if not _OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for hyperparameter optimization. "
                         "Install it with `pip install optuna`.")
    
    # Define the objective function
    def objective(trial, params):
        if verbose:
            logger.info(f"Trial {trial.number}: evaluating parameters {params}")
        
        try:
            # Train model with the suggested hyperparameters
            start_time = time.time()
            model = train_fn(params)
            train_time = time.time() - start_time
            
            # Evaluate the model
            eval_start_time = time.time()
            metric = eval_fn(model)
            eval_time = time.time() - eval_start_time
            
            if verbose:
                logger.info(f"Trial {trial.number}: metric={metric:.4f}, "
                          f"train_time={train_time:.2f}s, eval_time={eval_time:.2f}s")
            
            return metric
        
        except Exception as e:
            if verbose:
                logger.error(f"Trial {trial.number} failed: {e}")
            
            # Optuna expects Exceptions to be propagated
            raise
    
    # Create hyperparameter optimizer
    optimizer = HyperparameterOptimization(
        search_space=search_space,
        objective_fn=objective,
        direction=direction,
        search_algorithm=search_algorithm,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        study_name=study_name,
        storage=storage,
        pruner_cls=pruner,
        seed=seed,
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Save results if requested
    if save_path:
        optimizer.save_study(save_path)
    
    # Train final model with best parameters
    if verbose:
        logger.info(f"Training final model with best parameters: {results['best_params']}")
    
    best_model = train_fn(results["best_params"])
    
    # Return results
    return {
        "best_params": results["best_params"],
        "best_value": results["best_value"],
        "best_model": best_model,
        "study": results["study"],
        "optimizer": optimizer,
    }


class PruningCallback:
    """
    Callback for early stopping and pruning during training.
    
    This class provides a callback that can be used with training functions
    to enable early stopping and pruning during hyperparameter optimization.
    """
    
    def __init__(
        self,
        trial: "optuna.Trial",
        monitor: str = "val_loss",
        direction: str = "minimize",
    ):
        """
        Initialize pruning callback.
        
        Args:
            trial: Optuna trial object
            monitor: Metric to monitor
            direction: Optimization direction ('minimize' or 'maximize')
        """
        self.trial = trial
        self.monitor = monitor
        self.direction = direction
    
    def __call__(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        **kwargs,
    ) -> bool:
        """
        Call the callback during training.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
            model: Model being trained
            optimizer: Optimizer being used
            **kwargs: Additional arguments
            
        Returns:
            bool: Whether to stop training
        """
        # Skip if monitor metric not available
        if self.monitor not in metrics:
            logger.warning(f"Monitor metric '{self.monitor}' not in metrics. Skipping pruning check.")
            return False
        
        # Get monitor value
        monitor_value = metrics[self.monitor]
        
        # Report value to Optuna
        self.trial.report(monitor_value, epoch)
        
        # Check if trial should be pruned
        if self.trial.should_prune():
            logger.info(f"Trial pruned at epoch {epoch} with {self.monitor}={monitor_value:.4f}")
            return True
        
        return False