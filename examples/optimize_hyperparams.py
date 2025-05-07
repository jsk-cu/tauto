"""
Example usage of the TAuto hyperparameter optimization utilities.

This script demonstrates how to use the hyperparameter optimization utilities to:
1. Define a search space for hyperparameters
2. Optimize hyperparameters for a PyTorch model
3. Use different search algorithms and pruning strategies
4. Visualize and analyze optimization results
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
from pathlib import Path
import numpy as np

# Add parent directory to path to import tauto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tauto.optimize.hyperparams import (
    HyperparameterOptimization,
    create_search_space,
    optimize_hyperparameters,
    PruningCallback,
)
from tauto.utils import get_logger

logger = get_logger(__name__)

# Check if Optuna is available
try:
    import optuna
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    print("Optuna not available. Install it with `pip install optuna` to run this example.")
    sys.exit(1)

# Try to import plotly for visualization
try:
    import plotly
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Install it with `pip install plotly` for visualization.")


class CNNModel(nn.Module):
    """CNN model for MNIST classification."""
    
    def __init__(
        self,
        in_channels=1,
        conv_channels=(16, 32),
        kernel_size=3,
        fc_units=128,
        dropout=0.5,
        activation='relu',
    ):
        super().__init__()
        
        # Save hyperparameters
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.fc_units = fc_units
        self.dropout = dropout
        self.activation = activation
        
        # Define activation function
        if activation == 'relu':
            self.act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            self.act_fn = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.act_fn = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, conv_channels[0], kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels[0])
        self.pool1 = nn.MaxPool2d(2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channels[1])
        self.pool2 = nn.MaxPool2d(2)
        
        # Calculate size after convolutions and pooling
        # For MNIST: 28x28 -> 14x14 -> 7x7
        feature_size = 7 * 7 * conv_channels[1]
        
        # Fully connected layers
        self.fc1 = nn.Linear(feature_size, fc_units)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_units, 10)
    
    def forward(self, x):
        x = self.act_fn(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.act_fn(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.act_fn(self.fc1(x))
        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        return x


def create_mnist_dataset(num_samples=1000, batch_size=32):
    """Create a synthetic MNIST-like dataset."""
    # Generate random 28x28 images and random labels
    images = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, 10, (num_samples,))
    
    # Create dataset and split into train/val
    dataset = TensorDataset(images, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, optimizer, criterion, device,
               epochs=5, early_stopping=True, patience=3, trial=None):
    """Train a model with early stopping."""
    model = model.to(device)
    
    # Initialize variables for training
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None
    
    # Create pruning callback if trial is provided
    pruning_callback = None
    if trial is not None:
        pruning_callback = PruningCallback(trial, monitor='val_loss', direction='minimize')
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            train_samples += batch_size
        
        # Calculate average training loss
        train_loss /= train_samples
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update metrics
                batch_size = inputs.size(0)
                val_loss += loss.item() * batch_size
                val_samples += batch_size
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate average validation loss and accuracy
        val_loss /= val_samples
        val_accuracy = val_correct / val_samples
        val_losses.append(val_loss)
        
        # Print epoch metrics
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Check pruning if callback is provided
        if pruning_callback is not None:
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }
            should_stop = pruning_callback(epoch, metrics, model, optimizer)
            if should_stop:
                logger.info(f"Trial pruned at epoch {epoch+1}")
                break
        
        # Check early stopping
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
    
    # Load best model if early stopping was used
    if early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses, 'best_val_loss': best_val_loss}


def objective(trial, params):
    """Objective function for hyperparameter optimization."""
    # Extract hyperparameters
    conv_channels = (params['conv1_channels'], params['conv2_channels'])
    kernel_size = params['kernel_size']
    fc_units = params['fc_units']
    dropout = params['dropout']
    activation = params['activation']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    optimizer_name = params['optimizer']
    
    # Create model with suggested hyperparameters
    model = CNNModel(
        in_channels=1,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        fc_units=fc_units,
        dropout=dropout,
        activation=activation,
    )
    
    # Create data loaders
    train_loader, val_loader = create_mnist_dataset(num_samples=1000, batch_size=batch_size)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create optimizer
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=params.get('momentum', 0.9))
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=params.get('weight_decay', 0.0))
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Set loss function
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=5,
        early_stopping=True,
        patience=3,
        trial=trial,
    )
    
    # Return best validation loss (to be minimized)
    return history['best_val_loss']


def example_basic_optimization():
    """Example of basic hyperparameter optimization."""
    print("\n1. Basic Hyperparameter Optimization Example")
    print("---------------------------------------")
    
    # Define search space
    parameter_ranges = {
        'conv1_channels': (8, 32, 8),  # (low, high, step)
        'conv2_channels': (16, 64, 16),
        'kernel_size': (3, 5, 1),
        'fc_units': (64, 256, 64),
        'learning_rate': (1e-4, 1e-2, True),  # (low, high, log)
        'batch_size': (16, 64, 16),
    }
    
    categorical_params = {
        'optimizer': ['sgd', 'adam', 'rmsprop'],
        'activation': ['relu', 'leaky_relu', 'elu'],
    }
    
    conditional_params = {
        'momentum': ('optimizer', 'sgd'),
        'weight_decay': ('optimizer', 'adam'),
        'dropout': ('activation', 'relu'),
    }
    
    # Create search space
    search_space = create_search_space(
        parameter_ranges=parameter_ranges,
        categorical_params=categorical_params,
        conditional_params=conditional_params,
    )
    
    # Add additional constraints or parameter values
    search_space['momentum'] = {
        'type': 'float',
        'low': 0.8,
        'high': 0.99,
        'condition': ('optimizer', 'sgd'),
    }
    
    search_space['weight_decay'] = {
        'type': 'float',
        'low': 1e-5,
        'high': 1e-3,
        'log': True,
        'condition': ('optimizer', 'adam'),
    }
    
    search_space['dropout'] = {
        'type': 'float',
        'low': 0.1,
        'high': 0.7,
    }
    
    # Create optimizer
    optimizer = HyperparameterOptimization(
        search_space=search_space,
        objective_fn=objective,
        direction='minimize',
        search_algorithm='tpe',
        n_trials=10,  # Use a small number for this example
        pruner_cls='median',
    )
    
    # Run optimization
    print("Running hyperparameter optimization with TPE algorithm...")
    results = optimizer.optimize()
    
    # Print results
    print("\nOptimization Results:")
    print(f"Best value (validation loss): {results['best_value']:.4f}")
    print("Best hyperparameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    
    # Plot optimization history if plotly is available
    if _PLOTLY_AVAILABLE:
        print("\nCreating visualization plots...")
        # Create directory for plots
        os.makedirs("optimization_plots", exist_ok=True)
        
        # Plot optimization history
        optimizer.plot_optimization_history(path="optimization_plots/optimization_history.png")
        
        # Plot parameter importances
        optimizer.plot_param_importances(path="optimization_plots/param_importances.png")
        
        print("Plots saved in 'optimization_plots' directory")


def example_compare_search_algorithms():
    """Example comparing different search algorithms."""
    print("\n2. Comparing Search Algorithms Example")
    print("-----------------------------------")
    
    # Define a simpler search space for faster execution
    search_space = {
        'conv1_channels': {
            'type': 'int',
            'low': 8,
            'high': 32,
            'step': 8,
        },
        'conv2_channels': {
            'type': 'int',
            'low': 16,
            'high': 64,
            'step': 16,
        },
        'fc_units': {
            'type': 'int',
            'low': 64,
            'high': 256,
            'step': 64,
        },
        'learning_rate': {
            'type': 'float',
            'low': 1e-4,
            'high': 1e-2,
            'log': True,
        },
        'batch_size': 32,  # Fixed batch size
        'optimizer': 'adam',  # Fixed optimizer
        'activation': 'relu',  # Fixed activation
        'dropout': {
            'type': 'float',
            'low': 0.1,
            'high': 0.5,
        },
        'kernel_size': 3,  # Fixed kernel size
    }
    
    # Set up algorithms to compare
    algorithms = ['tpe', 'random', 'cmaes']
    results = {}
    
    # Run optimization with each algorithm
    n_trials = 10  # Use a small number for this example
    
    for algorithm in algorithms:
        print(f"\nRunning optimization with {algorithm.upper()} algorithm...")
        
        # Create optimizer
        optimizer = HyperparameterOptimization(
            search_space=search_space,
            objective_fn=objective,
            direction='minimize',
            search_algorithm=algorithm,
            n_trials=n_trials,
            pruner_cls='median',
        )
        
        # Run optimization
        start_time = time.time()
        opt_results = optimizer.optimize()
        elapsed_time = time.time() - start_time
        
        # Store results
        results[algorithm] = {
            'best_value': opt_results['best_value'],
            'best_params': opt_results['best_params'],
            'time': elapsed_time,
            'optimizer': optimizer,
        }
        
        # Print results
        print(f"Best value with {algorithm.upper()}: {opt_results['best_value']:.4f}")
        print(f"Time taken: {elapsed_time:.2f}s")
    
    # Compare results
    print("\nSearch Algorithm Comparison:")
    print(f"{'Algorithm':<10} {'Best Value':<15} {'Time (s)':<10}")
    print("-" * 35)
    
    for algorithm, result in results.items():
        print(f"{algorithm.upper():<10} {result['best_value']:<15.4f} {result['time']:<10.2f}")
    
    # Plot comparison if matplotlib is available
    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot best values
        algorithms = list(results.keys())
        best_values = [results[algo]['best_value'] for algo in algorithms]
        times = [results[algo]['time'] for algo in algorithms]
        
        ax[0].bar(algorithms, best_values)
        ax[0].set_ylabel('Best Validation Loss')
        ax[0].set_title('Performance by Search Algorithm')
        
        ax[1].bar(algorithms, times)
        ax[1].set_ylabel('Time (s)')
        ax[1].set_title('Runtime by Search Algorithm')
        
        plt.tight_layout()
        plt.savefig("optimization_plots/algorithm_comparison.png")
        print("Comparison plot saved as 'optimization_plots/algorithm_comparison.png'")
    except Exception as e:
        print(f"Error creating comparison plot: {e}")


def example_pruning_strategies():
    """Example demonstrating different pruning strategies."""
    print("\n3. Pruning Strategies Example")
    print("---------------------------")
    
    # Define a simpler search space for faster execution
    search_space = {
        'conv1_channels': {
            'type': 'int',
            'low': 8,
            'high': 32,
            'step': 8,
        },
        'conv2_channels': {
            'type': 'int',
            'low': 16,
            'high': 64,
            'step': 16,
        },
        'learning_rate': {
            'type': 'float',
            'low': 1e-4,
            'high': 1e-2,
            'log': True,
        },
        'batch_size': 32,
        'fc_units': 128,
        'dropout': 0.5,
        'optimizer': 'adam',
        'activation': 'relu',
        'kernel_size': 3,
    }
    
    # Set up pruners to compare
    pruners = [
        ('median', {'n_startup_trials': 5, 'n_warmup_steps': 1}),
        ('percentile', {'percentile': 75, 'n_startup_trials': 5, 'n_warmup_steps': 1}),
        ('hyperband', {'min_resource': 1, 'max_resource': 5, 'reduction_factor': 3}),
        ('nop', {}),  # No pruning
    ]
    
    results = {}
    n_trials = 15  # Use more trials to see pruning effects
    
    for pruner_name, pruner_kwargs in pruners:
        print(f"\nRunning optimization with {pruner_name.upper()} pruner...")
        
        # Create optimizer
        optimizer = HyperparameterOptimization(
            search_space=search_space,
            objective_fn=objective,
            direction='minimize',
            search_algorithm='tpe',
            n_trials=n_trials,
            pruner_cls=pruner_name,
            pruner_kwargs=pruner_kwargs,
        )
        
        # Run optimization
        start_time = time.time()
        opt_results = optimizer.optimize()
        elapsed_time = time.time() - start_time
        
        # Store results
        study = opt_results['study']
        pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        
        results[pruner_name] = {
            'best_value': opt_results['best_value'],
            'time': elapsed_time,
            'pruned_trials': pruned_trials,
            'completed_trials': completed_trials,
        }
        
        # Print results
        print(f"Best value: {opt_results['best_value']:.4f}")
        print(f"Pruned trials: {pruned_trials}/{n_trials} ({pruned_trials/n_trials*100:.1f}%)")
        print(f"Time taken: {elapsed_time:.2f}s")
    
    # Compare results
    print("\nPruning Strategy Comparison:")
    print(f"{'Pruner':<15} {'Best Value':<15} {'Pruned Trials':<15} {'Time (s)':<10}")
    print("-" * 55)
    
    for pruner_name, result in results.items():
        print(f"{pruner_name.upper():<15} {result['best_value']:<15.4f} "
              f"{result['pruned_trials']}/{n_trials} "
              f"{result['time']:<10.2f}")
    
    # Plot comparison if matplotlib is available
    try:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot best values
        pruners = list(results.keys())
        best_values = [results[p]['best_value'] for p in pruners]
        times = [results[p]['time'] for p in pruners]
        pruned_pcts = [results[p]['pruned_trials'] / n_trials * 100 for p in pruners]
        
        ax[0].bar(pruners, best_values)
        ax[0].set_ylabel('Best Validation Loss')
        ax[0].set_title('Performance by Pruning Strategy')
        
        ax[1].bar(pruners, times)
        ax[1].set_ylabel('Time (s)')
        ax[1].set_title('Runtime by Pruning Strategy')
        
        ax[2].bar(pruners, pruned_pcts)
        ax[2].set_ylabel('Pruned Trials (%)')
        ax[2].set_title('Pruning Rate by Strategy')
        
        plt.tight_layout()
        plt.savefig("optimization_plots/pruner_comparison.png")
        print("Comparison plot saved as 'optimization_plots/pruner_comparison.png'")
    except Exception as e:
        print(f"Error creating comparison plot: {e}")


def example_end_to_end_optimization():
    """Example demonstrating end-to-end optimization and model training."""
    print("\n4. End-to-End Optimization Example")
    print("--------------------------------")
    
    # Define search space for model and training parameters
    parameter_ranges = {
        'conv1_channels': (8, 32, 8),
        'conv2_channels': (16, 64, 16),
        'fc_units': (64, 256, 64),
        'learning_rate': (1e-4, 1e-2, True),
        'batch_size': (16, 64, 16),
    }
    
    categorical_params = {
        'optimizer': ['sgd', 'adam', 'rmsprop'],
        'activation': ['relu', 'leaky_relu', 'elu'],
    }
    
    # Create search space
    search_space = create_search_space(
        parameter_ranges=parameter_ranges,
        categorical_params=categorical_params,
    )
    
    # Additional parameters
    search_space['dropout'] = {
        'type': 'float',
        'low': 0.1,
        'high': 0.7,
    }
    
    search_space['kernel_size'] = {
        'type': 'int',
        'low': 3,
        'high': 5,
        'step': 2,
    }
    
    # Define functions for optimize_hyperparameters
    def train_fn(params):
        """Function that trains a model with given hyperparameters."""
        # Extract hyperparameters
        conv_channels = (params['conv1_channels'], params['conv2_channels'])
        kernel_size = params['kernel_size']
        fc_units = params['fc_units']
        dropout = params['dropout']
        activation = params['activation']
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']
        optimizer_name = params['optimizer']
        
        # Create model
        model = CNNModel(
            in_channels=1,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            fc_units=fc_units,
            dropout=dropout,
            activation=activation,
        )
        
        # Create data loaders
        train_loader, val_loader = create_mnist_dataset(num_samples=1000, batch_size=batch_size)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create optimizer
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        
        # Set loss function
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        model, _ = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=5,
            early_stopping=True,
            patience=3,
            trial=None,  # No trial for final training
        )
        
        return model
    
    def eval_fn(model):
        """Function that evaluates a trained model."""
        # Create data loaders for evaluation
        _, val_loader = create_mnist_dataset(num_samples=1000, batch_size=32)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Set loss function
        criterion = nn.CrossEntropyLoss()
        
        # Evaluate model
        model.eval()
        val_loss = 0.0
        val_samples = 0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update metrics
                batch_size = inputs.size(0)
                val_loss += loss.item() * batch_size
                val_samples += batch_size
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate average validation loss and accuracy
        val_loss /= val_samples
        val_accuracy = val_correct / val_samples
        
        print(f"Evaluation - Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        return val_loss
    
    # Create directory for saving results
    os.makedirs("optimization_results", exist_ok=True)
    
    # Run hyperparameter optimization
    print("Running end-to-end hyperparameter optimization...")
    results = optimize_hyperparameters(
        train_fn=train_fn,
        eval_fn=eval_fn,
        search_space=search_space,
        direction='minimize',
        n_trials=10,  # Use a small number for this example
        search_algorithm='tpe',
        pruner='median',
        save_path="optimization_results/end_to_end_study.pkl",
        verbose=True,
    )
    
    # Print results
    print("\nOptimization Complete!")
    print(f"Best validation loss: {results['best_value']:.4f}")
    print("Best hyperparameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    
    # Save best model
    best_model = results['best_model']
    torch.save(best_model.state_dict(), "optimization_results/best_model.pt")
    print("Best model saved as 'optimization_results/best_model.pt'")
    
    # Print optimization summary
    optimizer = results['optimizer']
    summary = optimizer.get_optimization_summary()
    
    print("\nOptimization Summary:")
    print(f"Trials: {summary['n_trials']} ({summary['n_completed_trials']} completed, "
          f"{summary['n_pruned_trials']} pruned)")
    print(f"Direction: {summary['direction']}")
    print(f"Search algorithm: {summary['search_algorithm']}")


def main():
    """Run the hyperparameter optimization examples."""
    print("TAuto Hyperparameter Optimization Examples")
    print("========================================")
    
    # Create directory for plots
    os.makedirs("optimization_plots", exist_ok=True)
    
    # Run examples
    try:
        # Start with basic example
        example_basic_optimization()
        
        # Ask if user wants to run more examples (these can be time-consuming)
        response = input("\nRun more examples? (y/n): ")
        if response.lower() != 'y':
            print("Exiting after basic example.")
            return
        
        # Run additional examples
        example_compare_search_algorithms()
        example_pruning_strategies()
        example_end_to_end_optimization()
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError running examples: {e}")
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main()