"""
Training optimization utilities for TAuto.

This module provides tools for optimizing the training process of PyTorch models,
including mixed precision training, gradient accumulation, and optimizer factories.
"""

import os
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import time
from pathlib import Path
import warnings
import functools

from tauto.utils import get_logger

logger = get_logger(__name__)


class MixedPrecisionTraining:
    """
    Mixed precision training using PyTorch AMP.
    
    This class provides utilities for training with mixed precision,
    which can significantly speed up training on GPUs that support FP16.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        scale_factor: float = 2.0,
        growth_interval: int = 2000,
    ):
        """
        Initialize mixed precision training.
        
        Args:
            enabled: Whether to enable mixed precision training
            dtype: Data type to use for mixed precision
            scale_factor: Factor by which to scale gradients
            growth_interval: Number of steps between gradient scale increases
        """
        self.enabled = enabled
        self.dtype = dtype
        
        if self.enabled:
            self.scaler = GradScaler('cuda',
                init_scale=scale_factor,
                growth_interval=growth_interval,
            )
        else:
            self.scaler = None
        
        logger.info(f"Initialized mixed precision training: enabled={enabled}, dtype={dtype}")
    
    def step(
        self,
        model: nn.Module,
        inputs: Any,
        targets: Any,
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        clip_grad_norm: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Perform a training step with mixed precision.
        
        Args:
            model: PyTorch model
            inputs: Input data
            targets: Target data
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            clip_grad_norm: Gradient clipping norm
            
        Returns:
            Dict[str, Any]: Dictionary containing loss and other metrics
        """
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass with autocast if enabled
        if self.enabled:
            with autocast('cuda', dtype=self.dtype):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Backward pass with scaler
            self.scaler.scale(loss).backward()
            
            # Optionally clip gradients
            if clip_grad_norm is not None:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            # Update weights with scaler
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Standard training step
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Optionally clip gradients
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            optimizer.step()
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step()
        
        # Compute additional metrics
        with torch.no_grad():
            # For classification, compute accuracy
            if isinstance(outputs, torch.Tensor) and outputs.dim() > 1 and outputs.size(1) > 1:
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == targets).sum().item()
                accuracy = correct / targets.size(0)
            else:
                accuracy = None
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "grad_scale": self.scaler.get_scale() if self.enabled else None,
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the state dict for the mixed precision trainer.
        
        Returns:
            Dict[str, Any]: State dict
        """
        if self.enabled:
            return {
                "enabled": self.enabled,
                "dtype": self.dtype,
                "scaler": self.scaler.state_dict(),
            }
        else:
            return {
                "enabled": self.enabled,
                "dtype": self.dtype,
            }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state dict for the mixed precision trainer.
        
        Args:
            state_dict: State dict to load
        """
        self.enabled = state_dict["enabled"]
        self.dtype = state_dict["dtype"]
        
        if self.enabled and "scaler" in state_dict:
            if self.scaler is None:
                self.scaler = GradScaler('cuda')
            self.scaler.load_state_dict(state_dict["scaler"])


class GradientAccumulation:
    """
    Gradient accumulation for training with larger effective batch sizes.
    
    This class provides utilities for training with gradient accumulation,
    which allows training with larger effective batch sizes than would
    otherwise fit in memory.
    """
    
    def __init__(
        self,
        accumulation_steps: int = 1,
        clip_grad_norm: Optional[float] = None,
    ):
        """
        Initialize gradient accumulation.
        
        Args:
            accumulation_steps: Number of steps to accumulate gradients over
            clip_grad_norm: Gradient clipping norm
        """
        self.accumulation_steps = max(1, accumulation_steps)
        self.clip_grad_norm = clip_grad_norm
        self.current_step = 0
        
        logger.info(f"Initialized gradient accumulation: steps={accumulation_steps}")
    
    def step(
        self,
        model: nn.Module,
        inputs: Any,
        targets: Any,
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        mixed_precision: Optional[MixedPrecisionTraining] = None,
    ) -> Dict[str, Any]:
        """
        Perform a training step with gradient accumulation.
        
        Args:
            model: PyTorch model
            inputs: Input data
            targets: Target data
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            mixed_precision: Mixed precision trainer
            
        Returns:
            Dict[str, Any]: Dictionary containing loss and other metrics
        """
        # Track whether to update weights this step
        should_update = (self.current_step + 1) % self.accumulation_steps == 0
        last_batch = should_update
        
        # If this is the first step in an accumulation cycle, zero gradients
        if self.current_step % self.accumulation_steps == 0:
            optimizer.zero_grad()
        
        # Calculate loss scaling based on accumulation steps
        loss_scale = 1.0 / self.accumulation_steps
        
        # If using mixed precision, delegate to that
        if mixed_precision is not None and mixed_precision.enabled:
            with autocast('cuda', dtype=mixed_precision.dtype):
                outputs = model(inputs)
                loss = criterion(outputs, targets) * loss_scale
            
            # Scale and accumulate gradients
            mixed_precision.scaler.scale(loss).backward()
            
            # Update weights on the final accumulation step
            if should_update:
                if self.clip_grad_norm is not None:
                    mixed_precision.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
                
                mixed_precision.scaler.step(optimizer)
                mixed_precision.scaler.update()
                
                # Update learning rate if scheduler is provided
                if scheduler is not None:
                    scheduler.step()
                
                # Reset accumulation counter
                self.current_step = 0
            else:
                self.current_step += 1
        else:
            # Standard training with accumulation
            outputs = model(inputs)
            loss = criterion(outputs, targets) * loss_scale
            loss.backward()
            
            # Update weights on the final accumulation step
            if should_update:
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
                
                optimizer.step()
                
                # Update learning rate if scheduler is provided
                if scheduler is not None:
                    scheduler.step()
                
                # Reset accumulation counter
                self.current_step = 0
            else:
                self.current_step += 1
        
        # Compute additional metrics
        with torch.no_grad():
            # For classification, compute accuracy
            if isinstance(outputs, torch.Tensor) and outputs.dim() > 1 and outputs.size(1) > 1:
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == targets).sum().item()
                accuracy = correct / targets.size(0)
            else:
                accuracy = None
        
        return {
            "loss": (loss.item() / loss_scale),  # Rescale to get true loss
            "accuracy": accuracy,
            "accumulation_step": self.current_step,
            "weights_updated": last_batch,
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the state dict for the gradient accumulation trainer.
        
        Returns:
            Dict[str, Any]: State dict
        """
        return {
            "accumulation_steps": self.accumulation_steps,
            "clip_grad_norm": self.clip_grad_norm,
            "current_step": self.current_step,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state dict for the gradient accumulation trainer.
        
        Args:
            state_dict: State dict to load
        """
        self.accumulation_steps = state_dict["accumulation_steps"]
        self.clip_grad_norm = state_dict["clip_grad_norm"]
        self.current_step = state_dict["current_step"]


class ModelCheckpointing:
    """
    Model checkpointing utilities.
    
    This class provides utilities for saving and loading model checkpoints
    during training.
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        filename_pattern: str = "checkpoint_epoch_{epoch:03d}.pt",
        save_freq: int = 1,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        max_to_keep: Optional[int] = 5,
        save_best_only: bool = False,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        """
        Initialize model checkpointing.
        
        Args:
            checkpoint_dir: Directory to save checkpoints to
            filename_pattern: Pattern for checkpoint filenames
            save_freq: Frequency (in epochs) to save checkpoints
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
            max_to_keep: Maximum number of checkpoints to keep
            save_best_only: Whether to only save the best checkpoint
            monitor: Metric to monitor for determining the best checkpoint
            mode: Mode for determining the best checkpoint ('min' or 'max')
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.filename_pattern = filename_pattern
        self.save_freq = save_freq
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.max_to_keep = max_to_keep
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.checkpoints = []
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.info(f"Initialized model checkpointing: dir={checkpoint_dir}, save_freq={save_freq}")
    
    def save_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, Any]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """
        Save a model checkpoint.
        
        Args:
            model: PyTorch model
            epoch: Current epoch
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            metrics: Dictionary of metrics
            additional_data: Additional data to save
            
        Returns:
            Optional[Path]: Path to saved checkpoint, if saved
        """
        # Check if we should save this checkpoint
        if self.save_best_only and metrics is not None:
            current_value = metrics.get(self.monitor)
            if current_value is None:
                logger.warning(f"Metric '{self.monitor}' not found in metrics. Skipping checkpoint.")
                return None
            
            if self.mode == 'min' and current_value >= self.best_value:
                return None
            elif self.mode == 'max' and current_value <= self.best_value:
                return None
            
            # Update best value
            self.best_value = current_value
        elif epoch % self.save_freq != 0:
            return None
        
        # Create checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }
        
        if self.save_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if self.save_scheduler and scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        if additional_data is not None:
            checkpoint.update(additional_data)
        
        # Generate filename
        filename = self.filename_pattern.format(epoch=epoch)
        filepath = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
        
        # Add to list of checkpoints
        self.checkpoints.append(filepath)
        
        # Remove old checkpoints if needed
        if self.max_to_keep is not None and len(self.checkpoints) > self.max_to_keep:
            # Get the old checkpoints to remove
            checkpoints_to_remove = self.checkpoints[:-self.max_to_keep]
            # Update the list of checkpoints to keep only the newest ones
            self.checkpoints = self.checkpoints[-self.max_to_keep:]
            
            # Remove old checkpoints
            for old_checkpoint in checkpoints_to_remove:
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")
        
        return filepath
    
    def load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """
        Load a model checkpoint.
        
        Args:
            model: PyTorch model
            checkpoint_path: Path to checkpoint
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            map_location: Location to map tensors to
            
        Returns:
            Dict[str, Any]: Loaded checkpoint data
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available and requested
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available and requested
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """
        Find the latest checkpoint in the checkpoint directory.
        
        Returns:
            Optional[Path]: Path to latest checkpoint, or None if not found
        """
        checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        
        if not checkpoints:
            return None
        
        # Sort by modification time (newest first)
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return latest_checkpoint
    
    def find_best_checkpoint(self, metric_name: Optional[str] = None) -> Optional[Path]:
        """
        Find the best checkpoint based on a metric.
        
        Args:
            metric_name: Name of the metric to use (defaults to self.monitor)
            
        Returns:
            Optional[Path]: Path to best checkpoint, or None if not found
        """
        metric_name = metric_name or self.monitor
        checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        
        if not checkpoints:
            return None
        
        # Function to extract metric value from checkpoint
        def get_metric_value(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'metrics' in checkpoint and metric_name in checkpoint['metrics']:
                    return checkpoint['metrics'][metric_name]
                else:
                    # If metric not found, use epoch as fallback
                    return checkpoint.get('epoch', 0)
            except Exception:
                # If loading fails, use modification time as fallback
                return checkpoint_path.stat().st_mtime
        
        # Sort checkpoints by metric
        if self.mode == 'min':
            best_checkpoint = min(checkpoints, key=get_metric_value)
        else:
            best_checkpoint = max(checkpoints, key=get_metric_value)
        
        return best_checkpoint


class OptimizerFactory:
    """
    Factory for creating optimizers with best practices.
    
    This class provides utilities for creating optimizers with
    recommended settings for different optimization algorithms.
    """
    
    @staticmethod
    def create_optimizer(
        model: nn.Module,
        optimizer_type: str = "adam",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad: bool = False,
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        Create an optimizer with recommended settings.
        
        Args:
            model: PyTorch model
            optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw', etc.)
            learning_rate: Learning rate
            weight_decay: Weight decay factor
            momentum: Momentum factor (for SGD)
            beta1: Beta1 factor (for Adam-like optimizers)
            beta2: Beta2 factor (for Adam-like optimizers)
            eps: Epsilon factor (for numerical stability)
            amsgrad: Whether to use AMSGrad variant (for Adam-like optimizers)
            **kwargs: Additional arguments to pass to the optimizer
            
        Returns:
            torch.optim.Optimizer: Optimizer
        """
        # Get parameters that require gradients
        parameters = [p for p in model.parameters() if p.requires_grad]
        
        # Create optimizer based on type
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                parameters,
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                parameters,
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
                **kwargs
            )
        elif optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
                **kwargs
            )
        elif optimizer_type == "rmsprop":
            optimizer = torch.optim.RMSprop(
                parameters,
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                eps=eps,
                **kwargs
            )
        elif optimizer_type == "adagrad":
            optimizer = torch.optim.Adagrad(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                eps=eps,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        logger.info(f"Created {optimizer_type} optimizer with lr={learning_rate}, weight_decay={weight_decay}")
        return optimizer
    
    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = "cosine",
        epochs: int = 10,
        steps_per_epoch: Optional[int] = None,
        min_lr: float = 0.0,
        patience: int = 10,
        factor: float = 0.1,
        **kwargs
    ) -> Any:
        """
        Create a learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler ('cosine', 'step', etc.)
            epochs: Number of epochs
            steps_per_epoch: Number of steps per epoch (for OneCycle)
            min_lr: Minimum learning rate
            patience: Patience for ReduceLROnPlateau
            factor: Factor for ReduceLROnPlateau
            **kwargs: Additional arguments to pass to the scheduler
            
        Returns:
            Any: Learning rate scheduler
        """
        scheduler_type = scheduler_type.lower()
        
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=min_lr,
                **kwargs
            )
        elif scheduler_type == "step":
            step_size = kwargs.get("step_size", epochs // 3)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=factor,
                **kwargs
            )
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                **kwargs
            )
        elif scheduler_type == "onecycle":
            if steps_per_epoch is None:
                raise ValueError("steps_per_epoch must be provided for OneCycle scheduler")
            
            max_lr = optimizer.param_groups[0]["lr"]
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=steps_per_epoch * epochs,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        logger.info(f"Created {scheduler_type} scheduler")
        return scheduler


def train_with_optimization(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    criterion: Optional[Callable] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epochs: int = 10,
    device: Optional[torch.device] = None,
    mixed_precision: bool = False,
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: Optional[float] = None,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    checkpoint_freq: int = 1,
    early_stopping: bool = False,
    early_stopping_patience: int = 5,
    callbacks: Optional[List[Callable]] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Train a model with various optimization techniques.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epochs: Number of epochs to train for
        device: Device to train on
        mixed_precision: Whether to use mixed precision training
        gradient_accumulation_steps: Number of steps to accumulate gradients over
        clip_grad_norm: Gradient clipping norm
        checkpoint_dir: Directory to save checkpoints to
        checkpoint_freq: Frequency (in epochs) to save checkpoints
        early_stopping: Whether to use early stopping
        early_stopping_patience: Patience for early stopping
        callbacks: List of callbacks to call after each epoch
        verbose: Whether to print progress
        
    Returns:
        Dict[str, List[float]]: Dictionary of metrics over time
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model.to(device)
    
    # Set criterion if not provided
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Set optimizer if not provided
    if optimizer is None:
        optimizer = OptimizerFactory.create_optimizer(model, "adam")
    
    # Set up mixed precision training
    mp_trainer = MixedPrecisionTraining(enabled=mixed_precision)
    
    # Set up gradient accumulation
    grad_accumulator = GradientAccumulation(
        accumulation_steps=gradient_accumulation_steps,
        clip_grad_norm=clip_grad_norm,
    )
    
    # Set up checkpointing if requested
    checkpointer = None
    if checkpoint_dir is not None:
        checkpointer = ModelCheckpointing(
            checkpoint_dir=checkpoint_dir,
            save_freq=checkpoint_freq,
            save_optimizer=True,
            save_scheduler=scheduler is not None,
            save_best_only=True if val_loader is not None else False,
            monitor="val_loss" if val_loader is not None else "train_loss",
            mode="min",
        )
    
    # Initialize metrics
    metrics_history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": [],
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_samples = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Use gradient accumulation and mixed precision
            metrics = grad_accumulator.step(
                model=model,
                inputs=data,
                targets=target,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler if scheduler is not None and isinstance(
                    scheduler, torch.optim.lr_scheduler.OneCycleLR
                ) else None,
                mixed_precision=mp_trainer,
            )
            
            # Update metrics
            batch_size = data.size(0)
            train_loss += metrics["loss"] * batch_size
            if metrics.get("accuracy") is not None:
                train_acc += metrics["accuracy"] * batch_size
            train_samples += batch_size
            
            # Print progress
            if verbose and (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                examples_per_sec = train_samples / elapsed
                
                print(f"Epoch: {epoch+1}/{epochs} [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {metrics['loss']:.4f} "
                      f"Acc: {metrics.get('accuracy', 0):.4f} "
                      f"Examples/sec: {examples_per_sec:.1f}")
        
        # Update epoch-level metrics
        if train_samples > 0:
            train_loss /= train_samples
            train_acc /= train_samples
        
        metrics_history["train_loss"].append(train_loss)
        metrics_history["train_acc"].append(train_acc)
        
        # Get current learning rate
        if optimizer is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            metrics_history["learning_rate"].append(current_lr)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_samples = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    # Move data to device
                    data, target = data.to(device), target.to(device)
                    
                    # Forward pass
                    output = model(data)
                    loss = criterion(output, target)
                    
                    # Update metrics
                    batch_size = data.size(0)
                    val_loss += loss.item() * batch_size
                    
                    # Compute accuracy if applicable
                    if isinstance(output, torch.Tensor) and output.dim() > 1 and output.size(1) > 1:
                        _, predicted = torch.max(output.data, 1)
                        correct = (predicted == target).sum().item()
                        val_acc += correct
                    
                    val_samples += batch_size
            
            # Update epoch-level metrics
            if val_samples > 0:
                val_loss /= val_samples
                val_acc /= val_samples
            
            metrics_history["val_loss"].append(val_loss)
            metrics_history["val_acc"].append(val_acc)
            
            # Update learning rate for ReduceLROnPlateau
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Check for early stopping
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        # Break out of the epoch loop
                        break
        elif scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        # Print epoch summary
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} completed in {time.time() - start_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            if val_loader is not None:
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint if requested
        if checkpointer is not None:
            epoch_metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
            }
            
            if val_loader is not None:
                epoch_metrics.update({
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                })
            
            checkpointer.save_checkpoint(
                model=model,
                epoch=epoch + 1,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=epoch_metrics,
                additional_data={
                    "mixed_precision": mp_trainer.state_dict() if mixed_precision else None,
                    "gradient_accumulation": grad_accumulator.state_dict(),
                },
            )
        
        # Call callbacks if provided
        if callbacks is not None:
            for callback in callbacks:
                callback(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics={
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss if val_loader is not None else None,
                        "val_acc": val_acc if val_loader is not None else None,
                        "learning_rate": current_lr,
                    },
                )
    
    return metrics_history