"""
Knowledge distillation utilities for TAuto.

This module provides tools for applying knowledge distillation techniques
to transfer knowledge from larger teacher models to smaller student models.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import copy
import warnings
from pathlib import Path

from tauto.utils import get_logger

logger = get_logger(__name__)


class KnowledgeDistillation:
    """
    Knowledge distillation for transferring knowledge from teacher to student models.
    
    This class provides utilities for training a smaller model (student)
    to mimic the behavior of a larger model (teacher).
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5,
        feature_layers: Optional[Dict[str, str]] = None,
        feature_weight: float = 0.0,
    ):
        """
        Initialize knowledge distillation.
        
        Args:
            teacher_model: Teacher model
            student_model: Student model
            temperature: Temperature for softening logits
            alpha: Weight for distillation loss (1-alpha for task loss)
            feature_layers: Dictionary mapping teacher layer names to student layer names
                           for feature distillation
            feature_weight: Weight for feature distillation loss
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.feature_layers = feature_layers
        self.feature_weight = feature_weight
        
        # Set teacher model to eval mode
        self.teacher_model.eval()
        
        logger.info(f"Initialized knowledge distillation: T={temperature}, alpha={alpha}")
    
    def get_feature_hooks(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Set up feature hooks for intermediate layer distillation.
        
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Teacher and student hooks
        """
        if not self.feature_layers:
            return {}, {}
        
        teacher_features = {}
        student_features = {}
        teacher_hooks = {}
        student_hooks = {}
        
        # Define hook function
        def get_features(name, features_dict):
            def hook(model, input, output):
                features_dict[name] = output
            return hook
        
        # Register hooks for teacher
        for t_name, _ in self.feature_layers.items():
            layer = self._get_layer_by_name(self.teacher_model, t_name)
            if layer is not None:
                teacher_hooks[t_name] = layer.register_forward_hook(
                    get_features(t_name, teacher_features)
                )
            else:
                logger.warning(f"Teacher layer {t_name} not found")
        
        # Register hooks for student
        for _, s_name in self.feature_layers.items():
            layer = self._get_layer_by_name(self.student_model, s_name)
            if layer is not None:
                student_hooks[s_name] = layer.register_forward_hook(
                    get_features(s_name, student_features)
                )
            else:
                logger.warning(f"Student layer {s_name} not found")
        
        return teacher_features, student_features
    
    def remove_hooks(self, hooks: Dict[str, Any]) -> None:
        """
        Remove hooks from models.
        
        Args:
            hooks: Dictionary of hooks to remove
        """
        for hook in hooks.values():
            hook.remove()
    
    def _get_layer_by_name(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        """
        Get a layer from a model by name.
        
        Args:
            model: PyTorch model
            name: Layer name
            
        Returns:
            Optional[nn.Module]: Layer if found, None otherwise
        """
        for n, m in model.named_modules():
            if n == name:
                return m
        return None
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        criterion: Callable,
        teacher_features: Optional[Dict[str, torch.Tensor]] = None,
        student_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss combining task loss and mimicry loss.
        
        Args:
            student_logits: Logits from the student model
            teacher_logits: Logits from the teacher model
            targets: Ground truth targets
            criterion: Loss function for task loss
            teacher_features: Features from the teacher model
            student_features: Features from the student model
            
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Total loss and loss components
        """
        # Task loss - regular cross-entropy with ground truth
        task_loss = criterion(student_logits, targets)
        
        # Distillation loss - KL divergence with teacher predictions
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = F.kl_div(log_probs, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        
        # Feature distillation loss
        feature_loss = 0.0
        if self.feature_weight > 0 and teacher_features and student_features:
            # Compute MSE loss between corresponding features
            for t_name, s_name in self.feature_layers.items():
                if t_name in teacher_features and s_name in student_features:
                    t_feature = teacher_features[t_name]
                    s_feature = student_features[s_name]
                    
                    # Adapt dimensions if needed
                    if t_feature.shape != s_feature.shape:
                        # For convolutional features (B, C, H, W)
                        if len(t_feature.shape) == 4:
                            # Match spatial dimensions using adaptive pooling
                            if t_feature.shape[2:] != s_feature.shape[2:]:
                                s_feature = F.adaptive_avg_pool2d(
                                    s_feature, 
                                    output_size=t_feature.shape[2:]
                                )
                            
                            # If channel dimensions don't match, use 1x1 conv to adapt
                            if t_feature.shape[1] != s_feature.shape[1]:
                                # Create a temporary 1x1 conv to match channels
                                channel_adapter = nn.Conv2d(
                                    s_feature.shape[1], 
                                    t_feature.shape[1],
                                    kernel_size=1, 
                                    bias=False
                                ).to(s_feature.device)
                                
                                # Initialize with identity-like mapping for minimal distortion
                                nn.init.kaiming_normal_(channel_adapter.weight)
                                
                                # Apply channel adaptation
                                s_feature = channel_adapter(s_feature)
                        
                        # For fully-connected features (B, F)
                        elif len(t_feature.shape) == 2:
                            if t_feature.shape[1] != s_feature.shape[1]:
                                # Use a linear layer to adapt feature dimensions
                                feature_adapter = nn.Linear(
                                    s_feature.shape[1],
                                    t_feature.shape[1],
                                    bias=False
                                ).to(s_feature.device)
                                
                                # Initialize with identity-like mapping
                                nn.init.xavier_uniform_(feature_adapter.weight)
                                
                                # Apply feature adaptation
                                s_feature = feature_adapter(s_feature)
                    
                    # Compute MSE loss
                    feature_loss += F.mse_loss(s_feature, t_feature)
        
        # Combine losses
        total_loss = (1 - self.alpha) * task_loss + self.alpha * distillation_loss
        
        # Add feature distillation loss if enabled
        if self.feature_weight > 0:
            total_loss += self.feature_weight * feature_loss
        
        # Return total loss and components
        loss_components = {
            "task_loss": task_loss.item(),
            "distillation_loss": distillation_loss.item(),
            "feature_loss": feature_loss if isinstance(feature_loss, float) else feature_loss.item(),
            "total_loss": total_loss.item(),
        }
        
        return total_loss, loss_components
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Perform a training step with knowledge distillation.
        
        Args:
            inputs: Input data
            targets: Target data
            optimizer: Optimizer for the student model
            criterion: Loss function for task loss
            scheduler: Learning rate scheduler
            
        Returns:
            Dict[str, float]: Training metrics
        """
        # Set teacher to eval, student to train
        self.teacher_model.eval()
        self.student_model.train()
        
        # Set up feature hooks if needed
        teacher_features, student_features = {}, {}
        teacher_hooks, student_hooks = {}, {}
        
        if self.feature_weight > 0 and self.feature_layers:
            teacher_features, student_features = self.get_feature_hooks()
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass through teacher model
        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
        
        # Forward pass through student model
        student_logits = self.student_model(inputs)
        
        # Compute distillation loss
        loss, loss_components = self.distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            targets=targets,
            criterion=criterion,
            teacher_features=teacher_features,
            student_features=student_features,
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step()
        
        # Remove hooks if they were set up
        if teacher_hooks:
            self.remove_hooks(teacher_hooks)
        if student_hooks:
            self.remove_hooks(student_hooks)
        
        # Compute metrics
        with torch.no_grad():
            # Compute accuracy
            _, student_predicted = torch.max(student_logits, 1)
            correct = (student_predicted == targets).sum().item()
            accuracy = correct / targets.size(0)
        
        # Update metrics
        metrics = {
            **loss_components,
            "accuracy": accuracy,
        }
        
        return metrics
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        device: torch.device,
        scheduler: Optional[Any] = None,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch with knowledge distillation.
        
        Args:
            train_loader: DataLoader for training data
            optimizer: Optimizer for the student model
            criterion: Loss function for task loss
            device: Device to train on
            scheduler: Learning rate scheduler
            val_loader: DataLoader for validation data
            
        Returns:
            Dict[str, float]: Training and validation metrics
        """
        # Move models to device
        self.teacher_model = self.teacher_model.to(device)
        self.student_model = self.student_model.to(device)
        
        # Training metrics
        train_loss = 0.0
        train_accuracy = 0.0
        samples = 0
        
        # Train loop
        self.teacher_model.eval()
        self.student_model.train()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Training step
            metrics = self.train_step(
                inputs=inputs,
                targets=targets,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) else None,
            )
            
            # Update metrics
            batch_size = inputs.size(0)
            train_loss += metrics["total_loss"] * batch_size
            train_accuracy += metrics["accuracy"] * batch_size
            samples += batch_size
        
        # Calculate average metrics
        train_loss /= samples
        train_accuracy /= samples
        
        epoch_metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
        }
        
        # Validation if provided
        if val_loader is not None:
            val_metrics = self.evaluate(
                val_loader=val_loader,
                criterion=criterion,
                device=device,
            )
            epoch_metrics.update(val_metrics)
            
            # Update learning rate for non-OneCycle schedulers
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["val_loss"])
                else:
                    scheduler.step()
        elif scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        return epoch_metrics
    
    def evaluate(
        self,
        val_loader: torch.utils.data.DataLoader,
        criterion: Callable,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Evaluate the student model.
        
        Args:
            val_loader: DataLoader for validation data
            criterion: Loss function
            device: Device to evaluate on
            
        Returns:
            Dict[str, float]: Validation metrics
        """
        # Set models to eval mode
        self.teacher_model.eval()
        self.student_model.eval()
        
        # Validation metrics
        val_loss = 0.0
        val_accuracy = 0.0
        samples = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                teacher_logits = self.teacher_model(inputs)
                student_logits = self.student_model(inputs)
                
                # Compute loss
                loss, _ = self.distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    targets=targets,
                    criterion=criterion,
                )
                
                # Compute accuracy
                _, predicted = torch.max(student_logits, 1)
                correct = (predicted == targets).sum().item()
                
                # Update metrics
                batch_size = inputs.size(0)
                val_loss += loss.item() * batch_size
                val_accuracy += correct
                samples += batch_size
        
        # Calculate average metrics
        val_loss /= samples
        val_accuracy /= samples
        
        return {
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        device: torch.device,
        epochs: int = 10,
        scheduler: Optional[Any] = None,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        patience: int = 5,
        verbose: bool = True,
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """
        Train the student model with knowledge distillation.
        
        Args:
            train_loader: DataLoader for training data
            optimizer: Optimizer for the student model
            criterion: Loss function for task loss
            device: Device to train on
            epochs: Number of epochs to train for
            scheduler: Learning rate scheduler
            val_loader: DataLoader for validation data
            checkpoint_dir: Directory to save checkpoints
            patience: Patience for early stopping
            verbose: Whether to print progress
            
        Returns:
            Tuple[nn.Module, Dict[str, List[float]]]: Trained student model and metrics
        """
        # Move models to device
        self.teacher_model = self.teacher_model.to(device)
        self.student_model = self.student_model.to(device)
        
        # Initialize metrics
        metrics_history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Create checkpoint directory if needed
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train for one epoch
            epoch_metrics = self.train_epoch(
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scheduler=scheduler,
                val_loader=val_loader,
            )
            
            # Update metrics history
            metrics_history["train_loss"].append(epoch_metrics["train_loss"])
            metrics_history["train_accuracy"].append(epoch_metrics["train_accuracy"])
            
            if "val_loss" in epoch_metrics:
                metrics_history["val_loss"].append(epoch_metrics["val_loss"])
                metrics_history["val_accuracy"].append(epoch_metrics["val_accuracy"])
                
                # Check for early stopping
                if epoch_metrics["val_loss"] < best_val_loss:
                    best_val_loss = epoch_metrics["val_loss"]
                    patience_counter = 0
                    
                    # Save the best model state
                    best_model_state = copy.deepcopy(self.student_model.state_dict())
                    
                    # Save checkpoint if requested
                    if checkpoint_dir is not None:
                        checkpoint_path = checkpoint_dir / f"student_model_best.pt"
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': self.student_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                            'loss': best_val_loss,
                            'metrics': epoch_metrics,
                        }, checkpoint_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Print progress
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} completed in {time.time() - start_time:.2f}s")
                print(f"  Train Loss: {epoch_metrics['train_loss']:.4f}, Train Acc: {epoch_metrics['train_accuracy']:.4f}")
                
                if "val_loss" in epoch_metrics:
                    print(f"  Val Loss: {epoch_metrics['val_loss']:.4f}, Val Acc: {epoch_metrics['val_accuracy']:.4f}")
        
        # Load the best model if available
        if best_model_state is not None:
            self.student_model.load_state_dict(best_model_state)
        
        return self.student_model, metrics_history


def distill_model(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[Callable] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    epochs: int = 10,
    temperature: float = 2.0,
    alpha: float = 0.5,
    feature_layers: Optional[Dict[str, str]] = None,
    feature_weight: float = 0.0,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    patience: int = 5,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Distill knowledge from a teacher model to a student model.
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for the student model
        criterion: Loss function for task loss
        scheduler: Learning rate scheduler
        device: Device to train on
        epochs: Number of epochs to train for
        temperature: Temperature for softening logits
        alpha: Weight for distillation loss (1-alpha for task loss)
        feature_layers: Dictionary mapping teacher layer names to student layer names
                      for feature distillation
        feature_weight: Weight for feature distillation loss
        checkpoint_dir: Directory to save checkpoints
        patience: Patience for early stopping
        verbose: Whether to print progress
        
    Returns:
        Tuple[nn.Module, Dict[str, List[float]]]: Trained student model and metrics
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set criterion
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Set optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    
    # Set up distillation
    distiller = KnowledgeDistillation(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=temperature,
        alpha=alpha,
        feature_layers=feature_layers,
        feature_weight=feature_weight,
    )
    
    # Train the student
    trained_student, metrics = distiller.train(
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs,
        scheduler=scheduler,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        patience=patience,
        verbose=verbose,
    )
    
    return trained_student, metrics