"""
Vision models for TAuto.

This module provides implementations and wrappers for common vision models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union

from tauto.models.registry import register_model
from tauto.utils import get_logger

logger = get_logger(__name__)

# Try to import torchvision, but provide fallbacks if not available
try:
    import torchvision
    from torchvision import models
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False
    logger.warning("torchvision not available. Limited vision models will be provided.")


class BasicConvNet(nn.Module):
    """
    Basic convolutional neural network for image classification.
    
    This model is a simple CNN with a few convolutional layers followed
    by fully connected layers. It's suitable for basic image classification
    tasks like MNIST, CIFAR-10, etc.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        hidden_channels: List[int] = [16, 32, 64],
        fc_dims: List[int] = [512, 128],
    ):
        """
        Initialize the model.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            hidden_channels: List of hidden channel dimensions for conv layers
            fc_dims: List of hidden dimensions for fully connected layers
        """
        super().__init__()
        
        # Create convolutional layers
        conv_layers = []
        channels = [in_channels] + hidden_channels
        
        for i in range(len(channels) - 1):
            conv_layers.extend([
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Create fully connected layers
        fc_layers = []
        
        # First FC layer takes input from flattened conv output
        # Size depends on input size and number of pooling layers
        # Assume input is 32x32 by default (e.g., CIFAR-10)
        # After 3 pooling layers with stride 2, we get 4x4 feature maps
        feature_size = 4 * 4 * hidden_channels[-1]
        
        fc_dims = [feature_size] + fc_dims + [num_classes]
        
        for i in range(len(fc_dims) - 1):
            fc_layers.extend([
                nn.Linear(fc_dims[i], fc_dims[i + 1]),
                nn.ReLU(inplace=True) if i < len(fc_dims) - 2 else nn.Identity(),
            ])
        
        self.fc_layers = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tensor: Output tensor of shape (B, num_classes)
        """
        # Pass through conv layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Pass through FC layers
        x = self.fc_layers(x)
        
        return x


# Register the basic conv model
register_model(
    name="basic_convnet",
    architecture="CNN",
    description="Basic convolutional neural network for image classification",
    task="image_classification",
    model_cls=BasicConvNet,
    input_size=(3, 32, 32),
    output_size=10,
    default_args={
        "in_channels": 3,
        "num_classes": 10,
        "hidden_channels": [16, 32, 64],
        "fc_dims": [512, 128],
    },
    pretrained_available=False,
)


# Register torchvision models if available
if _TORCHVISION_AVAILABLE:
    # Helper function to create torchvision models
    def _create_torchvision_model(model_fn, **kwargs):
        pretrained = kwargs.pop("pretrained", False)
        if pretrained:
            # Check if weights parameter is supported (newer torchvision)
            import inspect
            if "weights" in inspect.signature(model_fn).parameters:
                # Use the new API
                weights = "IMAGENET1K_V1" if pretrained else None
                return model_fn(weights=weights, **kwargs)
            else:
                # Use the old API
                return model_fn(pretrained=pretrained, **kwargs)
        else:
            return model_fn(**kwargs)
    
    # Register ResNet models
    register_model(
        name="resnet18",
        architecture="ResNet",
        description="ResNet-18 model from 'Deep Residual Learning for Image Recognition'",
        task="image_classification",
        factory_fn=lambda **kwargs: _create_torchvision_model(models.resnet18, **kwargs),
        input_size=(3, 224, 224),
        output_size=1000,
        paper_url="https://arxiv.org/abs/1512.03385",
        default_args={"pretrained": False},
        pretrained_available=True,
    )
    
    register_model(
        name="resnet50",
        architecture="ResNet",
        description="ResNet-50 model from 'Deep Residual Learning for Image Recognition'",
        task="image_classification",
        factory_fn=lambda **kwargs: _create_torchvision_model(models.resnet50, **kwargs),
        input_size=(3, 224, 224),
        output_size=1000,
        paper_url="https://arxiv.org/abs/1512.03385",
        default_args={"pretrained": False},
        pretrained_available=True,
        reference_speed={"inference_fp32": 23.0, "inference_fp16": 56.0},  # Examples in ms/batch
        reference_memory={"inference_fp32": 97.0, "inference_fp16": 53.0},  # Examples in MB
    )
    
    # Register EfficientNet models
    try:
        register_model(
            name="efficientnet_b0",
            architecture="EfficientNet",
            description="EfficientNet-B0 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks'",
            task="image_classification",
            factory_fn=lambda **kwargs: _create_torchvision_model(models.efficientnet_b0, **kwargs),
            input_size=(3, 224, 224),
            output_size=1000,
            paper_url="https://arxiv.org/abs/1905.11946",
            default_args={"pretrained": False},
            pretrained_available=True,
        )
    except AttributeError:
        # Old torchvision version without EfficientNet
        logger.warning("EfficientNet models not available in this torchvision version")
    
    # Register MobileNet models
    try:
        register_model(
            name="mobilenet_v2",
            architecture="MobileNet",
            description="MobileNetV2 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks'",
            task="image_classification",
            factory_fn=lambda **kwargs: _create_torchvision_model(models.mobilenet_v2, **kwargs),
            input_size=(3, 224, 224),
            output_size=1000,
            paper_url="https://arxiv.org/abs/1801.04381",
            default_args={"pretrained": False},
            pretrained_available=True,
            reference_speed={"inference_fp32": 9.0, "inference_fp16": 19.0},  # Examples in ms/batch
            reference_memory={"inference_fp32": 13.0, "inference_fp16": 7.0},  # Examples in MB
        )
    except AttributeError:
        # Old torchvision version without MobileNetV2
        logger.warning("MobileNetV2 model not available in this torchvision version")