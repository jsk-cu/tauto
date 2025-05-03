"""
Example usage of the TAuto model registry.

This script demonstrates how to use the model registry to:
1. List available models
2. Create models from the registry
3. Register custom models
4. Search for models with specific criteria
"""

import os
import sys
import torch
import torch.nn as nn

# Add parent directory to path to import tauto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tauto.models import (
    register_model,
    create_model,
    list_available_models,
    get_model_info,
)


def main():
    print("TAuto Model Registry Example")
    print("===========================")
    
    # List all available models
    print("\n1. Available Models")
    print("-----------------")
    models = list_available_models()
    print(f"Found {len(models)} registered models:")
    for model_name in models:
        model_info = get_model_info(model_name)
        print(f"  - {model_name} ({model_info.architecture}): {model_info.description}")
    
    # Create a model from the registry
    print("\n2. Creating Models")
    print("----------------")
    
    # Try to create a simple model
    if "basic_convnet" in models:
        print("Creating a basic CNN model:")
        model = create_model("basic_convnet", in_channels=1, num_classes=10)
        print(f"  Model created: {type(model).__name__}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test the model with a random input
        input_tensor = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            output = model(input_tensor)
        print(f"  Output shape: {output.shape}")
    
    # Try to create a pre-trained model if available
    if "resnet18" in models:
        print("\nCreating a pre-trained ResNet model:")
        model = create_model("resnet18", pretrained=True)
        print(f"  Model created: {type(model).__name__}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Register a custom model
    print("\n3. Registering Custom Models")
    print("--------------------------")
    
    class CustomMLP(nn.Module):
        def __init__(self, input_size=784, hidden_size=128, num_classes=10):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # Register the custom model
    register_model(
        name="custom_mlp",
        architecture="MLP",
        description="Custom MLP for MNIST classification",
        task="image_classification",
        model_cls=CustomMLP,
        input_size=(1, 28, 28),
        output_size=10,
        default_args={"input_size": 784, "hidden_size": 128, "num_classes": 10},
    )
    
    print("Registered a custom MLP model")
    
    # Verify that the model was registered
    print("Creating the custom model:")
    model = create_model("custom_mlp", hidden_size=256)
    print(f"  Model created: {type(model).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Search for models
    print("\n4. Searching for Models")
    print("---------------------")
    
    # Search by task
    image_models = list_available_models()
    image_models = [m for m in image_models if get_model_info(m).task == "image_classification"]
    print(f"Found {len(image_models)} image classification models:")
    for model_name in image_models:
        model_info = get_model_info(model_name)
        print(f"  - {model_name} ({model_info.architecture})")
    
    # Search by architecture
    cnn_models = list_available_models()
    cnn_models = [m for m in cnn_models if get_model_info(m).architecture == "CNN"]
    print(f"\nFound {len(cnn_models)} CNN models:")
    for model_name in cnn_models:
        model_info = get_model_info(model_name)
        print(f"  - {model_name}: {model_info.description}")
    
    # Search for pre-trained models
    pretrained_models = list_available_models()
    pretrained_models = [m for m in pretrained_models if get_model_info(m).pretrained_available]
    print(f"\nFound {len(pretrained_models)} models with pre-trained weights:")
    for model_name in pretrained_models:
        model_info = get_model_info(model_name)
        print(f"  - {model_name} ({model_info.architecture})")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()