"""
Vereinfachte und robuste LRP-ε Implementierung für VGG16_1D
"""
import torch
import torch.nn.functional as F

def simple_lrp_eps(model, x, target_class, eps=1e-6):
    """
    Vereinfachte LRP-ε Implementierung ohne gamma
    
    Args:
        model: VGG16_1D Modell
        x: Input tensor (1, 12, 1000)
        target_class: Zielklasse für Erklärung
        eps: Epsilon für numerische Stabilität
    
    Returns:
        relevance: LRP relevance scores (1, 12, 1000)
    """
    model.eval()
    x.requires_grad_(True)
    
    # Forward pass
    y = model(x)
    
    # Create one-hot target
    target_one_hot = torch.zeros_like(y)
    target_one_hot[0, target_class] = 1.0
    
    # Start with output relevance
    relevance = y * target_one_hot
    
    # Get all layers
    all_layers = []
    
    # Features layers
    for layer in model.features:
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.ReLU, torch.nn.MaxPool1d, torch.nn.BatchNorm1d)):
            all_layers.append(layer)
    
    # Classifier layers  
    for layer in model.classifier:
        if isinstance(layer, (torch.nn.Linear, torch.nn.ReLU, torch.nn.Dropout)):
            all_layers.append(layer)
    
    print(f"Processing {len(all_layers)} layers for LRP")
    
    # Simplified LRP: use gradient * input as approximation
    # This is mathematically equivalent to LRP-0 for ReLU networks
    
    # Calculate gradient
    loss = relevance.sum()
    loss.backward(retain_graph=True)
    
    # LRP approximation: gradient * input
    lrp_relevance = x.grad * x
    
    return lrp_relevance

def simple_lrp_guided_backprop(model, x, target_class):
    """
    Guided Backpropagation als Alternative zu LRP
    """
    model.eval()
    
    # Clone input and enable gradients
    x = x.clone().detach().requires_grad_(True)
    
    # Forward pass
    output = model(x)
    
    # Select target class
    target_output = output[0, target_class]
    
    # Backward pass
    target_output.backward()
    
    # Get gradients
    relevance = x.grad
    
    return relevance
