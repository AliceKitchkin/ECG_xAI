import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg_1D import VGG16_1D


def lrp_linear_eps_gamma(input, weight, bias, output, relevance, eps=1e-6, gamma=0):
    """
    Korrigierte LRP-ε/γ für Linear-Schichten basierend auf der ursprünglichen LRP-Formel
    """
    # LRP-gamma: modify weights
    w_pos = weight + gamma * weight.clamp(min=0)
    
    # Forward pass with modified weights
    z = F.linear(input, w_pos, bias)
    
    # Add epsilon for numerical stability
    z = z + eps * z.sign() + 1e-20
    
    # Backward pass: s = R_j / z_j
    s = relevance / z
    
    # Propagate relevance: R_i = sum_j(a_i * w_ij * s_j)
    # For linear layer: input * weight^T * s
    c = F.linear(s, w_pos.t())
    
    # Final relevance: element-wise multiplication
    return input * c

def lrp_conv1d_eps_gamma(input, module, relevance, eps=1e-6, gamma=0):
    """
    Korrigierte LRP-ε/γ für Conv1d basierend auf der ursprünglichen LRP-Formel
    """
    weight = module.weight
    bias = module.bias
    
    # LRP-gamma: modify weights
    w_pos = weight + gamma * weight.clamp(min=0)
    
    # Forward pass with modified weights
    z = F.conv1d(input, w_pos, bias, stride=module.stride, padding=module.padding, 
                dilation=module.dilation, groups=module.groups)
    
    # Ensure z and relevance have compatible shapes
    if z.shape != relevance.shape:
        if relevance.size(-1) != z.size(-1):
            relevance = F.interpolate(relevance.float(), size=z.size(-1), mode='linear', align_corners=False)
    
    # Add epsilon for numerical stability
    z = z + eps * z.sign() + 1e-20
    
    # Backward pass: s = R_j / z_j
    s = relevance / z
    
    # Propagate relevance: R_i = sum_j(a_i * w_ij * s_j)
    # For conv1d: use conv_transpose1d
    c = F.conv_transpose1d(s, w_pos, stride=module.stride, padding=module.padding, 
                          dilation=module.dilation, groups=module.groups)
    
    # Ensure output shape matches input shape
    if c.shape != input.shape:
        c = F.interpolate(c.float(), size=input.size(-1), mode='linear', align_corners=False)
    
    # Final relevance: element-wise multiplication
    return input * c

def lrp_maxpool1d(input, module, relevance):
    # For MaxPool1d, we need to "unpool" the relevance
    # Simple approach: use interpolation to match input size
    if relevance.shape != input.shape:
        relevance = F.interpolate(relevance.float(), size=input.size(-1), mode='linear', align_corners=False)
    return relevance

def lrp_sequential(x, modules, relevance, eps=1e-6, gamma=0):
    # LRP für nn.Sequential (umgekehrte Reihenfolge)
    activations = []
    
    # Forward pass to collect activations
    current_input = x
    for module in modules:
        activations.append(current_input)
        current_input = module(current_input)
    
    # Backward pass with LRP
    current_relevance = relevance
    for i, module in enumerate(reversed(modules)):
        activation_idx = len(modules) - 1 - i
        input_activation = activations[activation_idx]
        
        if isinstance(module, nn.ReLU):
            # ReLU: pass relevance through
            current_relevance = current_relevance
        elif isinstance(module, nn.MaxPool1d):
            # MaxPool: handle size changes
            current_relevance = lrp_maxpool1d(input_activation, module, current_relevance)
        elif isinstance(module, nn.Conv1d):
            current_relevance = lrp_conv1d_eps_gamma(input_activation, module, current_relevance, eps, gamma)
        elif isinstance(module, nn.Linear):
            current_relevance = lrp_linear_eps_gamma(input_activation, module.weight, module.bias, None, current_relevance, eps, gamma)
        elif isinstance(module, (nn.BatchNorm1d, nn.Dropout)):
            # Skip these layers - pass relevance through
            pass
        else:
            print(f"Warnung: LRP für {type(module)} nicht implementiert, überspringe...")
    
    return current_relevance

def lrp_vgg16_1d(model, x, target_class=None, eps=1e-6, gamma=0):
    model.eval()
    
    with torch.no_grad():
        output = model(x)
    
    if target_class is None:
        target_class = output.argmax(dim=1)
    
    # Ensure target_class is proper format
    if not torch.is_tensor(target_class):
        target_class = torch.tensor(target_class)
    target_class = target_class.to(dtype=torch.long)
    
    # Create one-hot relevance for target class
    one_hot = torch.zeros_like(output)
    one_hot[range(x.size(0)), target_class] = output[range(x.size(0)), target_class]
    relevance = one_hot
    
    # Get feature output for reshaping
    with torch.no_grad():
        feature_out = model.features(x)
    
    # LRP through classifier
    classifier_input = feature_out.view(x.size(0), -1)
    relevance = lrp_sequential(x=classifier_input,
                              modules=model.classifier,
                              relevance=relevance, eps=eps, gamma=gamma)
    
    # Reshape relevance back to feature map shape
    relevance = relevance.view_as(feature_out)
    
    # LRP through features
    relevance = lrp_sequential(x=x, modules=model.features, relevance=relevance, eps=eps, gamma=gamma)
    
    return relevance

# --- Beispielnutzung ---

if __name__ == "__main__":
    model = VGG16_1D(in_channels=12, num_classes=3, input_length=1000)
    x = torch.randn(1, 12, 1000)
    relevance = lrp_vgg16_1d(model, x, target_class=None, eps=1e-3, gamma=0.25)
    print("Relevance shape:", relevance.shape)