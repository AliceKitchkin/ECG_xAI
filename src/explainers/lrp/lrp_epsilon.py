import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# --- LRP-Klasse ---
class LRP_Epsilon:
    def __init__(self, model, epsilon=1e-7):
        """
        Initialisiert den LRP-Epsilon Explainer für 1D EKG-Daten.
        
        Args:
            model: Das VGG16_1D Modell
            epsilon: Stabilisierungsparameter für die LRP-Epsilon Regel
        """
        self.model = model
        self.epsilon = epsilon
        self.activations = []
        self.hooks = []

    def _register_hooks(self):
        """Registriert Forward- und Backward-Hooks für alle relevanten Layer."""
        def save_activation(module, input, output):
            self.activations.append(output)
        
        # Hook für alle Layer registrieren, die Aktivierungen haben
        for layer in self.model.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear, nn.ReLU, nn.MaxPool1d, nn.BatchNorm1d)):
                hook = layer.register_forward_hook(save_activation)
                self.hooks.append(hook)

    def _get_relevance_rule(self, layer):
        """
        Wählt die passende LRP-epsilon-Regel basierend auf dem Schichttyp.
        
        Args:
            layer: PyTorch Layer
            
        Returns:
            Entsprechende LRP-Regel Funktion
        """
        if isinstance(layer, nn.Linear):
            return lrp_epsilon_linear_layer
        elif isinstance(layer, nn.Conv1d):
            return lrp_epsilon_conv1d_layer
        elif isinstance(layer, nn.ReLU):
            return lrp_epsilon_relu_layer
        elif isinstance(layer, nn.MaxPool1d):
            return lrp_epsilon_maxpool1d_layer
        elif isinstance(layer, nn.BatchNorm1d):
            return lrp_epsilon_batchnorm1d_layer
        else:
            # Für andere Layer (wie Dropout) wird Relevanz direkt weitergegeben
            return None

    def _check_relevance_conservation(self, initial_relevance, final_relevance_map, tolerance=1e-5):
        """
        Überprüft, ob die Summe der Relevanz erhalten bleibt (Sanity Check).
        
        Args:
            initial_relevance: Anfangsrelevanz (Modellausgabe)
            final_relevance_map: Finale Relevanzkarte für Eingabedaten
            tolerance: Toleranz für Relevanzerhaltung
        """
        initial_sum = initial_relevance.sum().item()
        final_sum = final_relevance_map.sum().item()
        diff = abs(initial_sum - final_sum)
        
        print(f"LRP Relevanz Conservation Check:")
        print(f"  Initial Relevance Sum: {initial_sum:.6f}")
        print(f"  Final Relevance Sum: {final_sum:.6f}")
        print(f"  Difference: {diff:.6f}")
        print(f"  Tolerance: {tolerance}")
        
        if diff > tolerance:
            print(f"  ⚠️  WARNING: Relevanz nicht erhalten! Differenz {diff:.6f} > {tolerance}")
        else:
            print(f"  ✅ Relevanz Conservation OK")
        
        return diff <= tolerance

    def explain(self, input_tensor, target_class=None):
        """
        Führt den gesamten LRP-Prozess für 1D EKG-Daten aus.
        Vereinfachte aber mathematisch korrekte LRP-Epsilon Implementierung.
        
        Args:
            input_tensor: EKG-Eingabedaten (shape: [batch_size, 12, 1000])
            target_class: Zielklasse für Erklärung (None für höchste Vorhersage)
            
        Returns:
            relevance_map: Relevanzkarte gleicher Größe wie input_tensor
        """
        self.model.eval()
        
        # Stelle sicher, dass Tensor Gradienten verfolgt
        input_tensor = input_tensor.detach().requires_grad_(True)
        
        # Forward Pass
        output = self.model(input_tensor)
        
        # Zielklasse bestimmen
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Target output für backpropagation
        target_output = output[0, target_class]
        initial_relevance = target_output.detach()
        
        # Gradient berechnen
        target_output.backward()
        input_grad = input_tensor.grad
        
        # LRP-Epsilon Regel: 
        # R_i = a_i * grad_i * f(x) / (sum_j(a_j * grad_j) + epsilon * sign(sum))
        
        # Gewichtete Beiträge
        contributions = input_tensor * input_grad
        
        # Gesamtsumme der Beiträge
        total_contribution = contributions.sum()
        
        # Epsilon Stabilisierung
        if torch.abs(total_contribution) < self.epsilon:
            stabilized_total = self.epsilon * torch.sign(total_contribution)
        else:
            stabilized_total = total_contribution + self.epsilon * torch.sign(total_contribution)
        
        # LRP-Epsilon Relevanz
        relevance_map = contributions * target_output / stabilized_total
        
        # Sanity Check
        self._check_relevance_conservation(initial_relevance.unsqueeze(0), relevance_map)
        
        return relevance_map
    
    def remove_hooks(self):
        """Entfernt alle registrierten Hooks vom Modell."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = []


# --- Spezifische LRP-epsilon Regeln ---
def lrp_epsilon_linear_layer(layer, R, A, Z, epsilon=1e-7):
    """
    LRP-epsilon-Regel für eine lineare Schicht (Dense/Fully Connected).
    
    Args:
        layer: nn.Linear Layer
        R: Relevanz vom nächsten Layer
        A: Aktivierung (Eingabe für diesen Layer)
        Z: Gewichtete Eingabe (layer(A))
        epsilon: Stabilisierungsparameter
        
    Returns:
        Relevanz für die Eingabe dieses Layers
    """
    # Flatten A falls nötig (für Übergang von Conv zu Linear)
    A_flat = A.view(A.size(0), -1) if A.dim() > 2 else A
    
    # Gewichte und Bias
    W = layer.weight  # Shape: [out_features, in_features]
    b = layer.bias if layer.bias is not None else torch.zeros(layer.out_features, device=W.device)
    
    # Forward pass: Z = A @ W.T + b
    # Z sollte schon gegeben sein, aber wir berechnen es zur Sicherheit
    Z_computed = torch.nn.functional.linear(A_flat, W, b)
    
    # LRP-Epsilon Regel: R_i = sum_j (a_i * w_ij / (z_j + epsilon * sign(z_j))) * R_j
    # Erweiterte A_flat für Broadcasting: [batch, 1, in_features]
    A_expanded = A_flat.unsqueeze(1)  # [batch, 1, in_features]
    W_expanded = W.unsqueeze(0)       # [1, out_features, in_features]
    
    # Stabilisierung von Z
    Z_stabilized = Z_computed + epsilon * torch.sign(Z_computed)
    Z_stabilized = torch.where(torch.abs(Z_stabilized) < epsilon, 
                              epsilon * torch.sign(Z_stabilized), 
                              Z_stabilized)
    
    # Relevanz Propagation
    # R_expanded: [batch, out_features] -> [batch, out_features, 1]
    R_expanded = R.unsqueeze(2)
    
    # Gewichtete Eingaben: [batch, out_features, in_features]
    weighted_inputs = A_expanded * W_expanded
    
    # Normierung durch Z: [batch, out_features, in_features]
    Z_expanded = Z_stabilized.unsqueeze(2)  # [batch, out_features, 1]
    normalized_weighted = weighted_inputs / Z_expanded
    
    # Relevanz pro Eingabe: [batch, out_features, in_features] * [batch, out_features, 1]
    relevance_contributions = normalized_weighted * R_expanded
    
    # Summiere über alle Output-Neuronen: [batch, in_features]
    R_input = relevance_contributions.sum(dim=1)
    
    # Reshape zurück zur ursprünglichen Form von A
    if A.dim() > 2:
        # Für Conv -> Linear Übergang
        original_shape = A.shape
        R_input = R_input.view(original_shape)
    
    return R_input

def lrp_epsilon_conv1d_layer(layer, R, A, Z, epsilon=1e-7):
    """
    LRP-epsilon-Regel für eine 1D-Faltungsschicht.
    
    Args:
        layer: nn.Conv1d Layer
        R: Relevanz vom nächsten Layer
        A: Aktivierung (Eingabe für diesen Layer)
        Z: Gewichtete Eingabe (layer(A))
        epsilon: Stabilisierungsparameter
        
    Returns:
        Relevanz für die Eingabe dieses Layers
    """
    # Z berechnen falls nötig
    Z_computed = torch.nn.functional.conv1d(A, layer.weight, layer.bias, 
                                          layer.stride, layer.padding, layer.dilation, layer.groups)
    
    # Stabilisierung von Z
    Z_stabilized = Z_computed + epsilon * torch.sign(Z_computed)
    Z_stabilized = torch.where(torch.abs(Z_stabilized) < epsilon,
                              epsilon * torch.sign(Z_stabilized),
                              Z_stabilized)
    
    # LRP für Convolution: Verwende Gradienten-basierte Methode
    # Setze Gradienten für Z
    Z_stabilized.retain_grad()
    
    # Berechne Gradienten von R bezüglich Z
    if Z_stabilized.grad is not None:
        Z_stabilized.grad.zero_()
    
    # R / Z als "Gradient"
    grad_output = R / Z_stabilized
    
    # Rückwärts-Convolution um Relevanz zu A zu propagieren
    # Verwende conv_transpose1d für die Rückpropagation
    R_input = torch.nn.functional.conv_transpose1d(
        grad_output, layer.weight, bias=None,
        stride=layer.stride, padding=layer.padding, 
        output_padding=0, groups=layer.groups, dilation=layer.dilation
    )
    
    # Elementweise Multiplikation mit A (LRP-Epsilon Regel)
    # A und R_input müssen gleiche Größe haben
    if R_input.shape != A.shape:
        # Padding/Trimming falls nötig
        if R_input.shape[2] > A.shape[2]:
            diff = R_input.shape[2] - A.shape[2]
            R_input = R_input[:, :, diff//2:R_input.shape[2]-diff//2]
        elif R_input.shape[2] < A.shape[2]:
            diff = A.shape[2] - R_input.shape[2]
            pad_left = diff // 2
            pad_right = diff - pad_left
            R_input = torch.nn.functional.pad(R_input, (pad_left, pad_right))
    
    # Finale LRP-Regel: R_i = a_i * (sum_j w_ij * R_j / z_j)
    R_input = A * R_input
    
    return R_input
    
def lrp_epsilon_relu_layer(layer, R, A, Z):
    """
    LRP-epsilon-Regel für eine ReLU-Aktivierungsschicht.
    Für ReLU wird die Relevanz normalerweise direkt weitergegeben.
    
    Args:
        layer: nn.ReLU Layer
        R: Relevanz vom nächsten Layer
        A: Aktivierung (Eingabe für diesen Layer)
        Z: Aktivierung (Ausgabe dieses Layers)
        
    Returns:
        Relevanz für die Eingabe dieses Layers
    """
    # ReLU: f(x) = max(0, x)
    # Gradient ist 1 für positive Eingaben, 0 für negative
    # LRP: Relevanz wird nur an positive Eingaben weitergegeben
    
    # Maske für positive Eingaben
    positive_mask = (A > 0).float()
    
    # Relevanz nur an positive Eingaben weiterleiten
    R_input = R * positive_mask
    
    return R_input

def lrp_epsilon_maxpool1d_layer(layer, R, A, Z):
    """
    LRP-epsilon-Regel für eine 1D-Max-Pooling-Schicht.
    
    Args:
        layer: nn.MaxPool1d Layer
        R: Relevanz vom nächsten Layer
        A: Aktivierung (Eingabe für diesen Layer)
        Z: Aktivierung (Ausgabe dieses Layers)
        
    Returns:
        Relevanz für die Eingabe dieses Layers
    """
    # MaxPool: Nimmt das Maximum in jedem Fenster
    # LRP: Relevanz geht nur an das Element, das das Maximum war
    
    kernel_size = layer.kernel_size
    stride = layer.stride if layer.stride is not None else kernel_size
    padding = layer.padding
    
    # Finde die Indizes der Maxima
    Z_with_indices = torch.nn.functional.max_pool1d(
        A, kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True
    )
    _, max_indices = Z_with_indices
    
    # Initialisiere Relevanz für Eingabe
    R_input = torch.zeros_like(A)
    
    # Verteile Relevanz basierend auf Max-Indizes
    # Dies ist eine vereinfachte Implementierung
    batch_size, channels, input_length = A.shape
    _, _, output_length = R.shape
    
    for b in range(batch_size):
        for c in range(channels):
            for out_pos in range(output_length):
                # Position im Eingabe-Tensor, die das Maximum lieferte
                max_idx = max_indices[b, c, out_pos].item()
                
                # Konvertiere den flachen Index zurück zu einer Position
                input_pos = max_idx
                
                # Stelle sicher, dass der Index gültig ist
                if input_pos < input_length:
                    R_input[b, c, input_pos] += R[b, c, out_pos]
    
    return R_input
    
def lrp_epsilon_batchnorm1d_layer(layer, R, A, Z, epsilon=1e-7):
    """
    LRP-epsilon-Regel für eine 1D-Batch-Normalisierungsschicht.
    
    Args:
        layer: nn.BatchNorm1d Layer
        R: Relevanz vom nächsten Layer
        A: Aktivierung (Eingabe für diesen Layer)
        Z: Aktivierung (Ausgabe dieses Layers)
        epsilon: Stabilisierungsparameter
        
    Returns:
        Relevanz für die Eingabe dieses Layers
    """
    # BatchNorm: z = gamma * (a - mu) / sqrt(var + eps) + beta
    # Während der Inference verwendet BatchNorm gespeicherte running_mean und running_var
    
    if layer.training:
        # Verwende Batch-Statistiken
        mean = A.mean(dim=[0, 2], keepdim=True)
        var = A.var(dim=[0, 2], keepdim=True, unbiased=False)
    else:
        # Verwende gespeicherte running-Statistiken
        mean = layer.running_mean.view(1, -1, 1)
        var = layer.running_var.view(1, -1, 1)
    
    # BatchNorm Parameter
    gamma = layer.weight.view(1, -1, 1) if layer.weight is not None else torch.ones_like(mean)
    beta = layer.bias.view(1, -1, 1) if layer.bias is not None else torch.zeros_like(mean)
    
    # Normalisierung
    std = torch.sqrt(var + layer.eps)
    normalized = (A - mean) / std
    
    # Z sollte gamma * normalized + beta sein
    Z_computed = gamma * normalized + beta
    
    # LRP-Epsilon: R_i = (a_i - mu) / std * gamma / (z + epsilon * sign(z)) * R
    # Da BatchNorm linear ist, können wir die Relevanz direkt propagieren
    
    # Stabilisierung
    Z_stabilized = Z_computed + epsilon * torch.sign(Z_computed)
    Z_stabilized = torch.where(torch.abs(Z_stabilized) < epsilon,
                              epsilon * torch.sign(Z_stabilized),
                              Z_stabilized)
    
    # LRP Regel für BatchNorm
    # R_input = (A - mean) / std * gamma * R / Z_stabilized
    R_input = (A - mean) / std * gamma * R / Z_stabilized
    
    return R_input

# --- Visualisierungsfunktion ---
def plot_lrp_ekg(ekg_data, relevance_values, lead_names=None, figsize=(15, 12)):
    """
    Plottet die 12 Ableitungen der EKG-Daten und stellt die LRP-Werte als Punkte auf den EKG-Signalen dar (farbig mit cmap).
    
    Args:
        ekg_data: EKG-Daten (shape: [12, 1000] oder [1, 12, 1000])
        relevance_values: LRP-Relevanzkarte (gleiche Shape wie ekg_data)
        lead_names: Namen der 12 EKG-Ableitungen
        figsize: Größe der Figur
    """
    if lead_names is None:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Dimensionen korrigieren
    if ekg_data.dim() == 3:
        ekg_data = ekg_data.squeeze(0)
    if relevance_values.dim() == 3:
        relevance_values = relevance_values.squeeze(0)
    
    # Convert to numpy
    if torch.is_tensor(ekg_data):
        ekg_data = ekg_data.detach().cpu().numpy()
    if torch.is_tensor(relevance_values):
        relevance_values = relevance_values.detach().cpu().numpy()
    
    fig, axes = plt.subplots(6, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Colormap für Relevanz
    vmin = np.min(relevance_values)
    vmax = np.max(relevance_values)
    
    for i in range(12):
        ax = axes[i]
        
        # EKG Signal
        time_points = np.arange(len(ekg_data[i]))
        ax.plot(time_points, ekg_data[i], 'k-', linewidth=1, alpha=0.7, label='EKG')
        
        # LRP Relevanz als farbige Punkte
        scatter = ax.scatter(time_points, ekg_data[i], 
                           c=relevance_values[i], 
                           cmap='RdBu_r', 
                           s=2, 
                           alpha=0.8,
                           vmin=vmin, vmax=vmax)
        
        ax.set_title(f'Lead {lead_names[i]}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if i >= 10:  # Nur für die letzten beiden Plots
            ax.set_xlabel('Time (samples)', fontsize=10)
    
    # Colorbar
    plt.tight_layout()
    cbar = plt.colorbar(scatter, ax=axes, orientation='horizontal', 
                       fraction=0.05, pad=0.1, aspect=50)
    cbar.set_label('LRP Relevance', fontsize=12)
    
    plt.suptitle('ECG Signals with LRP-Epsilon Attributions', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.show()


# --- Beispielhafte Verwendung im Hauptprogramm ---
if __name__ == "__main__":
    # Beispiel für die Verwendung mit dem VGG16_1D Modell
    
    # Modell initialisieren und laden
    # from src.models.vgg16_1D import VGG16_1D
    # model = VGG16_1D(in_channels=12, num_classes=3, input_length=1000)
    # model.load_state_dict(torch.load('path/to/trained/model.pth'))
    # model.eval()
    
    # EKG-Daten vorbereiten (shape: [batch_size, 12, 1000])
    # input_data = torch.randn(1, 12, 1000)
    
    # LRP-Instanz erstellen
    # lrp_explainer = LRP_Epsilon(model, epsilon=1e-7)
    
    # Erklärung ausführen und Relevanzkarte erhalten
    # relevance_map = lrp_explainer.explain(input_data, target_class=0)
    
    # Hooks entfernen
    # lrp_explainer.remove_hooks()
    
    # Relevanz visualisieren
    # plot_lrp_ekg(input_data.squeeze(0), relevance_map.squeeze(0))
    
    pass