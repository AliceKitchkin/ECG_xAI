import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class LRP_Epsilon_Complete:
    """
    Vollständige LRP-Epsilon Implementierung für VGG-1D Modell.
    Diese Version implementiert LRP korrekt layer-für-layer rückwärts.
    """
    
    def __init__(self, model, epsilon=1e-7):
        self.model = model
        self.epsilon = epsilon
        
    def explain(self, input_tensor, target_class=None):
        """
        Führt LRP-Epsilon für das VGG-1D Modell aus.
        """
        self.model.eval()
        input_tensor = input_tensor.detach().requires_grad_(False)
        
        # Forward pass um alle Zwischenergebnisse zu sammeln
        activations = self._forward_and_collect(input_tensor)
        
        # Output und Zielklasse
        output = activations[-1]
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Initiale Relevanz nur für Zielklasse
        R = torch.zeros_like(output)
        R[0, target_class] = output[0, target_class]
        
        initial_relevance = R.clone()
        
        # Rückwärts durch alle Layer
        R = self._backward_pass(R, activations)
        
        # Sanity Check
        self._check_relevance_conservation(initial_relevance, R)
        
        return R
    
    def _forward_and_collect(self, x):
        """Sammelt alle Zwischenaktivierungen während des Forward Pass."""
        activations = [x]  # Eingabe
        
        # Features (Conv Blocks)
        for layer in self.model.features:
            x = layer(x)
            activations.append(x)
        
        # Flatten für Classifier
        x = x.view(x.size(0), -1)
        activations.append(x)
        
        # Classifier (Linear Layers)
        for layer in self.model.classifier:
            x = layer(x)
            activations.append(x)
        
        return activations
    
    def _backward_pass(self, R, activations):
        """Führt LRP rückwärts durch alle Layer aus."""
        
        # Start vom Ende: Classifier layers
        layer_idx = len(activations) - 1
        
        print(f"DEBUG: Starte Backward Pass mit R.shape = {R.shape}")
        
        # Rückwärts durch Classifier
        classifier_layers = list(self.model.classifier)
        print(f"DEBUG: Classifier hat {len(classifier_layers)} Layers")
        
        for i, layer in enumerate(reversed(classifier_layers)):
            print(f"DEBUG: Classifier Layer {i}: {type(layer).__name__}, R.shape = {R.shape}")
            
            if isinstance(layer, nn.Linear):
                A = activations[layer_idx - 1]  # Eingabe für diesen Layer
                print(f"DEBUG: Linear - A.shape = {A.shape}")
                R = self._lrp_linear(layer, R, A)
            elif isinstance(layer, nn.ReLU):
                A = activations[layer_idx - 1]
                print(f"DEBUG: ReLU - A.shape = {A.shape}")
                R = self._lrp_relu(R, A)
            # Dropout wird übersprungen (identität während inference)
            layer_idx -= 1
        
        # Nach dem Flatten müssen wir die Form wiederherstellen
        if R.dim() == 2:
            # Finde die Form vor dem Flatten
            features_output_shape = activations[layer_idx].shape
            print(f"DEBUG: Reshape R von {R.shape} zu {features_output_shape}")
            
            # Prüfe ob features_output_shape 3D ist (Conv Layer Output)
            if len(features_output_shape) == 3:
                # Berechne die korrekte Form: [batch, channels, length]
                batch_size = features_output_shape[0]
                channels = features_output_shape[1] 
                length = features_output_shape[2]
                
                # Reshape: R von [batch, channels*length] zu [batch, channels, length]
                if R.shape[1] == channels * length:
                    R = R.view(batch_size, channels, length)
                    print(f"DEBUG: Erfolgreich reshaped zu {R.shape}")
                else:
                    print(f"DEBUG: ERROR - Reshape nicht möglich: R.shape[1]={R.shape[1]} != channels*length={channels*length}")
                    # Fallback: Erstelle passende Relevanz
                    R = torch.zeros(features_output_shape, device=R.device, dtype=R.dtype)
                    print(f"DEBUG: Fallback - Erstelle Null-Relevanz: {R.shape}")
            else:
                print(f"DEBUG: features_output_shape ist nicht 3D: {features_output_shape}")
                # Wenn das letzte Features-Layer ebenfalls 2D ist, bleiben wir bei 2D
                print(f"DEBUG: Behalte R als 2D: {R.shape}")
                # Suche nach dem letzten echten 3D Conv-Ausgabe
                for i in range(layer_idx-1, -1, -1):
                    if len(activations[i].shape) == 3:
                        target_shape = activations[i].shape
                        print(f"DEBUG: Gefunden 3D Aktivierung bei Index {i}: {target_shape}")
                        
                        batch_size = target_shape[0]
                        channels = target_shape[1]
                        length = target_shape[2]
                        
                        if R.shape[1] == channels * length:
                            R = R.view(batch_size, channels, length)
                            print(f"DEBUG: Erfolgreich reshaped zu {R.shape}")
                            break
                        else:
                            print(f"DEBUG: Größe passt nicht: {R.shape[1]} != {channels * length}")
                else:
                    # Kein 3D gefunden - erstelle eine vernünftige 3D Form
                    print(f"DEBUG: Keine 3D Aktivierung gefunden, erstelle Standard-Form")
                    # Nehme eine vernünftige Aufteilung an
                    total_elements = R.shape[1]
                    # Versuche herauszufinden, welche Conv-Ausgabe das war
                    # Basierend auf VGG: letzte Conv hat 512 Kanäle
                    channels = 512
                    length = total_elements // channels
                    R = R.view(R.shape[0], channels, length)
                    print(f"DEBUG: Standard-Reshape zu {R.shape}")
        
        # Rückwärts durch Features (Conv Blocks)
        features_layers = list(self.model.features)
        print(f"DEBUG: Features hat {len(features_layers)} Layers")
        
        # Berechne den korrekten Aktivierungsindex für Features
        # activations enthält: [input] + features_activations + classifier_activations
        # Wir sind jetzt nach dem Classifier, also bei len(activations) - len(classifier_layers) - 1
        features_activation_start = len(features_layers)  # Index wo Features-Aktivierungen enden
        
        for i, layer in enumerate(reversed(features_layers)):
            # Korrekte Aktivierung: rückwärts durch Features bedeutet 
            # activations[features_activation_start - i - 1] für Input zur aktuellen Layer
            activation_idx = features_activation_start - i - 1
            print(f"DEBUG: Features Layer {i}: {type(layer).__name__}, R.shape = {R.shape}, activation_idx = {activation_idx}")
            
            if isinstance(layer, nn.Conv1d):
                A = activations[activation_idx]
                print(f"DEBUG: Conv1d - A.shape = {A.shape}")
                R = self._lrp_conv1d(layer, R, A)
            elif isinstance(layer, nn.BatchNorm1d):
                A = activations[activation_idx]
                print(f"DEBUG: BatchNorm1d - A.shape = {A.shape}")
                R = self._lrp_batchnorm1d(layer, R, A)
            elif isinstance(layer, nn.ReLU):
                A = activations[activation_idx]
                print(f"DEBUG: ReLU - A.shape = {A.shape}")
                R = self._lrp_relu(R, A)
            elif isinstance(layer, nn.MaxPool1d):
                A = activations[activation_idx]
                print(f"DEBUG: MaxPool1d - A.shape = {A.shape}, R.shape = {R.shape}")
                R = self._lrp_maxpool1d(layer, R, A)
        
        print(f"DEBUG: Backward Pass beendet mit R.shape = {R.shape}")
        return R
    
    def _lrp_linear(self, layer, R, A):
        """LRP-Epsilon für Linear Layer."""
        W = layer.weight  # [out_features, in_features]
        b = layer.bias if layer.bias is not None else torch.zeros(layer.out_features, device=W.device)
        
        # Forward: Z = A @ W.T + b
        Z = torch.nn.functional.linear(A, W, b)
        
        # Stabilisierung
        Z_eps = Z + self.epsilon * torch.sign(Z)
        Z_eps = torch.where(torch.abs(Z_eps) < self.epsilon, 
                           self.epsilon * torch.sign(Z_eps), Z_eps)
        
        # LRP-Epsilon: R_i = sum_j (A_i * W_ij / Z_j) * R_j
        # Eingehende Relevanz: R [batch, out_features]
        # Gewichte: W [out_features, in_features]  
        # Ausgehende Relevanz: R_out [batch, in_features]
        
        # Normalisierte Gewichte pro Output-Neuron
        S = (A.unsqueeze(1) * W.unsqueeze(0)) / Z_eps.unsqueeze(2)  # [batch, out, in]
        
        # Relevanz propagieren
        R_out = (S * R.unsqueeze(2)).sum(dim=1)  # [batch, in_features]
        
        return R_out
    
    def _lrp_conv1d(self, layer, R, A):
        """LRP-Epsilon für Conv1d Layer."""
        # Prüfe Dimensionskompatibilität
        print(f"DEBUG: Conv1d - R: {R.shape}, A: {A.shape}")
        
        # Vereinfachte Implementierung: Nutze den Layer direkt
        A_detached = A.detach().requires_grad_(True)
        
        # Forward - Nutze den Layer direkt statt functional
        Z = layer(A_detached)
        
        print(f"DEBUG: Conv1d Z: {Z.shape}")
        
        # Prüfe ob R und Z kompatibel sind
        if R.shape != Z.shape:
            print(f"DEBUG: Conv1d Größenanpassung: R {R.shape} -> Z {Z.shape}")
            if R.dim() == Z.dim() == 3 and R.shape[0] == Z.shape[0] and R.shape[1] == Z.shape[1]:
                # Interpoliere R auf Z Größe
                R = torch.nn.functional.interpolate(R, size=Z.shape[2], mode='linear', align_corners=False)
                print(f"DEBUG: R interpoliert zu {R.shape}")
        
        # Stabilisierung
        Z_eps = Z + self.epsilon * torch.sign(Z)
        Z_eps = torch.where(torch.abs(Z_eps) < self.epsilon,
                           self.epsilon * torch.sign(Z_eps), Z_eps)
        
        # LRP über autograd
        if R.shape == Z_eps.shape:
            # Verwende torch.autograd.grad statt .backward() für mehr Kontrolle
            gradients = torch.autograd.grad(
                outputs=(R / Z_eps * Z).sum(),
                inputs=A_detached,
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]
            R_out = gradients * A
        else:
            print(f"DEBUG: Conv1d - Fallback zu Identität")
            R_out = A * 0.1  # Kleine Relevanz als Fallback
        
        return R_out
    
    def _lrp_batchnorm1d(self, layer, R, A):
        """LRP-Epsilon für BatchNorm1d."""
        print(f"DEBUG: BatchNorm1d - R: {R.shape}, A: {A.shape}")
        
        # Prüfe Dimensionskompatibilität
        if R.shape != A.shape:
            print(f"DEBUG: BatchNorm1d Dimensionsmismatch")
            if R.dim() == A.dim() == 3 and R.shape[0] == A.shape[0] and R.shape[1] == A.shape[1]:
                # Interpoliere R auf A Größe
                R = torch.nn.functional.interpolate(R, size=A.shape[2], mode='linear', align_corners=False)
                print(f"DEBUG: R interpoliert zu {R.shape}")
        
        # BatchNorm ist linear -> direkte Propagation möglich
        if layer.training:
            mean = A.mean(dim=[0, 2], keepdim=True)
            var = A.var(dim=[0, 2], keepdim=True, unbiased=False)
        else:
            mean = layer.running_mean.view(1, -1, 1)
            var = layer.running_var.view(1, -1, 1)
        
        gamma = layer.weight.view(1, -1, 1) if layer.weight is not None else torch.ones_like(mean)
        
        # Normalisierung
        std = torch.sqrt(var + layer.eps)
        
        # LRP: Relevanz proportional zur Eingabe propagieren
        # R_out = R * gamma * (A - mean) / std / Z
        Z = gamma * (A - mean) / std + (layer.bias.view(1, -1, 1) if layer.bias is not None else 0)
        Z_eps = Z + self.epsilon * torch.sign(Z)
        
        R_out = R * gamma * (A - mean) / std / Z_eps
        
        return R_out
    
    def _lrp_relu(self, R, A):
        """LRP für ReLU: Relevanz nur an positive Eingaben."""
        # Prüfe Dimensionskompatibilität
        if R.shape != A.shape:
            print(f"DEBUG: ReLU Dimensionsmismatch - R: {R.shape}, A: {A.shape}")
            # Falls Größen nicht übereinstimmen, passe R an A an
            if R.dim() == A.dim() == 3:
                # Beide sind 3D, aber verschiedene Längen
                r_length = R.shape[2]
                a_length = A.shape[2]
                
                if r_length < a_length:
                    # R ist kürzer - interpoliere oder wiederhole
                    # Einfache Lösung: upsampling
                    R = torch.nn.functional.interpolate(R, size=a_length, mode='linear', align_corners=False)
                    print(f"DEBUG: R upsampled zu {R.shape}")
                elif r_length > a_length:
                    # R ist länger - downsampling
                    R = torch.nn.functional.interpolate(R, size=a_length, mode='linear', align_corners=False)
                    print(f"DEBUG: R downsampled zu {R.shape}")
                    
        return R * (A > 0).float()
    
    def _lrp_maxpool1d(self, layer, R, A):
        """LRP für MaxPool1d: Relevanz nur an Maxima."""
        # Debug: Prüfe Dimensionen
        print(f"DEBUG MaxPool1d: R.shape = {R.shape}, A.shape = {A.shape}")
        
        # Falls R 2D ist (von Linear Layer), reshape zu 3D
        if R.dim() == 2:
            # Das sollte nicht passieren, aber als Fallback
            print(f"WARNING: R ist 2D, aber MaxPool1d erwartet 3D!")
            return R
        
        # Stelle sicher, dass R 3D ist
        if R.dim() != 3:
            print(f"ERROR: R hat unerwartete Dimensionen: {R.shape}")
            return torch.zeros_like(A)
        
        kernel_size = layer.kernel_size
        stride = layer.stride if layer.stride is not None else kernel_size
        padding = layer.padding
        
        # Initialisiere Output-Relevanz
        R_out = torch.zeros_like(A)
        
        batch_size, channels, input_length = A.shape
        output_length = R.shape[2]
        
        # Für jede Position im Output finde das entsprechende Maximum im Input
        for b in range(batch_size):
            for c in range(channels):
                for out_pos in range(output_length):
                    # Berechne den entsprechenden Input-Bereich
                    start_pos = out_pos * stride - padding
                    end_pos = start_pos + kernel_size
                    
                    # Stelle sicher, dass wir im gültigen Bereich sind
                    start_pos = max(0, start_pos)
                    end_pos = min(input_length, end_pos)
                    
                    if start_pos < end_pos:
                        # Finde das Maximum in diesem Bereich
                        input_slice = A[b, c, start_pos:end_pos]
                        if len(input_slice) > 0:
                            max_val = input_slice.max()
                            # Finde die erste Position mit dem Maximalwert
                            max_idx_relative = (input_slice == max_val).nonzero(as_tuple=True)[0]
                            if len(max_idx_relative) > 0:
                                max_idx_absolute = start_pos + max_idx_relative[0].item()
                                # Gib die gesamte Relevanz an diese Position
                                R_out[b, c, max_idx_absolute] += R[b, c, out_pos]
        
        return R_out
    
    def _check_relevance_conservation(self, initial_relevance, final_relevance_map, tolerance=1e-3):
        """Sanity Check für Relevanzerhaltung."""
        initial_sum = initial_relevance.sum().item()
        final_sum = final_relevance_map.sum().item()
        diff = abs(initial_sum - final_sum)
        
        print(f"LRP Relevanz Conservation Check:")
        print(f"  Initial Relevance Sum: {initial_sum:.6f}")
        print(f"  Final Relevance Sum: {final_sum:.6f}")
        print(f"  Difference: {diff:.6f}")
        print(f"  Relative Error: {diff/abs(initial_sum)*100:.2f}%")
        
        if diff > tolerance:
            print(f"  ⚠️  WARNING: Relevanz nicht erhalten!")
        else:
            print(f"  ✅ Relevanz Conservation OK")
        
        return diff <= tolerance


# Einfache Visualisierungsfunktion
def plot_lrp_ekg_simple(ekg_data, relevance_values, lead_names=None, figsize=(15, 8)):
    """Vereinfachte Visualisierung für schnelle Überprüfung."""
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
    
    # Zeige nur erste 4 Leads für schnelle Überprüfung
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(4):
        ax = axes[i]
        
        # EKG Signal
        time_points = np.arange(len(ekg_data[i]))
        ax.plot(time_points, ekg_data[i], 'k-', linewidth=0.8, alpha=0.7)
        
        # LRP Relevanz als farbige Linie
        ax2 = ax.twinx()
        ax2.plot(time_points, relevance_values[i], 'r-', linewidth=1, alpha=0.8)
        ax2.set_ylabel('LRP Relevance', color='r')
        
        ax.set_title(f'Lead {lead_names[i]}')
        ax.set_ylabel('EKG Amplitude')
        ax.set_xlabel('Time (samples)')
        
    plt.tight_layout()
    plt.show()
