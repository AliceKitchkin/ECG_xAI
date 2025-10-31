import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any


def zero_grad(tensors: Tuple[torch.Tensor, ...]):
    """Setzt Gradienten aller Tensoren auf Null"""
    for t in tensors:
        if t.grad is not None:
            t.grad.zero_()


def lrp_rule_ratio(input_tensor: torch.Tensor, output_tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    LRP Ratio Rule: input * (output/input).detach()
    
    Args:
        input_tensor: Eingabetensor (aj)
        output_tensor: Ausgabetensor (c)
        eps: Numerische Stabilität
        
    Returns:
        Neuer Ausgabetensor mit angewandter LRP Regel
    """
    output = output_tensor.clone().detach()
    denom = input_tensor.detach()
    
    # Numerische Stabilität gegen Division durch Null
    nonzero_mask = denom.abs() > eps
    new_output = torch.zeros_like(output)
    
    new_output[nonzero_mask] = (input_tensor[nonzero_mask] / denom[nonzero_mask]) * output[nonzero_mask]
    new_output[~nonzero_mask] = output[~nonzero_mask]
    
    return new_output


class LRP_1D_New:
    """
    Layer-wise Relevance Propagation (LRP) für 1D Signale wie EKG-Daten.
    Implementierung basierend auf der 2D LRP Struktur, angepasst für 1D Daten.
    """
    
    LRP_BETA_ATTRIBUTE = "__act_for_lrp_beta"
    
    def __init__(self, model: nn.Module, eps: float = 1e-6):
        """
        Args:
            model: Das zu erklärende PyTorch Modell (z.B. VGG16_1D)
            eps: Kleiner Wert zur numerischen Stabilität
        """
        self.model = model
        self.eps = eps
        self.hooks = []
        self.activations = {}
        self.gradients = {}
        self.first_layer = None
        self._find_first_layer()
        
    def _find_first_layer(self):
        """Findet die erste Conv1d Schicht für LRP-beta"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv1d):
                self.first_layer = module
                break
        
        if self.first_layer is None:
            raise ValueError("Keine Conv1d Schicht im Modell gefunden!")
    
    def _lrp_beta_bounds(self, x: torch.Tensor, mean: Optional[torch.Tensor] = None, 
                        std: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Erstellt untere und obere Grenzen für LRP-beta
        
        Args:
            x: Input Tensor
            mean: Mittelwert für Normalisierung
            std: Standardabweichung für Normalisierung
            
        Returns:
            Tuple[lb, hb]: Untere und obere Grenzen
        """
        if mean is None:
            mean = torch.zeros(x.shape[1], 1)
        if std is None:
            std = torch.ones(x.shape[1], 1)
            
        # Stelle sicher, dass mean und std die richtige Form haben
        if mean.dim() == 1:
            mean = mean.unsqueeze(-1)
        if std.dim() == 1:
            std = std.unsqueeze(-1)
            
        # Normalisierte untere und obere Grenzen
        lb = (torch.zeros_like(x) - mean) / std  # Untere Grenze (0 normalisiert)
        hb = (torch.ones_like(x) - mean) / std   # Obere Grenze (1 normalisiert)
        
        return lb, hb
    
    def _lrp_beta(self, x: torch.Tensor, lb: torch.Tensor, hb: torch.Tensor) -> np.ndarray:
        """
        Berechnet die finale Relevanz nach der LRP-beta Regel
        
        Args:
            x: Input Tensor
            lb: Untere Grenze
            hb: Obere Grenze
            
        Returns:
            Relevanz-Heatmap als numpy array
        """
        heatmap = x * x.grad + lb * lb.grad + hb * hb.grad
        return heatmap.detach().cpu().numpy()
    
    def _setup_lrp_beta_hooks(self):
        """Setzt Forward Hooks für LRP-beta auf die erste Schicht"""
        def lrp_beta_forward_hook(module, input, output):
            # Speichere Aktivierungen für LRP-beta
            x = input[0]
            
            # Erstelle bounds für LRP-beta
            lb, hb = self._lrp_beta_bounds(x)
            lb = lb.requires_grad_(True)
            hb = hb.requires_grad_(True)
            
            # Speichere für späteren Zugriff
            setattr(module, self.LRP_BETA_ATTRIBUTE, (x, lb, hb))
        
        if self.first_layer is not None:
            hook = self.first_layer.register_forward_hook(lrp_beta_forward_hook)
            self.hooks.append(hook)
    
    def __enter__(self):
        """Context manager entry - Setup der Hooks"""
        self._setup_lrp_beta_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - Cleanup der Hooks"""
        self._remove_hooks()
    
    def _register_hooks(self):
        """Registriert Forward- und Backward-Hooks für alle relevanten Layer"""
        
        def forward_hook(module, input, output):
            # Speichere Aktivierungen
            module_name = self._get_module_name(module)
            self.activations[module_name] = output.clone()
            
        def backward_hook(module, grad_input, grad_output):
            # Speichere Gradienten
            module_name = self._get_module_name(module)
            if grad_output[0] is not None:
                self.gradients[module_name] = grad_output[0].clone()
        
        # Hooks für alle Conv1d und Linear Layer registrieren
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                h1 = module.register_forward_hook(forward_hook)
                h2 = module.register_full_backward_hook(backward_hook)
                self.hooks.extend([h1, h2])
    
    def _get_module_name(self, module):
        """Hilfsfunktion um Modulnamen zu finden"""
        for name, mod in self.model.named_modules():
            if mod is module:
                return name
        return str(id(module))
    
    def _remove_hooks(self):
        """Entfernt alle registrierten Hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
        self.gradients.clear()
        
        # Entferne LRP-beta Attribute
        if self.first_layer is not None and hasattr(self.first_layer, self.LRP_BETA_ATTRIBUTE):
            delattr(self.first_layer, self.LRP_BETA_ATTRIBUTE)
    
    def explain(self, x: torch.Tensor, target_class: int, 
                mean: Optional[torch.Tensor] = None, 
                std: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Erklärt eine Vorhersage mit LRP (ähnlich der 2D Variante)
        
        Args:
            x: Input Tensor, Shape [channels, length] (ohne Batch-Dimension)
            target_class: Zielklasse für die Erklärung
            mean: Mittelwert für Normalisierung (optional)
            std: Standardabweichung für Normalisierung (optional)
            
        Returns:
            Tuple[logits, heatmap]: Logits und Relevanz-Heatmap
        """
        input_shape = x.shape
        assert len(input_shape) == 2, f"Expected 2D input [channels, length], got {input_shape}"
        
        with self:
            # Füge Batch-Dimension hinzu falls nötig
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
                
            x = x.detach().requires_grad_(True)
            x.grad = None
            
            # Forward Pass
            logits = self.model(x)
            
            # Backward Pass für target class
            logits[:, target_class].backward()
            
            # Hole LRP-beta Daten von der ersten Schicht
            if hasattr(self.first_layer, self.LRP_BETA_ATTRIBUTE):
                aj, lb, hb = getattr(self.first_layer, self.LRP_BETA_ATTRIBUTE)
                
                # Vergleiche mit Input (sollten gleich sein)
                assert torch.allclose(x, aj, atol=1e-5), "Input und erste Schicht Aktivierung stimmen nicht überein"
                
                # Berechne LRP-beta Heatmap
                heatmap = self._lrp_beta(aj, lb, hb)
                heatmap = heatmap.squeeze()
            else:
                # Fallback: Einfache Gradient-basierte Relevanz
                heatmap = (x * x.grad).squeeze().detach().cpu().numpy()
            
            logits = logits.squeeze().detach().cpu().numpy()
            
            assert len(logits.shape) == 1, f"Expected 1D logits, got {logits.shape}"
            assert heatmap.shape == input_shape, f"Heatmap shape {heatmap.shape} != input shape {input_shape}"
            
            return logits, heatmap
    
    def explain_simple(self, x: torch.Tensor, target_class: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vereinfachte Erklärung ohne LRP-beta (nur Standard-LRP)
        
        Args:
            x: Input Tensor, Shape [channels, length]
            target_class: Zielklasse
            
        Returns:
            Tuple[logits, heatmap]: Logits und Relevanz-Heatmap
        """
        input_shape = x.shape
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        x = x.clone().detach().requires_grad_(True)
        
        # Forward Pass
        logits = self.model(x)
        
        # Backward Pass
        target_logit = logits[0, target_class]
        target_logit.backward()
        
        # Einfache Gradient-basierte Relevanz
        heatmap = x * x.grad
        
        return logits.squeeze().detach().cpu().numpy(), heatmap.squeeze().detach().cpu().numpy()


class SimpleEKGExplainer:
    """
    Einfache Wrapper-Klasse für die Verwendung in Jupyter Notebooks
    Entspricht der Funktionalität der 2D Variante, aber für 1D EKG-Daten
    """
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: Trainiertes PyTorch Modell
        """
        self.model = model
        self.lrp = LRP_1D_New(model)
    
    def explain_prediction(self, ekg_signal: torch.Tensor, target_class: Optional[int] = None) -> Dict[str, Any]:
        """
        Erklärt eine EKG-Vorhersage
        
        Args:
            ekg_signal: EKG Daten, Shape [channels, length]
            target_class: Klasse zu erklären. Wenn None, wird die vorhergesagte Klasse verwendet
            
        Returns:
            Dictionary mit Ergebnissen
        """
        self.model.eval()
        
        with torch.no_grad():
            if len(ekg_signal.shape) == 2:
                logits = self.model(ekg_signal.unsqueeze(0))
            else:
                logits = self.model(ekg_signal)
                
            predicted_class = torch.argmax(logits, dim=1).item()
        
        if target_class is None:
            target_class = predicted_class
            
        logits_np, heatmap = self.lrp.explain(ekg_signal, target_class)
        
        return {
            'predicted_class': predicted_class,
            'target_class': target_class,
            'logits': logits_np,
            'heatmap': heatmap,
            'confidence': torch.softmax(torch.tensor(logits_np), dim=0).numpy()
        }
    
    def plot_explanation(self, ekg_signal: torch.Tensor, target_class: Optional[int] = None, 
                        class_names: Optional[list] = None):
        """
        Erstellt Plots der Erklärung
        
        Args:
            ekg_signal: EKG Signal
            target_class: Zielklasse
            class_names: Namen der Klassen für bessere Visualisierung
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib ist nicht verfügbar. Bitte installieren Sie es für Visualisierungen.")
            return
        
        result = self.explain_prediction(ekg_signal, target_class)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(result['logits']))]
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # 1. Original EKG Signal
        ekg_np = ekg_signal.cpu().numpy() if torch.is_tensor(ekg_signal) else ekg_signal
        time_steps = np.arange(ekg_np.shape[1])
        
        for channel in range(ekg_np.shape[0]):
            axes[0].plot(time_steps, ekg_np[channel], label=f'Lead {channel+1}', alpha=0.7)
        
        axes[0].set_title(f'Original EKG Signal - Predicted: {class_names[result["predicted_class"]]}')
        axes[0].set_xlabel('Time Steps')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # 2. LRP Heatmap
        heatmap = result['heatmap']
        im = axes[1].imshow(heatmap, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        axes[1].set_title(f'LRP Relevance Heatmap - Target: {class_names[result["target_class"]]}')
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('EKG Channels (Leads)')
        axes[1].set_yticks(range(min(12, heatmap.shape[0])))
        axes[1].set_yticklabels([f'Lead {i+1}' for i in range(min(12, heatmap.shape[0]))])
        plt.colorbar(im, ax=axes[1])
        
        # 3. Aggregierte zeitliche Relevanz
        time_relevance = heatmap.sum(axis=0)  # Summe über alle Kanäle
        axes[2].plot(time_steps, time_relevance, color='red', linewidth=2)
        axes[2].set_title('Aggregated Temporal Relevance (Sum over all leads)')
        axes[2].set_xlabel('Time Steps')
        axes[2].set_ylabel('Total Relevance')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Zeige Confidence Scores
        print(f"\nConfidence Scores:")
        for i, (class_name, conf) in enumerate(zip(class_names, result['confidence'])):
            marker = " ←" if i == result['predicted_class'] else ""
            print(f"  {class_name}: {conf:.3f}{marker}")
        
        return result