import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, confusion_matrix


class ThresholdOptimizer:
    """
    Klasse zur Optimierung der Schwellenwerte pro Label basierend auf spezifischen Metriken.
    """

    def __init__(self, metrics_to_optimize, search_range=np.arange(0.1, 1.0, 0.05), beta=1.5):
        """
        Initialisiert den Optimizer.
        
        Args:
            metrics_to_optimize (dict): Dictionary, das für jeden Label-Index die zu optimierende Metrik festlegt.
                                        Beispiel: {0: 'recall_weighted', 1: 'precision_weighted', 2: 'recall_weighted'}
            search_range (np.array): Array von Schwellenwerten, die getestet werden sollen.
            beta (float): Der Gewichtungsfaktor für die modifizierten Metriken. Höherer Wert (>1) gewichtet Recall, niedrigerer Wert (<1) Precision.
        """
        self.metrics_to_optimize = metrics_to_optimize
        self.search_range = search_range
        self.optimized_thresholds = {}
        self.beta = beta
        self.beta2 = beta ** 2


    def optimize(self, y_true, y_probs):
        """
        Findet die optimalen Schwellenwerte für jedes Label.
        
        Args:
            y_true (np.array): Die wahren Labels.
            y_probs (np.array): Die vorhergesagten Wahrscheinlichkeiten vom Modell.
        
        Returns:
            dict: Ein Dictionary der optimalen Schwellenwerte pro Label.
        """
        num_labels = y_true.shape[1]
        
        for i in tqdm(range(num_labels), desc="Optimizing Thresholds"):
            best_threshold = 0.5
            best_score = -1
            
            metric_type = self.metrics_to_optimize.get(i)
            valid_metrics = ['recall', 'precision', 'recall_weighted', 'precision_weighted']
            
            if metric_type not in valid_metrics:
                print(f"Warnung: Keine gültige Metrik für Label {i} angegeben. Verwende Standard-Schwellenwert 0.5.")
                self.optimized_thresholds[i] = 0.5
                continue

            for threshold in self.search_range:
                y_pred_single = (y_probs[:, i] >= threshold).astype(int)
                
                tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred_single, labels=[0, 1]).ravel()

                if metric_type == 'recall':
                    score = tp / (tp + fn) if (tp + fn) > 0 else 0
                elif metric_type == 'precision':
                    score = tp / (tp + fp) if (tp + fp) > 0 else 0
                elif metric_type == 'recall_weighted':
                    score = tp / (tp + fp + self.beta2 * fn) if (tp + fp + self.beta2 * fn) > 0 else 0
                elif metric_type == 'precision_weighted':
                    score = tp / (tp + self.beta2 * fp + fn) if (tp + self.beta2 * fp + fn) > 0 else 0
                else:
                    score = -1 # Fehlerfall

                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            self.optimized_thresholds[i] = best_threshold
            print(f"Optimaler Schwellenwert für Label {i} ({metric_type}): {best_threshold:.2f} mit Score {best_score:.4f}")
            
        return self.optimized_thresholds