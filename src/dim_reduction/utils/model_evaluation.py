import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


class ModelEvaluation:
    
    @staticmethod
    def compute_metrics(y_true, y_pred, class_names=None):
        """
        Berechnet und gibt ein Dictionary mit verschiedenen Metriken zurück.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_per_class': f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            'precision_per_class': precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            'recall_per_class': recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        }
        return metrics
    

    @staticmethod
    def print_metrics(metrics, class_names=None):
        """
        Gibt die Metriken in einem lesbaren Format aus.
        """
        print("\n--- Evaluation Metrics ---")
        for k, v in metrics.items():
            if isinstance(v, list):
                if class_names is not None and len(v) == len(class_names):
                    print(f"{k}:")
                    for cname, val in zip(class_names, v):
                        print(f"  {cname}: {val:.4f}")
                else:
                    print(f"{k}: {v}")
            else:
                print(f"{k}: {v:.4f}" if isinstance(v, (float, int)) else f"{k}: {v}")
    

    @staticmethod
    def plot_loss_curves(train_losses, val_losses=None, train_losses_per_class=None, val_losses_per_class=None, class_names=None):
        plt.figure(figsize=(10,6))
        
        plt.plot(train_losses, label='Train Loss', color='black', linewidth=2)
        if val_losses is not None:
            plt.plot(val_losses, label='Val Loss', color='gray', linewidth=2)
        
        n_classes = len(train_losses_per_class[0]) if train_losses_per_class else (len(val_losses_per_class[0]) if val_losses_per_class else 0)
        
        if n_classes > 0:
            names = class_names if class_names and len(class_names) == n_classes else [f"Class {i}" for i in range(n_classes)]
            colors = plt.cm.tab10.colors

            if train_losses_per_class is not None:
                train_losses_per_class = np.array(train_losses_per_class)
                for i in range(n_classes):
                    plt.plot(train_losses_per_class[:, i], label=f'Train {names[i]}', linestyle='--', color=colors[i % len(colors)])
            
            if val_losses_per_class is not None:
                val_losses_per_class = np.array(val_losses_per_class)
                for i in range(n_classes):
                    plt.plot(val_losses_per_class[:, i], label=f'Val {names[i]}', linestyle=':', color=colors[i % len(colors)])
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve (gesamt & pro Klasse)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False):
        cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1), normalize='true' if normalize else None)
        plt.figure(figsize=(7,6))
        im = plt.imshow(cm, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        tick_marks = np.arange(len(class_names)) if class_names else np.arange(cm.shape[0])
        plt.xticks(tick_marks, class_names if class_names else tick_marks, rotation=45)
        plt.yticks(tick_marks, class_names if class_names else tick_marks)
        
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.show()


