import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_curve, auc, PrecisionRecallDisplay


class ModelEvaluation:
    
    @staticmethod
    def compute_metrics(y_true, y_pred):
        """
        Berechnet und gibt ein Dictionary mit verschiedenen Metriken zurück.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_per_class': f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_per_class': precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_per_class': recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        }
        return metrics
    

    @staticmethod
    def print_metrics(metrics, class_names=None, dataset='val'):
        """
        Gibt die Metriken in einem lesbaren Format aus.
        """
        if dataset == "train":
            set = "Training"
        elif dataset == "val":
            set = "Validation"
        else:
            set = "Test"
            
        print(f"\n--- Evaluation Metrics for {set} Dataset ---")
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
    def plot_loss_curves(figsize=None, history_path=None, show_per_class=True,
                        show_datasets=['train', 'val', 'test']):
        """
        Plottet Loss-Kurven für Gesamt- und pro-Klasse-Loss für Train, Val und Test.
        """
        if not os.path.exists(history_path):
            raise FileNotFoundError(f"History file not found: {history_path}")

        history = pd.read_csv(history_path)
        epochs = history['epoch'] if 'epoch' in history.columns else np.arange(len(history))

        plt.figure(figsize=figsize)

        # Plot overall losses basierend auf show_datasets Parameter
        if 'train' in show_datasets and 'train_loss' in history.columns:
            plt.plot(epochs, history['train_loss'], label='Train Loss', color='orange', linewidth=2)
        if 'val' in show_datasets and 'val_loss' in history.columns:
            plt.plot(epochs, history['val_loss'], label='Val Loss', color='lightblue', linewidth=2)
        if 'test' in show_datasets and 'test_loss' in history.columns:
            plt.plot(epochs, history['test_loss'], label='Test Loss', color='green', linewidth=2)

        # Plot per-class losses if present and enabled
        if show_per_class:
            for col in history.columns:
                if 'train' in show_datasets and col.startswith('train_loss_'):
                    plt.plot(epochs, history[col], '--', label=col)
                if 'val' in show_datasets and col.startswith('val_loss_'):
                    plt.plot(epochs, history[col], ':', label=col)
                if 'test' in show_datasets and col.startswith('test_loss_'):
                    plt.plot(epochs, history[col], '-.', label=col)

        plt.xlabel('Epoche')
        plt.ylabel('Loss / Score')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3, linewidth=0.7)
        plt.xlim(epochs.min(), epochs.max())
        plt.xticks(np.arange(0, epochs.max() + 1, 2))
        plt.ylim(0, 1.2)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_metrics_from_history(history_path, metrics=None, figsize=None):
        """
        Plottet beliebige Metriken aus einer Trainings-History-CSV.
        """
        if not os.path.exists(history_path):
            raise FileNotFoundError(f"History file not found: {history_path}")
        history = pd.read_csv(history_path)
        epochs = history['epoch'] if 'epoch' in history.columns else np.arange(len(history))

        if metrics is None:
            # Alle Spalten außer 'epoch' vorschlagen
            metrics = [col for col in history.columns if col != 'epoch']

        plt.figure(figsize=figsize)
        for metric in metrics:
            if metric in history.columns:
                plt.plot(epochs, history[metric], label=metric)
            else:
                print(f"Warnung: '{metric}' nicht in History gefunden.")

        plt.xlabel('Epochen')
        plt.ylabel('Score')
        plt.title('Metriken')
        plt.legend()
        plt.grid(True, alpha=0.3, linewidth=0.7)
        plt.xlim(epochs.min(), epochs.max())
        plt.xticks(np.arange(0, epochs.max() + 1, 2))
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.show()
    

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, figsize=None, dataset='val'):
        """
        Plottet die Konfusionsmatrix.
        """
        cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
        plt.figure(figsize=figsize)
        im = plt.imshow(cm, cmap='Blues')
        tick_marks = np.arange(len(class_names)) if class_names else np.arange(cm.shape[0])
        plt.xticks(tick_marks, class_names if class_names else tick_marks)
        plt.yticks(tick_marks, class_names if class_names else tick_marks)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        if dataset == "train":
            set = "Training"
        elif dataset == "val":
            set = "Validation"
        else:
            set = "Test"
        plt.title(f'Confusion Matrix on {set} Dataset')
        plt.xlabel('Vorhersage')
        plt.ylabel('Wahrheit')
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_learning_rate(history_path, figsize=None):
        """
        Plottet den Verlauf der Lernrate aus der Trainings-History.
        """
        history = pd.read_csv(history_path)

        if 'learning_rate' not in history.columns:
            raise ValueError("Die Spalte 'learning_rate' wurde in der History nicht gefunden.")

        plt.figure(figsize=figsize)
        plt.plot(history['epoch'], history['learning_rate'], color="#4D8446", linewidth=2)
        plt.xlabel('Epoche')
        plt.ylabel('Lernrate')
        plt.title('Lernratenverlauf')
        plt.grid(True, alpha=0.3, linewidth=0.7)
        plt.xlim(history['epoch'].min(), history['epoch'].max())
        plt.xticks(np.arange(0, history['epoch'].max()+1, 2))
        plt.tight_layout()
        plt.show()

    
    @staticmethod
    def plot_roc(y_true_test, y_probs_test, dataset, figsize=None):
        """
        Plottet die ROC-AUC.
        """
        # ROC-Kurve für die positive Klasse (NORM = 1)
        fpr, tpr, _ = roc_curve(y_true_test, y_probs_test[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='grey', lw=1.5, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.fill_between(fpr, tpr, alpha=0.2, color='grey')
        plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Zufälliger Klassifikator')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({dataset})')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3, linewidth=0.7)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_precision_recall_curve(y_true, y_probs, dataset, figsize=None):
        """
        Plottet die Precision-Recall-Kurve für beide Klassen.
        """
        plt.figure(figsize=figsize)
        disp_mi = PrecisionRecallDisplay.from_predictions(      # MI (Klasse 0)
            y_true == 0,
            y_probs[:, 0],
            name="MI",
            color="darkred",
            plot_chance_level=False,
            despine=True,
            ax=plt.gca()
        )
        disp_norm = PrecisionRecallDisplay.from_predictions(    # NORM (Klasse 1)
            y_true == 1,
            y_probs[:, 1],
            name="NORM",
            color="green",
            plot_chance_level=False,
            despine=True,
            ax=plt.gca()
        )

        # Manuell Chance Level Linien hinzufügen
        mi_prevalence = np.sum(y_true == 0) / len(y_true)
        norm_prevalence = np.sum(y_true == 1) / len(y_true)
        
        plt.axhline(y=mi_prevalence, color='grey', linestyle='--', linewidth=1, 
                    alpha=0.7, label=f'Zufallsrate MI ({mi_prevalence:.2f})')
        plt.axhline(y=norm_prevalence, color='grey', linestyle='--', linewidth=1, 
                    alpha=0.7, label=f'Zufallsrate NORM ({norm_prevalence:.2f})')

        plt.tight_layout()
        plt.grid(True, alpha=0.3, linewidth=0.7)
        plt.legend(loc="lower right")
        plt.xlabel('Recall (Sensitivität)')
        plt.ylabel('Precision (Spezifität)')
        plt.title(f"Precision-Recall Curve für MI und NORM ({dataset})")
        plt.show()



