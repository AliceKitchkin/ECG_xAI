
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# ------------------------------ CLASS ------------------------------
class ModelTrainer:

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device='cpu', scheduler=None, checkpoint_path=None):
        """
        Initialize the ModelTrainer.
        Args:
            model: The model to train
            train_loader: DataLoader for the training set
            val_loader: DataLoader for the validation set
            optimizer: Optimizer for model training
            criterion: Loss function
            device: Device to train on (default: 'cpu')
            scheduler: Optional learning rate scheduler (e.g. ReduceLROnPlateau)
        Returns:
            None
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path

        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        self.train_losses_per_class = []
        self.val_losses_per_class = []
        self.test_losses_per_class = []

        self.val_metrics_per_epoch = []
        self.train_metrics_per_epoch = []
        self.test_metrics_per_epoch = []

    
    def train(self, start_epoch=0, num_epochs=5, eval_fn=None, threshold=None):
        """
        Train the model.
        Args:
            num_epochs: Number of training epochs
            eval_fn: Optional evaluation function
            threshold: Classification threshold
            checkpoint_path: Path to save model checkpoints
        Returns:
            None
        """
        for epoch in range(start_epoch, num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}:")
            train_loss, train_per_class_loss, train_preds, train_labels = self.training_loop(threshold)
            val_loss, val_per_class_loss, val_preds, val_labels = self.validation_loop(threshold)
            test_loss, test_per_class_loss, test_preds, test_labels = self.test_loop(threshold)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.test_losses.append(test_loss)

            if train_per_class_loss is not None:
                self.train_losses_per_class.append(train_per_class_loss)
            if val_per_class_loss is not None:
                self.val_losses_per_class.append(val_per_class_loss)
            if test_per_class_loss is not None:
                self.test_losses_per_class.append(test_per_class_loss)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Test Loss: {test_loss:.4f}")
            
            if eval_fn is not None:
                # Speichere die Metriken für die Validierung
                train_metrics = self.handle_metrics(train_labels, train_preds, 'Training', eval_fn)
                val_metrics = self.handle_metrics(val_labels, val_preds, 'Validation', eval_fn)
                test_metrics = self.handle_metrics(test_labels, test_preds, 'Test', eval_fn)

                self.train_metrics_per_epoch.append(train_metrics)
                self.val_metrics_per_epoch.append(val_metrics)
                self.test_metrics_per_epoch.append(test_metrics)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)


    def training_loop(self, threshold):
        self.model.train()
        running_loss = 0.0
        all_y = []
        all_pred = []
        per_class_losses = []
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for X, y in progress_bar:
            loss, preds, labels, outputs, targets = self.training_step(X, y, threshold)
            running_loss += loss * X.size(0)
            all_y.append(labels)
            all_pred.append(preds)
            per_class_losses.append(self.compute_per_class_loss(outputs, targets))
            progress_bar.set_postfix({"batch_loss": loss})

        epoch_loss = self.compute_epoch_loss(running_loss, len(self.train_loader.dataset))
        avg_per_class_loss = np.mean(per_class_losses, axis=0) if per_class_losses else None
        
        return epoch_loss, avg_per_class_loss, np.concatenate(all_pred), np.concatenate(all_y)
    

    def training_step(self, X, y, threshold):
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(X)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        probs = torch.sigmoid(outputs)
        predicted = (probs >= threshold).int() if isinstance(threshold, (float, int)) else (probs >= torch.tensor(threshold, device=probs.device).view(1, -1)).int()
        return loss.item(), predicted.detach().cpu().numpy(), y.detach().cpu().numpy(), outputs, y


    def validation_loop(self, threshold):
        self.model.eval()
        running_loss = 0.0
        all_y = []
        all_pred = []
        per_class_losses = []
        
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y).item()
                running_loss += loss * X.size(0)

                probs = torch.sigmoid(outputs)
                predicted = (probs >= threshold).int() if isinstance(threshold, (float, int)) else (probs >= torch.tensor(threshold, device=probs.device).view(1, -1)).int()
                
                all_y.append(y.detach().cpu().numpy())
                all_pred.append(predicted.detach().cpu().numpy())
                per_class_losses.append(self.compute_per_class_loss(outputs, y))

        epoch_loss = self.compute_epoch_loss(running_loss, len(self.val_loader.dataset))
        avg_per_class_loss = np.mean(per_class_losses, axis=0) if per_class_losses else None
        
        return epoch_loss, avg_per_class_loss, np.concatenate(all_pred), np.concatenate(all_y)


    def test_loop(self, threshold):
        self.model.eval()
        running_loss = 0.0
        all_y = []
        all_pred = []
        per_class_losses = []
        
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y).item()
                running_loss += loss * X.size(0)

                probs = torch.sigmoid(outputs)
                predicted = (probs >= threshold).int() if isinstance(threshold, (float, int)) else (probs >= torch.tensor(threshold, device=probs.device).view(1, -1)).int()
                
                all_y.append(y.detach().cpu().numpy())
                all_pred.append(predicted.detach().cpu().numpy())
                per_class_losses.append(self.compute_per_class_loss(outputs, y))

        epoch_loss = self.compute_epoch_loss(running_loss, len(self.test_loader.dataset))
        avg_per_class_loss = np.mean(per_class_losses, axis=0) if per_class_losses else None
        
        return epoch_loss, avg_per_class_loss, np.concatenate(all_pred), np.concatenate(all_y)


    def compute_per_class_loss(self, outputs, targets):
        # Prüfe, ob das Kriterium eine pos_weight hat (wie bei create_weighted_criterion)
        pos_weight = getattr(self.criterion, 'pos_weight', None)

        if pos_weight is not None:
            bce = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        else:
            bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        losses = bce(outputs, targets).mean(dim=0).detach().cpu().numpy()
        return losses


    def compute_epoch_loss(self, running_loss, dataset_size):
        return running_loss / dataset_size


    def handle_metrics(self, y_true, y_pred, phase, eval_fn):
        metrics = eval_fn(y_true, y_pred)
        print(f"Metrics ({phase}): {metrics}")
        return metrics
    

    def inferencing(self, data_loader):
        """
        Führt einen vollständigen Inferenz-Durchlauf aus und gibt wahre Labels und Wahrscheinlichkeiten zurück.
        """
        self.model.eval()
        all_y_true = []
        all_y_probs = []
        
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                probs = torch.sigmoid(outputs)
                
                all_y_true.append(y.cpu().numpy())
                all_y_probs.append(probs.cpu().numpy())
                
        return np.concatenate(all_y_true), np.concatenate(all_y_probs)


    @staticmethod
    def create_weighted_criterion(y_train, class_names=None):
        """
        Erstellt eine gewichtete BCEWithLogitsLoss basierend auf Klassenhäufigkeiten.
        
        Args:
            y_train: Training labels (np.ndarray)
            class_names: Optional, Liste der Klassennamen für Debug-Output
        
        Returns:
            nn.BCEWithLogitsLoss: Gewichtete Loss-Funktion
        """
        # Klassen-Häufigkeiten berechnen
        if isinstance(y_train, np.ndarray) and y_train.ndim == 2:
            class_counts = y_train.sum(axis=0)
        else:
            class_counts = np.bincount(y_train)
        
        # Gewichte berechnen (inverse Häufigkeit + Normierung)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
        # Optional: Debug-Output
        if class_names:
            print("Class weights (higher values = rarer classes get more importance):")
            for name, weight in zip(class_names, class_weights):
                print(f"  {name}: {weight:.4f}")
        
        return nn.BCEWithLogitsLoss(pos_weight=class_weights)


# ------------------------------ SAVING ------------------------------
    def save_training_history(self, class_names=None, history_path='data/results/training_history/training_history.csv'):
        """
        Speichert die Trainings- und Validierungsverluste sowie alle Metriken
        in einer CSV-Datei, inkl. pro-Klasse Loss für Training und Validation je Epoche.
        Stellt sicher, dass das Zielverzeichnis existiert.
        """
        history = {
            'epoch': list(range(1, len(self.train_losses) + 1)),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }

        # Pro-Klasse Loss für Training und Validation
        def add_per_class_losses(losses_per_class, prefix):
            if losses_per_class:
                arr = np.array(losses_per_class)
                for cname, col in zip(class_names, arr.T):
                    history[f'{prefix}_{cname}'] = col.tolist()

        add_per_class_losses(self.train_losses_per_class, 'train_loss')
        add_per_class_losses(self.val_losses_per_class, 'val_loss')

        # Globale und pro-Klasse Metriken
        if self.val_metrics_per_epoch:
            for metric in ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']:
                history[f'val_{metric}'] = [m.get(metric) for m in self.val_metrics_per_epoch]

            def add_per_class_metric(metric_key, prefix):
                for i, cname in enumerate(class_names):
                    history[f'{prefix}_{cname}'] = [m.get(metric_key)[i] for m in self.val_metrics_per_epoch]

            add_per_class_metric('f1_per_class', 'val_f1')
            add_per_class_metric('precision_per_class', 'val_precision')
            add_per_class_metric('recall_per_class', 'val_recall')

        df = pd.DataFrame(history)

        # Ensure the directory exists
        dir_path = os.path.dirname(history_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        df.to_csv(history_path, index=False)
        print(f"Trainingshistorie wurde unter {history_path} gespeichert.")


    def save_model(self, path):
        """
        Save the model's state_dict to the specified path.
        Args:
            path: Path to save the model
        Returns:
            None
        """
        torch.save(self.model.state_dict(), path)
    

    def save_checkpoint(self, epoch, path):
        """
        Speichert den Modell-, Optimierer- und ggf. Scheduler-Zustand für die Fortsetzung des Trainings.
        Args:
            epoch: Die aktuelle Epoche, bis zu der das Training fortgesetzt werden soll
            path: Der Pfad, unter dem der Checkpoint gespeichert werden soll
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_losses_per_class': self.train_losses_per_class,
            'val_losses_per_class': self.val_losses_per_class
        }
        if self.scheduler is not None:
            try:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            except Exception as e:
                print(f"Warnung: Scheduler konnte nicht gespeichert werden: {e}")
        torch.save(checkpoint, path)
        print(f"Modell insgesamt trainierte Epochen: {epoch}")
        print(f"Checkpoint wird gespeichert unter: {path}")
    

# ------------------------------ LOADING ------------------------------
    def load_model(self, path):
        """
        Load the model's state_dict from the specified path.
        Args:
            path: Path to load the model
        Returns:
            None
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Modell erfolgreich aus '{path}' geladen.")


    def load_training_history(self, class_names=None, history_path='data/results/training_history/training_history.csv'):
        """
        Lädt die Trainingshistorie aus einer CSV-Datei und speichert sie in 
        den Instanzvariablen des Trainers. Fehlende Metrik-Spalten werden ignoriert.
        """
        try:
            df = pd.read_csv(history_path)
            self.train_losses = df['train_loss'].tolist() if 'train_loss' in df.columns else []
            self.val_losses = df['val_loss'].tolist() if 'val_loss' in df.columns else []
            self.val_metrics_per_epoch = []

            for _, row in df.iterrows():
                metrics_dict = {}
                # Globale Metriken laden (verwende get, falls Spalte fehlt)
                metrics_dict['accuracy'] = row.get('val_accuracy') if 'val_accuracy' in row else None
                metrics_dict['f1_weighted'] = row.get('val_f1_weighted') if 'val_f1_weighted' in row else None
                metrics_dict['precision_weighted'] = row.get('val_precision_weighted') if 'val_precision_weighted' in row else None
                metrics_dict['recall_weighted'] = row.get('val_recall_weighted') if 'val_recall_weighted' in row else None
                # Pro-Klasse Metriken laden (verwende get, falls Spalte fehlt)
                metrics_dict['f1_per_class'] = [row.get(f'val_f1_{cname}') if f'val_f1_{cname}' in row else None for cname in class_names]
                metrics_dict['precision_per_class'] = [row.get(f'val_precision_{cname}') if f'val_precision_{cname}' in row else None for cname in class_names]
                metrics_dict['recall_per_class'] = [row.get(f'val_recall_{cname}') if f'val_recall_{cname}' in row else None for cname in class_names]
                self.val_metrics_per_epoch.append(metrics_dict)
            print(f"Trainingshistorie aus '{history_path}' erfolgreich geladen.")
            
        except FileNotFoundError:
            print(f"Fehler: Datei '{history_path}' nicht gefunden.")
        except Exception as e:
            print(f"Fehler beim Laden der Trainingshistorie: {e}")

        
    def load_checkpoint(self, path):
        """Lädt den Modell-, Optimierer- und ggf. Scheduler-Zustand, um das Training fortzusetzen."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_losses_per_class = checkpoint['train_losses_per_class']
            self.val_losses_per_class = checkpoint['val_losses_per_class']

            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    print(f"Warnung: Scheduler konnte nicht geladen werden: {e}")

            start_epoch = checkpoint['epoch']
            print(f"Checkpoint aus {path} erfolgreich geladen. \nFortsetzung ab Epoche {start_epoch}.")

            return start_epoch
        
        except FileNotFoundError:
            print(f"Fehler: Checkpoint-Datei '{path}' nicht gefunden. Starte Training von Grund auf.")
            return 0 # Beginne von Epoche 0