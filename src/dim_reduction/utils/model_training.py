
import torch
import numpy as np
from tqdm import tqdm

# ------------------------------ CLASS ------------------------------
class ModelTrainer:

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device='cpu', scheduler=None):
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
        self.train_losses = []
        self.val_losses = []
        self.train_losses_per_class = []
        self.val_losses_per_class = []

    
    def train(self, num_epochs=10, eval_fn=None, threshold=0.5):
        for epoch in range(num_epochs):
            train_loss, train_per_class_loss, train_preds, train_labels = self.training_loop(threshold)
            val_loss, val_per_class_loss, val_preds, val_labels = self.validation_loop(threshold)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if train_per_class_loss is not None:
                self.train_losses_per_class.append(train_per_class_loss)
            if val_per_class_loss is not None:
                self.val_losses_per_class.append(val_per_class_loss)

            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            if eval_fn is not None:
                self.handle_metrics(train_labels, train_preds, 'Training', eval_fn)
                self.handle_metrics(val_labels, val_preds, 'Validation', eval_fn)

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


    def compute_per_class_loss(self, outputs, targets):
        bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        return bce(outputs, targets).mean(dim=0).detach().cpu().numpy()


    def compute_epoch_loss(self, running_loss, dataset_size):
        return running_loss / dataset_size


    def handle_metrics(self, y_true, y_pred, phase, eval_fn):
        metrics = eval_fn(y_true, y_pred)
        print(f"Metrics ({phase}): {metrics}")
    

    def inferencing(self, data_loader):
        """
        Führt einen vollständigen Inferenz-Durchlauf aus und gibt wahre Labels und 
        Wahrscheinlichkeiten zurück.
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


    def save_model(self, path):
        """
        Save the model's state_dict to the specified path.
        Args:
            path: Path to save the model
        Returns:
            None
        """
        torch.save(self.model.state_dict(), path)


    def load_model(self, path):
        """
        Load the model's state_dict from the specified path.
        Args:
            path: Path to load the model
        Returns:
            None
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))