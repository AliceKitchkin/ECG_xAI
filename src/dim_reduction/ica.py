import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

class ICA_1D:
    """
    Klasse für ICA-Analyse auf 1D-Mehrkanal-Zeitreihendaten (EKG).
    Ermöglicht die Auswahl, ob Ableitungen oder Zeitpunkte als Features verwendet werden.
    """

    def __init__(self, n_components=3, standardize=True, feature_mode='leads', random_state=0):
        """
        Initialisiert die ICA_1D-Klasse.
        
        Args:
            n_components (int): Anzahl der unabhängigen Komponenten.
            standardize (bool): Ob die Daten vor der ICA standardisiert werden sollen.
            feature_mode (str): Modi: 'leads' (Ableitungen sind Features) oder 
                                        'time' (Zeitpunkte sind Features).
            random_state (int): Zufallsseed für Reproduzierbarkeit.
        """
        if feature_mode not in ['leads', 'time']:
            raise ValueError("feature_mode muss 'leads' oder 'time' sein.")
            
        self.n_components = n_components
        self.standardize = standardize
        self.feature_mode = feature_mode
        self.ica = FastICA(n_components=n_components, random_state=random_state)
        self.fitted = False
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        Fittet die ICA auf die Trainingsdaten basierend auf dem gewählten Modus.
        Args:
            X: np.ndarray, shape (n_samples, n_leads, time) oder (n_samples, n_time, n_leads)
        """
        if X.ndim != 3:
            raise ValueError("Erwartetes Input-Format: (n_samples, n_leads, time) oder (n_samples, n_time, n_leads)")

        # Korrigieren der Daten-Shape durch Transponierung
        X = np.transpose(X, (0, 2, 1))
        n_samples, n_leads, n_time = X.shape

        if self.feature_mode == 'leads':
            # Daten umformen zu (n_samples * n_time, n_leads)
            X_reshaped = X.reshape(-1, n_leads)
        else: # feature_mode == 'time'
            # Daten umformen zu (n_samples * n_leads, n_time)
            X_reshaped = X.reshape(-1, n_time)

        if self.standardize:
            self.mean_ = X_reshaped.mean(axis=0)
            self.std_ = X_reshaped.std(axis=0) + 1e-8
            X_reshaped = (X_reshaped - self.mean_) / self.std_

        self.ica.fit(X_reshaped)
        self.fitted = True

    def transform(self, X, keep_shape=True):
        """
        Transformiert die Daten mit der gelernten ICA.
        Args:
            X: np.ndarray, shape (n_samples, n_leads, time)
        """
        if not self.fitted:
            raise RuntimeError("ICA wurde noch nicht gefittet!")
        if X.ndim != 3:
            raise ValueError("Erwartetes Input-Format: (n_samples, n_leads, time)")

        # Korrigieren der Daten-Shape durch Transponierung
        X = np.transpose(X, (0, 2, 1))
        n_samples, n_leads, n_time = X.shape

        if self.feature_mode == 'leads':
            X_reshaped = X.reshape(-1, n_leads)
            if self.standardize:
                X_reshaped = (X_reshaped - self.mean_) / self.std_
            transformed_data = self.ica.transform(X_reshaped)
            if keep_shape:
                return transformed_data.reshape(n_samples, n_time, self.n_components)
            else:
                return transformed_data
        else: # feature_mode == 'time'
            X_reshaped = X.reshape(-1, n_time)
            if self.standardize:
                X_reshaped = (X_reshaped - self.mean_) / self.std_
            transformed_data = self.ica.transform(X_reshaped)
            if keep_shape:
                return transformed_data.reshape(n_samples, n_leads, self.n_components)
            else:
                return transformed_data

    def plot_mixing_matrix(self, figsize=(10, 6)):
        """
        Plottet die Mischmatrix (wie die unabhängigen Komponenten auf die Originalkanäle wirken).
        """
        if not self.fitted:
            raise RuntimeError("ICA ist noch nicht gefittet!")
        plt.figure(figsize=figsize)
        plt.imshow(self.ica.mixing_, aspect='auto', cmap='bwr')
        plt.colorbar()
        plt.title("ICA-Mischmatrix")
        plt.xlabel("Unabhängige Komponenten")
        plt.ylabel("Original Features")
        plt.show()