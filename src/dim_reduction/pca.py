import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class PCA_1D:
    """
    Klasse für PCA-Analyse auf 1D-Mehrkanal-Zeitreihendaten (EKG).
    Ermöglicht die Auswahl, ob Ableitungen oder Zeitpunkte als Features verwendet werden.
    """

    def __init__(self, n_components=3, standardize=True, feature_mode='leads'):
        """
        Initialisiert die PCA_1D-Klasse.
        
        Args:
            n_components (int): Anzahl der Hauptkomponenten.
            standardize (bool): Ob die Daten vor der PCA standardisiert werden sollen.
            feature_mode (str): Modi: 'leads' (Ableitungen sind Features) oder 
                                        'time' (Zeitpunkte sind Features).
        """
        if feature_mode not in ['leads', 'time']:
            raise ValueError("feature_mode muss 'leads' oder 'time' sein.")
            
        self.n_components = n_components
        self.standardize = standardize
        self.feature_mode = feature_mode
        self.pca = PCA(n_components=n_components)
        self.fitted = False
        self.mean_ = None
        self.std_ = None


    def fit(self, X):
        """
        Fittet die PCA auf die Trainingsdaten basierend auf dem gewählten Modus.
        Erkennt für 'leads' automatisch, ob die Achsen vertauscht sind.
        Args:
            X: np.ndarray, shape (n_samples, n_leads, time) oder (n_samples, n_time, n_leads)
        """
        if X.ndim != 3:
            raise ValueError("Erwartetes Input-Format: (n_samples, n_leads, time) oder (n_samples, n_time, n_leads)")

        # Korrigieren der Daten-Shape durch Transponierung: 
        # ursprünglich: (Samples, Zeitpunkte, Ableitungen)
        # transponieren zu: (Samples, Ableitungen, Zeitpunkte)
        X = np.transpose(X, (0, 2, 1))
        n_samples, n_leads, n_time = X.shape

        if self.feature_mode == 'leads':
            # Daten umformen zu (n_samples * time, n_leads)
            X_reshaped = X.reshape(-1, n_leads)
        else: # feature_mode == 'time'
            # Daten umformen zu (n_samples * n_leads, time)
            X_reshaped = X.reshape(-1, n_time)

        if self.standardize:
            self.mean_ = X_reshaped.mean(axis=0)
            self.std_ = X_reshaped.std(axis=0) + 1e-8
            X_reshaped = (X_reshaped - self.mean_) / self.std_

        self.pca.fit(X_reshaped)
        self.fitted = True
        

    def transform(self, X, keep_shape=True):
        """
        Fittet die PCA auf die Trainingsdaten basierend auf dem gewählten Modus.
        
        Args:
            X: np.ndarray, shape (n_samples, n_leads, time)
        """
        if not self.fitted:
            raise RuntimeError("PCA wurde noch nicht gefittet!")
        if X.ndim != 3:
            raise ValueError("Erwartetes Input-Format: (n_samples, n_leads, time)")
        
        # Korrigieren der Daten-Shape durch Transponierung
        X = np.transpose(X, (0, 2, 1))
        n_samples, n_leads, n_time = X.shape

        if self.feature_mode == 'leads':
            X_reshaped = X.reshape(-1, n_leads)
            if self.standardize:
                X_reshaped = (X_reshaped - self.mean_) / self.std_
            transformed_data = self.pca.transform(X_reshaped)
            
            if keep_shape:
                return transformed_data.reshape(n_samples, n_time, self.n_components)
            else:
                return transformed_data
        
        else: # feature_mode == 'time'
            X_reshaped = X.reshape(-1, n_time)
            if self.standardize:
                X_reshaped = (X_reshaped - self.mean_) / self.std_
            transformed_data = self.pca.transform(X_reshaped)

            if keep_shape:
                return transformed_data.reshape(n_samples, n_leads, self.n_components)
            else:
                return transformed_data


    def plot_explained_variance(self, figsize=(10, 6)):
        """
        Plottet die kumulativ erklärte Varianz der Hauptkomponenten.
        Args:
            figsize (tuple): Größe des Plots (Breite, Höhe) in Zoll. Standard: (10, 6)
        """
        if not self.fitted:
            raise RuntimeError("PCA ist noch nicht gefittet!")
        
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        plt.figure(figsize=figsize)
        x_vals = range(1, len(cumulative_variance) + 1)
        plt.plot(x_vals, cumulative_variance, marker='o')

        if self.feature_mode == 'leads':
            mode_str = 'Ableitungen als Features'
        else:
            mode_str = 'Zeitpunkte als Features'

        plt.title(f'Kumulativ erklärte Varianz der PCA-Komponenten\n({mode_str})')
        plt.xlabel('Anzahl der Komponenten')
        plt.ylabel('Kumulative erklärte Varianz')
        plt.grid(True)
        plt.xlim(0, len(cumulative_variance) + 1)
        plt.ylim(bottom=0)

        # Dynamische xticks
        max_xticks = 15
        n = len(cumulative_variance)
        if n <= max_xticks:
            plt.xticks(x_vals)
        else:
            step = max(1, n // max_xticks)
            xticks = list(range(1, n + 1, step))
            if xticks[-1] != n:
                xticks.append(n)
            plt.xticks(xticks)
        plt.show()


    def plot_scree_plot(self, figsize=(10, 6)):
        """
        Plottet die individuell erklärte Varianz jeder Hauptkomponente.
        """
        if not self.fitted:
            raise RuntimeError("PCA ist noch nicht gefittet!")
        
        explained_variance_ratio = self.pca.explained_variance_ratio_
        plt.figure(figsize=figsize)
        x_vals = range(1, len(explained_variance_ratio) + 1)
        plt.bar(x_vals, explained_variance_ratio, alpha=0.5, align='center', label='Individuell erklärte Varianz')
        plt.step(x_vals, np.cumsum(explained_variance_ratio), where='mid', label='Kumulativ erklärte Varianz')

        if self.feature_mode == 'leads':
            mode_str = 'Ableitungen als Features'
        else:
            mode_str = 'Zeitpunkte als Features'
        plt.title(f'Scree-Plot zur Bestimmung der optimalen Komponentenanzahl\n({mode_str})')
        plt.xlabel('Hauptkomponente')
        plt.ylabel('Erklärte Varianz (%)')

        # Dynamische xticks
        max_xticks = 15
        n = len(explained_variance_ratio)
        if n <= max_xticks:
            plt.xticks(x_vals)
        else:
            step = max(1, n // max_xticks)
            xticks = list(range(1, n + 1, step))
            if xticks[-1] != n:
                xticks.append(n)
            plt.xticks(xticks)
        plt.legend(loc='best')
        plt.grid(True)
        plt.xlim(0, len(explained_variance_ratio) + 1)
        plt.ylim(bottom=0)
        plt.show()