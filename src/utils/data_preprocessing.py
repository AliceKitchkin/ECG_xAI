import os
import pandas as pd
import numpy as np
import pickle
import neurokit2 as nk
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ------------------------------ CLASS ------------------------------
class DataPreprocessor:
    def __init__(self):
        self.scaler = None

    def data_split(self, X_signals, y_labels, meta_df):
        """
        Split data into train, validation, and test sets using meta_df.strat_fold.
        Args:
            X_signals: Input signals.
            y_labels: Corresponding labels.
            meta_df: DataFrame containing metadata with stratified folds.
        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        X_train = X_signals[meta_df.strat_fold < 9]
        y_train = y_labels[meta_df.strat_fold < 9]
        X_val = X_signals[meta_df.strat_fold == 9]
        y_val = y_labels[meta_df.strat_fold == 9]
        X_test = X_signals[meta_df.strat_fold == 10]
        y_test = y_labels[meta_df.strat_fold == 10]

        return X_train, y_train, X_val, y_val, X_test, y_test


    def preprocess_signals(self, X_train, X_validation, X_test, outputfolder):
        """
        Preprocess the input signals by standardizing them.
        """
        # Alle Beats zu einem Vektor zusammenfassen
        all_train_values = np.concatenate(X_train).reshape(-1, 1).astype(float)
        self.scaler = StandardScaler().fit(all_train_values)

        self.save_scaler(outputfolder)

        X_train_std = self._apply_standardizer(X_train)
        X_val_std = self._apply_standardizer(X_validation)
        X_test_std = self._apply_standardizer(X_test)

        return np.array(X_train_std), np.array(X_val_std), np.array(X_test_std)


    def save_signals(self, X_train, y_train, X_val, y_val, X_test, y_test, outputfolder):
        """
        Save the preprocessed signals and labels to the specified output folder.
        """
        self.save_processed_data(X_train, y_train, outputfolder, 'train')
        self.save_processed_data(X_val, y_val, outputfolder, 'val')
        self.save_processed_data(X_test, y_test, outputfolder, 'test')
    

    @staticmethod
    def relabel_to_mi_norm(y, mlb):
        """
        Wandelt die Labels so um, dass nur noch MI und NORM existieren.
        Gibt Integer-Labels zurück: 0=MI, 1=NORM
        """
        mi_idx = mlb.classes_.tolist().index('MI') if 'MI' in mlb.classes_ else None
        norm_idx = mlb.classes_.tolist().index('NORM') if 'NORM' in mlb.classes_ else None
        new_y = []
        for row in y:
            if mi_idx is not None and row[mi_idx] == 1:
                new_y.append(0)  # MI
            elif norm_idx is not None and row[norm_idx] == 1:
                new_y.append(1)  # NORM
            # else: ignore
        return np.array(new_y)


    def save_scaler(self, output_folder):
        """Save the fitted scaler to a pickle file."""
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet.")
        os.makedirs(output_folder, exist_ok=True)
        with open(os.path.join(output_folder, 'standard_scaler.pkl'), 'wb') as ss_file:
            pickle.dump(self.scaler, ss_file)

    
    def load_scaler(self, output_folder):
        """Load a previously saved scaler."""
        scaler_path = os.path.join(output_folder, 'standard_scaler.pkl')
        with open(scaler_path, 'rb') as ss_file:
            self.scaler = pickle.load(ss_file)


    def _apply_standardizer(self, X):
        """Apply the fitted scaler to input data."""
        X = np.asarray(X)
        return self.scaler.transform(X.reshape(-1, 1)).reshape(X.shape)


    @staticmethod
    def save_processed_data(X, y, out_dir, prefix):
        """
        Speichert die Daten und Labels als .npy-Dateien im angegebenen Verzeichnis.
        """
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f'{prefix}_signals.npy'), X)
        np.save(os.path.join(out_dir, f'{prefix}_labels.npy'), y)


    @staticmethod
    def load_processed_data(out_dir, prefix):
        """
        Lädt die gespeicherten Daten und Labels als .npy-Dateien aus dem angegebenen Verzeichnis.
        """
        print(f'Loading processed data from {out_dir} with prefix {prefix}')
        X = np.load(os.path.join(out_dir, f'{prefix}_signals.npy'))
        y = np.load(os.path.join(out_dir, f'{prefix}_labels.npy'))
        
        return X, y