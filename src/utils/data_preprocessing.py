import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


# ------------------------------ CLASS ------------------------------
class DataPreprocessor:
    def __init__(self):
        self.scaler = None

    def data_split(self, X_signals, y_labels, meta_df):
        """
        Split data into train, validation, and test sets using meta_df.strat_fold.
        Converts one-hot labels to class indices if needed.
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

        # Ensure labels are class indices, not one-hot
        y_train = self._ensure_class_indices(y_train)
        y_val = self._ensure_class_indices(y_val)
        y_test = self._ensure_class_indices(y_test)

        return X_train, y_train, X_val, y_val, X_test, y_test


    def preprocess_signals(self, X_train, X_validation, X_test, outputfolder):
        """
        Preprocess the input signals by standardizing them.
        """
        self.scaler = StandardScaler()
        self.scaler.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))

        # 2. Standard Scaler speichern
        save_standard_scaler(self.scaler, outputfolder)

        # 3. Standardisieren 
        X_train_std = apply_standardizer(X_train, self.scaler)
        X_val_std = apply_standardizer(X_validation, self.scaler)
        X_test_std = apply_standardizer(X_test, self.scaler)

        return X_train_std, X_val_std, X_test_std


    def save_signals(self, X_train, y_train, X_val, y_val, X_test, y_test, outputfolder):
        """
        Save the preprocessed signals and labels to the specified output folder.
        """
        # 4. Speichern (optional, kann im Notebook aufgerufen werden)
        save_processed_data(X_train, y_train, outputfolder, 'train')
        save_processed_data(X_val, y_val, outputfolder, 'val')
        save_processed_data(X_test, y_test, outputfolder, 'test')
        return None


    @staticmethod
    def _ensure_class_indices(y):
        import numpy as np
        return np.argmax(y, axis=1) if y.ndim > 1 else y
    

    @staticmethod
    def relabel_to_mi_norm_other(y, mlb):
        """
        Wandelt die Labels so um, dass nur noch MI, NORM und OTHER existieren.
        Alles außer MI und NORM wird zu OTHER.
        Args:
            y: np.ndarray, shape (n_samples, n_classes), multi-hot
            mlb: MultiLabelBinarizer
        Returns:
            np.ndarray, shape (n_samples, 3)
        """
        mi_idx = mlb.classes_.tolist().index('MI') if 'MI' in mlb.classes_ else None
        norm_idx = mlb.classes_.tolist().index('NORM') if 'NORM' in mlb.classes_ else None
        new_y = []
        for row in y:
            if mi_idx is not None and row[mi_idx] == 1:
                new_y.append([1, 0, 0])  # MI
            elif norm_idx is not None and row[norm_idx] == 1:
                new_y.append([0, 1, 0])  # NORM
            else:
                new_y.append([0, 0, 1])  # OTHER
        return np.array(new_y)



def save_standard_scaler(scaler, output_folder):
    """
    Save the StandardScaler instance to a pickle file. Create the folder if it does not exist.
    """
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'standard_scaler.pkl'), 'wb') as ss_file:
        pickle.dump(scaler, ss_file)
    return None


def apply_standardizer(X, scaler):
    """
    Apply the StandardScaler to the input data.
    """
    if scaler is None:
        raise ValueError("Scaler has not been fitted. Call preprocess_signals first.")
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(scaler.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp


def save_processed_data(X, y, out_dir, prefix):
    """
    Speichert die Daten und Labels als .npy-Dateien im angegebenen Verzeichnis.
    """
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'{prefix}_signals.npy'), X)
    np.save(os.path.join(out_dir, f'{prefix}_labels.npy'), y)


def load_processed_data(out_dir, prefix, class_names=None):
    """
    Lädt die gespeicherten Daten und Labels als .npy-Dateien aus dem angegebenen Verzeichnis.
    Optional: Wandelt eindimensionale Labels in One-Hot-Labels um, wenn class_names übergeben wird.
    """
    X = np.load(os.path.join(out_dir, f'{prefix}_signals.npy'))
    y = np.load(os.path.join(out_dir, f'{prefix}_labels.npy'))
    
    if class_names is not None:
        import numpy as np
        if y.ndim == 1:
            y = np.eye(len(class_names))[y]
    return X, y
