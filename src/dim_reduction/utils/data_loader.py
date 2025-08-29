
import os
import numpy as np
import pandas as pd
import ast
import pickle
from wfdb import rdsamp
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer


# ------------------------------ CLASS ------------------------------
class EKGDataLoader:
    def __init__(self, data_path, sampling_frequency=100, task='superdiagnostic', experiment='default', output_folder='output/'):
        self.data_path = data_path
        self.sampling_frequency = sampling_frequency
        self.task = task
        self.experiment = experiment
        self.output_folder = output_folder


    def load_ptbxl_data(self, min_samples=0):
        """
        Lädt PTB-XL EKG-Daten und gibt die EKG-Signale (X), die Labels (y) und den MultiLabelBinarizer zurück.
        """
        datafolder = self.data_path
        sampling_frequency = self.sampling_frequency
        task = self.task
        experiment = self.experiment
        outputfolder_for_mlb = os.path.join(os.path.dirname(datafolder), 'output', experiment)
        os.makedirs(outputfolder_for_mlb, exist_ok=True)

        # 1. Lade PTB-XL Daten
        df, raw_labels = load_dataset(datafolder, sampling_frequency)

        # 2. Labels aggregieren
        labels = compute_label_aggregations(raw_labels, datafolder, task)

        # 3. Relevante Daten auswählen und in One-Hot umwandeln
        X, Y, y, mlb = select_data(df, labels, task, min_samples=min_samples, output_folder=outputfolder_for_mlb)

        return X, Y, y, mlb


# ----------------------- 1. Lade PTB-XL Daten
def load_dataset(path, sampling_rate, release=False):
    if os.path.exists(path + 'ptbxl_database.csv'):
        print(f'Loading PTB-XL data from: {path} ptbxl_database.csv')
        # load and convert annotation data
        Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data_ptbxl(Y, sampling_rate, path)
        return X, Y
    else:
        print(f'Error loading data')
        return None, None


def load_raw_data_ptbxl(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path + 'raw100.npy', allow_pickle=True)
        else:
            data = [rdsamp(path + f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path + 'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path + 'raw500.npy', allow_pickle=True)
        else:
            data = [rdsamp(path + f) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path + 'raw500.npy', 'wb'), protocol=4)
    return data


# ----------------------- 2. Labels aggregieren
def compute_label_aggregations(df, folder, ctype):
    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(folder + 'scp_statements.csv', index_col=0)

    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]

        if ctype == 'diagnostic':
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))

    elif ctype == 'all':
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df


# ----------------------- # 3. Relevante Daten auswählen und in One-Hot umwandeln
def select_data(XX, YY, ctype, min_samples, output_folder):
    # convert multi_label to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
    else:
        pass

    # save Label_Binarizer
    with open(output_folder + 'mlb.pkl', 'wb') as tokenizer:
        pickle.dump(mlb, tokenizer)

    return X, Y, y, mlb

