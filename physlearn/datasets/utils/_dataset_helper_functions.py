import os
import re
import json
import sklearn.utils
import pandas as pd

from sklearn.model_selection import train_test_split


def _json_dump(train_test_data, folder, n_qubits):
    assert isinstance(train_test_data, dict)
    assert isinstance(train_test_data['X_train'], pd.DataFrame)
    assert isinstance(train_test_data['X_test'], pd.DataFrame)
    assert isinstance(train_test_data['y_train'], (pd.Series, pd.DataFrame))
    assert isinstance(train_test_data['y_test'], (pd.Series, pd.DataFrame))
    assert isinstance(folder, str)
    assert isinstance(n_qubits, int)

    train_test_data_json = {'X_train': train_test_data['X_train'].to_json(),
                            'X_test': train_test_data['X_test'].to_json(),
                            'y_train': train_test_data['y_train'].to_json(),
                            'y_test': train_test_data['y_test'].to_json()}

    with open(folder + '_{}'.format(n_qubits) + 'q' + '.json', 'w') as outfile:
        json.dump(train_test_data_json, outfile)


def _json_load(folder, n_qubits):
    file = folder + '_{}'.format(n_qubits) + 'q' + '.json'
    with open(file, 'r') as json_file:
        get_train_test_data = json.load(json_file)

    train_test_data = {}
    train_test_data['X_train'] = pd.read_json(get_train_test_data['X_train'])
    train_test_data['X_test'] = pd.read_json(get_train_test_data['X_test'])
    train_test_data['y_train'] = pd.read_json(get_train_test_data['y_train'])
    train_test_data['y_test'] = pd.read_json(get_train_test_data['y_test'])

    return train_test_data


def _train_test_split(X, y, test_size, random_state):
    # Set shuffle=False since shuffling is
    # already handled by Pandas
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state,
                                                        shuffle=False)

    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}


def _df_shuffle(df, drop=True):
    return sklearn.utils.shuffle(df).reset_index(drop=drop)
    

def _iqr_outlier_mask(df):
    # Compute first and third quantiles
    # then remove outliers from df
    first = df.quantile(0.25)
    third = df.quantile(0.75)
    iqr = third - first
    return ((df < (first - 1.5*iqr)) | (df > (third + 1.5*iqr))).any(axis=1)


def _path_google_data(n_qubits):
    root = os.path.dirname(__file__).replace('utils', '')
    if re.search('C:', root):
        root = root + 'data\\'
    else:
        root = root + 'data/'
    if n_qubits == 3:
        path = root + 'google_3q_random.csv'
    elif n_qubits == 5:
        path = root + 'google_5q_random.csv'
    return path


def _path_google_json_folder():
    root = os.path.dirname(__file__).replace('utils', '')
    if re.search('C:', root):
        folder = root + 'google_json\\'
    else:
        folder = root + 'google_json/'
    return folder