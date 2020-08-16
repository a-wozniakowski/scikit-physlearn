"""
The :mod:`physlearn.datasets.google.utils._dataset_helper_functions` module
provides basic utilities for wrangling, loading, and dumping supervised
learning data. 
"""

# Author: Alex Wozniakowski
# License: MIT

import os
import re
import json
import typing

import pandas as pd

import sklearn.model_selection
import sklearn.utils

DataFrame_or_Series = typing.Union[pd.DataFrame, pd.Series]


def _json_dump(train_test_data: dict, folder: str, n_qubits=None) -> None:
    """
    Serializes the training and test data dictionary as a JSON formatted stream.

    Parameters
    ----------
    train_test_data : dict
        A dictionary with keys: 'X_train', 'X_test', 'y_train', and 'y_test'.

    folder : str
        Directory in which the training and test data is dumped.

    n_qubits : int or None, optional (default=None)
        Number of qubits. If specified, then this value
        is utilied in the file name.
    """

    assert isinstance(train_test_data, dict)
    assert isinstance(train_test_data['X_train'], pd.DataFrame)
    assert isinstance(train_test_data['X_test'], pd.DataFrame)
    assert isinstance(train_test_data['y_train'], (pd.Series, pd.DataFrame))
    assert isinstance(train_test_data['y_test'], (pd.Series, pd.DataFrame))
    assert isinstance(folder, str)

    train_test_data_json = {'X_train': train_test_data['X_train'].to_json(),
                            'X_test': train_test_data['X_test'].to_json(),
                            'y_train': train_test_data['y_train'].to_json(),
                            'y_test': train_test_data['y_test'].to_json()}

    if n_qubits is not None:
        assert isinstance(n_qubits, int)
        file = folder + '_{}'.format(n_qubits) + 'q_{}'.format(pd.Timestamp.now().isoformat())
    else:
        file = folder + '_{}'.format(pd.Timestamp.now().isoformat())

    with open(file + '.json', 'w') as outfile:
        json.dump(train_test_data_json, outfile)


def _json_load(filename: str) -> dict:
    """
    Deserializes the training and test data dictionary, which was serialized
    as a JSON formatted stream.

    Parameters
    ----------
    filename : str
        Name of the file in which the training and test data dictionary has been dumped.
    """

    with open(filename, 'r') as json_file:
        get_train_test_data = json.load(json_file)

    train_test_data = {}
    train_test_data['X_train'] = pd.read_json(get_train_test_data['X_train'])
    train_test_data['X_test'] = pd.read_json(get_train_test_data['X_test'])
    train_test_data['y_train'] = pd.read_json(get_train_test_data['y_train'])
    train_test_data['y_test'] = pd.read_json(get_train_test_data['y_test'])

    return train_test_data


def _train_test_split(X: DataFrame_or_Series, y: DataFrame_or_Series, test_size: float,
                      random_state: int) -> dict:
    """
    Splits the X and y data intro training and test data according to the specified
    fraction of the test size.

    Parameters
    ----------
    X : DataFrame or Series
        The design matrix, where each row corresponds to an example and the
        column(s) correspond to the feature(s).

    y : DataFrame or Series
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

    test_size : float
        The decimal amount of test data.

    random_state : int, RandomState instance or None.
        Determines random number generation in sklearn.model_selection.train_test_split.

    Notes
    -----
    As shuffling is handled by sklearn.utils.shuffle, there is no shuffling parameter.
    """

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y,
                                                                                test_size=test_size,
                                                                                random_state=random_state,
                                                                                shuffle=False)

    return dict(X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test)


def _shuffle(data: DataFrame_or_Series, drop=True) -> DataFrame_or_Series:
    """
    Shuffles the pandas data object.

    Parameters
    ----------
    data : DataFrame or Series
        The pandas data that is to be shuffled.

    drop : bool
        Resets the index of the pandas data object.

    """

    return sklearn.utils.shuffle(data).reset_index(drop=drop)
    

def _iqr_outlier_mask(data: DataFrame_or_Series) -> DataFrame_or_Series:
    """
    Computes the interquartile range for the pandas data object,
    then it masks the outliers.

    Parameters
    ----------
    data : DataFrame or Series
        The pandas data that is to be masked.

    """

    first = data.quantile(0.25)
    third = data.quantile(0.75)
    iqr = third - first
    return ((data < (first - 1.5*iqr)) | (data > (third + 1.5*iqr))).any(axis=1)


def _path_to_google_data() -> str:
    """
    Finds the path to the Google quantum computer calibration data.
    """

    root = os.path.dirname(__file__).replace('utils', '')
    return os.path.join(root, 'data', 'google_5q_random.csv')


def _path_to_google_json_folder() -> str:
    """
    Finds the path to the folder, which contains the serialized Google
    quantum computer calibration data.
    """

    root = os.path.dirname(__file__).replace('utils', '')
    return os.path.join(root, 'google_json')
