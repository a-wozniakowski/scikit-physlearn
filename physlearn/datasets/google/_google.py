"""
The :mod:`physlearn.datasets.google._google` module provides utilities
for wrangling, loading, and saving the Google quantum computer
calibration data.

Notes
-----
The calibration data was collected by Benjamin Chiaro during his
time as a graduate student at UC Santa Barbara. The quantum device
contains 9 qubits, wherein the 5 rightmost qubits and 4 interleaving
couplers were utilized during experimentation. The 4 leftmost qubits
and couplers were left idle during experimentation.
"""

# Author: Alex Wozniakowski
# License: MIT

import os
import typing

import pandas as pd

import sklearn.model_selection

from physlearn.datasets.google.base import BaseDataFrame
from physlearn.datasets.google.utils._dataset_helper_functions import (_shuffle,
                                                                       _iqr_outlier_mask,
                                                                       _train_test_split,
                                                                       _json_dump,
                                                                       _json_load,
                                                                       _path_to_google_data,
                                                                       _path_to_google_json_folder)


class GoogleDataFrame(BaseDataFrame):
    """Represents the Google quantum computer calibration data with a DataFrame."""

    def __init__(self, n_qubits, path):
        super().__init__(path=path)

        self.n_qubits = n_qubits
        self._validate_dataframe_options()

    def _validate_dataframe_options(self):
        assert isinstance(self.path, str)
        assert isinstance(self.n_qubits, int) and self.n_qubits > 0
        
    @property    
    def get_df_with_correct_columns(self) -> pd.DataFrame:
        """Split DataFrame into train and test split."""

        df = self.get_df

        if self.n_qubits == 5:
            # Select every fifth row, as well as
            # the relevant columns.
            df = df.iloc[1::5, :].loc[:, 'qubit_voltages':' .29']

            df.columns = ['qvolt5', 'qvolt6', 'qvolt7', 'qvolt8', 'qvolt9',
                          'cvolt4', 'cvolt5', 'cvolt6', 'cvolt7', 'cvolt8',
                          'pref105', 'pref106', 'pref107', 'pref108', 'pref109',
                          'precoup5', 'precoup6', 'precoup7', 'precoup8',
                          'postf105', 'post106', 'postf107', 'postf108', 'postf109',
                          'postcoup5', 'postcoup6', 'postcoup7', 'postcoup8',
                          'peig1', 'peig2', 'peig3', 'peig4', 'peig5',
                          'eeig1', 'eeig2', 'eeig3', 'eeig4', 'eeig5']
            
            # This coupler was idle during the experiment.
            df = df.drop(['cvolt4'], axis=1)
        
        return df


class GoogleData(GoogleDataFrame):
    """Wrangles the Google quantum computer calibration data for multi-target regression."""

    def __init__(self, n_qubits=5, test_split=0.3, random_state=0,
                 remove_outliers=False, shuffle=True):

        super().__init__(n_qubits=n_qubits,
                         path=_path_to_google_data())

        self.test_split = test_split        
        self.random_state = random_state
        self.remove_outliers = remove_outliers
        self.shuffle = shuffle
        self._validate_data_options()

    def _validate_data_options(self):
        if self.n_qubits != 5:
            raise ValueError('This object only supports 5 qubits, '
                             'but %s qubits were specified.'
                             % (self.n_qubits))
        assert self.test_split > 0.0 and self.test_split < 1.0
        assert isinstance(self.random_state, int)
        assert isinstance(self.remove_outliers, bool)
        assert isinstance(self.shuffle, bool)

    def _train_test_split(self) -> dict:
        """Get the DataFrame, then split it into training and test data."""

        if self.shuffle:
            df = _shuffle(data=self.get_df_with_correct_columns)
        else:
            df = self.get_df_with_correct_columns

        # Compute interquartile range for outlier removal.
        if self.remove_outliers:
            df = df[~_iqr_outlier_mask(data=df)]

        qubit_coupler_voltages = ['qvolt5', 'qvolt6', 'qvolt7', 'qvolt8', 'qvolt9',
                                  'cvolt5', 'cvolt6', 'cvolt7', 'cvolt8']        
        google_predictions = ['peig1', 'peig2', 'peig3', 'peig4', 'peig5']
        measured_eigenvalues = ['eeig1', 'eeig2', 'eeig3', 'eeig4', 'eeig5']
        mat_entries = ['postf105', 'post106', 'postf107', 'postf108', 'postf109',
                       'postcoup5', 'postcoup6', 'postcoup7', 'postcoup8']

        return _train_test_split(df[qubit_coupler_voltage + google_predictions],
                                 df[measured_eigenvalues + mat_entry],
                                 test_size=self.test_split,
                                 random_state=self.random_state)

    def save_train_test_split_to_json(self) -> None:
        """Serializes the training and test data as a JSON formatted stream.

        It automatically dumps the data into the Google JSON folder.
        """

        _json_dump(train_test_data=self._train_test_split(),
                   folder=_path_to_google_json_folder(),
                   n_qubits=self.n_qubits)

    @property    
    def load_benchmark(self) -> dict:
        """Deserializes the benchmark dataset."""

        folder = _path_to_google_json_folder()
        return _json_load(filename=os.path.join(folder, '_{}'.format(self.n_qubits) + 'q.json'))


def load_benchmark(return_split=False) -> typing.Union[sklearn.model_selection.train_test_split, dict]:
    """Deserializes the benchmark dataset for the multi-target regression task.
    
    If the return split parameter is true, then the benchmark dataset is
    returned in the familiar X_train, X_test, y_train, and y_test format.

    Parameters
    ----------
    return_split : bool
        If True, then the benchmark dataset is returned in the form of
        X_train, X_test, y_train, and y_test.
    """

    data = GoogleData(n_qubits=5).load_benchmark

    if return_split:
        X_train, X_test = data['X_train'].iloc[:, -5:], data['X_test'].iloc[:, -5:]
        y_train, y_test = data['y_train'].iloc[:, :5], data['y_test'].iloc[:, :5]
        return X_train, X_test, y_train, y_test
    else:
        return data
