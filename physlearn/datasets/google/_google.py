"""
The :mod:`physlearn.datasets.google._google` module provides utilities
for wrangling, serializing, and deserializing superconducting quantum
computing calibration data.

Notes
-----
The calibration data was collected by Benjamin Chiaro during his
time as a graduate student at UC Santa Barbara. The Google quantum computer
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

from dataclasses import dataclass, field

from physlearn.datasets.google.base import BaseDataFrame
from physlearn.datasets.google.utils._helper_functions import (_shuffle,
                                                               _iqr_outlier_mask,
                                                               _train_test_split,
                                                               _json_dump,
                                                               _json_load,
                                                               _path_to_google_data,
                                                               _path_to_google_json_folder)

sklearn_train_test_split_or_dict = typing.Union[sklearn.model_selection.train_test_split, dict]


@dataclass
class GoogleDataFrame(BaseDataFrame):
    """Represents the Google quantum computer calibration data with a DataFrame.

    Parameters
    ----------
    path : str
        Path to the csv file with calibration data.

    n_qubits : int
        Number of qubits in the experiment.

    See Also
    --------
    :class:`physlearn.datasets.GoogleData` : Class for wrangling the calibration data.

    Examples
    --------
    >>> from physlearn.datasets import GoogleDataFrame
    >>> from physlearn.datasets.google.utils._helper_functions import _path_to_google_data
    >>> df = GoogleDataFrame(path=_path_to_google_data(), n_qubits=5)
    >>> df.get_df_with_correct_columns.head().iloc[0, :3]
    qvolt5   -0.008238
    qvolt6   -0.006896
    qvolt7   -0.026120
    Name: 1, dtype: float64
    """

    n_qubits: int

    def __post_init__(self):
        self._validate_dataframe_options()

    def _validate_dataframe_options(self):
        assert isinstance(self.path, str)
        assert isinstance(self.n_qubits, int) and self.n_qubits > 0
        
    @property    
    def get_df_with_correct_columns(self) -> pd.DataFrame:
        """Drops the undesired columns from the raw calibration data.

        Returns
        -------
        df : DataFrame
        """

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


@dataclass
class GoogleData(GoogleDataFrame):
    """Wrangles the calibration data for multi-target regression.

    Parameters
    ----------
    path : str, optional (default=None)
        Path to the csv file with calibration data.

    n_qubits : int, optional (default=5)
        Number of qubits in the experiment. Currently, supports 5 qubits.

    test_split : float, optional (default=0.3)
        The proportion of labeled examples withheld from training.

    random_state : int, RandomState instance, or None, optional (default=0)
        Determines the random number generation in the training and test
        examples split.

    remove_outliers : bool, optional (default=False)
        If True, then it removes labeled examples that are not
        within the interquartile range of the DataFrame.

    shuffle : bool, optional (default=True)
        If True, then it shuffles the DataFrame rows prior
        to splitting the DataFrame into training and test
        examples.

    See Also
    --------
    :class:`physlearn.datasets.GoogleDataFrame` : Class for representing the calibration data.

    Examples
    --------
    >>> from physlearn.datasets import GoogleData
    >>> data = GoogleData()
    >>> data.load_benchmark['X_train'].iloc[0, :3]
    qvolt5    0.003398
    qvolt6   -0.018080
    qvolt7   -0.009895
    Name: 0, dtype: float64

    References
    ----------
    - Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
      "Boosting on the shoulders of giants in quantum device calibration",
      arXiv preprint arXiv:2005.06194 (2020).
    """

    path : str = field(default=None)
    n_qubits: int = field(default=5)
    test_split: float = field(default=0.3)
    random_state: int = field(default=0)
    remove_outliers: bool = field(default=False)
    shuffle: bool = field(default=True)

    def __post_init__(self):
        if self.path is None:
            self.path = _path_to_google_data()
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
        """Get the DataFrame, then split it into training and test data.

        Returns
        -------
        X_train, X_test, y_train, and y_test : DataFrame(s)
        """

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
        """Deserializes the benchmark dataset.

        Returns
        -------
        data : dict
        """

        folder = _path_to_google_json_folder()
        return _json_load(filename=os.path.join(folder, '_{}'.format(self.n_qubits) + 'q.json'))


def load_benchmark(return_split=False) -> sklearn_train_test_split_or_dict:
    """Deserializes the benchmark dataset for the multi-target regression task.
    
    If the return split parameter is true, then the benchmark dataset is
    returned in the familiar X_train, X_test, y_train, and y_test format.

    Parameters
    ----------
    return_split : bool
        If True, then the benchmark dataset is returned in the form of
        X_train, X_test, y_train, and y_test.

    Returns
    -------
    X_train, X_test, y_train, and y_test or data : DataFrame(s) or dict

    References
    ----------
    - Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
      "Boosting on the shoulders of giants in quantum device calibration",
      arXiv preprint arXiv:2005.06194 (2020).
    """

    data = GoogleData(n_qubits=5).load_benchmark

    if return_split:
        X_train, X_test = data['X_train'].iloc[:, -5:], data['X_test'].iloc[:, -5:]
        y_train, y_test = data['y_train'].iloc[:, :5], data['y_test'].iloc[:, :5]
        return X_train, X_test, y_train, y_test
    else:
        return data
