"""
Learning from Google calibration data experiment.

Calibration data was collected by Benjamin Chiaro during his
time as a graduate student at UC Santa Barbara.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

from .base import BaseDataFrame
from .utils._dataset_helper_functions import (_df_shuffle, _iqr_outlier_mask,
                                              _train_test_split, _json_dump,
                                              _json_load, _path_google_data,
                                              _path_google_json_folder)


class GoogleDataFrame(BaseDataFrame):
    """Supervised DataFrame object for Google calibration data."""

    def __init__(self, n_qubits, path):
        assert isinstance(n_qubits, int) and n_qubits > 0
        assert isinstance(path, str)
        
        self.n_qubits = n_qubits

        super().__init__(path=path)
        
    @property    
    def get_df_with_correct_columns(self):
        """Split DataFrame into train and test split."""

        # Load the DataFrame
        df = self.get_df

        if self.n_qubits == 3:
            # Select every third row, and select the relevant columns
            df = df.iloc[1::3, :].loc[:, 'qubit_voltages':' .10']

            df.columns = ['qvolt7','qvolt8','qvolt9',
                          'cvolt6', 'cvolt7', 'cvolt8',
                          'pfreq7', 'pfreq8', 'pfreq9',
                          'pcoup7', 'pcoup8',
                          'peig1', 'peig2', 'peig3',
                          'eeig1', 'eeig2', 'eeig3']
            
            df = df.drop(['cvolt6'], axis=1)
            
        elif self.n_qubits == 5:
            # Select every fifth row, and select the relevant columns
            df = df.iloc[1::5, :].loc[:, 'qubit_voltages':' .29']

            df.columns = ['qvolt5', 'qvolt6', 'qvolt7', 'qvolt8', 'qvolt9',
                          'cvolt4', 'cvolt5', 'cvolt6', 'cvolt7', 'cvolt8',
                          'pref105', 'pref106', 'pref107', 'pref108', 'pref109',
                          'precoup5', 'precoup6', 'precoup7', 'precoup8',
                          'postf105', 'post106', 'postf107', 'postf108', 'postf109',
                          'postcoup5', 'postcoup6', 'postcoup7', 'postcoup8',
                          'peig1', 'peig2', 'peig3', 'peig4', 'peig5',
                          'eeig1', 'eeig2', 'eeig3', 'eeig4', 'eeig5']
            
            df = df.drop(['cvolt4'], axis=1)
        
        return df


class GoogleData(GoogleDataFrame):
    """Supervised object for Google calibration data experiment."""

    def __init__(self, n_qubits=5, test_split=0.3, random_state=0,
                 remove_outliers=False, shuffle=True):

        assert(isinstance(test_split, float))
        assert(test_split > 0.0 and test_split < 1.0)
        assert(isinstance(random_state, int))
        assert isinstance(remove_outliers, bool)
        assert isinstance(shuffle, bool)

        self.test_split = test_split        
        self.random_state = random_state
        self.remove_outliers = remove_outliers
        self.shuffle = shuffle

        super().__init__(n_qubits=n_qubits,
                         path=_path_google_data(n_qubits=n_qubits))

    def _get_train_test_data_split(self):
        """Split DataFrame into train and test split."""

        df = self.get_df_with_correct_columns

        if self.shuffle:
            df = _df_shuffle(df)

        # Compute interquartile range for outlier removal.
        if self.remove_outliers:
            mask = _iqr_outlier_mask(df)
            df = df[~mask]

        if self.n_qubits == 3:
            qubit_coupler_voltage = ['qvolt7', 'qvolt8', 'qvolt9',
                                     'cvolt7', 'cvolt8']
            google_pred_eigenvalue = ['peig1', 'peig2', 'peig3']
            target_eigenvalue = ['eeig1', 'eeig2', 'eeig3']
        elif self.n_qubits == 5:
            qubit_coupler_voltage = ['qvolt5', 'qvolt6', 'qvolt7', 'qvolt8', 'qvolt9',
                                     'cvolt5', 'cvolt6', 'cvolt7', 'cvolt8']        
            google_pred_eigenvalue = ['peig1', 'peig2', 'peig3', 'peig4', 'peig5']
            target_eigenvalue = ['eeig1', 'eeig2', 'eeig3', 'eeig4', 'eeig5']
            mat_entry = ['postf105', 'post106', 'postf107', 'postf108', 'postf109',
                         'postcoup5', 'postcoup6', 'postcoup7', 'postcoup8']

        feature = qubit_coupler_voltage + google_pred_eigenvalue

        if self.n_qubits == 3:
            target = target_eigenvalue
        elif self.n_qubits == 5:
            target = target_eigenvalue + mat_entry

        X = df[feature]
        y = df[target]

        return _train_test_split(X, y, test_size=self.test_split,
                                 random_state=self.random_state)

    def save_train_test_data_split_json(self):
        """Save Google benchmark dataset in json file format."""

        _json_dump(train_test_data=self._get_train_test_data_split(),
                   folder=_path_google_json_folder(),
                   n_qubits=self.n_qubits)

    @property    
    def load_benchmark(self):
        """Load Google benchmark dataset in json file format."""

        return _json_load(folder=_path_google_json_folder(),
                          n_qubits=self.n_qubits)


def load_benchmark():
    """Load and return Google benchmark dataset (multi-target regression)."""

    return GoogleData(n_qubits=5).load_benchmark