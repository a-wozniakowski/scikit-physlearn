"""
The :mod:`physlearn.datasets.google.base` module provides the abstract
base class and the base class for representing data with pandas.
"""

# Author: Alex Wozniakowski
# License: MIT

import re

import pandas as pd

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class AbstractDataFrame(ABC):
    """Abstract base class for the supervised DataFrame."""

    @property    
    @abstractmethod
    def get_df(self):
        """Retrieves the DataFrame."""

        pass


@dataclass
class BaseDataFrame(AbstractDataFrame):
    """Base class for the supervised DataFrame.

    Parameters
    ----------
    path : str
        Path to the csv file with calibration data.

    Examples
    --------
    >>> from physlearn.datasets.google.base import BaseDataFrame
    >>> from physlearn.datasets.google.utils._helper_functions import _path_to_google_data
    >>> df = BaseDataFrame(path=_path_to_google_data())
    >>> df.get_df.iloc[:, 2]
    0     -0.008238
    1     -0.008238
    2     -0.008238
    3     -0.008238
    4     -0.008238
             ...
    675    0.007770
    676    0.007770
    677    0.007770
    678    0.007770
    679    0.007770
    Name: qubit_voltages, Length: 680, dtype: float64
    """

    path: str

    def __post_init__(self):
        self._validate_dataframe_options()

    def _validate_dataframe_options(self):
        assert isinstance(self.path, str)

    @property    
    def get_df(self) -> pd.DataFrame:
        """Reads a file into a pandas DataFrame.

        Supports comma-separated values (csv), Excel, or JSON formatted files.

        Returns
        -------
        df : DataFrame
        """

        try:
            if re.search('.csv', self.path):
                df = pd.read_csv(self.path)
            elif re.search('.xls', self.path):
                df = pd.read_excel(self.path)
            elif re.search('.xlsx', self.path):
                df = pd.read_excel(self.path)
            elif re.search('.xlsm', self.path):
                df = pd.read_excel(self.path)
            elif re.search('.xlsb', self.path):
                df = pd.read_excel(self.path)
            elif re.search('.odf', self.path):
                df = pd.read_excel(self.path)
            elif re.search('.ods', self.path):
                df = pd.read_excel(self.path)
            elif re.search('.odt', self.path):
                df = pd.read_excel(self.path)
            elif re.search('.json', self.path):
                df = pd.read_json(self.path)
        except:
            raise AssertionError('The %s does not contain a supported file format. '
                                 'Please use either a comma-seperated values '
                                 '(csv), Excel, or JSON formatted file.'
                                 % (self.path))

        return df
