"""
The :mod:`physlearn.datasets.google.base` module provides the abstract
base class and the base class for pandas data representations.
"""

# Author: Alex Wozniakowski
# License: MIT

import re

import pandas as pd

from abc import ABC, abstractmethod


class AbstractDataFrame(ABC):
    """
    Abstract base class for the supervised DataFrame.
    """

    @property    
    @abstractmethod
    def get_df(self):
        """
        Retrieves the DataFrame.
        """

        pass


class BaseDataFrame(AbstractDataFrame):
    """
    Base class for the Supervised DataFrame.
    """

    def __init__(self, path):
        self.path = path

    def _validate_dataframe_options(self):
        assert isinstance(path, str)

    @property    
    def get_df(self) -> pd.DataFrame:
        """
        Reads either a comma-separated values (csv), Excel, or JSON
        formatted file into a pandas DataFrame.
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
