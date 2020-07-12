"""
Base class for the supervised DataFrame.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>


from abc import ABC, abstractmethod

import re
import pandas as pd


class AbstractDataFrame(ABC):
    """
    Abstract base class for the supervised DataFrame.
    """

    @property    
    @abstractmethod
    def get_df(self):
        """Load DataFrame."""


class BaseDataFrame(AbstractDataFrame):
    """
    Supervised DataFrame object.
    """

    def __init__(self, path):
        assert(isinstance(path, str))
        self.path = path

    @property    
    def get_df(self):
        """Load DataFrame with csv or xlsx file format."""

        csv_file = '\.csv'
        excel_file = '\.xlsx'

        if re.search(csv_file, self.path):
            df = pd.read_csv(self.path)
        elif re.search(excel_file, self.path):
            df = pd.read_excel(self.path)
        else:
            raise AssertionError('Please use csv or xlsx file.')    
        
        return df