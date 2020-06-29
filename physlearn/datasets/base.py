from abc import ABC, abstractmethod

import re
import pandas as pd


class AbstractDataFrame(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_df(self):
        pass


class BaseDataFrame(AbstractDataFrame):

    def __init__(self, path):
        assert(isinstance(path, str))
        self.path = path

    def get_df(self):
        csv_file = '\.csv'
        excel_file = '\.xlsx'

        if re.search(csv_file, self.path):
            df = pd.read_csv(self.path)
        elif re.search(excel_file, self.path):
            df = pd.read_excel(self.path)
        else:
            raise AssertionError('Please use csv or xlsx file.')    
        
        return df