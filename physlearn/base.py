"""
Base class for main regressor object.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

from abc import ABC, abstractmethod


class AdditionalRegressorMixin(ABC):
    """
    Mixin class for main regressor object.
    """

    @abstractmethod
    def dump(self, value, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass

    @abstractmethod
    def fit(self, X, y=None, sample_weight=None):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
