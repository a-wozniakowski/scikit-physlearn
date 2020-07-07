"""
Base class for the model dictionary and a regressor mixin class.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>


from abc import ABC, abstractmethod


class AbstractModelDictionaryInterface(ABC):
    """
    Abstract base class for the model dictionary interface.
    """

    @abstractmethod
    def set_params(self, params):
        pass


class AdditionalRegressorMixin(ABC):
    """
    Mixin class for regressor object.
    """

    @abstractmethod
    def dump(self, value, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
