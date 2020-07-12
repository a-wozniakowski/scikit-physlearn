"""
Abstract base class for the model dictionary and a mixin class for regression.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>


from abc import ABC, abstractmethod


class AbstractModelDictionaryInterface(ABC):
    """
    Abstract base class for the model dictionary interface.
    """

    @abstractmethod
    def set_params(self, params):
        """Set parameters of model choice."""


class AdditionalRegressorMixin(ABC):
    """
    Mixin class for regressor object.
    """

    @abstractmethod
    def dump(self, value, filename):
        """Save a file."""

    @abstractmethod
    def load(self, filename):
        """Load a file."""

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """Fit model in supervised fashion."""
    
    @abstractmethod
    def predict(self, X):
        """Generate predictions."""
