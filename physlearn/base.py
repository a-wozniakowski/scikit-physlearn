"""
Abstract base class for the estimator dictionary and a mixin class for regression.
"""

# Author: Alex Wozniakowski
# License: MIT


from abc import ABC, abstractmethod


class AbstractEstimatorDictionaryInterface(ABC):
    """
    Abstract base class for the estimator dictionary interface.

    Notes
    -----
    All estimators should be retrieved from the estimator dictionary,
    thereby enabling a case-insensitive estimator API.
    """

    @abstractmethod
    def set_params(self):
        """Set the parameters."""


class AdditionalRegressorMixin(ABC):
    """
    An additional mixin to include with the :class:`sklearn.base.RegressorMixin 
    <https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html>`_.

    Notes
    -----
    The Scikit-learn regressor mixin includes a ``score`` method.
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
