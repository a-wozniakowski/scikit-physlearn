"""
The :mod:`physlearn.base` module provides a base class and mixin class
for the estimator dictionary and regressor amalgamation, respectively.
"""

# Author: Alex Wozniakowski
# License: MIT


from abc import ABC, abstractmethod


class AbstractEstimatorDictionaryInterface(ABC):
    """Abstract base class for the estimator dictionary interface.

    Notes
    -----
    All estimators should be retrieved from the estimator dictionary,
    thereby enabling a case-insensitive estimator API.
    """

    @abstractmethod
    def set_params(self):
        """Set the (hyper)parameters."""


class AdditionalRegressorMixin(ABC):
    """Mixin class to include with :class:`sklearn.base.RegressorMixin`.

    Notes
    -----
    The Scikit-learn regressor mixin includes a ``score`` method.
    """

    @abstractmethod
    def dump(self, value, filename):
        """Serializes the value."""

    @abstractmethod
    def load(self, filename):
        """Deserializes the file object."""

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """Fit a model in supervised fashion."""
    
    @abstractmethod
    def predict(self, X):
        """Generate predictions with a model."""
