"""
Machine learning package for Python.
====================================
"""

# Author: Alex Wozniakowski
# Licence: MIT

from __future__ import absolute_import


import os


from .supervised.interface import RegressorDictionaryInterface
from .supervised.regression import BaseRegressor, Regressor
from .pipeline import ModifiedPipeline, make_pipeline
from .loss import (LeastSquaresError, LeastAbsoluteError,
                   HuberLossFunction, QuantileLossFunction)
from .supervised.interpretation.interpret_regressor import ShapInterpret
from .supervised.model_selection.learning_curve import (LearningCurve,
                                                        plot_learning_curve)


VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
with open(VERSION_FILE) as f:
    __version__ = f.read().strip()


__all__ = ['ModifiedPipeline', 'make_pipeline',
           'LeastSquaresError', 'LeastAbsoluteError',
           'HuberLossFunction', 'QuantileLossFunction',
           'BaseRegressor', 'Regressor',
           'RegressorDictionaryInterface',
           'ShapInterpret', 'LearningCurve',
           'plot_learning_curve']
