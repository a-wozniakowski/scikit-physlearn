"""
Machine learning package for Python.
====================================
"""

# Author: Alex Wozniakowski
# Licence: MIT

from __future__ import absolute_import


__version__ = '0.1.5'


from .supervised.interface import RegressorDictionaryInterface
from .supervised.regression import BaseRegressor, Regressor
from .pipeline import ModifiedPipeline, make_pipeline
from .loss import (LeastSquaresError, LeastAbsoluteError,
                   HuberLossFunction, QuantileLossFunction)
from .supervised.interpretation.interpret_regressor import ShapInterpret
from .supervised.model_selection.learning_curve import (LearningCurve,
                                                        plot_learning_curve)


__all__ = ['ModifiedPipeline', 'make_pipeline',
           'LeastSquaresError', 'LeastAbsoluteError',
           'HuberLossFunction', 'QuantileLossFunction',
           'BaseRegressor', 'Regressor',
           'RegressorDictionaryInterface',
           'ShapInterpret', 'LearningCurve',
           'plot_learning_curve']
