"""
Machine learning package for Python.
====================================

# Author: Alex Wozniakowski
# Licence: MIT
"""

from __future__ import absolute_import


__version__ = '0.1.4'


try:
    from .supervised.regression import BaseRegressor, Regressor
    from .supervised.interface import RegressorDictionaryInterface
    from .supervised.model_selection.bayesian_search import _bayesoptcv
    from .supervised.model_selection.learning_curve import LearningCurve, plot_learning_curve
    from .supervised.interpretation.interpret_regressor import ShapInterpret
except ImportError:
    pass

try:
    from .pipeline import _make_pipeline, ModifiedPipeline
except ImportError:
    pass

try:
    from .loss import (LeastSquaresError, LeastAbsoluteError, HuberLossFunction,
                       QuantileLossFunction)
except ImportError:
    pass


__all__ = ['BaseRegressor', 'Regressor',
           'RegressorDictionaryInterface',
           '_bayesoptcv', 'LearningCurve',
           'plot_learning_curve', 'ShapInterpret',
           '_make_pipeline', 'ModifiedPipeline',
           'LeastSquaresError', 'LeastAbsoluteError',
           'HuberLossFunction', 'QuantileLossFunction']
