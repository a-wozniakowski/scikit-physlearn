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


dir_path = os.path.dirname(os.path.realpath(__file__))

if os.path.isfile(os.path.join(dir_path, 'VERSION.txt')):
    with open(os.path.join(dir_path, 'VERSION.txt')) as version_file:
        __version__ = version_file.read().strip()


__all__ = ['ModifiedPipeline', 'make_pipeline',
           'LeastSquaresError', 'LeastAbsoluteError',
           'HuberLossFunction', 'QuantileLossFunction',
           'BaseRegressor', 'Regressor',
           'RegressorDictionaryInterface',
           'ShapInterpret', 'LearningCurve',
           'plot_learning_curve']
